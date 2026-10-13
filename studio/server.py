"""
MultiFace Studio — multi-face mapping on top of the FaceFusion engine.

Every person found in the target (photo or video) gets its own source face,
and everything is rendered in a single pass.
"""
import argparse
import base64
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Callable, Dict, List, Optional

STUDIO_PATH = Path(__file__).resolve().parent
ROOT_PATH = STUDIO_PATH.parent
ENGINE_PATH = ROOT_PATH / 'engine'
WORKSPACE_PATH = ROOT_PATH / 'workspace'
UPLOAD_PATH = WORKSPACE_PATH / 'uploads'
OUTPUT_PATH = WORKSPACE_PATH / 'outputs'

CLI = argparse.ArgumentParser(description = 'MultiFace Studio')
CLI.add_argument('--host', default = '127.0.0.1')
CLI.add_argument('--port', type = int, default = 7870)
CLI.add_argument('--open-browser', action = 'store_true')
CLI_ARGS = CLI.parse_args()

# facefusion resolves its modules and models relative to the working directory and parses sys.argv while building its program
os.chdir(ENGINE_PATH)
sys.path.insert(0, str(ENGINE_PATH))
sys.argv = [ 'facefusion.py', 'headless-run' ]

import cv2
import numpy
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import facefusion.choices
import facefusion.execution
from facefusion import content_analyser, download, ffmpeg, logger, process_manager, state_manager
from facefusion.args import apply_args
from facefusion.audio import create_empty_audio_frame
from facefusion.core import common_pre_check, conditional_process, processors_pre_check
from facefusion.face_analyser import get_many_faces
from facefusion.face_mapper import match_face_mappings
from facefusion.face_store import clear_static_faces
from facefusion.filesystem import is_image, is_video
from facefusion.processors.core import get_processors_modules
from facefusion.processors.modules.background_remover import core as background_remover_core
from facefusion.processors.modules.face_enhancer import choices as face_enhancer_choices
from facefusion.processors.modules.face_enhancer import core as face_enhancer_core
from facefusion.processors.modules.face_swapper import choices as face_swapper_choices
from facefusion.processors.modules.face_sticker import core as face_sticker_core
from facefusion.processors.modules.face_swapper import core as face_swapper_core
from facefusion.program import create_program
from facefusion.temp_helper import resolve_temp_frame_paths
from facefusion.types import Face, FaceMapping, Mask, VisionFrame
from facefusion.vision import count_video_frame_total, detect_image_resolution, detect_video_fps, detect_video_resolution, extract_vision_mask, fit_cover_frame, merge_vision_mask, obscure_frame, read_static_image, read_video_frame, restrict_frame
from facefusion.workflows import image_to_image, image_to_video

ENGINE_LOCK = threading.RLock()
PEOPLE_EXEMPLAR_LIMIT = 12
PEOPLE_EXEMPLAR_SPREAD = 0.08
VIDEO_SCAN_SAMPLES = 60
STICKER_MAX_SIZE = 1024
STICKER_CUTOUT_MODEL = 'modnet'
STICKER_DEFAULT_OPTIONS : Dict[str, Any] =\
{
	'scale': 1.6,
	'offset_x': 0.0,
	'offset_y': -0.12,
	'rotation': 0.0,
	'follow_rotation': True,
	'depth': 0.7,
	'mirror': False,
	'round': False,
	'cutout': False
}
STICKER_OPTION_RANGES : Dict[str, Any] =\
{
	'scale': (0.2, 5.0),
	'offset_x': (-2.0, 2.0),
	'offset_y': (-2.0, 2.0),
	'rotation': (-180.0, 180.0),
	'depth': (0.0, 1.0)
}

SESSION : Dict[str, Any] =\
{
	'target': None,
	'people': [],
	'library': [],
	'next_person_id': 1,
	'trim': [ None, None ],
	'output': None
}
TASK : Dict[str, Any] =\
{
	'id': None,
	'kind': None,
	'status': 'idle',
	'progress': 0.0,
	'stage': '',
	'message': '',
	'result': None,
	'cancel': False
}
SETTINGS : Dict[str, Any] = {}
DOWNLOAD : Dict[str, Any] =\
{
	'file_name': None,
	'path': None,
	'total': 0
}
DOWNLOAD_VISIBLE_SIZE = 2 * 1024 * 1024

FRIENDLY_SWAPPER_MODELS =\
{
	'hyperswap_1a_256': 'HyperSwap 1A',
	'hyperswap_1b_256': 'HyperSwap 1B',
	'hyperswap_1c_256': 'HyperSwap 1C',
	'inswapper_128': 'InSwapper 128',
	'inswapper_128_fp16': 'InSwapper 128 FP16',
	'simswap_256': 'SimSwap 256',
	'simswap_unofficial_512': 'SimSwap 512',
	'ghost_1_256': 'Ghost 1',
	'ghost_2_256': 'Ghost 2',
	'ghost_3_256': 'Ghost 3',
	'hififace_unofficial_256': 'HiFiFace',
	'blendswap_256': 'BlendSwap',
	'uniface_256': 'UniFace'
}


# ---------------------------------------------------------------- engine

def init_engine() -> None:
	program = create_program()
	args = vars(program.parse_args([ 'headless-run' ]))
	apply_args(args, state_manager.init_item)
	logger.init('info')

	state_manager.init_item('face_selector_mode', 'mapping')
	state_manager.init_item('face_mappings', [])
	state_manager.init_item('source_paths', [])
	state_manager.init_item('video_memory_strategy', 'tolerant')

	# facefusion defaults to flac, which QuickTime and Safari cannot play inside mp4
	if 'aac' in ffmpeg.get_available_encoder_set().get('audio'):
		state_manager.init_item('output_audio_encoder', 'aac')

	SETTINGS.update(
	{
		'face_swapper_model': state_manager.get_item('face_swapper_model'),
		'face_swapper_pixel_boost': state_manager.get_item('face_swapper_pixel_boost'),
		'face_swapper_weight': state_manager.get_item('face_swapper_weight'),
		'face_enhancer': False,
		'face_enhancer_model': state_manager.get_item('face_enhancer_model'),
		'face_enhancer_blend': state_manager.get_item('face_enhancer_blend'),
		'occlusion': False,
		'face_mask_blur': state_manager.get_item('face_mask_blur'),
		'reference_face_distance': state_manager.get_item('reference_face_distance'),
		'face_detector_model': state_manager.get_item('face_detector_model'),
		'face_detector_score': state_manager.get_item('face_detector_score'),
		'output_image_quality': 92,
		'output_video_quality': 85,
		'execution_provider': (state_manager.get_item('execution_providers') or [ 'cpu' ])[0]
	})
	apply_settings()


def run_setup() -> Dict[str, Any]:
	TASK['stage'] = 'Préparation des modèles'

	if not ensure_models(collect_wanted_processors()):
		raise TaskError('Impossible de télécharger les modèles de base (connexion internet ?).')
	return {}


def apply_settings() -> None:
	face_mappings = build_face_mappings()
	processors = []

	if any(face_mapping.get('source_paths') for face_mapping in face_mappings):
		processors.append('face_swapper')
	if SETTINGS.get('face_enhancer'):
		processors.append('face_enhancer')
	# stickers go last so no other processor paints over them
	if any(face_mapping.get('sticker_path') for face_mapping in face_mappings):
		processors.append('face_sticker')

	face_mask_types = [ 'box' ]

	if SETTINGS.get('occlusion'):
		face_mask_types.append('occlusion')

	face_detector_model = SETTINGS.get('face_detector_model')

	state_manager.set_item('processors', processors)
	state_manager.set_item('face_selector_mode', 'mapping')
	state_manager.set_item('face_swapper_model', SETTINGS.get('face_swapper_model'))
	state_manager.set_item('face_swapper_pixel_boost', SETTINGS.get('face_swapper_pixel_boost'))
	state_manager.set_item('face_swapper_weight', SETTINGS.get('face_swapper_weight'))
	state_manager.set_item('face_enhancer_model', SETTINGS.get('face_enhancer_model'))
	state_manager.set_item('face_enhancer_blend', SETTINGS.get('face_enhancer_blend'))
	state_manager.set_item('face_mask_types', face_mask_types)
	state_manager.set_item('face_mask_blur', SETTINGS.get('face_mask_blur'))
	state_manager.set_item('reference_face_distance', SETTINGS.get('reference_face_distance'))
	state_manager.set_item('face_detector_model', face_detector_model)
	state_manager.set_item('face_detector_size', facefusion.choices.face_detector_set.get(face_detector_model)[-1])
	state_manager.set_item('face_detector_score', SETTINGS.get('face_detector_score'))
	state_manager.set_item('output_image_quality', SETTINGS.get('output_image_quality'))
	state_manager.set_item('output_video_quality', SETTINGS.get('output_video_quality'))
	state_manager.set_item('execution_providers', [ SETTINGS.get('execution_provider') ])
	state_manager.set_item('face_mappings', face_mappings)
	state_manager.set_item('source_paths', collect_source_paths())
	state_manager.set_item('target_path', SESSION.get('target').get('path') if SESSION.get('target') else None)


def is_model_downloaded(module : Any, model_name : str) -> bool:
	model_set = module.create_static_model_set('full').get(model_name)

	if model_set:
		return all(os.path.isfile(source.get('path')) for source in model_set.get('sources').values())
	return False


def ensure_models(processors : Optional[List[str]] = None) -> bool:
	processors = processors or state_manager.get_item('processors')
	process_manager.check()
	try:
		return common_pre_check() and all(processor_module.pre_check() for processor_module in get_processors_modules(processors))
	finally:
		process_manager.end()


def collect_wanted_processors() -> List[str]:
	# the swapper model is fetched ahead of time, even before any face is assigned
	processors = [ 'face_swapper' ] + state_manager.get_item('processors')

	if SETTINGS.get('face_enhancer'):
		processors.append('face_enhancer')
	return list(dict.fromkeys(processors))


# ---------------------------------------------------------------- people & library

def build_face_mappings() -> List[FaceMapping]:
	face_mappings : List[FaceMapping] = []

	for person in SESSION.get('people'):
		source = find_library_item(person.get('source_id'))
		is_sticker = bool(source and source.get('kind') == 'sticker')
		face_mappings.append(
		{
			'reference_embeddings': person.get('embeddings'),
			'source_paths': [ source.get('path') ] if source and not is_sticker else [],
			'sticker_path': resolve_sticker_path(source, person.get('sticker')) if is_sticker else None,
			'sticker_options': dict(person.get('sticker'), round = False) if is_sticker else None
		})
	return face_mappings


def resolve_sticker_variant(library_item : Dict[str, Any], sticker_options : Dict[str, Any]) -> str:
	if sticker_options.get('cutout') and library_item.get('cutout_path'):
		return 'cutout'
	if sticker_options.get('round') and library_item.get('round_path'):
		return 'round'
	return 'original'


def resolve_sticker_path(library_item : Dict[str, Any], sticker_options : Dict[str, Any]) -> str:
	sticker_variant = resolve_sticker_variant(library_item, sticker_options)

	if sticker_variant == 'original':
		return library_item.get('path')
	return library_item.get(sticker_variant + '_path')


def collect_source_paths() -> List[str]:
	source_paths = []

	for person in SESSION.get('people'):
		source = find_library_item(person.get('source_id'))

		if source and source.get('kind') == 'face' and source.get('path') not in source_paths:
			source_paths.append(source.get('path'))
	return source_paths


def find_person(person_id : int) -> Optional[Dict[str, Any]]:
	for person in SESSION.get('people'):
		if person.get('id') == person_id:
			return person
	return None


def find_library_item(library_id : Optional[str]) -> Optional[Dict[str, Any]]:
	for library_item in SESSION.get('library'):
		if library_item.get('id') == library_id:
			return library_item
	return None


def get_bounding_box_area(face : Face) -> float:
	return float((face.bounding_box[2] - face.bounding_box[0]) * (face.bounding_box[3] - face.bounding_box[1]))


def create_thumbnail(vision_frame : VisionFrame, face : Face, size : int = 256) -> str:
	height, width = vision_frame.shape[:2]
	start_x, start_y, end_x, end_y = face.bounding_box
	center_x = (start_x + end_x) / 2
	center_y = (start_y + end_y) / 2
	half_side = max(end_x - start_x, end_y - start_y) * 0.8
	crop_start_x = int(max(0, center_x - half_side))
	crop_start_y = int(max(0, center_y - half_side))
	crop_end_x = int(min(width, center_x + half_side))
	crop_end_y = int(min(height, center_y + half_side))
	crop_vision_frame = vision_frame[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

	if crop_vision_frame.size == 0:
		crop_vision_frame = vision_frame
	crop_vision_frame = fit_cover_frame(crop_vision_frame, (size, size))
	return encode_data_url(crop_vision_frame, 88)


def encode_data_url(vision_frame : VisionFrame, quality : int) -> str:
	_, buffer = cv2.imencode('.jpg', vision_frame[:, :, :3], [ cv2.IMWRITE_JPEG_QUALITY, quality ])
	return 'data:image/jpeg;base64,' + base64.b64encode(buffer.tobytes()).decode()


def calculate_face_quality(face : Face) -> float:
	return get_bounding_box_area(face) * float(face.score_set.get('detector')) * (1.0 - min(abs(face.angle), 90) / 180)


def sort_faces_reading_order(faces : List[Face]) -> List[Face]:
	rows : List[List[Face]] = []

	for face in sorted(faces, key = lambda face: (face.bounding_box[1] + face.bounding_box[3]) / 2):
		face_center_y = (face.bounding_box[1] + face.bounding_box[3]) / 2
		face_height = face.bounding_box[3] - face.bounding_box[1]

		if rows:
			row_face = rows[-1][0]
			row_center_y = (row_face.bounding_box[1] + row_face.bounding_box[3]) / 2

			if abs(face_center_y - row_center_y) < face_height * 0.6:
				rows[-1].append(face)
				continue
		rows.append([ face ])

	return [ face for row in rows for face in sorted(row, key = lambda face: face.bounding_box[0]) ]


def merge_faces_into_people(vision_frame : VisionFrame, faces : List[Face], frame_number : int) -> int:
	people = SESSION.get('people')
	face_mappings : List[FaceMapping] = [ { 'reference_embeddings': person.get('embeddings'), 'source_paths': [] } for person in people ]
	match_pairs = match_face_mappings(faces, face_mappings, SETTINGS.get('reference_face_distance'))
	mapping_people = { id(face_mapping): person for face_mapping, person in zip(face_mappings, people) }
	matched_face_ids = set()
	created_total = 0

	for face, face_mapping in match_pairs:
		person = mapping_people.get(id(face_mapping))
		matched_face_ids.add(id(face))
		person['count'] += 1
		exemplar_distances = [ 1 - numpy.dot(face.embedding_norm, embedding) for embedding in person.get('embeddings') ]

		# keep a few distinct angles of each person so matching survives head turns
		if len(person.get('embeddings')) < PEOPLE_EXEMPLAR_LIMIT and min(exemplar_distances) > PEOPLE_EXEMPLAR_SPREAD:
			person['embeddings'].append(face.embedding_norm)

		face_quality = calculate_face_quality(face)

		if face_quality > person.get('quality'):
			person['quality'] = face_quality
			person['thumbnail'] = create_thumbnail(vision_frame, face)
			person['frame_number'] = frame_number

	for face in sort_faces_reading_order(faces):
		if id(face) not in matched_face_ids:
			people.append(
			{
				'id': SESSION.get('next_person_id'),
				'embeddings': [ face.embedding_norm ],
				'count': 1,
				'quality': calculate_face_quality(face),
				'thumbnail': create_thumbnail(vision_frame, face),
				'frame_number': frame_number,
				'gender': face.gender,
				'source_id': None
			})
			SESSION['next_person_id'] += 1
			created_total += 1

	return created_total


def read_target_frame(frame_number : int) -> Optional[VisionFrame]:
	target = SESSION.get('target')

	if target:
		if target.get('type') == 'image':
			return read_static_image(target.get('path'))
		return read_video_frame(target.get('path'), frame_number)
	return None


def serialize_session() -> Dict[str, Any]:
	target = SESSION.get('target')
	people = []

	for person in SESSION.get('people'):
		source = find_library_item(person.get('source_id'))
		is_sticker = bool(source and source.get('kind') == 'sticker')
		people.append(
		{
			'id': person.get('id'),
			'thumbnail': person.get('thumbnail'),
			'count': person.get('count'),
			'frame_number': person.get('frame_number'),
			'source_id': person.get('source_id'),
			'source_kind': source.get('kind') if source else None,
			'source_thumbnail': source.get('thumbnail') if source else None,
			'sticker': person.get('sticker') if is_sticker else None,
			'sticker_image': '/api/library/' + source.get('id') + '/image?variant=' + resolve_sticker_variant(source, person.get('sticker')) if is_sticker else None,
			'sticker_size': [ source.get('width'), source.get('height') ] if is_sticker else None
		})

	return\
	{
		'target': { key: value for key, value in target.items() if key != 'path' } if target else None,
		'people': people,
		'library': [ { 'id': item.get('id'), 'name': item.get('name'), 'thumbnail': item.get('thumbnail'), 'kind': item.get('kind'), 'has_alpha': item.get('has_alpha') } for item in SESSION.get('library') ],
		'trim': SESSION.get('trim'),
		'output': SESSION.get('output'),
		'settings': SETTINGS,
		'task': serialize_task()
	}


def serialize_task() -> Dict[str, Any]:
	task = { key: value for key, value in TASK.items() if key != 'cancel' }
	task['download'] = serialize_download()
	return task


def serialize_download() -> Optional[Dict[str, Any]]:
	download_path = DOWNLOAD.get('path')
	download_total = DOWNLOAD.get('total')

	# hash files are a few bytes, only real models deserve a progress card
	if download_path and download_total > DOWNLOAD_VISIBLE_SIZE:
		download_received = os.path.getsize(download_path) if os.path.isfile(download_path) else 0
		return\
		{
			'name': DOWNLOAD.get('file_name').rsplit('.', 1)[0],
			'received': download_received,
			'total': download_total,
			'progress': min(1.0, download_received / download_total)
		}
	return None


def install_download_hooks() -> None:
	original_conditional_download = download.conditional_download

	def conditional_download(download_directory_path : str, urls : List[str]) -> None:
		for url in urls:
			file_name = os.path.basename(urlparse(url).path)
			DOWNLOAD.update(
			{
				'file_name': file_name,
				'path': os.path.join(download_directory_path, file_name),
				'total': download.get_static_download_size(url)
			})
			try:
				original_conditional_download(download_directory_path, [ url ])
			finally:
				DOWNLOAD.update({ 'file_name': None, 'path': None, 'total': 0 })

	# download.py resolves conditional_download at call time, so the hashes and sources helpers pick this up
	download.conditional_download = conditional_download


def save_upload(upload : UploadFile, folder : Path) -> Path:
	folder.mkdir(parents = True, exist_ok = True)
	file_name = Path(upload.filename or 'upload').name
	file_path = folder / (uuid.uuid4().hex[:8] + '_' + file_name)

	with open(file_path, 'wb') as file:
		shutil.copyfileobj(upload.file, file)
	return file_path


def ensure_idle() -> None:
	if TASK.get('status') == 'running':
		raise HTTPException(409, 'Une tâche est déjà en cours.')


# ---------------------------------------------------------------- tasks

def start_task(kind : str, runner : Callable[[], Optional[Dict[str, Any]]]) -> Dict[str, Any]:
	ensure_idle()
	TASK.update(
	{
		'id': uuid.uuid4().hex[:8],
		'kind': kind,
		'status': 'running',
		'progress': 0.0,
		'stage': '',
		'message': '',
		'result': None,
		'cancel': False
	})

	def run() -> None:
		with ENGINE_LOCK:
			try:
				result = runner()

				if TASK.get('cancel'):
					TASK.update({ 'status': 'cancelled', 'stage': 'Annulé' })
				else:
					TASK.update({ 'status': 'done', 'progress': 1.0, 'result': result })
			except TaskError as exception:
				TASK.update({ 'status': 'error', 'message': str(exception) })
			except Exception as exception:
				logger.error(repr(exception), __name__)
				TASK.update({ 'status': 'error', 'message': 'Erreur inattendue : ' + str(exception) })

	threading.Thread(target = run, daemon = True).start()
	return serialize_task()


class TaskError(Exception):
	pass


def run_video_scan(samples : int) -> Dict[str, Any]:
	target = SESSION.get('target')
	frame_start, frame_end = resolve_trim()
	frame_numbers = sorted(set(numpy.linspace(frame_start, max(frame_start, frame_end - 1), samples).astype(int).tolist()))
	created_total = 0
	TASK['stage'] = 'Analyse des visages'

	for index, frame_number in enumerate(frame_numbers):
		if TASK.get('cancel'):
			break
		vision_frame = read_video_frame(target.get('path'), frame_number)

		if vision_frame is not None:
			created_total += merge_faces_into_people(vision_frame, get_many_faces([ vision_frame ]), frame_number)
		TASK['progress'] = (index + 1) / len(frame_numbers)
		TASK['message'] = str(len(SESSION.get('people'))) + ' personne(s) trouvée(s)'

	SESSION['people'].sort(key = lambda person: person.get('count'), reverse = True)
	apply_settings()
	return { 'created': created_total }


def resolve_trim() -> List[int]:
	target = SESSION.get('target')
	frame_total = target.get('frame_total') or 1
	trim_start, trim_end = SESSION.get('trim')
	trim_start = max(0, min(trim_start or 0, frame_total - 1))
	trim_end = max(trim_start + 1, min(trim_end or frame_total, frame_total))
	return [ trim_start, trim_end ]


def run_render() -> Dict[str, Any]:
	target = SESSION.get('target')
	target_path = Path(target.get('path'))
	output_stem = re.sub(r'[^\w\-. ]', '_', target.get('name').rsplit('.', 1)[0]).strip() or 'resultat'
	output_name = output_stem + '_multiface_' + time.strftime('%Y%m%d-%H%M%S') + target_path.suffix
	output_path = OUTPUT_PATH / output_name
	OUTPUT_PATH.mkdir(parents = True, exist_ok = True)

	TASK['stage'] = 'Vérification des modèles'
	apply_settings()
	state_manager.set_item('target_path', str(target_path))
	state_manager.set_item('output_path', str(output_path))

	if target.get('type') == 'video':
		trim_start, trim_end = resolve_trim()
		state_manager.set_item('trim_frame_start', trim_start if trim_start > 0 else None)
		state_manager.set_item('trim_frame_end', trim_end if trim_end < target.get('frame_total') else None)
		state_manager.set_item('output_video_fps', target.get('fps'))
	else:
		state_manager.set_item('trim_frame_start', None)
		state_manager.set_item('trim_frame_end', None)

	if not ensure_models():
		raise TaskError('Impossible de télécharger les modèles nécessaires (connexion internet ?).')

	TASK['stage'] = 'Préparation'
	error_code = conditional_process()

	if error_code == 2:
		raise TaskError('Aucune source valide : ajoutez au moins un visage net ou une image.')
	if error_code == 3:
		raise TaskError('Le contenu a été refusé par l\'analyseur de contenu de FaceFusion.')
	if error_code == 4 or TASK.get('cancel'):
		return {}
	if error_code > 0 or not output_path.is_file():
		raise TaskError('Le rendu a échoué, consultez le terminal pour les détails.')

	SESSION['output'] =\
	{
		'name': output_name,
		'type': target.get('type'),
		'url': '/api/output/' + output_name
	}
	return SESSION.get('output')


def wrap_stage(module : Any, function_name : str, stage : str, progress_start : float) -> None:
	original_function = getattr(module, function_name)

	def wrapper(*args : Any, **kwargs : Any) -> Any:
		TASK['stage'] = stage
		TASK['progress'] = max(TASK.get('progress'), progress_start)
		return original_function(*args, **kwargs)

	setattr(module, function_name, wrapper)


def install_progress_hooks() -> None:
	original_process_video = image_to_video.process_video
	original_process_temp_frame = image_to_video.process_temp_frame
	frame_counter = { 'done': 0, 'total': 1 }
	frame_counter_lock = threading.Lock()

	def process_video() -> Any:
		frame_counter['done'] = 0
		frame_counter['total'] = max(1, len(resolve_temp_frame_paths(state_manager.get_item('target_path'))))
		TASK['stage'] = 'Remplacement des visages'
		return original_process_video()

	def process_temp_frame(temp_frame_path : str, frame_number : int) -> bool:
		result = original_process_temp_frame(temp_frame_path, frame_number)

		with frame_counter_lock:
			frame_counter['done'] += 1
			TASK['progress'] = 0.1 + 0.8 * frame_counter.get('done') / frame_counter.get('total')
			TASK['message'] = str(frame_counter.get('done')) + ' / ' + str(frame_counter.get('total')) + ' images'
		return result

	image_to_video.process_video = process_video
	image_to_video.process_temp_frame = process_temp_frame
	wrap_stage(image_to_video, 'setup', 'Analyse du contenu', 0.0)
	wrap_stage(image_to_video, 'extract_frames', 'Extraction des images', 0.03)
	wrap_stage(image_to_video, 'merge_frames', 'Assemblage de la vidéo', 0.9)
	wrap_stage(image_to_video, 'restore_audio', 'Restauration de l\'audio', 0.96)
	wrap_stage(image_to_image, 'setup', 'Analyse du contenu', 0.0)
	wrap_stage(image_to_image, 'prepare_image', 'Préparation', 0.1)
	wrap_stage(image_to_image, 'process_image', 'Remplacement des visages', 0.2)
	wrap_stage(image_to_image, 'finalize_image', 'Finalisation', 0.9)


# ---------------------------------------------------------------- api

app = FastAPI(title = 'MultiFace Studio')


@app.middleware('http')
async def revalidate_interface(request : Any, call_next : Callable[[Any], Any]) -> Any:
	response = await call_next(request)

	# the interface changes with updates, browsers must not keep a stale copy
	if not request.url.path.startswith('/api/'):
		response.headers['Cache-Control'] = 'no-cache'
	return response


class ScanPayload(BaseModel):
	mode : str = 'frame'
	frame_number : int = 0
	samples : int = VIDEO_SCAN_SAMPLES


class AssignPayload(BaseModel):
	source_id : Optional[str] = None


class TrimPayload(BaseModel):
	start : Optional[int] = None
	end : Optional[int] = None


@app.get('/api/state')
def get_state() -> Dict[str, Any]:
	return serialize_session()


@app.get('/api/options')
def get_options() -> Dict[str, Any]:
	return\
	{
		'face_swapper_models': [ { 'value': model, 'label': FRIENDLY_SWAPPER_MODELS.get(model, model), 'downloaded': is_model_downloaded(face_swapper_core, model) } for model in face_swapper_choices.face_swapper_models ],
		'face_swapper_pixel_boosts': face_swapper_choices.face_swapper_set,
		'face_enhancer_models': [ { 'value': model, 'label': model, 'downloaded': is_model_downloaded(face_enhancer_core, model) } for model in face_enhancer_choices.face_enhancer_models ],
		'face_detector_models': facefusion.choices.face_detector_models,
		'execution_providers': facefusion.execution.get_available_execution_providers(),
		'platform': sys.platform
	}


@app.post('/api/target')
def upload_target(file : UploadFile = File(...)) -> Dict[str, Any]:
	ensure_idle()
	target_path = save_upload(file, UPLOAD_PATH / 'targets')

	with ENGINE_LOCK:
		if is_image(str(target_path)):
			width, height = detect_image_resolution(str(target_path))
			target = { 'type': 'image', 'frame_total': 1, 'fps': None }
		elif is_video(str(target_path)):
			width, height = detect_video_resolution(str(target_path))
			target = { 'type': 'video', 'frame_total': count_video_frame_total(str(target_path)), 'fps': detect_video_fps(str(target_path)) }
		else:
			target_path.unlink(missing_ok = True)
			raise HTTPException(415, 'Format non supporté. Utilisez une photo (jpg, png, webp…) ou une vidéo (mp4, mov…).')

		target.update(
		{
			'id': uuid.uuid4().hex[:8],
			'name': Path(file.filename or target_path.name).name,
			'path': str(target_path),
			'width': width,
			'height': height
		})
		SESSION.update({ 'target': target, 'people': [], 'trim': [ None, None ], 'output': None })
		clear_static_faces()

		vision_frame = read_target_frame(0)
		if vision_frame is not None:
			merge_faces_into_people(vision_frame, get_many_faces([ vision_frame ]), 0)
		apply_settings()
	return serialize_session()


@app.get('/api/frame')
def get_frame(frame_number : int = 0, max_size : int = 1920) -> Response:
	vision_frame = read_target_frame(frame_number)

	if vision_frame is None:
		raise HTTPException(404, 'Aucune image')
	vision_frame = restrict_frame(vision_frame, (max_size, max_size))
	_, buffer = cv2.imencode('.jpg', vision_frame, [ cv2.IMWRITE_JPEG_QUALITY, 88 ])
	return Response(buffer.tobytes(), media_type = 'image/jpeg', headers = { 'Cache-Control': 'max-age=3600' })


@app.get('/api/faces')
def get_faces(frame_number : int = 0) -> Dict[str, Any]:
	ensure_idle()

	with ENGINE_LOCK:
		vision_frame = read_target_frame(frame_number)

		if vision_frame is None:
			return { 'faces': [] }
		height, width = vision_frame.shape[:2]
		faces = get_many_faces([ vision_frame ])
		face_mappings : List[FaceMapping] = [ { 'reference_embeddings': person.get('embeddings'), 'source_paths': [] } for person in SESSION.get('people') ]
		mapping_person_ids = { id(face_mapping): person.get('id') for face_mapping, person in zip(face_mappings, SESSION.get('people')) }
		person_ids = {}

		for face, face_mapping in match_face_mappings(faces, face_mappings, SETTINGS.get('reference_face_distance')):
			person_ids[id(face)] = mapping_person_ids.get(id(face_mapping))

	return\
	{
		'faces':
		[
			{
				'box': [ float(face.bounding_box[0] / width), float(face.bounding_box[1] / height), float(face.bounding_box[2] / width), float(face.bounding_box[3] / height) ],
				'anchor': serialize_anchor(face, width, height),
				'person_id': person_ids.get(id(face))
			}
			for face in faces
		]
	}


def serialize_anchor(face : Face, width : int, height : int) -> Dict[str, float]:
	center_x, center_y, face_size, face_angle, face_pitch, face_yaw = face_sticker_core.calculate_sticker_anchor(face)
	return { 'x': center_x / width, 'y': center_y / height, 'size': face_size / width, 'angle': face_angle, 'pitch': face_pitch, 'yaw': face_yaw }


@app.post('/api/scan')
def scan(payload : ScanPayload) -> Dict[str, Any]:
	ensure_idle()
	target = SESSION.get('target')

	if not target:
		raise HTTPException(400, 'Ajoutez d\'abord une photo ou une vidéo.')

	if payload.mode == 'video' and target.get('type') == 'video':
		samples = max(4, min(400, payload.samples))
		start_task('scan', lambda: run_video_scan(samples))
		return serialize_session()

	with ENGINE_LOCK:
		vision_frame = read_target_frame(payload.frame_number)

		if vision_frame is not None:
			created_total = merge_faces_into_people(vision_frame, get_many_faces([ vision_frame ]), payload.frame_number)
		apply_settings()
	session = serialize_session()
	session['created'] = created_total if vision_frame is not None else 0
	return session


@app.post('/api/people/reset')
def reset_people() -> Dict[str, Any]:
	ensure_idle()

	with ENGINE_LOCK:
		SESSION['people'] = []
		apply_settings()
	return serialize_session()


@app.delete('/api/people/{person_id}')
def delete_person(person_id : int) -> Dict[str, Any]:
	ensure_idle()

	with ENGINE_LOCK:
		SESSION['people'] = [ person for person in SESSION.get('people') if person.get('id') != person_id ]
		apply_settings()
	return serialize_session()


@app.post('/api/people/{person_id}/source')
def assign_source(person_id : int, payload : AssignPayload) -> Dict[str, Any]:
	ensure_idle()
	person = find_person(person_id)
	source_id = payload.source_id or None

	if not person:
		raise HTTPException(404, 'Personne introuvable')
	if source_id and not find_library_item(source_id):
		raise HTTPException(404, 'Source introuvable')

	with ENGINE_LOCK:
		assign_library_item(person, find_library_item(source_id))
		apply_settings()
	return serialize_session()


def assign_library_item(person : Dict[str, Any], library_item : Optional[Dict[str, Any]]) -> None:
	person['source_id'] = library_item.get('id') if library_item else None

	if library_item and library_item.get('kind') == 'sticker':
		sticker_options = dict(STICKER_DEFAULT_OPTIONS, **(person.get('sticker') or {}))
		# cut-out images keep their own shape, plain photos get a soft round cut
		sticker_options['round'] = not library_item.get('has_alpha')
		person['sticker'] = sticker_options


@app.post('/api/people/{person_id}/sticker')
def update_sticker(person_id : int, payload : Dict[str, Any]) -> Dict[str, Any]:
	ensure_idle()
	person = find_person(person_id)

	if not person or not person.get('sticker'):
		raise HTTPException(404, 'Aucune image associée à cette personne')

	with ENGINE_LOCK:
		if payload.get('reset'):
			source = find_library_item(person.get('source_id'))
			person['sticker'] = dict(STICKER_DEFAULT_OPTIONS, round = not (source and source.get('has_alpha')))

		for key, value in payload.items():
			if key in STICKER_OPTION_RANGES:
				minimum, maximum = STICKER_OPTION_RANGES.get(key)
				person['sticker'][key] = max(minimum, min(maximum, float(value)))
			if key in [ 'follow_rotation', 'mirror', 'round', 'cutout' ]:
				person['sticker'][key] = bool(value)

		if person.get('sticker').get('cutout'):
			source = find_library_item(person.get('source_id'))

			if not create_sticker_cutout(source):
				person['sticker']['cutout'] = False
				apply_settings()
				raise HTTPException(503, 'Le détourage a échoué (modèle indisponible ?).')
		apply_settings()
	return serialize_session()


def create_sticker_cutout(library_item : Dict[str, Any]) -> bool:
	if library_item.get('cutout_path') and os.path.isfile(library_item.get('cutout_path')):
		return True

	state_manager.set_item('background_remover_model', STICKER_CUTOUT_MODEL)

	if not ensure_models([ 'background_remover' ]):
		return False

	sticker_frame = face_sticker_core.read_sticker_frame.__wrapped__(library_item.get('path'))
	sticker_height, sticker_width = sticker_frame.shape[:2]
	# the matting model sees transparent padding as black, which it treats as background
	sticker_alpha = sticker_frame[:, :, 3:4].astype(numpy.float32) / 255
	composite_frame = (sticker_frame[:, :, :3].astype(numpy.float32) * sticker_alpha).astype(numpy.uint8)
	cutout_mask = background_remover_core.normalize_vision_mask(background_remover_core.forward(background_remover_core.prepare_temp_frame(composite_frame)))
	cutout_mask = cv2.resize(cutout_mask, (sticker_width, sticker_height)).astype(numpy.float32) / 255
	cutout_mask *= create_head_mask(composite_frame)
	cutout_mask *= create_border_mask(sticker_height, sticker_width)
	cutout_frame = sticker_frame.copy()
	cutout_frame[:, :, 3] = (sticker_alpha[:, :, 0] * cutout_mask * 255).clip(0, 255).astype(numpy.uint8)

	cutout_path = library_item.get('path').rsplit('.', 1)[0] + '.cutout.png'
	cv2.imwrite(cutout_path, cutout_frame)
	library_item['cutout_path'] = cutout_path
	return True


def create_round_frame(sticker_frame : VisionFrame) -> VisionFrame:
	# the oval hugs the visible pixels, not the canvas, so padded or tight portraits still get a real round cut
	sticker_alpha = sticker_frame[:, :, 3]
	visible_rows = numpy.where(sticker_alpha.max(axis = 1) > 10)[0]
	visible_columns = numpy.where(sticker_alpha.max(axis = 0) > 10)[0]

	if not visible_rows.size or not visible_columns.size:
		return sticker_frame

	start_y, end_y = visible_rows[0], visible_rows[-1] + 1
	start_x, end_x = visible_columns[0], visible_columns[-1] + 1
	round_mask = numpy.zeros(sticker_alpha.shape, dtype = numpy.float32)
	cv2.ellipse(round_mask, (int((start_x + end_x) / 2), int((start_y + end_y) / 2)), (int((end_x - start_x) * 0.47), int((end_y - start_y) * 0.47)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
	round_mask = cv2.GaussianBlur(round_mask, (0, 0), max(end_x - start_x, end_y - start_y) * 0.015)
	round_frame = sticker_frame.copy()
	round_frame[:, :, 3] = (sticker_alpha * round_mask).astype(numpy.uint8)
	return round_frame


def create_border_mask(frame_height : int, frame_width : int) -> Mask:
	# hair cut by the photo frame fades out instead of ending on a straight line
	border_size = max(frame_height, frame_width) * 0.05
	rows = numpy.arange(frame_height, dtype = numpy.float32)
	columns = numpy.arange(frame_width, dtype = numpy.float32)
	row_mask = numpy.clip(numpy.minimum(rows, frame_height - 1 - rows) / border_size, 0, 1)
	column_mask = numpy.clip(numpy.minimum(columns, frame_width - 1 - columns) / border_size, 0, 1)
	return numpy.minimum(row_mask[:, None], column_mask[None, :])


def create_head_mask(vision_frame : VisionFrame) -> Mask:
	# keep the head and a bit of neck: everything below the chin fades out, shoulders included
	frame_height, frame_width = vision_frame.shape[:2]
	head_mask = numpy.ones((frame_height, frame_width), dtype = numpy.float32)
	faces = get_many_faces([ numpy.ascontiguousarray(vision_frame) ])

	if faces:
		face = max(faces, key = get_bounding_box_area)
		face_height = face.bounding_box[3] - face.bounding_box[1]
		fade_end = min(face.bounding_box[3] + face_height * 0.3, frame_height * 0.97)
		fade_start = min(face.bounding_box[3] + face_height * 0.02, fade_end - 1)
		rows = numpy.arange(frame_height, dtype = numpy.float32)
		head_mask *= numpy.interp(rows, [ fade_start, fade_end ], [ 1.0, 0.0 ])[:, None]
	return head_mask


@app.post('/api/people/assign-in-order')
def assign_in_order(kind : str = 'face') -> Dict[str, Any]:
	ensure_idle()

	with ENGINE_LOCK:
		library_items = [ library_item for library_item in SESSION.get('library') if library_item.get('kind') == kind ]

		for person, library_item in zip(SESSION.get('people'), library_items):
			assign_library_item(person, library_item)
		apply_settings()
	return serialize_session()


@app.post('/api/people/clear-sources')
def clear_sources() -> Dict[str, Any]:
	ensure_idle()

	with ENGINE_LOCK:
		for person in SESSION.get('people'):
			person['source_id'] = None
		apply_settings()
	return serialize_session()


@app.post('/api/library')
def upload_library(files : List[UploadFile] = File(...), kind : str = 'face') -> Dict[str, Any]:
	ensure_idle()
	added = []
	rejected = []

	with ENGINE_LOCK:
		for upload in files:
			if kind == 'sticker':
				library_item = create_sticker_item(upload)

				if library_item:
					SESSION['library'].append(library_item)
					added.append(library_item.get('id'))
				else:
					rejected.append(upload.filename)
				continue

			source_path = save_upload(upload, UPLOAD_PATH / 'sources')
			source_vision_frame = read_static_image(str(source_path)) if is_image(str(source_path)) else None
			source_faces = get_many_faces([ source_vision_frame ]) if source_vision_frame is not None else []

			if not source_faces:
				source_path.unlink(missing_ok = True)
				rejected.append(upload.filename)
				continue

			source_face = max(source_faces, key = get_bounding_box_area)
			library_item =\
			{
				'id': uuid.uuid4().hex[:8],
				'kind': 'face',
				'name': Path(upload.filename or source_path.name).name,
				'path': str(source_path),
				'thumbnail': create_thumbnail(source_vision_frame, source_face)
			}
			SESSION['library'].append(library_item)
			added.append(library_item.get('id'))
		apply_settings()

	session = serialize_session()
	session.update({ 'added': added, 'rejected': rejected })
	return session


def create_sticker_item(upload : UploadFile) -> Optional[Dict[str, Any]]:
	upload_path = save_upload(upload, UPLOAD_PATH / 'stickers')
	# read uncached: the normalized png may overwrite this very path
	sticker_frame = face_sticker_core.read_sticker_frame.__wrapped__(str(upload_path))

	if sticker_frame is None:
		upload_path.unlink(missing_ok = True)
		return None

	has_alpha = bool((sticker_frame[:, :, 3] < 250).any())

	# a plain photo of someone is meant as their head, not the whole picture
	if not has_alpha:
		sticker_frame = crop_sticker_to_head(sticker_frame)

	sticker_height, sticker_width = sticker_frame.shape[:2]
	sticker_scale = min(1.0, STICKER_MAX_SIZE / max(sticker_width, sticker_height))

	if sticker_scale < 1:
		sticker_frame = cv2.resize(sticker_frame, (round(sticker_width * sticker_scale), round(sticker_height * sticker_scale)), interpolation = cv2.INTER_AREA)

	sticker_path = upload_path.with_suffix('.png')
	cv2.imwrite(str(sticker_path), sticker_frame)
	round_path = str(sticker_path).rsplit('.', 1)[0] + '.round.png'
	cv2.imwrite(round_path, create_round_frame(sticker_frame))

	if sticker_path != upload_path:
		upload_path.unlink(missing_ok = True)
	return\
	{
		'id': uuid.uuid4().hex[:8],
		'kind': 'sticker',
		'name': Path(upload.filename or sticker_path.name).name,
		'path': str(sticker_path),
		'round_path': round_path,
		'thumbnail': create_sticker_thumbnail(sticker_frame),
		'has_alpha': has_alpha,
		'width': sticker_frame.shape[1],
		'height': sticker_frame.shape[0]
	}


def crop_sticker_to_head(sticker_frame : VisionFrame) -> VisionFrame:
	sticker_faces = get_many_faces([ numpy.ascontiguousarray(sticker_frame[:, :, :3]) ])

	if not sticker_faces:
		return sticker_frame

	sticker_height, sticker_width = sticker_frame.shape[:2]
	start_x, start_y, end_x, end_y = max(sticker_faces, key = get_bounding_box_area).bounding_box
	center_x = (start_x + end_x) / 2
	center_y = (start_y + end_y) / 2 - (end_y - start_y) * 0.08
	face_size = max(end_x - start_x, end_y - start_y)
	# twice the face when the photo allows it, never padding a tight portrait so the round cut bites real pixels
	crop_size = int(round(max(face_size * 1.25, min(face_size * 2, sticker_width, sticker_height))))
	crop_start_x = int(round(center_x - crop_size / 2))
	crop_start_y = int(round(center_y - crop_size / 2))

	if crop_size <= sticker_width:
		crop_start_x = max(0, min(crop_start_x, sticker_width - crop_size))
	if crop_size <= sticker_height:
		crop_start_y = max(0, min(crop_start_y, sticker_height - crop_size))
	# only a photo smaller than the crop gets transparent padding
	crop_frame = numpy.zeros((crop_size, crop_size, 4), dtype = numpy.uint8)
	inside_start_x = max(0, crop_start_x)
	inside_start_y = max(0, crop_start_y)
	inside_end_x = min(sticker_width, crop_start_x + crop_size)
	inside_end_y = min(sticker_height, crop_start_y + crop_size)
	crop_frame[inside_start_y - crop_start_y:inside_end_y - crop_start_y, inside_start_x - crop_start_x:inside_end_x - crop_start_x] = sticker_frame[inside_start_y:inside_end_y, inside_start_x:inside_end_x]
	return crop_frame


def create_sticker_thumbnail(sticker_frame : VisionFrame, size : int = 256) -> str:
	sticker_height, sticker_width = sticker_frame.shape[:2]
	thumbnail_scale = size / max(sticker_width, sticker_height)
	thumbnail_width = max(1, round(sticker_width * thumbnail_scale))
	thumbnail_height = max(1, round(sticker_height * thumbnail_scale))
	thumbnail_frame = numpy.zeros((size, size, 4), dtype = numpy.uint8)
	start_x = (size - thumbnail_width) // 2
	start_y = (size - thumbnail_height) // 2
	thumbnail_frame[start_y:start_y + thumbnail_height, start_x:start_x + thumbnail_width] = cv2.resize(sticker_frame, (thumbnail_width, thumbnail_height), interpolation = cv2.INTER_AREA)
	_, buffer = cv2.imencode('.png', thumbnail_frame)
	return 'data:image/png;base64,' + base64.b64encode(buffer.tobytes()).decode()


@app.get('/api/library/{library_id}/image')
def get_library_image(library_id : str, variant : str = 'original') -> FileResponse:
	library_item = find_library_item(library_id)

	if not library_item:
		raise HTTPException(404, 'Image introuvable')
	if variant in [ 'round', 'cutout' ] and library_item.get(variant + '_path'):
		return FileResponse(library_item.get(variant + '_path'))
	return FileResponse(library_item.get('path'))


@app.delete('/api/library/{library_id}')
def delete_library_item(library_id : str) -> Dict[str, Any]:
	ensure_idle()

	with ENGINE_LOCK:
		SESSION['library'] = [ item for item in SESSION.get('library') if item.get('id') != library_id ]

		for person in SESSION.get('people'):
			if person.get('source_id') == library_id:
				person['source_id'] = None
		apply_settings()
	return serialize_session()


@app.post('/api/settings')
def update_settings(payload : Dict[str, Any]) -> Dict[str, Any]:
	ensure_idle()
	validators : Dict[str, Callable[[Any], bool]] =\
	{
		'face_swapper_model': lambda value: value in face_swapper_choices.face_swapper_models,
		'face_swapper_pixel_boost': lambda value: isinstance(value, str),
		'face_swapper_weight': lambda value: 0 <= float(value) <= 1,
		'face_enhancer': lambda value: isinstance(value, bool),
		'face_enhancer_model': lambda value: value in face_enhancer_choices.face_enhancer_models,
		'face_enhancer_blend': lambda value: 0 <= int(value) <= 100,
		'occlusion': lambda value: isinstance(value, bool),
		'face_mask_blur': lambda value: 0 <= float(value) <= 1,
		'reference_face_distance': lambda value: 0.05 <= float(value) <= 0.95,
		'face_detector_model': lambda value: value in facefusion.choices.face_detector_models,
		'face_detector_score': lambda value: 0.05 <= float(value) <= 0.95,
		'output_image_quality': lambda value: 10 <= int(value) <= 100,
		'output_video_quality': lambda value: 10 <= int(value) <= 100,
		'execution_provider': lambda value: value in facefusion.execution.get_available_execution_providers()
	}

	with ENGINE_LOCK:
		for key, value in payload.items():
			if key in validators and validators[key](value):
				SETTINGS[key] = value

		pixel_boost_choices = face_swapper_choices.face_swapper_set.get(SETTINGS.get('face_swapper_model'))

		if SETTINGS.get('face_swapper_pixel_boost') not in pixel_boost_choices:
			SETTINGS['face_swapper_pixel_boost'] = pixel_boost_choices[0]

		if 'face_detector_model' in payload or 'face_detector_score' in payload:
			clear_static_faces()
		apply_settings()

		# fetch freshly picked models right away instead of on the first preview
		if any(key in payload for key in [ 'face_detector_model', 'face_swapper_model', 'face_enhancer', 'face_enhancer_model', 'occlusion' ]) and not ensure_models(collect_wanted_processors()):
			raise HTTPException(503, 'Impossible de télécharger le modèle (connexion internet ?).')
	return serialize_session()


@app.post('/api/trim')
def update_trim(payload : TrimPayload) -> Dict[str, Any]:
	ensure_idle()
	SESSION['trim'] = [ payload.start, payload.end ]
	return serialize_session()


@app.get('/api/preview')
def get_preview(frame_number : int = 0, max_size : int = 1600, stickers : int = 1) -> Response:
	ensure_idle()

	with ENGINE_LOCK:
		vision_frame = read_target_frame(frame_number)

		if vision_frame is None:
			raise HTTPException(404, 'Aucune image')
		apply_settings()

		if not ensure_models():
			raise HTTPException(503, 'Impossible de télécharger les modèles nécessaires.')

		target_vision_frame = restrict_frame(vision_frame, (max_size, max_size))
		temp_vision_frame = target_vision_frame.copy()
		temp_vision_mask = extract_vision_mask(temp_vision_frame)
		flagged = content_analyser.analyse_frame(target_vision_frame)

		if flagged:
			temp_vision_frame = obscure_frame(temp_vision_frame)
		else:
			preview_processors = [ processor for processor in state_manager.get_item('processors') if stickers or processor != 'face_sticker' ]

			for processor_module in get_processors_modules(preview_processors):
				logger.disable()
				ready = processor_module.pre_process('preview')
				logger.enable()

				if ready:
					temp_vision_frame, temp_vision_mask = processor_module.process_frame(
					{
						'reference_vision_frame': target_vision_frame,
						'source_audio_frame': create_empty_audio_frame(),
						'source_voice_frame': create_empty_audio_frame(),
						'source_vision_frames': [],
						'target_vision_frame': target_vision_frame,
						'temp_vision_frame': temp_vision_frame,
						'temp_vision_mask': temp_vision_mask
					})
			temp_vision_frame = merge_vision_mask(temp_vision_frame, temp_vision_mask)[:, :, :3]

	_, buffer = cv2.imencode('.jpg', temp_vision_frame, [ cv2.IMWRITE_JPEG_QUALITY, 90 ])
	return Response(buffer.tobytes(), media_type = 'image/jpeg', headers = { 'Cache-Control': 'no-store', 'X-Content-Flagged': '1' if flagged else '0' })


@app.post('/api/render')
def render() -> Dict[str, Any]:
	target = SESSION.get('target')

	if not target:
		raise HTTPException(400, 'Ajoutez d\'abord une photo ou une vidéo.')
	if not any(person.get('source_id') for person in SESSION.get('people')):
		raise HTTPException(400, 'Associez au moins un visage ou une image à une personne.')
	start_task('render', run_render)
	return serialize_session()


@app.get('/api/task')
def get_task() -> Dict[str, Any]:
	return serialize_task()


@app.get('/api/download')
def get_download() -> Dict[str, Any]:
	return { 'download': serialize_download() }


@app.post('/api/task/cancel')
def cancel_task() -> Dict[str, Any]:
	if TASK.get('status') == 'running':
		TASK['cancel'] = True
		TASK['stage'] = 'Annulation…'

		if TASK.get('kind') == 'render':
			process_manager.stop()
	return serialize_task()


@app.get('/api/output/{output_name}')
def get_output(output_name : str) -> FileResponse:
	output_path = (OUTPUT_PATH / output_name).resolve()

	if output_path.parent != OUTPUT_PATH.resolve() or not output_path.is_file():
		raise HTTPException(404, 'Fichier introuvable')
	return FileResponse(output_path, filename = output_name, content_disposition_type = 'inline')


@app.post('/api/reveal')
def reveal_output() -> Dict[str, Any]:
	output = SESSION.get('output')
	output_path = OUTPUT_PATH / output.get('name') if output else None
	OUTPUT_PATH.mkdir(parents = True, exist_ok = True)

	if sys.platform == 'darwin':
		subprocess.Popen([ 'open', '-R', str(output_path) ] if output_path else [ 'open', str(OUTPUT_PATH) ])
	elif sys.platform == 'win32':
		subprocess.Popen([ 'explorer', '/select,', str(output_path) ] if output_path else [ 'explorer', str(OUTPUT_PATH) ])
	else:
		subprocess.Popen([ 'xdg-open', str(OUTPUT_PATH) ])
	return { 'ok': True }


@app.exception_handler(HTTPException)
def handle_http_exception(_ : Any, exception : HTTPException) -> JSONResponse:
	return JSONResponse({ 'error': exception.detail }, status_code = exception.status_code)


app.mount('/', StaticFiles(directory = STUDIO_PATH / 'static', html = True), name = 'static')


def main() -> None:
	init_engine()
	install_progress_hooks()
	install_download_hooks()
	start_task('setup', run_setup)
	url = 'http://' + CLI_ARGS.host + ':' + str(CLI_ARGS.port)
	threading.Thread(target = announce_when_ready, args = (url,), daemon = True).start()
	uvicorn.run(app, host = CLI_ARGS.host, port = CLI_ARGS.port, log_level = 'warning')


def announce_when_ready(url : str) -> None:
	# launchers (Pinokio) open the printed url right away, so only print it once the port accepts connections
	while True:
		try:
			with socket.create_connection((CLI_ARGS.host, CLI_ARGS.port), timeout = 1):
				break
		except OSError:
			time.sleep(0.2)
	print('\n  MultiFace Studio prêt → ' + url + '\n', flush = True)

	if CLI_ARGS.open_browser:
		webbrowser.open(url)


if __name__ == '__main__':
	main()
