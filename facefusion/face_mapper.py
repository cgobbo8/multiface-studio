import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy

from facefusion import process_manager, state_manager
from facefusion.common_helper import get_first
from facefusion.face_analyser import get_average_face, get_many_faces
from facefusion.types import Face, FaceMapping, VisionFrame
from facefusion.vision import read_image, read_static_image, read_static_images

# seconds a briefly undetected face keeps being processed, bridged between the frames around the miss
FACE_TRACK_GAP = 0.25
FACE_TRACK_SET : Dict[str, List[Tuple[Face, FaceMapping]]] = {}
FACE_TRACK_FRAMES : List[Dict[int, Face]] = []
FACE_TRACK_TOKEN : List[Tuple[int, int]] = []


def get_face_mappings() -> List[FaceMapping]:
	return state_manager.get_item('face_mappings') or []


def has_face_mapping_source() -> bool:
	for face_mapping in get_face_mappings():
		if get_face_mapping_source_face(face_mapping):
			return True
	return False


def has_face_mapping_sticker() -> bool:
	return any(face_mapping.get('sticker_path') for face_mapping in get_face_mappings())


def calculate_face_mapping_distance(face : Face, face_mapping : FaceMapping) -> float:
	face_distances = [ 1 - numpy.dot(face.embedding_norm, reference_embedding) for reference_embedding in face_mapping.get('reference_embeddings') ]

	if face_distances:
		return float(numpy.interp(min(face_distances), [ 0, 2 ], [ 0, 1 ]))
	return 1.0


def match_face_mappings(target_faces : List[Face], face_mappings : List[FaceMapping], face_distance : float) -> List[Tuple[Face, FaceMapping]]:
	match_candidates = []
	match_pairs = []
	used_target_indices = set()
	used_mapping_indices = set()

	for target_index, target_face in enumerate(target_faces):
		for mapping_index, face_mapping in enumerate(face_mappings):
			current_face_distance = calculate_face_mapping_distance(target_face, face_mapping)

			if current_face_distance < face_distance:
				match_candidates.append((current_face_distance, target_index, mapping_index))

	# closest pairs win first so two similar looking people never share the same mapping
	for _, target_index, mapping_index in sorted(match_candidates):
		if target_index not in used_target_indices and mapping_index not in used_mapping_indices:
			used_target_indices.add(target_index)
			used_mapping_indices.add(mapping_index)
			match_pairs.append((target_faces[target_index], face_mappings[mapping_index]))

	return match_pairs


def select_mapped_faces(target_vision_frame : VisionFrame) -> List[Tuple[Face, FaceMapping]]:
	return [ (target_face, face_mapping) for target_face, face_mapping in match_target_faces(target_vision_frame) if face_mapping.get('source_paths') ]


def select_sticker_faces(target_vision_frame : VisionFrame) -> List[Tuple[Face, FaceMapping]]:
	return [ (target_face, face_mapping) for target_face, face_mapping in match_target_faces(target_vision_frame) if face_mapping.get('sticker_path') ]


def match_target_faces(target_vision_frame : VisionFrame) -> List[Tuple[Face, FaceMapping]]:
	# inside a prepared video the tracked faces win, they include the bridged misses
	if FACE_TRACK_SET:
		tracked_pairs = FACE_TRACK_SET.get(create_frame_key(target_vision_frame))

		if tracked_pairs is not None:
			return tracked_pairs

	# every mapping takes part so swapped, stickered and untouched people never steal each other's faces
	target_faces = get_many_faces([ target_vision_frame ])
	return match_face_mappings(target_faces, get_face_mappings(), state_manager.get_item('reference_face_distance'))


def create_frame_key(vision_frame : VisionFrame) -> str:
	# crc32 collides too often over thousands of frames, a wrong hit would paste faces from another frame
	return hashlib.blake2b(numpy.ascontiguousarray(vision_frame).data, digest_size = 16).hexdigest()


def get_tracked_faces() -> List[Dict[int, Face]]:
	return FACE_TRACK_FRAMES


def clear_face_tracks() -> None:
	FACE_TRACK_SET.clear()
	FACE_TRACK_FRAMES.clear()
	FACE_TRACK_TOKEN.clear()


def prepare_face_tracks(temp_frame_paths : List[str]) -> None:
	track_token = (id(temp_frame_paths), len(temp_frame_paths))

	# several processors prepare the same render, the video is analysed once
	if track_token in FACE_TRACK_TOKEN:
		return

	clear_face_tracks()
	FACE_TRACK_TOKEN.append(track_token)
	face_mappings = get_face_mappings()
	track_indices = [ index for index, face_mapping in enumerate(face_mappings) if face_mapping.get('source_paths') or face_mapping.get('sticker_path') ]

	if not track_indices:
		return

	frame_total = len(temp_frame_paths)
	frame_keys : List[Optional[str]] = [ None ] * frame_total
	frame_faces : List[List[Face]] = [ [] for _ in range(frame_total) ]
	matched_faces : List[Dict[int, Face]] = [ {} for _ in range(frame_total) ]

	with ThreadPoolExecutor(max_workers = state_manager.get_item('execution_thread_count')) as executor:
		for frame_number, frame_key, target_faces, frame_matches in executor.map(analyse_track_frame, range(frame_total), temp_frame_paths):
			frame_keys[frame_number] = frame_key
			frame_faces[frame_number] = target_faces
			matched_faces[frame_number] = frame_matches

	tracked_faces = [ dict(frame_matches) for frame_matches in matched_faces ]
	gap_total = max(1, round((state_manager.get_item('output_video_fps') or 25.0) * FACE_TRACK_GAP))

	for mapping_index in track_indices:
		known_numbers = [ frame_number for frame_number in range(frame_total) if mapping_index in matched_faces[frame_number] ]

		for start_number, end_number in zip(known_numbers[:-1], known_numbers[1:]):
			start_face = matched_faces[start_number].get(mapping_index)
			end_face = matched_faces[end_number].get(mapping_index)

			if 1 < end_number - start_number <= gap_total + 1 and is_face_bridgeable(start_face, end_face):
				for frame_number in range(start_number + 1, end_number):
					estimate_face = interpolate_face(start_face, end_face, (frame_number - start_number) / (end_number - start_number))
					bridge_face = find_bridge_face(estimate_face, frame_faces[frame_number], tracked_faces[frame_number])

					if bridge_face:
						tracked_faces[frame_number][mapping_index] = bridge_face

	FACE_TRACK_FRAMES.extend(tracked_faces)

	for frame_number, frame_key in enumerate(frame_keys):
		if frame_key:
			FACE_TRACK_SET[frame_key] = [ (face, face_mappings[mapping_index]) for mapping_index, face in tracked_faces[frame_number].items() ]


def analyse_track_frame(frame_number : int, temp_frame_path : str) -> Tuple[int, Optional[str], List[Face], Dict[int, Face]]:
	if process_manager.is_stopping():
		return frame_number, None, [], {}

	# uncached read: a whole video must not pile up in memory, the face store keeps the detections
	vision_frame = read_image(temp_frame_path)

	if vision_frame is None:
		return frame_number, None, [], {}

	face_mappings = get_face_mappings()
	mapping_indices = { id(face_mapping): index for index, face_mapping in enumerate(face_mappings) }
	target_faces = get_many_faces([ vision_frame ])
	frame_matches = { mapping_indices.get(id(face_mapping)): target_face for target_face, face_mapping in match_face_mappings(target_faces, face_mappings, state_manager.get_item('reference_face_distance')) }
	return frame_number, create_frame_key(vision_frame), target_faces, frame_matches


def get_face_size(face : Face) -> float:
	return float(max(face.bounding_box[2] - face.bounding_box[0], face.bounding_box[3] - face.bounding_box[1]))


def get_face_center(face : Face) -> Tuple[float, float]:
	return float((face.bounding_box[0] + face.bounding_box[2]) / 2), float((face.bounding_box[1] + face.bounding_box[3]) / 2)


def is_face_bridgeable(start_face : Face, end_face : Face) -> bool:
	# a short miss leaves the head almost where it was: any real move means a cut or another person, never bridged
	start_size = get_face_size(start_face)
	end_size = get_face_size(end_face)
	face_shift = math.dist(get_face_center(start_face), get_face_center(end_face))
	return 0.75 < end_size / max(start_size, 1e-6) < 1.33 and face_shift < (start_size + end_size) / 4


def interpolate_face(start_face : Face, end_face : Face, progress : float) -> Face:
	nearest_face = start_face if progress < 0.5 else end_face
	landmark_set = { landmark_key: start_face.landmark_set.get(landmark_key) * (1 - progress) + end_face.landmark_set.get(landmark_key) * progress for landmark_key in start_face.landmark_set }
	bounding_box = start_face.bounding_box * (1 - progress) + end_face.bounding_box * progress
	return nearest_face._replace(bounding_box = bounding_box, landmark_set = landmark_set)


def calculate_face_overlap(face : Face, other_face : Face) -> float:
	start_x = max(face.bounding_box[0], other_face.bounding_box[0])
	start_y = max(face.bounding_box[1], other_face.bounding_box[1])
	end_x = min(face.bounding_box[2], other_face.bounding_box[2])
	end_y = min(face.bounding_box[3], other_face.bounding_box[3])
	overlap_area = max(0.0, end_x - start_x) * max(0.0, end_y - start_y)
	union_area = get_bounding_box_area(face) + get_bounding_box_area(other_face) - overlap_area
	return float(overlap_area / union_area) if union_area > 0 else 0.0


def find_bridge_face(estimate_face : Face, target_faces : List[Face], frame_tracked_faces : Dict[int, Face]) -> Optional[Face]:
	claimed_face_ids = { id(face) for face in frame_tracked_faces.values() }
	overlap_pairs = [ (calculate_face_overlap(estimate_face, target_face), target_face) for target_face in target_faces ]
	best_overlap, best_face = max(overlap_pairs, key = lambda overlap_pair: overlap_pair[0], default = (0.0, None))

	if best_face is not None and best_overlap > 0.3:
		# the face is there but belongs to someone else in this frame: stay away from it
		if id(best_face) in claimed_face_ids:
			return None
		# detected but not recognised (profile, blur): its own landmarks beat any estimate
		return best_face

	# never stack two people on one spot, even when the other one was bridged too
	if any(calculate_face_overlap(estimate_face, tracked_face) > 0.3 for tracked_face in frame_tracked_faces.values()):
		return None
	return estimate_face


def get_face_mapping_source_face(face_mapping : FaceMapping) -> Optional[Face]:
	source_faces = []

	for source_vision_frame in read_static_images(face_mapping.get('source_paths') or []):
		temp_faces = get_many_faces([ source_vision_frame ])

		if temp_faces:
			source_faces.append(max(temp_faces, key = get_bounding_box_area))

	return get_average_face(source_faces)


def get_face_mapping_source_frame(face_mapping : FaceMapping) -> Optional[VisionFrame]:
	source_path = get_first(face_mapping.get('source_paths'))

	if source_path:
		return read_static_image(source_path)
	return None


def get_bounding_box_area(face : Face) -> float:
	return (face.bounding_box[2] - face.bounding_box[0]) * (face.bounding_box[3] - face.bounding_box[1])
