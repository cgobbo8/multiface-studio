from argparse import ArgumentParser
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy

from facefusion import face_masker, logger, state_manager, translator
from facefusion.face_analyser import scale_face
from facefusion.face_helper import warp_face_by_face_landmark_5
from facefusion.face_mapper import clear_face_tracks, has_face_mapping_filter, prepare_face_tracks, select_filter_faces
from facefusion.filesystem import in_directory, is_image, is_video, same_file_extension
from facefusion.processors.types import ProcessorOutputs
from facefusion.types import ApplyStateItem, Args, Face, InferencePool, ProcessMode, VisionFrame

# a person gets one filter per group, except the drawn props which add up
FACE_FILTER_SET : Dict[str, str] =\
{
	'big_eyes': 'shape',
	'big_mouth': 'shape',
	'big_nose': 'shape',
	'big_head': 'shape',
	'tiny_face': 'shape',
	'long_face': 'shape',
	'wide_face': 'shape',
	'alien': 'shape',
	'baby': 'shape',
	'swirl': 'shape',
	'cyclops': 'shape',
	'upside_down': 'shape',
	'hair_pink': 'hair',
	'hair_blue': 'hair',
	'hair_green': 'hair',
	'hair_red': 'hair',
	'hair_platinum': 'hair',
	'hair_rainbow': 'hair',
	'skin_green': 'skin',
	'skin_blue': 'skin',
	'skin_yellow': 'skin',
	'skin_red': 'skin',
	'clown_nose': 'prop',
	'demon_eyes': 'prop',
	'blush': 'prop',
	'moustache': 'prop',
	'pixel_glasses': 'prop'
}
FACE_FILTER_MULTIPLE_GROUPS = [ 'prop' ]
# rgb colour and how much light is added first: dark hair takes no colour without it
HAIR_COLOURS : Dict[str, Tuple[Tuple[int, int, int], float]] =\
{
	'hair_pink': ((255, 70, 170), 0.18),
	'hair_blue': ((40, 120, 255), 0.18),
	'hair_green': ((50, 210, 90), 0.18),
	'hair_red': ((235, 45, 35), 0.18),
	'hair_platinum': ((250, 240, 215), 0.55)
}
SKIN_COLOURS : Dict[str, Tuple[int, int, int]] =\
{
	'skin_green': (120, 175, 60),
	'skin_blue': (70, 130, 220),
	'skin_yellow': (255, 210, 40),
	'skin_red': (205, 45, 40)
}
# classes of the face parser (CelebAMask-HQ): skin, ears, nose and neck make the skin, 17 is the hair
PARSER_SKIN_CLASSES = [ 1, 7, 8, 10, 14 ]
PARSER_HAIR_CLASSES = [ 17 ]
PIXEL_GLASSES =\
[
	'#####################',
	'.#ww#####...#ww#####.',
	'.##ww####...##ww####.',
	'..######.....######..',
	'...####.......####...'
]
# right half of the moustache in eye distances from its center, the left half is mirrored
MOUSTACHE_OUTLINE = [ (0.0, -0.06), (0.14, -0.12), (0.3, -0.1), (0.44, -0.03), (0.56, -0.02), (0.66, -0.1), (0.72, -0.22), (0.68, -0.32), (0.6, -0.3), (0.63, -0.2), (0.58, -0.1), (0.48, 0.08), (0.33, 0.14), (0.18, 0.14), (0.07, 0.1), (0.0, 0.06) ]


class FaceGeometry(NamedTuple):
	left_eye : numpy.ndarray
	right_eye : numpy.ndarray
	nose : numpy.ndarray
	mouth : numpy.ndarray
	eye_center : numpy.ndarray
	axis_x : numpy.ndarray
	axis_y : numpy.ndarray
	unit : float


WarpMap = Callable[[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]]
WarpPart = Tuple[numpy.ndarray, float, WarpMap]


def get_inference_pool() -> InferencePool:
	pass


def clear_inference_pool() -> None:
	pass


def register_args(program : ArgumentParser) -> None:
	pass


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	pass


def pre_check() -> bool:
	# the hair and skin colours need the face parser
	return face_masker.pre_check()


def pre_process(mode : ProcessMode) -> bool:
	if not has_face_mapping_filter():
		logger.error(translator.get('choose_image_source') + translator.get('exclamation_mark'), __name__)
		return False
	if mode in [ 'output', 'preview' ] and not is_image(state_manager.get_item('target_path')) and not is_video(state_manager.get_item('target_path')):
		logger.error(translator.get('choose_image_or_video_target') + translator.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not in_directory(state_manager.get_item('output_path')):
		logger.error(translator.get('specify_image_or_video_output') + translator.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not same_file_extension(state_manager.get_item('target_path'), state_manager.get_item('output_path')):
		logger.error(translator.get('match_target_and_output_extension') + translator.get('exclamation_mark'), __name__)
		return False
	return True


def post_process() -> None:
	clear_face_tracks()


def prepare_frames(temp_frame_paths : List[str]) -> None:
	# filtered faces follow the shared tracks, short detection misses keep their filter
	if state_manager.get_item('face_selector_mode') == 'mapping':
		prepare_face_tracks(temp_frame_paths)


def normalize_face_filters(face_filters : List[str]) -> List[str]:
	normalized_filters : List[str] = []

	for face_filter in face_filters:
		face_filter_group = FACE_FILTER_SET.get(face_filter)

		if face_filter_group and face_filter not in normalized_filters:
			# the newest pick of a group replaces the previous one
			if face_filter_group not in FACE_FILTER_MULTIPLE_GROUPS:
				normalized_filters = [ normalized_filter for normalized_filter in normalized_filters if FACE_FILTER_SET.get(normalized_filter) != face_filter_group ]
			normalized_filters.append(face_filter)
	return normalized_filters


def create_face_geometry(face : Face) -> FaceGeometry:
	# the detector's own points: the live mode skips the 68 landmarks, and they flip on small faces anyway
	face_landmark_5 = face.landmark_set.get('5').astype(numpy.float32)
	left_eye, right_eye, nose = face_landmark_5[0], face_landmark_5[1], face_landmark_5[2]
	eye_vector = right_eye - left_eye
	unit = max(float(numpy.linalg.norm(eye_vector)), 1.0)
	axis_x = eye_vector / unit
	axis_y = numpy.array([ -axis_x[1], axis_x[0] ], dtype = numpy.float32)
	return FaceGeometry(left_eye, right_eye, nose, (face_landmark_5[3] + face_landmark_5[4]) / 2, (left_eye + right_eye) / 2, axis_x, axis_y, unit)


def to_frame_point(geometry : FaceGeometry, origin : numpy.ndarray, offset_x : float, offset_y : float) -> numpy.ndarray:
	return origin + (geometry.axis_x * offset_x + geometry.axis_y * offset_y) * geometry.unit


def to_face_coordinates(geometry : FaceGeometry, origin : numpy.ndarray, x : numpy.ndarray, y : numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
	delta_x, delta_y = x - origin[0], y - origin[1]
	return delta_x * geometry.axis_x[0] + delta_y * geometry.axis_x[1], delta_x * geometry.axis_y[0] + delta_y * geometry.axis_y[1]


def smoothstep(edge_start : float, edge_end : float, value : numpy.ndarray) -> numpy.ndarray:
	value = numpy.clip((value - edge_start) / max(edge_end - edge_start, 1e-6), 0, 1)
	return value * value * (3 - 2 * value)


def calculate_region(temp_vision_frame : VisionFrame, center : numpy.ndarray, radius : float) -> Optional[Tuple[int, int, int, int]]:
	frame_height, frame_width = temp_vision_frame.shape[:2]
	start_x, start_y = max(0, int(center[0] - radius)), max(0, int(center[1] - radius))
	end_x, end_y = min(frame_width, int(center[0] + radius) + 1), min(frame_height, int(center[1] + radius) + 1)

	if end_x <= start_x or end_y <= start_y:
		return None
	return start_x, start_y, end_x, end_y


def create_region_grid(region : Tuple[int, int, int, int]) -> Tuple[numpy.ndarray, numpy.ndarray]:
	start_x, start_y, end_x, end_y = region
	grid_y, grid_x = numpy.mgrid[start_y:end_y, start_x:end_x].astype(numpy.float32)
	return grid_x, grid_y


def blend_region(temp_vision_frame : VisionFrame, region : Tuple[int, int, int, int], overlay_frame : numpy.ndarray, overlay_alpha : numpy.ndarray) -> VisionFrame:
	start_x, start_y, end_x, end_y = region
	crop_frame = temp_vision_frame[start_y:end_y, start_x:end_x].astype(numpy.float32)
	overlay_alpha = overlay_alpha[:, :, None]
	temp_vision_frame[start_y:end_y, start_x:end_x] = (crop_frame * (1 - overlay_alpha) + overlay_frame * overlay_alpha).clip(0, 255).astype(numpy.uint8)
	return temp_vision_frame


# ---------------------------------------------------------------- deformations

def create_zoom_map(center : numpy.ndarray, radius : float, zoom : float, plateau : float = 0.0) -> WarpMap:
	# zoom above 1 enlarges the middle, below 1 shrinks it, the rim of the circle never moves
	def warp_map(x : numpy.ndarray, y : numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
		delta_x, delta_y = x - center[0], y - center[1]
		distance = numpy.sqrt(delta_x * delta_x + delta_y * delta_y) / radius

		if zoom < 1:
			# a shrink stretches the ring around it: this curve keeps that stretch even, without streaks
			factor = 1 + (1 / zoom - 1) * (1 - numpy.minimum(distance, 1)) ** 2
		else:
			blend = smoothstep(plateau, 1, distance)
			factor = (1 - blend) / zoom + blend
		return center[0] + delta_x * factor, center[1] + delta_y * factor
	return warp_map


def create_stretch_map(geometry : FaceGeometry, center : numpy.ndarray, radius_x : float, radius_y : float, zoom_x : float, zoom_y : float) -> WarpMap:
	# a zoom of its own along the eyes and along the nose
	def warp_map(x : numpy.ndarray, y : numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
		face_x, face_y = to_face_coordinates(geometry, center, x, y)
		blend = smoothstep(0.35, 1, numpy.sqrt((face_x / radius_x) ** 2 + (face_y / radius_y) ** 2))
		face_x = face_x * ((1 - blend) / zoom_x + blend)
		face_y = face_y * ((1 - blend) / zoom_y + blend)
		return center[0] + face_x * geometry.axis_x[0] + face_y * geometry.axis_y[0], center[1] + face_x * geometry.axis_x[1] + face_y * geometry.axis_y[1]
	return warp_map


def create_swirl_map(center : numpy.ndarray, radius : float, angle : float) -> WarpMap:
	def warp_map(x : numpy.ndarray, y : numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
		delta_x, delta_y = x - center[0], y - center[1]
		turn = angle * (1 - smoothstep(0, 1, numpy.sqrt(delta_x * delta_x + delta_y * delta_y) / radius))
		turn_cos, turn_sin = numpy.cos(turn), numpy.sin(turn)
		return center[0] + delta_x * turn_cos - delta_y * turn_sin, center[1] + delta_x * turn_sin + delta_y * turn_cos
	return warp_map


def create_shape_parts(face_filter : str, geometry : FaceGeometry) -> List[WarpPart]:
	unit = geometry.unit
	face_center = to_frame_point(geometry, geometry.eye_center, 0, 0.45)
	forehead = to_frame_point(geometry, geometry.eye_center, 0, -0.85)
	chin = to_frame_point(geometry, geometry.mouth, 0, 0.35)
	nose = to_frame_point(geometry, geometry.nose, 0, -0.12)

	if face_filter == 'big_eyes':
		return [ (eye, unit * 0.62, create_zoom_map(eye, unit * 0.62, 1.8)) for eye in (geometry.left_eye, geometry.right_eye) ]
	if face_filter == 'big_mouth':
		return [ (geometry.mouth, unit * 0.85, create_zoom_map(geometry.mouth, unit * 0.85, 1.8)) ]
	if face_filter == 'big_nose':
		return [ (nose, unit * 0.7, create_zoom_map(nose, unit * 0.7, 2.3)) ]
	if face_filter == 'big_head':
		head_center = to_frame_point(geometry, geometry.eye_center, 0, 0.15)
		return [ (head_center, unit * 3.6, create_zoom_map(head_center, unit * 3.6, 1.5, 0.5)) ]
	if face_filter == 'tiny_face':
		return [ (face_center, unit * 1.9, create_zoom_map(face_center, unit * 1.9, 0.55)) ]
	if face_filter == 'long_face':
		return [ (face_center, unit * 2.6, create_stretch_map(geometry, face_center, unit * 1.6, unit * 2.6, 0.92, 1.45)) ]
	if face_filter == 'wide_face':
		return [ (face_center, unit * 2.2, create_stretch_map(geometry, face_center, unit * 2.2, unit * 2.0, 1.45, 0.75)) ]
	if face_filter == 'alien':
		return\
		[
			(forehead, unit * 1.7, create_zoom_map(forehead, unit * 1.7, 1.8)),
			(chin, unit * 1.2, create_zoom_map(chin, unit * 1.2, 0.6)),
			*[ (eye, unit * 0.6, create_zoom_map(eye, unit * 0.6, 2.3)) for eye in (geometry.left_eye, geometry.right_eye) ]
		]
	if face_filter == 'baby':
		return\
		[
			(forehead, unit * 1.4, create_zoom_map(forehead, unit * 1.4, 1.2)),
			(chin, unit * 1.3, create_zoom_map(chin, unit * 1.3, 0.85)),
			(nose, unit * 0.5, create_zoom_map(nose, unit * 0.5, 0.7)),
			(geometry.mouth, unit * 0.6, create_zoom_map(geometry.mouth, unit * 0.6, 0.72)),
			*[ (eye, unit * 0.6, create_zoom_map(eye, unit * 0.6, 1.9)) for eye in (geometry.left_eye, geometry.right_eye) ]
		]
	if face_filter == 'swirl':
		return [ (face_center, unit * 2.2, create_swirl_map(face_center, unit * 2.2, 1.4)) ]
	return []


def warp_face(temp_vision_frame : VisionFrame, warp_parts : List[WarpPart]) -> VisionFrame:
	# the whole chain in one remap: each map sends a pixel to where its colour comes from
	regions = [ calculate_region(temp_vision_frame, center, radius) for center, radius, _ in warp_parts ]
	regions = [ region for region in regions if region ]

	if not regions:
		return temp_vision_frame
	region = min(region[0] for region in regions), min(region[1] for region in regions), max(region[2] for region in regions), max(region[3] for region in regions)
	map_x, map_y = create_region_grid(region)

	for _, _, warp_map in reversed(warp_parts):
		map_x, map_y = warp_map(map_x, map_y)
	start_x, start_y, end_x, end_y = region
	warp_frame = cv2.remap(temp_vision_frame, map_x.astype(numpy.float32), map_y.astype(numpy.float32), cv2.INTER_LINEAR, borderMode = cv2.BORDER_REPLICATE)
	temp_vision_frame = temp_vision_frame.copy()
	temp_vision_frame[start_y:end_y, start_x:end_x] = warp_frame
	return temp_vision_frame


def flip_feature(temp_vision_frame : VisionFrame, geometry : FaceGeometry, center : numpy.ndarray, half_width : float, half_height : float) -> VisionFrame:
	# mirrored upside down inside a soft ellipse of the face frame
	region = calculate_region(temp_vision_frame, center, max(half_width, half_height) * geometry.unit * 1.1)

	if not region:
		return temp_vision_frame
	grid_x, grid_y = create_region_grid(region)
	face_x, face_y = to_face_coordinates(geometry, center, grid_x, grid_y)
	map_x = center[0] + face_x * geometry.axis_x[0] - face_y * geometry.axis_y[0]
	map_y = center[1] + face_x * geometry.axis_x[1] - face_y * geometry.axis_y[1]
	flip_frame = cv2.remap(temp_vision_frame, map_x.astype(numpy.float32), map_y.astype(numpy.float32), cv2.INTER_LINEAR, borderMode = cv2.BORDER_REPLICATE)
	distance = numpy.sqrt((face_x / (half_width * geometry.unit)) ** 2 + (face_y / (half_height * geometry.unit)) ** 2)
	return blend_region(temp_vision_frame, region, flip_frame.astype(numpy.float32), 1 - smoothstep(0.6, 1, distance))


def turn_upside_down(temp_vision_frame : VisionFrame, geometry : FaceGeometry) -> VisionFrame:
	temp_vision_frame = temp_vision_frame.copy()

	for eye in (geometry.left_eye, geometry.right_eye):
		temp_vision_frame = flip_feature(temp_vision_frame, geometry, to_frame_point(geometry, eye, 0, -0.12), 0.42, 0.32)
	return flip_feature(temp_vision_frame, geometry, geometry.mouth, 0.55, 0.3)


def close_eye(temp_vision_frame : VisionFrame, geometry : FaceGeometry, eye : numpy.ndarray) -> VisionFrame:
	# the smooth skin under the eye is lifted over it, faded out on every side
	region = calculate_region(temp_vision_frame, eye, geometry.unit * 0.5)

	if not region:
		return temp_vision_frame
	grid_x, grid_y = create_region_grid(region)
	lift = geometry.axis_y * geometry.unit * 0.4
	skin_frame = cv2.remap(temp_vision_frame, grid_x + lift[0], grid_y + lift[1], cv2.INTER_LINEAR, borderMode = cv2.BORDER_REPLICATE)
	face_x, face_y = to_face_coordinates(geometry, eye, grid_x, grid_y)
	distance = numpy.sqrt((face_x / (geometry.unit * 0.4)) ** 2 + (face_y / (geometry.unit * 0.22)) ** 2)
	return blend_region(temp_vision_frame, region, skin_frame.astype(numpy.float32), 1 - smoothstep(0.55, 1, distance))


def create_cyclops(temp_vision_frame : VisionFrame, geometry : FaceGeometry) -> VisionFrame:
	source_vision_frame = temp_vision_frame
	temp_vision_frame = temp_vision_frame.copy()

	for eye in (geometry.left_eye, geometry.right_eye):
		temp_vision_frame = close_eye(temp_vision_frame, geometry, eye)

	# the eye seen the widest, twice as big, in the middle
	nose_x = numpy.dot(geometry.nose - geometry.eye_center, geometry.axis_x)
	eye = geometry.left_eye if nose_x > 0 else geometry.right_eye
	center = to_frame_point(geometry, geometry.eye_center, 0, -0.08)
	region = calculate_region(temp_vision_frame, center, geometry.unit * 0.8)

	if not region:
		return temp_vision_frame
	grid_x, grid_y = create_region_grid(region)
	eye_frame = cv2.remap(source_vision_frame, eye[0] + (grid_x - center[0]) / 2, eye[1] + (grid_y - center[1]) / 2, cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)
	face_x, face_y = to_face_coordinates(geometry, center, grid_x, grid_y)
	distance = numpy.sqrt((face_x / (geometry.unit * 0.75)) ** 2 + (face_y / (geometry.unit * 0.45)) ** 2)
	return blend_region(temp_vision_frame, region, eye_frame.astype(numpy.float32), 1 - smoothstep(0.6, 1, distance))


def apply_shape(temp_vision_frame : VisionFrame, geometry : FaceGeometry, face_filter : str) -> VisionFrame:
	if face_filter == 'cyclops':
		return create_cyclops(temp_vision_frame, geometry)
	if face_filter == 'upside_down':
		return turn_upside_down(temp_vision_frame, geometry)
	return warp_face(temp_vision_frame, create_shape_parts(face_filter, geometry))


# ---------------------------------------------------------------- hair and skin

def parse_head(temp_vision_frame : VisionFrame, face : Face) -> Tuple[numpy.ndarray, numpy.ndarray]:
	# the ffhq crop frames the whole head, hair included, like the images the parser learnt from
	crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, face.landmark_set.get('5/68'), 'ffhq_512', (512, 512))
	prepare_vision_frame = crop_vision_frame[:, :, ::-1].astype(numpy.float32) / 255.0
	prepare_vision_frame = (prepare_vision_frame - numpy.array([ 0.485, 0.456, 0.406 ], dtype = numpy.float32)) / numpy.array([ 0.229, 0.224, 0.225 ], dtype = numpy.float32)
	prepare_vision_frame = numpy.expand_dims(prepare_vision_frame.transpose(2, 0, 1), axis = 0)
	region_labels = face_masker.forward_parse_face(prepare_vision_frame).argmax(0).astype(numpy.uint8)
	return region_labels, affine_matrix


def paste_region_mask(temp_vision_frame : VisionFrame, region_labels : numpy.ndarray, affine_matrix : numpy.ndarray, region_classes : List[int]) -> Tuple[Optional[Tuple[int, int, int, int]], numpy.ndarray]:
	region_mask = cv2.GaussianBlur(numpy.isin(region_labels, region_classes).astype(numpy.float32), (0, 0), 2)
	inverse_matrix = cv2.invertAffineTransform(affine_matrix)
	corners = cv2.transform(numpy.array([ [ [ 0, 0 ], [ 512, 0 ], [ 512, 512 ], [ 0, 512 ] ] ], dtype = numpy.float32), inverse_matrix)[0]
	frame_height, frame_width = temp_vision_frame.shape[:2]
	start_x, start_y = max(0, int(corners[:, 0].min())), max(0, int(corners[:, 1].min()))
	end_x, end_y = min(frame_width, int(corners[:, 0].max()) + 1), min(frame_height, int(corners[:, 1].max()) + 1)

	if end_x <= start_x or end_y <= start_y:
		return None, region_mask
	# only the head's own box is warped back, not the whole frame
	inverse_matrix[:, 2] -= [ start_x, start_y ]
	return (start_x, start_y, end_x, end_y), cv2.warpAffine(region_mask, inverse_matrix, (end_x - start_x, end_y - start_y), flags = cv2.INTER_LINEAR, borderValue = 0)


def dye_hair(temp_vision_frame : VisionFrame, geometry : FaceGeometry, region : Tuple[int, int, int, int], hair_mask : numpy.ndarray, face_filter : str) -> VisionFrame:
	start_x, start_y, end_x, end_y = region
	crop_frame = temp_vision_frame[start_y:end_y, start_x:end_x]
	gray_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY).astype(numpy.float32) / 255

	if face_filter == 'hair_rainbow':
		grid_x, grid_y = create_region_grid(region)
		face_x, _ = to_face_coordinates(geometry, geometry.eye_center, grid_x, grid_y)
		hue = ((face_x / geometry.unit + 1.8) / 3.6 * 150).clip(0, 150).astype(numpy.uint8)
		colour_frame = cv2.cvtColor(numpy.dstack([ hue, numpy.full_like(hue, 230), numpy.full_like(hue, 255) ]), cv2.COLOR_HSV2BGR).astype(numpy.float32)
		lift = 0.18
	else:
		colour, lift = HAIR_COLOURS.get(face_filter)
		colour_frame = numpy.array(colour[::-1], dtype = numpy.float32)[None, None]
	# the strands keep their light and shade under the new colour
	light_frame = numpy.clip(gray_frame * (2.4 - lift * 1.5) + lift, 0, 1.15)[:, :, None]
	return blend_region(temp_vision_frame, region, colour_frame * light_frame, hair_mask * 0.85)


def paint_skin(temp_vision_frame : VisionFrame, region : Tuple[int, int, int, int], skin_mask : numpy.ndarray, face_filter : str) -> VisionFrame:
	start_x, start_y, end_x, end_y = region
	crop_frame = temp_vision_frame[start_y:end_y, start_x:end_x].astype(numpy.float32) / 255
	colour = SKIN_COLOURS.get(face_filter)
	lab_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2LAB)
	target_lab = cv2.cvtColor(numpy.array([ [ colour[::-1] ] ], dtype = numpy.float32) / 255, cv2.COLOR_BGR2LAB)[0, 0]
	weight = skin_mask > 0.5
	paint_frame = lab_frame.copy()

	# the new tone keeps a little of the skin's own variations, its light stays almost untouched
	for channel in (1, 2):
		channel_mean = lab_frame[:, :, channel][weight].mean() if weight.any() else 0
		paint_frame[:, :, channel] = target_lab[channel] + (lab_frame[:, :, channel] - channel_mean) * 0.5
	paint_frame[:, :, 0] = lab_frame[:, :, 0] * 0.8 + target_lab[0] * 0.2
	paint_frame = cv2.cvtColor(paint_frame, cv2.COLOR_LAB2BGR) * 255
	return blend_region(temp_vision_frame, region, paint_frame, skin_mask * 0.92)


def apply_colours(temp_vision_frame : VisionFrame, face : Face, geometry : FaceGeometry, face_filters : List[str]) -> VisionFrame:
	hair_filter = next((face_filter for face_filter in face_filters if FACE_FILTER_SET.get(face_filter) == 'hair'), None)
	skin_filter = next((face_filter for face_filter in face_filters if FACE_FILTER_SET.get(face_filter) == 'skin'), None)

	if not hair_filter and not skin_filter:
		return temp_vision_frame
	region_labels, affine_matrix = parse_head(temp_vision_frame, face)
	temp_vision_frame = temp_vision_frame.copy()

	if skin_filter:
		region, skin_mask = paste_region_mask(temp_vision_frame, region_labels, affine_matrix, PARSER_SKIN_CLASSES)

		if region:
			temp_vision_frame = paint_skin(temp_vision_frame, region, skin_mask, skin_filter)
	if hair_filter:
		region, hair_mask = paste_region_mask(temp_vision_frame, region_labels, affine_matrix, PARSER_HAIR_CLASSES)

		if region:
			temp_vision_frame = dye_hair(temp_vision_frame, geometry, region, hair_mask, hair_filter)
	return temp_vision_frame


# ---------------------------------------------------------------- drawn props

def draw_clown_nose(temp_vision_frame : VisionFrame, geometry : FaceGeometry) -> VisionFrame:
	radius = geometry.unit * 0.21
	center = to_frame_point(geometry, geometry.nose, 0, -0.02)
	region = calculate_region(temp_vision_frame, center, radius + 2)

	if not region:
		return temp_vision_frame
	grid_x, grid_y = create_region_grid(region)
	face_x, face_y = to_face_coordinates(geometry, center, grid_x, grid_y)
	face_x, face_y = face_x / radius, face_y / radius
	distance = numpy.sqrt(face_x * face_x + face_y * face_y)
	# lit from the top left of the face, with a sharp highlight
	light = numpy.clip(1 - numpy.sqrt((face_x + 0.35) ** 2 + (face_y + 0.4) ** 2) * 0.85, 0, 1)
	shine = numpy.clip(1 - numpy.sqrt((face_x + 0.35) ** 2 + (face_y + 0.4) ** 2) * 3.2, 0, 1) ** 2 * 210
	ball_frame = numpy.dstack([ 25 + 45 * light + shine, 25 + 40 * light + shine, 140 + 115 * light + shine ])
	return blend_region(temp_vision_frame.copy(), region, ball_frame, numpy.clip((1 - distance) * radius * 0.6, 0, 1))


def draw_demon_eyes(temp_vision_frame : VisionFrame, geometry : FaceGeometry) -> VisionFrame:
	temp_vision_frame = temp_vision_frame.copy()

	for eye in (geometry.left_eye, geometry.right_eye):
		radius = geometry.unit * 0.34
		region = calculate_region(temp_vision_frame, eye, radius)

		if region:
			start_x, start_y, end_x, end_y = region
			grid_x, grid_y = create_region_grid(region)
			distance = numpy.sqrt((grid_x - eye[0]) ** 2 + (grid_y - eye[1]) ** 2) / radius
			glow = numpy.clip(1 - distance, 0, 1) ** 2.2
			core = numpy.clip(1 - distance * 4.5, 0, 1)
			crop_frame = temp_vision_frame[start_y:end_y, start_x:end_x].astype(numpy.float32)
			crop_frame += glow[:, :, None] * numpy.array([ 20, 30, 330 ], dtype = numpy.float32) + core[:, :, None] * 160
			temp_vision_frame[start_y:end_y, start_x:end_x] = crop_frame.clip(0, 255).astype(numpy.uint8)
	return temp_vision_frame


def draw_blush(temp_vision_frame : VisionFrame, geometry : FaceGeometry) -> VisionFrame:
	temp_vision_frame = temp_vision_frame.copy()

	for side, eye in ((-1, geometry.left_eye), (1, geometry.right_eye)):
		center = to_frame_point(geometry, eye, side * 0.1, 0.58)
		region = calculate_region(temp_vision_frame, center, geometry.unit * 0.32)

		if region:
			grid_x, grid_y = create_region_grid(region)
			face_x, face_y = to_face_coordinates(geometry, center, grid_x, grid_y)
			distance = numpy.sqrt((face_x / (geometry.unit * 0.3)) ** 2 + (face_y / (geometry.unit * 0.19)) ** 2)
			blush_frame = numpy.array([ 150, 110, 255 ], dtype = numpy.float32)[None, None]
			temp_vision_frame = blend_region(temp_vision_frame, region, numpy.broadcast_to(blush_frame, grid_x.shape + (3,)), 0.42 * (1 - smoothstep(0, 1, distance)))
	return temp_vision_frame


def draw_shape(temp_vision_frame : VisionFrame, geometry : FaceGeometry, origin : numpy.ndarray, outline : List[Tuple[float, float]], colour : Tuple[int, int, int]) -> VisionFrame:
	points = numpy.array([ to_frame_point(geometry, origin, point_x, point_y) for point_x, point_y in outline ])
	region = calculate_region(temp_vision_frame, points.mean(axis = 0), float(numpy.abs(points - points.mean(axis = 0)).max()) + 3)

	if not region:
		return temp_vision_frame
	start_x, start_y, end_x, end_y = region
	shape_mask = numpy.zeros((end_y - start_y, end_x - start_x), dtype = numpy.uint8)
	# drawn with sub-pixel precision so the edges stay smooth at any size
	cv2.fillPoly(shape_mask, [ numpy.round((points - [ start_x, start_y ]) * 16).astype(numpy.int32) ], 255, cv2.LINE_AA, 4)
	colour_frame = numpy.array(colour, dtype = numpy.float32)[None, None]
	return blend_region(temp_vision_frame.copy(), region, numpy.broadcast_to(colour_frame, shape_mask.shape + (3,)), shape_mask.astype(numpy.float32) / 255)


def draw_moustache(temp_vision_frame : VisionFrame, geometry : FaceGeometry) -> VisionFrame:
	origin = to_frame_point(geometry, geometry.nose, 0, 0.27)
	outline = MOUSTACHE_OUTLINE + [ (-point_x, point_y) for point_x, point_y in reversed(MOUSTACHE_OUTLINE) ]
	return draw_shape(temp_vision_frame, geometry, origin, outline, (28, 30, 42))


def draw_pixel_glasses(temp_vision_frame : VisionFrame, geometry : FaceGeometry) -> VisionFrame:
	pattern_height, pattern_width = len(PIXEL_GLASSES), len(PIXEL_GLASSES[0])
	pixel_size = geometry.unit * 2.0 / pattern_width
	pattern_frame = numpy.zeros((pattern_height, pattern_width, 4), dtype = numpy.float32)

	for row, line in enumerate(PIXEL_GLASSES):
		for column, cell in enumerate(line):
			if cell == '#':
				pattern_frame[row, column] = [ 12, 12, 12, 1 ]
			if cell == 'w':
				pattern_frame[row, column] = [ 255, 255, 255, 1 ]
	# big square pixels first, then a smooth warp onto the tilted face
	upscale = 8
	pattern_frame = cv2.resize(pattern_frame, (pattern_width * upscale, pattern_height * upscale), interpolation = cv2.INTER_NEAREST)
	origin = to_frame_point(geometry, geometry.eye_center, -pattern_width * pixel_size / 2 / geometry.unit, -0.16)
	region = calculate_region(temp_vision_frame, geometry.eye_center, geometry.unit * 1.2)

	if not region:
		return temp_vision_frame
	start_x, start_y, end_x, end_y = region
	scale = pixel_size / upscale
	affine_matrix = numpy.array(
	[
		[ geometry.axis_x[0] * scale, geometry.axis_y[0] * scale, origin[0] - start_x ],
		[ geometry.axis_x[1] * scale, geometry.axis_y[1] * scale, origin[1] - start_y ]
	], dtype = numpy.float32)
	warp_frame = cv2.warpAffine(pattern_frame, affine_matrix, (end_x - start_x, end_y - start_y), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = (0, 0, 0, 0))
	alpha = warp_frame[:, :, 3].clip(0, 1)
	return blend_region(temp_vision_frame.copy(), region, warp_frame[:, :, :3] / numpy.maximum(alpha, 1e-3)[:, :, None], alpha)


PROP_PAINTERS : Dict[str, Callable[[VisionFrame, FaceGeometry], VisionFrame]] =\
{
	'clown_nose': draw_clown_nose,
	'demon_eyes': draw_demon_eyes,
	'blush': draw_blush,
	'moustache': draw_moustache,
	'pixel_glasses': draw_pixel_glasses
}


def filter_face(temp_vision_frame : VisionFrame, face : Face, face_filters : List[str]) -> VisionFrame:
	geometry = create_face_geometry(face)
	# colours first, then the props, then the shape: a big head carries its red nose with it
	temp_vision_frame = apply_colours(temp_vision_frame, face, geometry, face_filters)

	for face_filter in face_filters:
		if face_filter in PROP_PAINTERS:
			temp_vision_frame = PROP_PAINTERS.get(face_filter)(temp_vision_frame, geometry)

	for face_filter in face_filters:
		if FACE_FILTER_SET.get(face_filter) == 'shape':
			temp_vision_frame = apply_shape(temp_vision_frame, geometry, face_filter)
	return temp_vision_frame


def process_frame(inputs : dict) -> ProcessorOutputs:
	target_vision_frame = inputs.get('target_vision_frame')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')

	for target_face, face_mapping in select_filter_faces(target_vision_frame):
		target_face = scale_face(target_face, target_vision_frame, temp_vision_frame)
		temp_vision_frame = filter_face(temp_vision_frame, target_face, face_mapping.get('filters'))

	return temp_vision_frame, temp_vision_mask
