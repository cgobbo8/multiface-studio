import math
from argparse import ArgumentParser
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy

from facefusion import logger, state_manager, translator
from facefusion.face_analyser import scale_face
from facefusion.face_mapper import has_face_mapping_sticker, select_sticker_faces
from facefusion.filesystem import in_directory, is_image, is_video, same_file_extension
from facefusion.processors.types import ProcessorOutputs
from facefusion.types import ApplyStateItem, Args, Face, FaceStickerOptions, InferencePool, ProcessMode, VisionFrame
from facefusion.vision import read_static_image, read_static_video_frame

FaceStickerAnchor = Tuple[float, float, float, float, float, float]
# generic head (nose tip, chin, outer eye corners, mouth corners), y up and the nose pointing to +z
HEAD_MODEL_POINTS = numpy.array(
[
	[ 0.0, 0.0, 0.0 ],
	[ 0.0, -330.0, -65.0 ],
	[ -225.0, 170.0, -135.0 ],
	[ 225.0, 170.0, -135.0 ],
	[ -150.0, -150.0, -125.0 ],
	[ 150.0, -150.0, -125.0 ]
], dtype = numpy.float64)
HEAD_PITCH_BIAS = 10.0
HEAD_ANGLE_LIMIT = 75.0
STICKER_PERSPECTIVE = 3.0


def get_inference_pool() -> InferencePool:
	pass


def clear_inference_pool() -> None:
	pass


def register_args(program : ArgumentParser) -> None:
	pass


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	pass


def pre_check() -> bool:
	return True


def pre_process(mode : ProcessMode) -> bool:
	if not has_face_mapping_sticker():
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
	read_sticker_frame.cache_clear()
	prepare_sticker_frame.cache_clear()
	read_static_image.cache_clear()
	read_static_video_frame.cache_clear()


@lru_cache(maxsize = 32)
def read_sticker_frame(sticker_path : str) -> Optional[VisionFrame]:
	sticker_frame = cv2.imdecode(numpy.fromfile(sticker_path, dtype = numpy.uint8), cv2.IMREAD_UNCHANGED)

	if sticker_frame is None:
		return None
	if sticker_frame.dtype != numpy.uint8:
		sticker_frame = (sticker_frame / 257).astype(numpy.uint8)
	if sticker_frame.ndim == 2:
		return cv2.cvtColor(sticker_frame, cv2.COLOR_GRAY2BGRA)
	if sticker_frame.shape[2] == 3:
		return cv2.cvtColor(sticker_frame, cv2.COLOR_BGR2BGRA)
	return sticker_frame


@lru_cache(maxsize = 32)
def prepare_sticker_frame(sticker_path : str, mirror : bool, round_shape : bool) -> Optional[VisionFrame]:
	sticker_frame = read_sticker_frame(sticker_path)

	if sticker_frame is None:
		return None
	if mirror:
		sticker_frame = sticker_frame[:, ::-1]
	sticker_frame = numpy.ascontiguousarray(sticker_frame)

	if round_shape:
		sticker_height, sticker_width = sticker_frame.shape[:2]
		round_mask = numpy.zeros((sticker_height, sticker_width), dtype = numpy.float32)
		cv2.ellipse(round_mask, (sticker_width // 2, sticker_height // 2), (int(sticker_width * 0.46), int(sticker_height * 0.46)), 0, 0, 360, 1.0, -1)
		round_mask = cv2.GaussianBlur(round_mask, (0, 0), max(sticker_width, sticker_height) * 0.02)
		sticker_frame = sticker_frame.copy()
		sticker_frame[:, :, 3] = (sticker_frame[:, :, 3] * round_mask).astype(numpy.uint8)
	return sticker_frame


def calculate_sticker_anchor(face : Face) -> FaceStickerAnchor:
	start_x, start_y, end_x, end_y = face.bounding_box
	face_landmark_5 = face.landmark_set.get('5/68')
	left_eye, right_eye = face_landmark_5[0], face_landmark_5[1]
	face_angle = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
	face_size = max(end_x - start_x, end_y - start_y)
	face_pitch, face_yaw = estimate_head_pose(face)
	return float((start_x + end_x) / 2), float((start_y + end_y) / 2), float(face_size), face_angle, face_pitch, face_yaw


def estimate_head_pose(face : Face) -> Tuple[float, float]:
	face_landmark_68 = face.landmark_set.get('68')
	start_x, start_y, end_x, end_y = face.bounding_box
	face_size = max(end_x - start_x, end_y - start_y)
	image_points = numpy.array([ face_landmark_68[30], face_landmark_68[8], face_landmark_68[36], face_landmark_68[45], face_landmark_68[48], face_landmark_68[54] ], dtype = numpy.float64)
	focal_length = face_size * 4
	camera_matrix = numpy.array([ [ focal_length, 0, (start_x + end_x) / 2 ], [ 0, focal_length, (start_y + end_y) / 2 ], [ 0, 0, 1 ] ], dtype = numpy.float64)
	# start from a frontal head in front of the camera, otherwise the solver may land on the mirrored pose
	rotation_vector = numpy.array([ [ math.pi ], [ 0.0 ], [ 0.0 ] ])
	translation_vector = numpy.array([ [ 0.0 ], [ 0.0 ], [ focal_length * 600 / max(face_size, 1) ] ])
	is_solved, rotation_vector, translation_vector = cv2.solvePnP(HEAD_MODEL_POINTS, image_points, camera_matrix, None, rotation_vector, translation_vector, useExtrinsicGuess = True, flags = cv2.SOLVEPNP_ITERATIVE)

	if not is_solved:
		return 0.0, 0.0

	rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
	face_normal = rotation_matrix @ numpy.array([ 0.0, 0.0, 1.0 ])
	# camera space: x right, y down, z away; positive pitch looks up, positive yaw looks to the image right
	face_pitch = math.degrees(math.atan2(-face_normal[1], -face_normal[2])) + HEAD_PITCH_BIAS
	face_yaw = math.degrees(math.atan2(face_normal[0], -face_normal[2]))
	return float(numpy.clip(face_pitch, -HEAD_ANGLE_LIMIT, HEAD_ANGLE_LIMIT)), float(numpy.clip(face_yaw, -HEAD_ANGLE_LIMIT, HEAD_ANGLE_LIMIT))


def project_sticker_corners(width : float, height : float, roll : float, pitch : float, yaw : float) -> numpy.ndarray:
	# same maths as the css transform "perspective() rotateZ() rotateX() rotateY()" used by the studio preview
	perspective = STICKER_PERSPECTIVE * max(width, height)
	corners = numpy.array([ [ -width / 2, -height / 2, 0 ], [ width / 2, -height / 2, 0 ], [ width / 2, height / 2, 0 ], [ -width / 2, height / 2, 0 ] ], dtype = numpy.float64)
	radians_x, radians_y, radians_z = math.radians(pitch), math.radians(yaw), math.radians(roll)
	rotate_x = numpy.array([ [ 1, 0, 0 ], [ 0, math.cos(radians_x), -math.sin(radians_x) ], [ 0, math.sin(radians_x), math.cos(radians_x) ] ])
	rotate_y = numpy.array([ [ math.cos(radians_y), 0, math.sin(radians_y) ], [ 0, 1, 0 ], [ -math.sin(radians_y), 0, math.cos(radians_y) ] ])
	rotate_z = numpy.array([ [ math.cos(radians_z), -math.sin(radians_z), 0 ], [ math.sin(radians_z), math.cos(radians_z), 0 ], [ 0, 0, 1 ] ])
	points = corners @ rotate_y.T @ rotate_x.T @ rotate_z.T
	return points[:, :2] / (1 - points[:, 2:3] / perspective)


def paste_sticker(temp_vision_frame : VisionFrame, target_face : Face, sticker_frame : VisionFrame, sticker_options : FaceStickerOptions) -> VisionFrame:
	center_x, center_y, face_size, face_angle, face_pitch, face_yaw = calculate_sticker_anchor(target_face)
	angle = (face_angle if sticker_options.get('follow_rotation') else 0.0) + sticker_options.get('rotation')
	depth = sticker_options.get('depth') or 0.0
	radians = math.radians(angle)
	offset_x = sticker_options.get('offset_x') * face_size
	offset_y = sticker_options.get('offset_y') * face_size
	# offsets live in the face frame, so "up" stays above the head when it tilts
	center_x += offset_x * math.cos(radians) - offset_y * math.sin(radians)
	center_y += offset_x * math.sin(radians) + offset_y * math.cos(radians)

	sticker_height, sticker_width = sticker_frame.shape[:2]
	resize_scale = face_size * sticker_options.get('scale') / max(sticker_width, sticker_height)
	resize_width = max(1, round(sticker_width * resize_scale))
	resize_height = max(1, round(sticker_height * resize_scale))
	resize_frame = cv2.resize(sticker_frame, (resize_width, resize_height), interpolation = cv2.INTER_AREA if resize_scale < 1 else cv2.INTER_CUBIC)

	# premultiplied alpha keeps the borders from bleeding black while rotating
	resize_alpha = resize_frame[:, :, 3:4].astype(numpy.float32) / 255
	resize_frame = numpy.dstack([ resize_frame[:, :, :3].astype(numpy.float32) * resize_alpha, resize_alpha * 255 ])

	corners = project_sticker_corners(resize_width, resize_height, angle, face_pitch * depth, face_yaw * depth) + [ center_x, center_y ]
	frame_height, frame_width = temp_vision_frame.shape[:2]
	start_x = max(0, int(math.floor(corners[:, 0].min())))
	start_y = max(0, int(math.floor(corners[:, 1].min())))
	end_x = min(frame_width, int(math.ceil(corners[:, 0].max())))
	end_y = min(frame_height, int(math.ceil(corners[:, 1].max())))

	if end_x <= start_x or end_y <= start_y:
		return temp_vision_frame

	source_corners = numpy.array([ [ 0, 0 ], [ resize_width, 0 ], [ resize_width, resize_height ], [ 0, resize_height ] ], dtype = numpy.float32)
	perspective_matrix = cv2.getPerspectiveTransform(source_corners, (corners - [ start_x, start_y ]).astype(numpy.float32))
	warp_frame = cv2.warpPerspective(resize_frame, perspective_matrix, (end_x - start_x, end_y - start_y), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = (0, 0, 0, 0))
	warp_alpha = warp_frame[:, :, 3:4] / 255

	paste_vision_frame = temp_vision_frame.copy()
	crop_vision_frame = paste_vision_frame[start_y:end_y, start_x:end_x].astype(numpy.float32)
	paste_vision_frame[start_y:end_y, start_x:end_x] = (crop_vision_frame * (1 - warp_alpha) + warp_frame[:, :, :3]).clip(0, 255).astype(numpy.uint8)
	return paste_vision_frame


def process_frame(inputs : dict) -> ProcessorOutputs:
	target_vision_frame = inputs.get('target_vision_frame')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')

	for target_face, face_mapping in select_sticker_faces(target_vision_frame):
		sticker_options = face_mapping.get('sticker_options')
		sticker_frame = prepare_sticker_frame(face_mapping.get('sticker_path'), bool(sticker_options.get('mirror')), bool(sticker_options.get('round')))

		if sticker_frame is not None:
			target_face = scale_face(target_face, target_vision_frame, temp_vision_frame)
			temp_vision_frame = paste_sticker(temp_vision_frame, target_face, sticker_frame, sticker_options)

	return temp_vision_frame, temp_vision_mask
