import math
from argparse import ArgumentParser
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy

from facefusion import logger, state_manager, translator
from facefusion.face_analyser import scale_face
from facefusion.face_mapper import clear_face_tracks, get_face_mappings, get_tracked_faces, has_face_mapping_sticker, prepare_face_tracks, select_sticker_faces
from facefusion.filesystem import in_directory, is_image, is_video, same_file_extension
from facefusion.processors.types import ProcessorOutputs
from facefusion.types import ApplyStateItem, Args, Face, FaceMapping, FaceStickerOptions, InferencePool, ProcessMode, VisionFrame
from facefusion.vision import read_static_image, read_static_video_frame

FaceStickerAnchor = Tuple[float, float, float, float, float, float]
# head pose from the detector's 5 points, gains calibrated against solvePnP on sharp faces
HEAD_YAW_GAIN = 2.8
HEAD_PITCH_GAIN = 1.2
HEAD_PITCH_NEUTRAL = 0.69
HEAD_ANGLE_LIMIT = 75.0
STICKER_PERSPECTIVE = 3.0
# frames on each side averaged against jitter, never across a cut
STICKER_TRACK_SMOOTHING = 1
STICKER_TRACKS : Dict[int, numpy.ndarray] = {}


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
	STICKER_TRACKS.clear()
	clear_face_tracks()
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
	# the detector's own points: '5/68' switches to the 68 landmarks around their confidence threshold and makes the tilt flicker
	face_landmark_5 = face.landmark_set.get('5')
	left_eye, right_eye = face_landmark_5[0], face_landmark_5[1]
	face_angle = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
	face_size = max(end_x - start_x, end_y - start_y)
	face_pitch, face_yaw = estimate_head_pose(face)
	return float((start_x + end_x) / 2), float((start_y + end_y) / 2), float(face_size), face_angle, face_pitch, face_yaw


def estimate_head_pose(face : Face) -> Tuple[float, float]:
	# the 68 landmarks turn to noise on small or blurry faces (low landmarker score), the detector's 5 points stay sane
	face_landmark_5 = face.landmark_set.get('5').astype(numpy.float64)
	left_eye, right_eye = face_landmark_5[0], face_landmark_5[1]
	roll = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
	rotation_matrix = numpy.array([ [ math.cos(roll), math.sin(roll) ], [ -math.sin(roll), math.cos(roll) ] ])
	# upright face frame centred on the nose: x to the image right, y down
	face_points = (face_landmark_5 - face_landmark_5[2]) @ rotation_matrix.T
	eye_center = (face_points[0] + face_points[1]) / 2
	mouth_center = (face_points[3] + face_points[4]) / 2
	eye_distance = max(face_points[1][0] - face_points[0][0], 1e-6)
	eye_mouth_distance = max(mouth_center[1] - eye_center[1], 1e-6)
	# the nose drifts sideways when the head turns, and towards the eyes when it looks up
	yaw_ratio = -eye_center[0] / eye_distance
	pitch_ratio = -eye_center[1] / eye_mouth_distance
	face_yaw = math.degrees(math.asin(numpy.clip(yaw_ratio * HEAD_YAW_GAIN, -1, 1)))
	face_pitch = math.degrees(math.asin(numpy.clip((HEAD_PITCH_NEUTRAL - pitch_ratio) * HEAD_PITCH_GAIN, -1, 1)))
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


def paste_sticker(temp_vision_frame : VisionFrame, sticker_anchor : FaceStickerAnchor, sticker_frame : VisionFrame, sticker_options : FaceStickerOptions) -> VisionFrame:
	center_x, center_y, face_size, face_angle, face_pitch, face_yaw = sticker_anchor
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


def paste_mapping_sticker(temp_vision_frame : VisionFrame, sticker_anchor : FaceStickerAnchor, face_mapping : FaceMapping) -> VisionFrame:
	sticker_options = face_mapping.get('sticker_options')
	sticker_frame = prepare_sticker_frame(face_mapping.get('sticker_path'), bool(sticker_options.get('mirror')), bool(sticker_options.get('round')))

	if sticker_frame is None:
		return temp_vision_frame
	return paste_sticker(temp_vision_frame, sticker_anchor, sticker_frame, sticker_options)


def prepare_frames(temp_frame_paths : List[str]) -> None:
	# videos get one track per person: glitches dropped, short misses bridged, motion smoothed
	STICKER_TRACKS.clear()
	face_mappings = get_face_mappings()
	sticker_mapping_indices = [ index for index, face_mapping in enumerate(face_mappings) if face_mapping.get('sticker_path') ]

	if not sticker_mapping_indices:
		return

	# the shared face tracks already hold every detection plus the bridged misses
	prepare_face_tracks(temp_frame_paths)
	raw_anchors = numpy.full((len(face_mappings), len(temp_frame_paths), 6), numpy.nan)

	for frame_number, tracked_faces in enumerate(get_tracked_faces()):
		for mapping_index, tracked_face in tracked_faces.items():
			if mapping_index in sticker_mapping_indices:
				raw_anchors[mapping_index, frame_number] = calculate_sticker_anchor(tracked_face)

	video_fps = state_manager.get_item('output_video_fps') or 25.0

	for mapping_index in sticker_mapping_indices:
		STICKER_TRACKS[mapping_index] = build_sticker_track(raw_anchors[mapping_index], video_fps)


def build_sticker_track(raw_anchors : numpy.ndarray, video_fps : float) -> numpy.ndarray:
	# short misses are already bridged by the shared face tracks, here the motion only gets smoothed
	frame_total = len(raw_anchors)
	sticker_track = raw_anchors.copy()
	known_numbers = numpy.flatnonzero(~numpy.isnan(raw_anchors[:, 0]))
	segment_ids = numpy.full(frame_total, -1)
	segment_id = -1
	previous_number = None

	# a hole or a jump (cut, another person taking over) starts a new segment, nothing is blended across it
	for frame_number in known_numbers:
		if previous_number is None or frame_number != previous_number + 1 or is_anchor_jump(raw_anchors[previous_number], raw_anchors[frame_number]):
			segment_id += 1
		segment_ids[frame_number] = segment_id
		previous_number = frame_number

	# the roll wraps around at ±180°: unwrapped within each segment, an upside-down head is not averaged back to upright
	for segment in range(segment_id + 1):
		segment_numbers = numpy.flatnonzero(segment_ids == segment)
		sticker_track[segment_numbers, 3] = numpy.degrees(numpy.unwrap(numpy.radians(raw_anchors[segment_numbers, 3])))

	angle_track = sticker_track[:, 3:].copy()

	# a lone pose glitch between two agreeing neighbours takes their angles
	for frame_number in known_numbers:
		if 0 < frame_number < frame_total - 1 and segment_ids[frame_number - 1] == segment_ids[frame_number] == segment_ids[frame_number + 1]:
			neighbour_angles = (angle_track[frame_number - 1] + angle_track[frame_number + 1]) / 2
			is_agreeing = numpy.abs(angle_track[frame_number - 1] - angle_track[frame_number + 1]).max() < 15
			is_glitch = numpy.abs(angle_track[frame_number] - neighbour_angles).max() > 25

			if is_agreeing and is_glitch:
				sticker_track[frame_number, 3:] = neighbour_angles

	smooth_track = sticker_track.copy()

	for frame_number in known_numbers:
		window_numbers = [ number for number in range(frame_number - STICKER_TRACK_SMOOTHING, frame_number + STICKER_TRACK_SMOOTHING + 1) if 0 <= number < frame_total and segment_ids[number] == segment_ids[frame_number] ]
		smooth_track[frame_number] = sticker_track[window_numbers].mean(axis = 0)
	smooth_track[:, 3] = (smooth_track[:, 3] + 180) % 360 - 180
	return smooth_track


def is_anchor_jump(anchor : numpy.ndarray, next_anchor : numpy.ndarray) -> bool:
	face_size = (anchor[2] + next_anchor[2]) / 2
	return math.dist(anchor[:2], next_anchor[:2]) > face_size * 0.35 or not 0.75 < next_anchor[2] / max(anchor[2], 1e-6) < 1.33


def process_frame(inputs : dict) -> ProcessorOutputs:
	target_vision_frame = inputs.get('target_vision_frame')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')
	frame_number = inputs.get('frame_number')

	if STICKER_TRACKS and frame_number is not None:
		face_mappings = get_face_mappings()

		for mapping_index, sticker_track in STICKER_TRACKS.items():
			if frame_number < len(sticker_track) and not numpy.isnan(sticker_track[frame_number, 0]):
				temp_vision_frame = paste_mapping_sticker(temp_vision_frame, tuple(sticker_track[frame_number]), face_mappings[mapping_index])
		return temp_vision_frame, temp_vision_mask

	# photos and previews: no track, the head found in this very frame is used
	for target_face, face_mapping in select_sticker_faces(target_vision_frame):
		target_face = scale_face(target_face, target_vision_frame, temp_vision_frame)
		temp_vision_frame = paste_mapping_sticker(temp_vision_frame, calculate_sticker_anchor(target_face), face_mapping)

	return temp_vision_frame, temp_vision_mask
