from argparse import ArgumentParser
from functools import lru_cache
from typing import Callable, Dict, Optional

import cv2
import numpy

from facefusion import logger, state_manager, translator
from facefusion.filesystem import in_directory, is_image, is_video, same_file_extension
from facefusion.processors.types import ProcessorOutputs
from facefusion.types import ApplyStateItem, Args, InferencePool, ProcessMode, VisionFrame


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
	if not get_frame_filter():
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
	create_vignette.cache_clear()


def get_frame_filter() -> Optional[str]:
	frame_filter = state_manager.get_item('frame_filter')
	return frame_filter if frame_filter in FRAME_FILTER_PAINTERS else None


@lru_cache(maxsize = 8)
def create_vignette(width : int, height : int, strength : float) -> numpy.ndarray:
	grid_y, grid_x = numpy.mgrid[0:height, 0:width].astype(numpy.float32)
	distance = numpy.sqrt(((grid_x - width / 2) / (width / 2)) ** 2 + ((grid_y - height / 2) / (height / 2)) ** 2) / numpy.sqrt(2)
	return (1 - strength * distance ** 2.2)[:, :, None]


def create_grain(width : int, height : int, amount : float) -> numpy.ndarray:
	# drawn at half size: cheaper, and the grain gets the size of real film grain
	grain = numpy.random.normal(0, amount, (max(1, height // 2), max(1, width // 2))).astype(numpy.float32)
	return cv2.resize(grain, (width, height), interpolation = cv2.INTER_LINEAR)[:, :, None]


def paint_comic(vision_frame : VisionFrame) -> VisionFrame:
	height, width = vision_frame.shape[:2]
	small_frame = cv2.resize(vision_frame, (max(1, width // 2), max(1, height // 2)), interpolation = cv2.INTER_AREA)

	for _ in range(2):
		small_frame = cv2.bilateralFilter(small_frame, 7, 60, 7)
	small_frame = (small_frame // 40) * 40 + 20
	colour_frame = cv2.resize(small_frame, (width, height), interpolation = cv2.INTER_LINEAR)
	gray_frame = cv2.medianBlur(cv2.cvtColor(vision_frame, cv2.COLOR_BGR2GRAY), 7)
	edge_mask = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, max(3, width // 140 | 1), 7)
	return cv2.bitwise_and(colour_frame, colour_frame, mask = edge_mask)


def paint_silent_film(vision_frame : VisionFrame) -> VisionFrame:
	height, width = vision_frame.shape[:2]
	gray_frame = cv2.cvtColor(vision_frame, cv2.COLOR_BGR2GRAY).astype(numpy.float32)[:, :, None]
	# a contrasty print that flickers a little from one image to the next
	gray_frame = (gray_frame - 128) * 1.3 + 128 + numpy.random.uniform(-7, 7)
	gray_frame = gray_frame * create_vignette(width, height, 0.75) + create_grain(width, height, 9)
	film_frame = gray_frame * numpy.array([ 0.9, 0.97, 1.0 ], dtype = numpy.float32)

	if numpy.random.random() < 0.35:
		scratch_x = numpy.random.randint(0, width)
		film_frame[:, scratch_x:scratch_x + max(1, width // 900)] += 70
	return film_frame.clip(0, 255).astype(numpy.uint8)


def paint_sketch(vision_frame : VisionFrame) -> VisionFrame:
	width = vision_frame.shape[1]
	gray_frame = cv2.cvtColor(vision_frame, cv2.COLOR_BGR2GRAY)
	blur_frame = cv2.GaussianBlur(255 - gray_frame, (0, 0), max(1.0, width / 250))
	# colour dodge: flat areas turn to paper, edges stay as pencil strokes
	sketch_frame = cv2.divide(gray_frame, 255 - blur_frame, scale = 256).astype(numpy.float32)[:, :, None]
	sketch_frame = 255 - (255 - sketch_frame) * 1.35
	return (sketch_frame * numpy.array([ 0.93, 0.97, 1.0 ], dtype = numpy.float32)).clip(0, 255).astype(numpy.uint8)


def paint_pixel(vision_frame : VisionFrame) -> VisionFrame:
	height, width = vision_frame.shape[:2]
	block_size = max(4, width // 160)
	small_frame = cv2.resize(vision_frame, (max(1, width // block_size), max(1, height // block_size)), interpolation = cv2.INTER_AREA)
	small_frame = (small_frame // 48) * 48 + 24
	return cv2.resize(small_frame, (width, height), interpolation = cv2.INTER_NEAREST)


def paint_thermal(vision_frame : VisionFrame) -> VisionFrame:
	width = vision_frame.shape[1]
	gray_frame = cv2.GaussianBlur(cv2.cvtColor(vision_frame, cv2.COLOR_BGR2GRAY), (0, 0), max(1.0, width / 400))
	gray_frame = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX)
	return cv2.applyColorMap(gray_frame, cv2.COLORMAP_INFERNO)


def paint_night_vision(vision_frame : VisionFrame) -> VisionFrame:
	height, width = vision_frame.shape[:2]
	gray_frame = cv2.equalizeHist(cv2.cvtColor(vision_frame, cv2.COLOR_BGR2GRAY)).astype(numpy.float32)[:, :, None]
	gray_frame = gray_frame * 0.9 + 25 + create_grain(width, height, 14)
	# a dim line every few rows, like the tube of the scope
	gray_frame[::3] *= 0.85
	night_frame = gray_frame * numpy.array([ 0.25, 1.0, 0.35 ], dtype = numpy.float32) * create_vignette(width, height, 0.95)
	return night_frame.clip(0, 255).astype(numpy.uint8)


def paint_vhs(vision_frame : VisionFrame) -> VisionFrame:
	height, width = vision_frame.shape[:2]
	shift = max(2, width // 320)
	# the tape blurs and offsets the colour much more than the light
	ycrcb_frame = cv2.cvtColor(vision_frame, cv2.COLOR_BGR2YCrCb)
	ycrcb_frame[:, :, 1:] = cv2.blur(ycrcb_frame[:, :, 1:], (shift * 4 + 1, 1))
	ycrcb_frame[:, :, 1] = numpy.roll(ycrcb_frame[:, :, 1], shift, axis = 1)
	ycrcb_frame[:, :, 2] = numpy.roll(ycrcb_frame[:, :, 2], -shift, axis = 1)
	vhs_frame = cv2.cvtColor(ycrcb_frame, cv2.COLOR_YCrCb2BGR).astype(numpy.float32)
	vhs_frame[::2] *= 0.86
	vhs_frame += create_grain(width, height, 6)

	# now and then a band of the picture slips sideways
	if numpy.random.random() < 0.3:
		band_y = numpy.random.randint(0, height)
		band_height = max(2, height // 30)
		vhs_frame[band_y:band_y + band_height] = numpy.roll(vhs_frame[band_y:band_y + band_height], numpy.random.randint(shift * 2, shift * 6), axis = 1)
	return vhs_frame.clip(0, 255).astype(numpy.uint8)


FRAME_FILTER_PAINTERS : Dict[str, Callable[[VisionFrame], VisionFrame]] =\
{
	'comic': paint_comic,
	'sketch': paint_sketch,
	'silent_film': paint_silent_film,
	'pixel': paint_pixel,
	'thermal': paint_thermal,
	'night_vision': paint_night_vision,
	'vhs': paint_vhs
}


def process_frame(inputs : dict) -> ProcessorOutputs:
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')
	frame_filter = get_frame_filter()

	if frame_filter:
		temp_vision_frame = FRAME_FILTER_PAINTERS.get(frame_filter)(temp_vision_frame)
	return temp_vision_frame, temp_vision_mask
