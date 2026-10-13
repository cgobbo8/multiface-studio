from typing import List, Optional, Tuple

import numpy

from facefusion import state_manager
from facefusion.common_helper import get_first
from facefusion.face_analyser import get_average_face, get_many_faces
from facefusion.types import Face, FaceMapping, VisionFrame
from facefusion.vision import read_static_image, read_static_images


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
	# every mapping takes part so swapped, stickered and untouched people never steal each other's faces
	target_faces = get_many_faces([ target_vision_frame ])
	return match_face_mappings(target_faces, get_face_mappings(), state_manager.get_item('reference_face_distance'))


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
