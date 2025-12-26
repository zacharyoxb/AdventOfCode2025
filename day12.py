""" Day 12 """

from dataclasses import dataclass
import re
from typing import Callable, Optional, TypeAlias
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

PlacementArea: TypeAlias = NDArray[np.int8]
PresentMatrix: TypeAlias = NDArray[np.int8]
PresentMatrices: TypeAlias = list[NDArray[np.int8]]


def get_all_orientations(present_matrix: PresentMatrix):
    """ Gets all possible orientations of a present """
    orientations = set()

    # Gets every rotation from k=0 to k=3
    for k in range(4):
        # Rotate
        rotated = np.rot90(present_matrix, k)
        orientations.add(tuple(rotated.flatten()))

        # Flip
        mirrored = np.fliplr(rotated)
        orientations.add(tuple(mirrored.flatten()))

    # Convert from hashable tuple back to matrix
    return [np.array(orientation).reshape(present_matrix.shape)
            for orientation in orientations]


def get_matrix_mask(matrix: PresentMatrix, area_shape: tuple[int, int], x: int, y: int):
    """ Create a mask by placing matrix in a larger array at position (x, y). """
    mask = np.zeros(area_shape, dtype=matrix.dtype)

    height, width = matrix.shape
    target_h, target_w = area_shape

    # Calculate slice boundaries (clamp to bounds)
    start_x = max(0, x - 1)
    end_x = min(target_h, start_x + height)

    start_y = max(0, y - 1)
    end_y = min(target_w, start_y + width)

    # Adjust matrix slice if needed (when near edges)
    matrix_start_x = max(0, 1 - x)
    matrix_start_y = max(0, 1 - y)

    matrix_end_x = min(height, height - (start_x + height - target_h))
    matrix_end_y = min(width, width - (start_y + width - target_w))

    # Place matrix slice
    mask[start_x:end_x, start_y:end_y] = \
        matrix[matrix_start_x:matrix_end_x, matrix_start_y:matrix_end_y]

    return mask


def find_best_placement(placement_area: PlacementArea,
                        present_matrix: PresentMatrix
                        ) -> Optional[tuple[int, Callable]]:
    """ Gets best fitting placement for present_matrix if possible
    Returns the score of the mask and the mask of the present if successful """

    # Matrix is empty
    if not present_matrix.any():
        return None

    # get padded placement_area
    padded = np.pad(placement_area, pad_width=1,
                    mode='constant', constant_values=1)

    rows, cols = placement_area.shape

    best_score = -1
    best_x, best_y = -1, -1

    # Iterate through centre positions
    for x in range(rows):
        for y in range(cols):
            window = padded[x:x+3, y:y+3]

            if np.any(window & present_matrix):
                continue

            # Compute score (max score is 9 for 3x3 grid)
            score = 9 - np.count_nonzero(window == 0)

            if score > best_score:
                best_score = score
                best_x, best_y = x, y

    if best_score == -1:
        return None

    def mask_call():
        return get_matrix_mask(present_matrix, (rows, cols), best_x, best_y)

    return best_score, mask_call


@dataclass
class Placement:
    """ Represents a placement of a present. """
    score: int
    present_idx: int
    mask_function: Callable


def presents_can_fit(
    placement_area: PlacementArea,
    present_matrices: PresentMatrices,
    present_count: list[int]
):
    """ Checks if all presents can fit in area """

    # incase any of the present_counts start off at 0
    present_matrices = [present_matrix if present_count[i] >
                        0 else np.zeros((3, 3), np.int8)
                        for i, present_matrix in enumerate(present_matrices)]

    # get all orientations for all presents
    orientations = []
    for matrix in present_matrices:
        if len(matrix) > 0:
            matrix_orientations = get_all_orientations(matrix)
            orientations.append(matrix_orientations)
        else:
            orientations.append([])

    # while still more presents to fit
    while sum(present_count) > 0:
        # if area remaining is less than area of presents to place, return early
        area_remaining = placement_area.size - np.count_nonzero(placement_area)
        total_present_area = 0
        for present, count in zip(present_matrices, present_count):
            total_present_area += (present.size -
                                   np.count_nonzero(present)) * count
        if total_present_area > area_remaining:
            return False

        # stores score, function to get mask, and index of present
        placements: list[Placement] = []

        # find the best fit out of all presents and all orientations
        for n_present, present_orientations in enumerate(orientations):
            for present_orientation in present_orientations:
                score_and_func = find_best_placement(
                    placement_area, present_orientation)

                # If present cannot fit, skip
                if not score_and_func:
                    continue

                score, func = score_and_func

                # Otherwise add to placements
                placements.append(Placement(score, n_present, func))

        # No possible placements
        if not placements:
            return False

        # get highest scoring placement, add to area
        placement = max(placements, key=lambda placement: placement.score)
        mask = placement.mask_function()
        placement_area ^= mask

        # lower counter
        present_count[placement.present_idx] -= 1

        # if present counter at 0, set present to 0s so it won't be selected
        if present_count[placement.present_idx] == 0:
            present_matrices[placement.present_idx] = np.zeros(
                (3, 3), np.uint8)

    return True


def day12(present_matrices: PresentMatrices, placement_info: list[str, str, list[int]]):
    """ Main function """
    fit_count = 0

    for width_str, height_str, present_count in tqdm(placement_info):
        height, width = int(height_str), int(width_str)
        placement_area = []
        for _ in range(height):
            placement_area.append([0] * width)
        present_count = list(map(int, present_count.strip().split(' ')))

        placement_area = np.array(placement_area, np.int8)
        present_matrices = np.array(present_matrices, np.int8)

        if presents_can_fit(
                placement_area, present_matrices, present_count):
            fit_count += 1

    print(fit_count)


if __name__ == "__main__":
    raw_lines = []
    with open("inputs/day12/input.txt", encoding="UTF-8") as f:
        raw_lines = f.readlines()
    FULL_TEXT = " ".join(raw_lines)

    # get all shapes matrixes from text
    SHAPE_PATTERN = r"\d+:\s*((?:[.#]{3}\s*){3})"
    shape_matches = re.findall(SHAPE_PATTERN, FULL_TEXT)

    extracted_present_matrices: list[list[int]] = []

    for extracted_matrix in shape_matches:
        split_matrix = extracted_matrix.split()

        binary_matrix = [[1 if char == '#' else 0 for char in row]
                         for row in split_matrix]
        extracted_present_matrices.append(binary_matrix)

    # get all areas we will insert into and indexes of shapes
    REGION_PATTERN = r'(\d+)x(\d+):\s*((?:\d+\s*)+\n)'
    region_matches = re.findall(REGION_PATTERN, FULL_TEXT)

    day12(extracted_present_matrices, region_matches)
