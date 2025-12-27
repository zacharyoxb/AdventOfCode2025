""" Day 12 """

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import re
from typing import Optional, TypeAlias
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

PresentMatrix: TypeAlias = NDArray[np.int8]
PresentMatrices: TypeAlias = list[NDArray[np.int8]]

WINDOW_ROW_MASK = 0x7  # 111 in binary


@dataclass
class Placement:
    """ Represents a placement of a present. """
    score: int
    bitmask: int


def find_best_placement(placement_area: int,
                        area_size: tuple[int, int],
                        present: int,
                        ) -> Optional[Placement]:
    """ Gets best fitting placement for present_matrix if possible
    Returns the score of the mask and the mask of the present if successful

    Args:
        placement_area (int): binary number representing the space of the area
        area_size (tuple[int, int]): width, height of the placement area
        present (int): int representing the dimensions of the present in binary

    Returns:
        Optional[Placement]: If not None, best possible placement of present.


    Extracting windows (w = width):
    [i-w-1, i-w, i-w+1]
    [i-1,    i,    i+1]
    [i+w-1, i+w, i+w+1]
    """

    width, height = area_size

    best_score = -1
    best_bitmask = 0

    for i in range(width * height):
        # skip loop if 3x3 square with centre i overlaps horizontal boundary
        if (i % width) == 0 or ((i+1) % width) == 0:
            continue

        # skip loop if 3x3 square with centre i overlaps vertical boundary
        if (i // width) == 0 or (i // width) == height-1:
            continue

        # get 3 rows from window with i in centre
        window = 0
        for row_offset in range(-width, width+1, width):
            window = window << 3
            shift = (i-1) + row_offset
            window |= (placement_area >> shift) & WINDOW_ROW_MASK

        # collision detected
        if present & window:
            continue

        # Compute score (max score is 9 for 3x3 grid)
        score = bin(present ^ window).count('1')

        # if score is 9 or score is more than best score, make mask
        bitmask = 0
        if score == 9 or score > best_score:
            for row_offset in range(-width, width+1, width):
                shift = (i-1) + row_offset
                bitmask |= (present & WINDOW_ROW_MASK) << shift
                present = present >> 3

        # If score is 9, immediately return
        if score == 9:
            return Placement(score, bitmask)

        if score > best_score:
            best_score = score
            best_bitmask = bitmask

    if best_score == -1:
        return None

    return Placement(best_score, best_bitmask)


def presents_can_fit(
    area_size: tuple[int, int],
    orientations: list[list[int]],
    present_count: list[int]
):
    """ Checks if all presents can fit in area. """

    # If any of counts are 0, replace ints with a 0
    orientations = [orientation if present_count[i] >
                    0 else [0] for i, orientation in enumerate(orientations)]

    area = 0

    while sum(present_count) > 0:
        best_present_idx = -1
        best_placement: Placement = None

        # check every orientation of every present
        for npresent, present_orientations in enumerate(orientations):
            # no more of this present to place
            if present_orientations == [0]:
                continue

            # all orientations for single present
            for present_orientation in present_orientations:
                placement = find_best_placement(
                    area, area_size, present_orientation)

                # if present doesn't fit
                if not placement:
                    continue

                # if placement is better than best
                if best_placement is None or placement.score > best_placement.score:
                    best_present_idx = npresent
                    best_placement = placement

        # if no fits found, return false
        if best_present_idx == -1:
            return False

        # place best present in bitboard
        area |= best_placement.bitmask

        # remove from count
        present_count[best_present_idx] -= 1
        if present_count[best_present_idx] == 0:
            orientations[best_present_idx] = [0]

    return True


def get_all_orientations(present_matrix: PresentMatrix):
    """ Gets all possible orientations of a present """
    # convert to np array to make rotation easier
    present_matrix = np.array(present_matrix)
    orientations = set()

    # Gets every rotation from k=0 to k=3
    for k in range(4):
        # Rotate
        rotated = np.rot90(present_matrix, k)
        orientations.add(tuple(rotated.flatten()))

        # Flip
        mirrored = np.fliplr(rotated)
        orientations.add(tuple(mirrored.flatten()))

    reshaped = [np.array(orientation).reshape(present_matrix.shape)
                for orientation in orientations]

    return reshaped


def matrix_to_bitmask(matrix):
    """ Convert 3x3 matrix to 9-bit integer """
    bits = 0
    for i, val in enumerate(matrix.flatten()):
        if val:
            bits |= 1 << i
    return bits


def _get_bitmask_orientations(present_matrices: PresentMatrices) -> list[list[int]]:
    """ Gets bitmask representations of all unique orientations of matrixes"""
    # get bitmask representations for orientations of presents
    orientations: list[int] = []
    for matrix in present_matrices:
        matrix_orientations = get_all_orientations(matrix)
        bitmasks = list(map(matrix_to_bitmask, matrix_orientations))
        orientations.append(bitmasks)
    return orientations


def _process_task(args):
    return presents_can_fit(*args)


def _get_args(present_matrices: list[list[int]],
              placement_info: list[str, str, list[int]]
              ):
    args = []
    for width_str, height_str, present_count_str in placement_info:
        height, width = int(height_str), int(width_str)
        orientations = _get_bitmask_orientations(present_matrices)
        present_count_str = present_count_str.split()
        present_count = list(map(int, present_count_str))
        args.append(((height, width), orientations, present_count))
    return args


def day12(present_matrices: list[list[int]], placement_info: list[str, str, list[int]]):
    """ Main function """
    fit_count = 0

    args_list = _get_args(present_matrices, placement_info)

    with ProcessPoolExecutor(1) as executor:
        futures = [executor.submit(_process_task, args) for args in args_list]

        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                try:
                    if future.result():
                        fit_count += 1
                except TimeoutError as e:
                    print(f"Timeout error! {e}")
                finally:
                    pbar.update(1)

    print(
        f"All iterations complete, total areas which fit all presents: {fit_count}")


if __name__ == "__main__":
    raw_lines = []
    with open("inputs/day12/testinput.txt", encoding="UTF-8") as f:
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
    REGION_PATTERN = r'(\d+)x(\d+):\s*((?:\d+\s*)+)(?=\n|$)'
    region_matches = re.findall(REGION_PATTERN, FULL_TEXT)

    day12(extracted_present_matrices, region_matches)
