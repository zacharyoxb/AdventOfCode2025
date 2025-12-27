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

PresentOrientation: TypeAlias = list[int]

WINDOW_ROW_MASK = 0x7  # 111 in binary


@dataclass
class Placement:
    """ Represents a placement of a present. """
    score: int
    bitmask: PresentOrientation
    bitmask_range: tuple[int, int]


def find_best_placement(placement_area: list[int],
                        area_size: tuple[int, int],
                        present: PresentOrientation,
                        ) -> Optional[Placement]:
    """ Gets best fitting placement for present_matrix if possible
    Returns the score of the mask and the mask of the present if successful

    Args:
        placement_area (list[int]): binary numbers representing each row of area
        area_size (tuple[int, int]): width, height of the placement area
        present (PresentOrientation): list of ints representing each present row

    Returns:
        Optional[Placement]: If not None, best possible placement of present.
    """

    width, height = area_size

    best_score = -1
    best_bitmask = []
    best_bitmask_range = (-1, -1)

    for i in range(width * height):
        # skip loop if 3x3 square with centre i overlaps horizontal boundary
        if (i % width) == 0 or ((i+1) % width) == 0:
            continue

        # skip loop if 3x3 square with centre i overlaps vertical boundary
        if (i // width) == 0 or (i // width) == height-1:
            continue

        # get 3 rows from window with i in centre
        window: list[int] = []
        row_idx = i // width
        for row in range(row_idx-1, row_idx+2):
            shift = (i-1) % width
            window.append((placement_area[row] >> shift) & WINDOW_ROW_MASK)

        # if any collisions exist
        if any(map(lambda pair: pair[0] & pair[1], zip(present, window))):
            continue

        # Compute score (max score is 9 for 3x3 grid)
        score = 0
        for present_row, window_row in zip(present, window):
            score += bin(present_row ^ window_row).count('1')

        # if score is more than best score, make mask
        bitmask = []
        if score > best_score:
            for present_row, window_row in zip(present, window):
                bitmask.append(present_row ^ window_row)

        # If score is 9, immediately return
        if score == 9:
            return Placement(score, bitmask, (row_idx-1, row_idx+2))

        if score > best_score:
            best_score = score
            best_bitmask = bitmask
            best_bitmask_range = (row_idx-1, row_idx+2)

    if best_score == -1:
        return None

    return Placement(best_score, best_bitmask, best_bitmask_range)


def presents_can_fit(
    area_size: tuple[int, int],
    orientations: list[list[PresentOrientation]],
    present_count: list[int]
):
    """ Checks if all presents can fit in area. """

    # if any of present counts are 0, replace ints with a 0
    orientations = [orientation if present_count[i] >
                    0 else [0] for i, orientation in enumerate(orientations)]

    # one int for each row
    area = [0] * area_size[1]

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
        for present_row, idx in enumerate(range(*best_placement.bitmask_range)):
            area[idx] |= best_placement.bitmask[present_row]

        # remove from count
        present_count[best_present_idx] -= 1
        if present_count[best_present_idx] == 0:
            orientations[best_present_idx] = [0]

    return True


def get_all_orientations(present_matrix: PresentMatrix) -> PresentMatrix:
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
    """ Convert 3x3 matrix to three 3-digit binary numbers (one per row) """
    binary_rows = []

    for row in matrix:
        # Convert each row to a 3-digit binary number
        binary_str = ''
        for val in row:
            binary_str += '1' if val else '0'

        binary_rows.append(int(binary_str, 2))

    return binary_rows


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

    with ProcessPoolExecutor() as executor:
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
