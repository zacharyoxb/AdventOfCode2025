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
L_WINDOW_CELL = 0x4  # 100 in binary
R_WINDOW_CELL = 0x1  # 001 in binary


@dataclass
class Placement:
    """ Represents a placement of a present. """
    score: float
    bitmask: PresentOrientation
    bitmask_range: tuple[int, int]


def _window_out_of_bounds(i: int, width: int, height: int) -> bool:
    # skip loop if 3x3 square with centre i overlaps horizontal boundary
    if (i % width) == 0 or ((i+1) % width) == 0:
        return True

    # skip loop if 3x3 square with centre i overlaps vertical boundary
    if (i // width) == 0 or (i // width) == height-1:
        return True
    return False


def _get_window(placement_area: list[int], i: int, width: int) -> list[int]:
    # get 3 rows from window with i in centre
    window: list[int] = []
    top_row_idx = (i // width) - 1
    bottom_row_idx = (i // width) + 1
    for row in range(top_row_idx, bottom_row_idx+1):
        shift = (i-1) % width
        window.append((placement_area[row] >> shift) & WINDOW_ROW_MASK)
    return window


def _get_vertical_adjacency_score(
        placement_area: list[int],
        i: int,
        width: int,
        present: PresentOrientation
) -> float:
    adjacent_cells = 0

    adj_idx = (i // width) - 2, (i // width) + 2
    shift = (i-1) % width

    # get vertically adjacent points above top (MASK if out of bounds)
    top_adjacent = WINDOW_ROW_MASK
    if adj_idx[0] > -1:
        top_adjacent = (placement_area[adj_idx[0]]
                        >> shift) & WINDOW_ROW_MASK
    adjacent_cells += bin(top_adjacent & present[0]).count('1')

    # get vertically adjacent points below bottom (MASK if out of bounds)
    adj_mask = WINDOW_ROW_MASK
    if adj_idx[1] < len(placement_area):
        adj_mask = (
            placement_area[adj_idx[1]] >> shift) & WINDOW_ROW_MASK
    adjacent_cells += bin(adj_mask & present[2]).count('1')

    return adjacent_cells / 10


def _get_horizontal_adjacency_score(
        placement_area: list[int],
        i: int,
        width: int,
        present: PresentOrientation
) -> float:
    adj_idx = (i // width) - 1, (i // width), (i // width) + 1
    l_shift = (i-1) % width
    r_shift = (i+1) % width

    adj_to_left_boundary = ((i % width) - 2) < 0
    adj_to_right_boundary = ((i % width) + 2) >= width

    adjacent_cells = 0

    # for every index in horizontal adj idx
    for nrow, idx in enumerate(adj_idx):
        left_window_cell = present[nrow] & L_WINDOW_CELL
        right_window_cell = present[nrow] & R_WINDOW_CELL

        # check if adjacent to boundary/present on the left side
        if adj_to_left_boundary and left_window_cell:
            adjacent_cells += 1
        elif not adj_to_left_boundary:
            if (placement_area[idx] >> l_shift-1) & 1 and left_window_cell:
                adjacent_cells += 1

        # check if adjacent to boundary/present on the right side
        if adj_to_right_boundary and right_window_cell:
            adjacent_cells += 1
        elif not adj_to_right_boundary:
            if (placement_area[idx] >> r_shift+1) & 1 and right_window_cell:
                adjacent_cells += 1
    return adjacent_cells / 10


def _get_adjacency_score(
        placement_area: list[int],
        i: int,
        width: int,
        present: PresentOrientation
) -> float:
    adjacent_cells = 0

    adjacent_cells += _get_vertical_adjacency_score(
        placement_area, i, width, present)

    adjacent_cells += _get_horizontal_adjacency_score(
        placement_area, i, width, present)

    return adjacent_cells


def _get_valid_orientations(
        current_orientations: list[PresentOrientation],
        window: PresentOrientation) -> list[PresentOrientation]:
    valid_orientations = []
    for orientation in current_orientations:
        # Check if orientation doesn't overlap with window
        if not any(orientation_part & window_part
                   for orientation_part, window_part in zip(orientation, window)):
            valid_orientations.append(orientation)
    return valid_orientations


def _get_best_orientation(
        placement_area: list[int],
        i: int,
        width: int,
        orientations: list[PresentOrientation],
        window: PresentOrientation
) -> tuple[int, PresentOrientation]:
    """ Gets best orientation based on score. The score consists of two things:
        1. Amount of 1s in xor of window and orientation. This is the same regardless
           of orientation and consistent with each present. This is more important so
           each 1 adds 1 to the score
        2. Amount of non empty cells / boundaries touched by the shape. Less important
            so only adds .1 to the score
    """
    base_score = 0
    best_adj_score = -1
    best_orientation = None

    # adds 1 based on xor (same for all orientations)
    for present_row, window_row in zip(orientations[0], window):
        base_score += bin(present_row ^ window_row).count('1')

    # get adjacency score, add to score
    for orientation in orientations:
        adj_score = _get_adjacency_score(placement_area, i, width, orientation)
        if adj_score > best_adj_score:
            best_adj_score = adj_score
            best_orientation = orientation

    # make bitmask of best orientation
    bitmask = []
    for present_row, window_row in zip(best_orientation, window):
        bitmask.append(present_row ^ window_row)

    score = base_score + best_adj_score

    return score, bitmask


def find_best_placement(placement_area: list[int],
                        area_size: tuple[int, int],
                        orientations: list[PresentOrientation],
                        ) -> Optional[Placement]:
    """ Gets best fitting placement for present_matrix if possible
    Returns the score of the mask and the mask of the present if successful

    Args:
        placement_area (list[int]): binary numbers representing each row of area
        area_size (tuple[int, int]): width, height of the placement area
        present (list[PresentOrientation]): list of list of ints representing each 
            present orientation row

    Returns:
        Optional[Placement]: If not None, best possible placement of present.
    """

    width, height = area_size

    best_score = -1
    best_bitmask = []
    best_bitmask_range = (-1, -1)

    for i in range(width * height):
        if _window_out_of_bounds(i, width, height):
            continue

        # get 3 rows from window with i in centre
        window: list[int] = _get_window(placement_area, i, width)

        # get orientations without collisions
        valid_orientations = _get_valid_orientations(orientations, window)

        # if no orientation can fit
        if not valid_orientations:
            continue

        # get best orientation
        score, bitmask = _get_best_orientation(
            placement_area, i, width, orientations, window)

        # update best score
        if score > best_score:
            best_score = score
            best_bitmask = bitmask
            best_bitmask_range = (i//width-1, i//width+2)

    if best_score == -1:
        return None

    return Placement(best_score, best_bitmask, best_bitmask_range)


def presents_can_fit(
    area_size: tuple[int, int],
    orientations: list[list[PresentOrientation]],
    present_count: list[int]
):
    """ Checks if all presents can fit in area. """

    # one int for each row
    area = [0] * area_size[1]

    while sum(present_count) > 0:
        best_present_idx = -1
        best_placement: Placement = None

        # check each present to see if it fits
        for npresent, present_orientations in enumerate(orientations):
            # if count for this present is 0
            if present_count[npresent] == 0:
                continue

            placement = find_best_placement(
                area, area_size, present_orientations)

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

        # reduce count
        present_count[best_present_idx] -= 1

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

    # _process_task(args_list[0])
    # _process_task(args_list[1])
    # _process_task(args_list[2])

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
