""" Day 12 """

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import math
import re
from typing import Generator, Optional, TypeAlias
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


def _point_cannot_fit(i: int, original_i: int, width: int, height: int) -> bool:
    # if point i passes vertical boundary
    if (i < 0) or (i // width) > height-1:
        return True
    # if point i is not on the same line as original
    if ((original_i // width) != (i // width)) and ((original_i % width) != (i % width)):
        return True
    return False


def _adj_cell_is_1(
    placement_area: list[int],
    width: int,
    point: int
) -> bool:
    if placement_area[point // width] >> point % width & 1:
        return True
    return False


def _all_occupied_present_cells(
        window_i: int,
        window: PresentOrientation,
        width: int
) -> list[int]:
    pos_offset = [
        -width-1, -width, -width+1,
        -1,          0,          1,
        width-1,   width,  width+1
    ]

    # convert window to list of binary digits
    window_bin = [list(bin(row)[2:]) for row in window]
    # flatten
    window_cell = [item for row in window_bin for item in row]

    cell_i = []

    # for every cell in window
    for bin_digit, offset in zip(window_cell, pos_offset):
        if bin_digit == '1':
            cell_i.append(window_i + offset)
    return cell_i


def _all_adjacent_cells(
        cell_idx: int,
        width: int
) -> Generator[int, None, None]:
    pos_offset = [
        -width-1, -width, -width+1,
        -1,                      1,
        width-1,   width,  width+1
    ]

    for offset in pos_offset:
        yield cell_idx + offset


def _get_adjacency_score(
        placement_area: list[int],
        window_i: int,
        area_size: tuple[int, int],
        present: PresentOrientation
) -> float:
    adj_score = 0
    non_adj_point = 0

    width, height = area_size
    # Get i positions of all occupied present cells
    cell_idxs = _all_occupied_present_cells(window_i, present, width)

    # If all of the 3x3 present is filled, don't check centre adjacency
    if len(cell_idxs) == 9:
        cell_idxs.pop(4)

    # Otherwise, for all cells
    for cell_idx in cell_idxs:
        # for cells adjacent to cell not in cells
        for adj_cell_idx in _all_adjacent_cells(cell_idx, width):
            # adj cell is one of the occupied present cells
            if adj_cell_idx in cell_idxs:
                continue
            # adj cell can't fit on side or on top of adj cell (edge placement)
            if _point_cannot_fit(adj_cell_idx, cell_idx, width, height):
                adj_score += 1
            # otherwise if position is occupied, add 1 to score
            elif _adj_cell_is_1(placement_area, width, adj_cell_idx):
                adj_score += 1
            # add 1 to non_adj point so we can normalise
            else:
                non_adj_point += 1

    # normalise
    max_adj = adj_score + non_adj_point
    norm_adj_score = adj_score / max_adj

    return norm_adj_score


def _get_xor_score(
        orientation: PresentOrientation,
        window: PresentOrientation
) -> float:
    xor_score = 0
    for present_row, window_row in zip(orientation, window):
        xor_score += bin(present_row ^ window_row).count('1')

    max_xor = 9
    norm_xor = xor_score / max_xor
    return norm_xor


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
        window_i: int,
        area_size: tuple[int, int],
        orientations: list[PresentOrientation],
        window: PresentOrientation
) -> tuple[int, PresentOrientation]:
    """ Gets best orientation based on score. The score consists of these things:
        1. Amount of empty space filled
        2. Amount of filled cells/boundaries shape would be adjacent to
    """

    width, _ = area_size

    best_total_score = -1.0
    best_orientation = None

    for orientation in orientations:
        # get xor score
        norm_xor = _get_xor_score(orientation, window)

        # get adjacency score
        norm_adj = _get_adjacency_score(
            placement_area, window_i, area_size, orientation)

        # weighted combination
        total_score = (
            norm_xor * 0.2 +       # Fill empty space
            norm_adj * 0.8         # Compactness
        )

        if total_score > best_total_score:
            best_total_score = total_score
            best_orientation = orientation

    # make bitmask of best orientation
    bitmask = []
    for present_row, window_row in zip(best_orientation, window):
        bitmask.append((present_row ^ window_row) << (window_i-1) % width)

    return best_total_score * 100, bitmask


def _window_out_of_bounds(window_i: int, width: int, height: int) -> bool:
    # if 3x3 square with centre i overlaps horizontal boundary
    if (window_i % width) == 0 or ((window_i+1) % width) == 0:
        return True

    # if 3x3 square with centre i overlaps vertical boundary
    if (window_i // width) == 0 or (window_i // width) == height-1:
        return True
    return False


def _get_window(placement_area: list[int], window_i: int, width: int) -> list[int]:
    # get 3 rows from window with i in centre
    window: list[int] = []
    top_row_idx = (window_i // width) - 1
    bottom_row_idx = (window_i // width) + 1
    for row in range(top_row_idx, bottom_row_idx+1):
        shift = (window_i-1) % width
        window.append((placement_area[row] >> shift) & WINDOW_ROW_MASK)
    return window


def find_best_placement(placement_area: list[int],
                        area_size: tuple[int, int],
                        orientations: list[PresentOrientation],
                        ) -> Optional[Placement]:
    """ Gets best fitting placement for present_matrix if possible.
    Chooses best positioning furthest up the area.
    Returns the score of the mask and the mask of the present if successful.

    Args:
        placement_area (list[int]): binary numbers representing each row of area
        area_size (tuple[int, int]): width, height of the placement area
        present (list[PresentOrientation]): list of list of ints representing each 
            present orientation row

    Returns:
        Optional[Placement]: If not None, best possible placement of present.
    """

    width, height = area_size

    best_col = math.inf
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
            placement_area, i, area_size, orientations, window)

        # update best row/score
        if (i // width) <= best_col and (score > best_score):
            best_col = i // width
            best_score = score
            best_bitmask = bitmask
            best_bitmask_range = (i//width-1, i//width+2)

    if best_col == math.inf:
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

    task1 = _process_task(args_list[0])
    task2 = _process_task(args_list[1])
    task3 = _process_task(args_list[2])

    print(f"task1: {task1}, task2: {task2}, task3: {task3}")

    # with ProcessPoolExecutor(1) as executor:
    #     futures = [executor.submit(_process_task, args) for args in args_list]

    #     with tqdm(total=len(futures)) as pbar:
    #         for future in as_completed(futures):
    #             try:
    #                 if future.result():
    #                     fit_count += 1
    #             except TimeoutError as e:
    #                 print(f"Timeout error! {e}")
    #             finally:
    #                 pbar.update(1)

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
