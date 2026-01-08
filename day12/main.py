""" Main day 12 file """
import re

import numpy as np

from ga_types import Present, PresentMatrix
from ga_types.present_packing_ga import PresentPackingGA


def can_fit(
        width: int,
        height: int,
        presents: list[Present],
        present_count: list[int]
) -> bool:
    """ Checks if all presents can fit """
    genetic_alg = PresentPackingGA(
        (width, height), presents, present_count)

    return genetic_alg.eu_mu_plus_lambda_custom()


def info_to_list(placement_info: list[tuple[str, str, str]]) -> list[tuple[int, int, list[int]]]:
    """ Get string placement info, converts to list containing 
    tuple of int height, int width, and an int list with all present counts.
    """
    # get height and width of container and amount of presents to place
    args = []
    for height_str, width_str, present_count_str in placement_info:
        width, height = int(height_str), int(width_str)
        present_count_str = present_count_str.split()
        present_count = list(map(int, present_count_str))
        args.append((width, height, present_count))
    return args


def day12(present_matrices: list[PresentMatrix], placement_info: list[tuple[str, str, str]]):
    """ Main function """
    fit_count = 0
    presents: list[Present] = []

    # create presents
    for present in present_matrices:
        presents.append(Present.from_matrix(present))

    # get height and width of container and amount of presents to place
    args = info_to_list(placement_info)

    for width, height, present_count in args:
        if can_fit(width, height, presents, present_count):
            fit_count += 1

    print(f"Amount of placement areas that can fit all presents: {fit_count}")


if __name__ == "__main__":
    raw_lines = []
    with open("inputs/testinput.txt", encoding="UTF-8") as f:
        raw_lines = f.readlines()
    FULL_TEXT = " ".join(raw_lines)

    # get all shapes matrixes from text
    SHAPE_PATTERN = r"\d+:\s*((?:[.#]{3}\s*){3})"
    shape_matches = re.findall(SHAPE_PATTERN, FULL_TEXT)

    extracted_present_matrices: list[PresentMatrix] = []

    for extracted_matrix in shape_matches:
        split_matrix = extracted_matrix.split()

        binary_matrix: list[list[int]] = [[1 if char == '#' else 0 for char in row]
                                          for row in split_matrix]
        extracted_present_matrices.append(np.array(binary_matrix, np.int32))

    # get all areas we will insert into and indexes of shapes
    REGION_PATTERN = r'(\d+)x(\d+):\s*((?:\d+\s*)+)(?=\n|$)'
    region_matches = re.findall(REGION_PATTERN, FULL_TEXT)

    day12(extracted_present_matrices, region_matches)
