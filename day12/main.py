""" Main day 12 file """
import re
import copy

import numpy as np

from ga_types import Present, PresentMatrix


def can_fit(height: int, width: int, presents: list[Present]):
    """ Checks if all presents can fit """


def day12(present_matrices: list[PresentMatrix], placement_info: list[tuple[str, str, str]]):
    """ Main function """
    fit_count = 0

    presents: list = []

    # create presents
    for i, present in enumerate(present_matrices):
        presents.append(Present.from_matrix(present))

    # get height and width of container and amount of presents to place
    args = []
    for height_str, width_str, present_count_str in placement_info:
        height, width = int(height_str), int(width_str)
        present_count_str = present_count_str.split()
        present_count = list(map(int, present_count_str))
        args.append((height, width, present_count))

    for height, width, present_count in args:
        iter_presents: list[Present] = []

        # add all presents based on present_count
        for i, count in enumerate(present_count):
            if count == 0:
                continue

            clones = [copy.deepcopy(presents[i]) for _ in range(count)]
            iter_presents.extend(clones)

        if can_fit(height, width, iter_presents):
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
        extracted_present_matrices.append(np.array(binary_matrix))

    # get all areas we will insert into and indexes of shapes
    REGION_PATTERN = r'(\d+)x(\d+):\s*((?:\d+\s*)+)(?=\n|$)'
    region_matches = re.findall(REGION_PATTERN, FULL_TEXT)

    day12(extracted_present_matrices, region_matches)
