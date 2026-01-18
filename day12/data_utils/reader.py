""" Reads data from text """

from dataclasses import dataclass
import re

import torch


def get_presents(file_name="input.txt") -> torch.Tensor:
    """ Extracts present matrices from data as a PyTorch tensor """
    raw_lines = []
    with open(f"inputs/{file_name}", encoding="UTF-8") as f:
        raw_lines = f.readlines()
    full_text = " ".join(raw_lines)

    # get all shapes matrices from text
    present_pattern = r"\d+:\s*((?:[.#]{3}\s*){3})"
    shape_matches = re.findall(present_pattern, full_text)

    extracted_present_tensors: list[torch.Tensor] = []

    for extracted_matrix in shape_matches:
        split_matrix = extracted_matrix.split()

        binary_matrix: list[list[int]] = [[1 if char == '#' else 0 for char in row]
                                          for row in split_matrix]
        # Convert to PyTorch tensor with float16 dtype
        tensor_matrix = torch.tensor(binary_matrix, dtype=torch.float16)
        extracted_present_tensors.append(tensor_matrix)

    return torch.stack(extracted_present_tensors)


@dataclass
class PlacementInfo:
    """ Stores config for each placement problem """
    width: int
    height: int
    present_count: torch.Tensor


def get_placement_info(file_name="input.txt") -> list[PlacementInfo]:
    """ Gets placement info on area and how many of each present to place """
    raw_lines = []
    with open(f"inputs/{file_name}", encoding="UTF-8") as f:
        raw_lines = f.readlines()
    full_text = " ".join(raw_lines)

    # get all areas we will insert into and indexes of shapes
    region_pattern = r'(\d+)x(\d+):\s*((?:\d+\s*)+)(?=\n|$)'
    region_matches = re.findall(region_pattern, full_text)

    # get height and width of container and amount of presents to place
    args = []
    for height_str, width_str, present_count_str in region_matches:
        width, height = int(height_str), int(width_str)
        present_count_str = present_count_str.split()
        present_count = list(map(int, present_count_str))
        args.append(PlacementInfo(width, height, torch.Tensor(present_count)))
    return args
