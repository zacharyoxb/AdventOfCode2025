""" Helper module for GA fitness: places in area, gives score """
from dataclasses import dataclass, field

import numpy as np

from ga_types import Gene, Present
from ga_types.present import PresentMatrix


@dataclass
class PlacementArea:
    """ Manages the area to place in and the scoring of placements """
    width: int
    height: int
    presents: list[Present]
    area: PresentMatrix = field(init=False)

    def __post_init__(self):
        self.area = np.zeros((self.height, self.width), np.int32)

    def _get_present_to_place(self, placement_gene: Gene) -> PresentMatrix:
        return self.presents[placement_gene.present_idx].masks[placement_gene.orientation]

    def _get_placement_window(self, placement_gene: Gene):
        x_start = placement_gene.x - 1
        x_end = placement_gene.x + 2  # (slice end is exclusive)
        y_start = placement_gene.y - 1
        y_end = placement_gene.y + 2

        window = self.area[y_start:y_end, x_start:x_end]

        return window.copy()

    def place_present(self, placement_gene: Gene):
        """ Places present in area """
        top_left_x = placement_gene.x - 1
        top_left_y = placement_gene.y - 1

        present = self._get_present_to_place(placement_gene)

        self.area[top_left_y:top_left_y+3, top_left_x:top_left_x+3] += present

    def get_non_empty(self):
        """ Get amount of empty squares in area, normalised """
        non_empty_cells = self.area.size - int(np.sum(self.area == 0))
        norm_non_empty = non_empty_cells / self.area.size
        return norm_non_empty

    def get_non_collisions(self):
        """ Get amount of collisions in area, normalised """
        collisions = int(np.sum(self.area > 1))
        non_collisions = self.area.size - collisions
        return non_collisions / self.area.size

    def get_border_adjacency_score(self):
        """ Score for non-empty cells adjacent to borders """
        # Get border indices
        height, width = self.area.shape

        # Create masks for border rows and columns
        top_border = self.area[0, :] > 0
        bottom_border = self.area[-1, :] > 0
        left_border = self.area[:, 0] > 0
        right_border = self.area[:, -1] > 0

        # Count border-adjacent non-empty cells
        border_non_empty = (np.sum(top_border) + np.sum(bottom_border) +
                            np.sum(left_border) + np.sum(right_border))

        # Normalize by maximum possible border cells
        max_border_cells = 2 * (height + width)
        norm_score = border_non_empty / max_border_cells if max_border_cells > 0 else 0

        return float(norm_score)
