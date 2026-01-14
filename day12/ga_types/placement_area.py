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
    placed = 0
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

    def place_present(self, placement_gene: Gene) -> bool:
        """ Places present in area, returns True if successfully placed """
        top_left_x = placement_gene.x - 1
        top_left_y = placement_gene.y - 1

        present = self._get_present_to_place(placement_gene)

        if np.any(self.area[top_left_y:top_left_y+3, top_left_x:top_left_x+3] & present):
            return False

        self.area[top_left_y:top_left_y+3, top_left_x:top_left_x+3] += present
        self.placed += 1

        return True

    def place_all_presents(self, genes: list[Gene]):
        """ Place all presents without checking for overlaps.
        Should only be used for plotting data. """

        for gene in genes:
            top_left_x = gene.x - 1
            top_left_y = gene.y - 1
            present = self._get_present_to_place(gene)
            self.area[top_left_y:top_left_y+3,
                      top_left_x:top_left_x+3] += present
