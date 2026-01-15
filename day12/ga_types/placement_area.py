""" Helper module for GA fitness: places in area, gives score """
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import generic_filter

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

    def _place_present(self, placement_gene: Gene) -> bool:
        """ Attempts to place present, will not place if it would cause a collision."""
        top_left_x = placement_gene.x - 1
        top_left_y = placement_gene.y - 1

        present = self._get_present_to_place(placement_gene)

        if np.any(self.area[top_left_y:top_left_y+3, top_left_x:top_left_x+3] & present):
            return False

        self.area[top_left_y:top_left_y+3, top_left_x:top_left_x+3] += present

        return True

    def _calc_contact_amount(self):
        # convert to array with just 1s and 0s
        bin_array = np.where(self.area > 0, 1, 0)

        def get_window_adj(window):
            if window[4] == 0:
                return 0
            return int(np.sum(window))

        # get filtered array
        filtered = generic_filter(
            bin_array, get_window_adj, 3, mode='constant', cval=1.0)

        # sum to get adj point total
        adj_points = np.sum(filtered)

        # calculate max adjacent points
        max_points = self.area.size * 8

        # normalise
        return adj_points / max_points

    def calculate_fitness(self, individual: list[Gene]) -> tuple[float]:
        """ Calculates fitness of an individual. """
        placed = 0
        for gene in individual:
            if self._place_present(gene):
                placed += 1

        # placement success
        placement_ratio = placed / len(individual)

        # if placement ratio is already 1, we can ignore everything else
        if placement_ratio == 1.0:
            return (placement_ratio,)

        # packing quality
        adj_score = self._calc_contact_amount()

        fitness = (
            0.6 * placement_ratio +
            0.4 * adj_score
        )

        return (fitness,)

    def place_all_presents(self, genes: list[Gene]):
        """ Place all presents without checking for overlaps.
        Should only be used for plotting data. """

        for gene in genes:
            top_left_x = gene.x - 1
            top_left_y = gene.y - 1
            present = self._get_present_to_place(gene)
            self.area[top_left_y:top_left_y+3,
                      top_left_x:top_left_x+3] += present
