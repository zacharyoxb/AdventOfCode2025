""" Helper module for GA fitness: places in area, gives score """
from dataclasses import dataclass, field

import numpy as np
from ga_types import Gene, Present
from ga_types.present import PresentMatrix


@dataclass
class PlacementMetrics:
    """ Represents scoring of each placement """
    collisions: int
    norm_xor: float
    norm_adj_score: float


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

        return window

    def _get_adjacency_score(self, placement_gene: Gene, present: PresentMatrix) -> float:
        top_left_x = placement_gene.x - 1
        top_left_y = placement_gene.y - 1

        # get coords of all present 1s
        present_coords = []
        for idx in np.ndindex(present.shape):  # (row, col) iterator
            if present[idx] == 1:
                present_coords.append(
                    (idx[1] + top_left_x, idx[0] + top_left_y))

        adj_coords = set()

        for x, y in present_coords:
            adj_top_left_x = x - 1
            adj_top_left_y = y - 1

            # get coords adjacent to coord not in present_coords
            adj_coords.update({
                (adj_top_left_x + i, adj_top_left_y + j)
                for i in range(3)
                for j in range(3)
                if (adj_top_left_x + i, adj_top_left_y + j) not in present_coords
            })

        # filter out out of bounds coords
        filtered_coords = [
            (x, y)
            for x, y in adj_coords
            if 0 <= x < self.width and 0 <= y < self.height
        ]

        # add amount of out of bounds coords
        adj_items = len(adj_coords) - len(filtered_coords)

        # get count of points with other presents in
        adj_items += sum(1 for x, y in filtered_coords if
                         self.area[y, x] == 1)

        # return normalised adjacency score
        return adj_items / len(adj_coords)

    def analyse_placement(
        self,
        placement_gene: Gene,
    ) -> PlacementMetrics:
        """ Analyses how good placement is, returns placement metrics """

        area_window = self._get_placement_window(placement_gene)
        present = self._get_present_to_place(placement_gene)

        # get state before present was placed, clamp incase -1
        area_window[present] -= 1
        area_window = np.maximum(area_window, 0)

        # get scores
        collisions = int(np.sum(area_window * present))
        xor_score = int(np.count_nonzero(
            (area_window > 0) ^ (present.astype(bool))))

        # Calculate adjacency score
        norm_adj_score = self._get_adjacency_score(
            placement_gene, present)

        # Normalize xor
        norm_xor = xor_score / 9 if xor_score > 0 else 0

        return PlacementMetrics(
            collisions,
            norm_xor,
            norm_adj_score,
        )

    def place_present(self, placement_gene: Gene):
        """ Places present in area """
        top_left_x = placement_gene.x - 1
        top_left_y = placement_gene.y - 1

        present = self._get_present_to_place(placement_gene)

        self.area[top_left_y:top_left_y+3, top_left_x:top_left_x+3] += present
