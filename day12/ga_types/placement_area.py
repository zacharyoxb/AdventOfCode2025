""" Helper module for GA fitness: places in area, gives score """
from dataclasses import dataclass, field
from typing import List

from ga_types import Gene, Present, PresentOrientation


@dataclass
class PlacementMetrics:
    """ Represents scoring of each placement """
    norm_xor: float
    norm_collisions: float
    norm_adj_score: float


@dataclass
class PlacementArea:
    """ Manages the area to place in and the scoring of placements """
    width: int
    height: int
    presents: list[Present]
    area: List[int] = field(init=False)

    # Const
    WINDOW_ROW_MASK = 0x7  # 111 in binary

    def __post_init__(self):
        self.area = [0] * self.height

    def _get_adjacency_score(self, placement_gene: Gene) -> float:
        top_left_x = placement_gene.x - 1
        top_left_y = placement_gene.y - 1
        present = self._get_present_to_place(placement_gene)

        # get coords of all present 1s
        present_coords = [
            (top_left_x + i, top_left_y + j)
            for i, row in enumerate(present)
            for j, digit in enumerate(bin(row)[2:])
            if digit == '1'
        ]

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
        filtered_coords = {
            (x, y)
            for x, y in adj_coords
            if 0 <= x < self.width and 0 <= y < self.height
        }

        # add amount of out of bounds coords
        adj_items = len(adj_coords) - len(filtered_coords)

        # get count of points with other presents in
        adj_items += sum(1 for x, y in filtered_coords if (
            self.area[x] >> y) & 1)

        # return normalised adjacency score
        return adj_items / len(adj_coords)

    def _get_present_to_place(self, placement_gene: Gene) -> PresentOrientation:
        return self.presents[placement_gene.present_idx].masks[placement_gene.orientation]

    def _analyze_placement(
        self,
        placement_gene: Gene,
    ) -> tuple[PlacementMetrics, list[int]]:
        collisions = 0
        xor_score = 0
        bitmask = []

        # Process each row of the present
        for present_row, window_row in self._get_present_window_rows(placement_gene):
            collisions += bin(present_row & window_row).count('1')
            xor_score += bin(present_row ^ window_row).count('1')
            bitmask.append((present_row ^ window_row) << placement_gene.y-1)

        # Calculate adjacency score
        norm_adj_score = self._get_adjacency_score(
            placement_gene)

        # Normalize metrics
        norm_xor = xor_score / 9 if xor_score > 0 else 0
        norm_collisions = collisions / 9 if collisions > 0 else 0

        return (
            PlacementMetrics(
                norm_xor=norm_xor,
                norm_collisions=norm_collisions,
                norm_adj_score=norm_adj_score,
            ),
            bitmask
        )

    def _get_window(self, x: int, y: int) -> list[int]:
        # get 3 rows from window with i in centre
        window: list[int] = []
        top_row_idx = x-1
        bottom_row_idx = x+1
        for row in range(top_row_idx, bottom_row_idx+1):
            shift = y-1
            window.append((self.area[row] >> shift) & self.WINDOW_ROW_MASK)
        return window

    def _get_present_window_rows(self, placement_gene: Gene):
        present = self._get_present_to_place(placement_gene)
        window = self._get_window(placement_gene.x, placement_gene.y)
        for present_row, window_row in zip(present, window):
            yield present_row, window_row

    def _apply_placement(self, placement_gene: Gene, bitmask: list[int]):
        mask_start, mask_end = placement_gene.x-1, placement_gene.x+2
        for present_row, area_idx in enumerate(range(mask_start, mask_end)):
            self.area[area_idx] |= bitmask[present_row]

    def place_present(self, placement_gene: Gene) -> PlacementMetrics:
        """ Places present in area returns score """

        # Analyze the placement
        placement_metrics, bitmask = self._analyze_placement(placement_gene)

        # Apply the placement to the area
        self._apply_placement(placement_gene, bitmask)

        # Calculate and return fitness
        return placement_metrics
