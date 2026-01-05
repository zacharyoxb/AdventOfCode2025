""" Helper module for GA fitness: places in area, gives score """
from dataclasses import dataclass, field
from typing import Generator, List

from ga_types.gene import Gene
from ga_types.present import Present


@dataclass
class PlacementMetrics:
    """ Represents scoring of each placement """
    collisions: int
    xor_score: int
    norm_xor: float
    norm_collisions: float
    norm_adj_score: float
    bitmask: list


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
        """ Initialize the area with zeros (empty rows) """
        self.area = [0] * self.height

    def _point_cannot_fit(
            self,
            adj_i: int,
            original_i: int,
    ) -> bool:
        # if point i passes vertical boundary
        if (adj_i < 0) or (adj_i // self.width) >= self.height:
            return True

        # if point i is not adjacent to original (exceeded horizontal boundary)
        diff_row = (original_i // self.width) != (adj_i // self.width)
        diff_column = (original_i % self.width) != (adj_i % self.width)
        if diff_row and diff_column:
            return True

        return False

    def _adj_cell_is_1(
        self,
        point: int,
    ) -> bool:
        row = point // self.width
        shift = point % self.width
        if (self.area[row] >> shift) & 1:
            return True
        return False

    def _all_occupied_present_cells(
            self,
            window_i: int,
            window: list[int],
    ) -> list[int]:
        pos_offset = [
            -self.width-1, -self.width, -self.width+1,
            -1,          0,          1,
            self.width-1,   self.width,  self.width+1
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
            self,
            cell_idx: int,
    ) -> Generator[int, None, None]:
        # above, left, right, bottom
        pos_offset = [
            -self.width,
            -1,
            1,
            self.width,
        ]

        for offset in pos_offset:
            yield cell_idx + offset

    def _get_adjacency_score(
            self,
            window_i: int,
            present: list[int]
    ) -> float:
        # set of all adjacent indexes out of bounds / 1
        adj_indxs = set()
        # set of all adjacent indexes that are 0
        all_adj_cells = set()

        # Get i positions of all occupied present cells
        cell_idxs = self._all_occupied_present_cells(
            window_i, present)

        # If all of the 3x3 present is filled, don't check centre adjacency
        if len(cell_idxs) == 9:
            cell_idxs.pop(4)

        # Otherwise, for all cells
        for cell_idx in cell_idxs:
            # for cells adjacent to cell not in cells
            for adj_cell_idx in self._all_adjacent_cells(cell_idx):
                # adj cell is one of the occupied present cells
                if adj_cell_idx in cell_idxs:
                    continue

                # add to max possible adj
                all_adj_cells.add(adj_cell_idx)

                # adj cell can't fit on side or on top of adj cell (edge placement)
                if self._point_cannot_fit(adj_cell_idx, cell_idx):
                    adj_indxs.add(adj_cell_idx)

                # otherwise if position is occupied, add 1 to score
                elif self._adj_cell_is_1(adj_cell_idx):
                    adj_indxs.add(adj_cell_idx)

        # normalise
        norm_adj_score = len(adj_indxs) / len(all_adj_cells)

        return norm_adj_score

    def _get_window(self, x: int, y: int) -> list[int]:
        # get 3 rows from window with i in centre
        window: list[int] = []
        top_row_idx = x-1
        bottom_row_idx = x+1
        for row in range(top_row_idx, bottom_row_idx+1):
            shift = y-1
            window.append((self.area[row] >> shift) & self.WINDOW_ROW_MASK)
        return window

    def _get_present_to_place(self, placement_gene: Gene):
        return self.presents[placement_gene.present_idx].masks[placement_gene.orientation]

    def _analyze_placement(self, placement_gene: Gene, present):
        collisions = 0
        xor_score = 0
        bitmask = []

        # Process each row of the present
        for present_row, window_row in self._get_present_window_rows(placement_gene, present):
            collisions += bin(present_row & window_row).count('1')
            xor_score += bin(present_row ^ window_row).count('1')
            bitmask.append((present_row ^ window_row) << placement_gene.y-1)

        # Calculate adjacency score
        window_i = (self.width * placement_gene.x) + placement_gene.y
        norm_adj_score = self._get_adjacency_score(window_i, present)

        # Normalize metrics
        norm_xor = 9 / xor_score if xor_score > 0 else 0
        norm_collisions = 9 / collisions if collisions > 0 else 0

        return PlacementMetrics(
            collisions=collisions,
            xor_score=xor_score,
            norm_xor=norm_xor,
            norm_collisions=norm_collisions,
            norm_adj_score=norm_adj_score,
            bitmask=bitmask
        )

    def _get_present_window_rows(self, placement_gene: Gene, present):
        window = self._get_window(placement_gene.x, placement_gene.y)
        for present_row, window_row in zip(present, window):
            yield present_row, window_row

    def _apply_placement(self, placement_gene: Gene, bitmask: list[int]):
        mask_start, mask_end = placement_gene.x-1, placement_gene.x+2
        for present_row, area_idx in enumerate(range(mask_start, mask_end)):
            self.area[area_idx] |= bitmask[present_row]

    def _calculate_fitness(self, metrics: PlacementMetrics):
        # non-collision > 0, collision < 0
        positive_score = 0.4 * metrics.norm_xor + 0.6 * metrics.norm_adj_score

        if metrics.collisions == 0:
            return positive_score

        return -1.0 - (positive_score * 0.5) - (metrics.norm_collisions * 0.5)

    def place_present(self, placement_gene: Gene):
        """ Places present in area returns score """
        # Get the present to place
        present = self._get_present_to_place(placement_gene)

        # Analyze the placement
        placement_metrics = self._analyze_placement(placement_gene, present)

        # Apply the placement to the area
        self._apply_placement(placement_gene,
                              placement_metrics.bitmask
                              )

        # Calculate and return fitness
        return self._calculate_fitness(placement_metrics)
