""" Contains a representation of the data in the genes of the GA. """
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Gene:
    """ Holds data for each gene in GA """
    present_idx: int
    x: int
    y: int
    orientation: int

    @classmethod
    def create_random_batch(
        cls,
        present_idx: int,
        x_range: Tuple[int, int],
        y_range: Tuple[int, int],
        batch_size: int = 1
    ) -> list['Gene']:
        """ Create batch using numpy for better performance with large batches """
        x_min, x_max = x_range
        y_min, y_max = y_range

        # Generate all random values at once
        xs = np.random.randint(x_min, x_max + 1, size=batch_size)
        ys = np.random.randint(y_min, y_max + 1, size=batch_size)
        orientations = np.random.randint(0, 8, size=batch_size)

        return [
            cls(present_idx=present_idx, x=int(x),
                y=int(y), orientation=int(rot))
            for x, y, rot in zip(xs, ys, orientations)
        ]
