""" Contains a representation of the data in the genes of the GA. """
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Gene:
    """ Holds data for each gene in GA """
    present_idx: int
    orientation: int
    x: int
    y: int

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

    @classmethod
    def get_solution_genes(cls, solution: list[int]) -> list['Gene']:
        """ Gets genes from solution returned by genetic algorithm. """
        genes = []
        gene_matrix = np.reshape(solution, (-1, 4))

        for row in gene_matrix:
            genes.append(Gene(row[0], row[1], row[2], row[3]))

        return genes
