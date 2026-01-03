""" Genetic algorithm for placement """

from typing import List

import numpy as np

from ga_types import Present


class PresentPackingGA:
    """ Genetic algorithm for packing presents """

    def __init__(self,
                 container_width: int,
                 container_height: int,
                 presents: List[Present],
                 ):
        """
        GA for packing presents in a container

        Args:
            container_width, container_height: Size of container
            presents: List of Present objects to pack
        """
        self.container_dims = (container_width, container_height)
        self.presents = presents

        # Number of genes = number of presents * 3 (i, idx, rotation)
        self.num_genes = len(self.presents) * 3

        # Store results
        self.best_solution = None
        self.best_fitness = -float('inf')
