""" Genetic algorithm for placement """

from dataclasses import astuple
import pygad as pg

from ga_types import Present
from ga_types import Gene


class PresentPackingGA:
    """ Genetic algorithm for packing presents """

    def __init__(self,
                 container_width: int,
                 container_height: int,
                 presents: list[Present],
                 present_count: list[int],
                 ):
        """
        GA for packing presents in a container

        Args:
            container_width, container_height: Size of container
            presents: List of Present objects to pack
            present_count: amount of each present to put in the population
        """

        # Store presents and container
        self.container_dims = (container_width, container_height)
        self.presents = presents
        self.present_count = present_count

        # create starting population
        pop = self._get_population(
            present_count, container_height, container_width)

        gene_space = self._get_gene_space(
            present_count, container_height, container_width)

        # Init GA
        self.ga = pg.GA(
            num_generations=100,
            num_parents_mating=10,
            fitness_func=self.fitness_func,
            initial_population=pop,
            gene_type=int,
            parent_selection_type="rws",
            crossover_type="single_point",
            crossover_probability=0.1,
            mutation_type="random",
            mutation_probability=0.3,
            mutation_percent_genes="default",
            random_mutation_min_val=-100,
            random_mutation_max_val=100,
            gene_space=gene_space,
        )

        # Number of genes = number of presents * 3 (idx, x, y, rotation)
        self.num_genes = len(self.presents) * 3

    def _get_population(
            self,
            present_count: list[int],
            container_height: int,
            container_width: int
    ) -> list[tuple[int, int, int, int]]:
        pop = []
        for i, count in enumerate(present_count):
            gene_batch = Gene.create_random_batch(
                i, (2, container_height-2), (2, container_width-2), count)
            tuple_genes = list(map(astuple, gene_batch))
            pop.extend(tuple_genes)
        return pop

    def _get_gene_space(self,
                        present_count: list[int],
                        container_height: int,
                        container_width: int,
                        ) -> list[list[int]]:
        gene_space = []
        for i, count in enumerate(present_count):
            for _ in range(count):
                # For each gene: [idx, x, y, rotation]
                gene_space.append([i])  # Fixed present index
                gene_space.append(list(range(container_width - 1)))
                gene_space.append(list(range(container_height - 1)))
                gene_space.append(list(range(8)))
        return gene_space

    def fitness_func(self, _ga_instance, _solution, _solution_idx):
        """ Fitness function - checks how well present fits in area """
        return 1
