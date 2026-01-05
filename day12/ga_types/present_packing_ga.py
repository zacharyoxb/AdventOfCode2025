""" Genetic algorithm for placement """

import random

from deap import base, creator, tools

from ga_types import Present


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
        self.toolbox = base.Toolbox()

        self.required_present_indices = []
        for present_idx, count in enumerate(self.present_count):
            self.required_present_indices.extend([present_idx] * count)

        self._setup_deap_types()
        self.setup_deap()

    def _setup_deap_types(self):
        """ Setup DEAP types """
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax  # type: ignore
        if hasattr(creator, "Individual"):
            del creator.Individual  # type: ignore

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list,
                       fitness=creator.FitnessMax)

    def setup_deap(self):
        """ Sets up deap algorithm """

        # Register attributes not including idx
        height, width = self.container_dims
        self.toolbox.register("attr_orientation", random.randint, 0, 7)
        self.toolbox.register("attr_x", random.randint, 0, width-2)
        self.toolbox.register("attr_y", random.randint, 0, height-2)

        # Form genes / individuals / populations
        self.toolbox.register("gene", self.create_gene)
        self.toolbox.register("individual", self.create_individual)
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)

        # custom GA functions
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", self.two_point_crossover)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def create_gene(self, present_idx) -> tuple[int, int, int, int]:
        """ Creates gene of present_idx """
        return (
            present_idx,
            self.toolbox.attr_orientation(),
            self.toolbox.attr_x(),
            self.toolbox.attr_y(),
        )

    def create_individual(self) -> 'creator.Individual':
        """ Create individual with exact present counts """
        # Start with required indices
        present_indices = self.required_present_indices.copy()

        # Create genes for each required present
        individual = []
        for present_idx in present_indices:
            individual.append(self.create_gene(present_idx))

        return creator.Individual(individual)

    def evaluate(self, individual: 'creator.Individual') -> tuple:
        """ Evaluates placement """
        return (1, 1, 1)

    def two_point_crossover(
            self,
            parent1: 'creator.Individual',
            parent2: 'creator.Individual'
    ) -> tuple['creator.Individual', 'creator.Individual']:
        """
        Applies 2 point crossover to parents, returns offspring
        """
        # get 2 points within range randomly, sort to ascending order
        cx1, cx2 = sorted(random.sample(range(1, len(parent1)), 2))
        child1 = parent1[:cx1] + parent2[cx1:cx2] + parent1[cx2:]
        child2 = parent2[:cx1] + parent1[cx1:cx2] + parent2[cx2:]
        return child1, child2

    def mutate(self, individual: 'creator.Individual', indpb: float):
        """ Mutates placement """
