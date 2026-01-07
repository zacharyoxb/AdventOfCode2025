""" Genetic algorithm for placement """

import random

from deap import algorithms, base, creator, tools

from ga_types import Present, Gene, PlacementArea, PlacementMetrics


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
        if hasattr(creator, "MultiFitness"):
            del creator.MultiFitness  # type: ignore
        if hasattr(creator, "Individual"):
            del creator.Individual  # type: ignore

        creator.create("MultiFitness", base.Fitness, weights=(1.0, -3.0, 1.0))
        creator.create("Individual", list,
                       fitness=creator.MultiFitness)

    def setup_deap(self):
        """ Sets up deap algorithm """

        # Register attributes not including idx
        height, width = self.container_dims
        self.toolbox.register("attr_orientation", random.randint, 0, 7)
        self.toolbox.register("attr_x", random.randint, 1, width-2)
        self.toolbox.register("attr_y", random.randint, 1, height-2)

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
        present_indices = self.required_present_indices

        # Create genes for each required present
        individual = []
        for present_idx in present_indices:
            individual.append(self.create_gene(present_idx))

        return creator.Individual(individual)

    def two_point_crossover(
            self,
            parent1: 'creator.Individual',
            parent2: 'creator.Individual'
    ) -> tuple['creator.Individual', 'creator.Individual']:
        """
        Applies 2 point crossover to parents, returns offspring
        """
        # if parents have less than 2 genes, do 1 point
        if len(parent1) < 3:
            child1 = creator.Individual((parent1[0], parent2[1]))
            child2 = creator.Individual((parent2[0], parent1[1]))
            return child1, child2

        # get 2 points within range randomly, sort to ascending order
        cx1, cx2 = sorted(random.sample(range(1, len(parent1)), 2))
        child1 = creator.Individual(
            (parent1[:cx1] + parent2[cx1:cx2] + parent1[cx2:])
        )
        child2 = creator.Individual(
            (parent2[:cx1] + parent1[cx1:cx2] + parent2[cx2:])
        )
        return child1, child2

    def mutate(self, individual: 'creator.Individual') -> 'creator.Individual':
        """ Mutates placement """
        width, height = self.container_dims
        min_x, max_x = 1, width-2
        min_y, max_y = 1, height-2

        for i, _ in enumerate(individual):
            idx, orientation, x, y = individual[i]

            # Orientation: circular mutation (0.4 prob)
            if random.random() < 0.4:
                orientation = orientation + random.choice([-1, 1])
                orientation = (orientation+8) % 8

            # Coordinates: Uniform mutation with step size (Either mutates x or y)
            if random.random() < 0.5:
                step = random.randint(-10, 10)
                x = x + step
                x = max(min_x, min(max_x, x))  # Clamp to bounds
            else:
                step = random.randint(-10, 10)
                y = y + step
                y = max(min_y, min(max_y, y))  # Clamp to bounds

            individual[i] = (idx, orientation, x, y)

        return (individual,)

    def evaluate(self, individual: 'creator.Individual') -> tuple:
        """ Evaluates placement """
        area = PlacementArea(*self.container_dims, self.presents)

        metrics_list: list[PlacementMetrics] = []

        for gene_data in individual:
            gene = Gene(*gene_data)
            placement_metrics = area.place_present(gene)
            metrics_list.append(placement_metrics)

        num_placements = len(metrics_list)

        total_norm_xor = 0.0
        total_norm_collisions = 0.0
        total_norm_adj_score = 0.0

        for metrics in metrics_list:
            total_norm_xor += metrics.norm_xor
            total_norm_collisions += metrics.norm_collisions
            total_norm_adj_score += metrics.norm_adj_score

        avg_xor = total_norm_xor / num_placements
        avg_collisions = total_norm_collisions / num_placements
        avg_adj = total_norm_adj_score / num_placements

        return (avg_xor, avg_collisions, avg_adj)

    def run_can_fit(self, cxpb=0.4, mutpb=0.2, ngen=100) -> bool:
        """ Runs evolutionary algorithm, returns true if found fitting solution """
        population = self.toolbox.population(n=100)
        hof = tools.HallOfFame(1)
        algorithms.eaSimple(population=population,
                            toolbox=self.toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, halloffame=hof)

        best_individual = hof[0]
        _, collisions, _ = best_individual.fitness.values
        print(best_individual.fitness.values)
        if collisions == 0:
            return True
        return False
