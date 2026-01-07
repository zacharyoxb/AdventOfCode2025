""" Genetic algorithm for placement """

import random

from deap import algorithms, base, creator, tools

from ga_types import Present, Gene, PlacementArea, PlacementMetrics


class PresentPackingGA:
    """ Genetic algorithm for packing presents """

    def __init__(self,
                 container_dims: tuple[int, int],
                 presents: list[Present],
                 present_count: list[int]
                 ):
        """
        GA for packing presents in a container

        Args:
            container_width, container_height: Size of container
            presents: List of Present objects to pack
            present_count: amount of each present to put in the population
        """

        # Store presents and container
        self.container_dims = container_dims
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
            del creator.MultiFitness
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("MultiFitness", base.Fitness, weights=(-1.0, 1.0, 1.0))
        creator.create("Individual", list,
                       fitness=creator.MultiFitness)

    def setup_deap(self):
        """ Sets up deap algorithm """

        # Register attributes not including idx
        width, height = self.container_dims
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
        self.toolbox.register("mutate", self.mutate,
                              orientpb=0.6, xpb=0.5, ypb=0.5)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("elitism", tools.selBest, k=10)

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

        # invalidate fitness
        del child1.fitness.values
        del child2.fitness.values

        return child1, child2

    def _reflect_value(self, value: int, lower: int, upper: int):
        span = upper - lower
        # The key is using % (2*span) for the modulo
        t = (value - lower) % (2 * span)
        if t > span:
            t = 2 * span - t
        return t + lower

    def mutate(
            self,
            individual: 'creator.Individual',
            orientpb: int,
            xpb: int,
            ypb: int
    ) -> 'creator.Individual':
        """ Mutates placement """
        width, height = self.container_dims

        for i, _ in enumerate(individual):
            del_fitness = False
            idx, orientation, x, y = individual[i]

            # Orientation: circular mutation (0.4 prob)
            if random.random() < orientpb:
                orientation = orientation + random.choice([-1, 1])
                orientation = (orientation+8) % 8
                del_fitness = True

            # Coordinates: Uniform mutation with step size
            if random.random() < xpb:
                x = self._reflect_value(
                    x + random.randint(-50, 50), 1, width-2)
                del_fitness = True
            if random.random() < ypb:
                y = self._reflect_value(
                    y + random.randint(-50, 50), 1, height-2)
                del_fitness = True

            if del_fitness:
                del individual.fitness.values
            individual[i] = (idx, orientation, x, y)

        return (individual,)

    def evaluate(self, individual: 'creator.Individual') -> tuple:
        """ Evaluates placement """
        area = PlacementArea(*self.container_dims, self.presents)

        genes = []

        # place all presents
        for gene_data in individual:
            gene = Gene(*gene_data)
            area.place_present(gene)
            genes.append(gene)

        metrics_list: list[PlacementMetrics] = []

        # evaluate placements
        for gene in genes:
            metrics_list.append(area.analyse_placement(gene))

        num_placements = len(metrics_list)

        total_collisions = 0.0
        total_norm_xor = 0.0
        total_norm_adj_score = 0.0

        for metrics in metrics_list:
            total_collisions += metrics.collisions
            total_norm_xor += metrics.norm_xor
            total_norm_adj_score += metrics.norm_adj_score

        avg_xor = total_norm_xor / num_placements
        avg_adj = total_norm_adj_score / num_placements

        return (total_collisions, avg_xor, avg_adj)

    def run_can_fit(self, mu=100, lambda_=400, cxpb=0.6, mutpb=0.4, ngen=200) -> bool:
        """ Runs evolutionary algorithm, returns true if found fitting solution """
        population = self.toolbox.population(n=mu+lambda_)
        hof = tools.HallOfFame(10)

        algorithms.eaMuPlusLambda(population=population, toolbox=self.toolbox,
                                  mu=mu, lambda_=lambda_, cxpb=cxpb, mutpb=mutpb,
                                  ngen=ngen, halloffame=hof)

        best_individual = hof[0]
        collisions, _xor_score, _adj_score = best_individual.fitness.values
        print(
            f"Collisions: {collisions}, xor score: {_xor_score}, adjacency score: {_adj_score}"
        )

        if collisions:
            return False
        return True
