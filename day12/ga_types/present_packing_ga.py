""" Genetic algorithm for placement """

from dataclasses import dataclass
import random

from deap import algorithms, base, creator, tools
import numpy as np

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
        self.toolbox.register("select", tools.selNSGA2)
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

             # Occasionally swap two rectangles' positions entirely
            if random.random() < 0.1:
                j = random.randrange(len(individual))
                individual[i], individual[j] = individual[j], individual[i]

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

    @dataclass
    class GAConfig:
        """ Config for ga algorithm"""
        mu = 100
        lambda_ = 400
        cxpb = 0.6
        mutpb = 0.4
        ngen = 200

    def _print_generation_header(self):
        """Prints the generation statistics header"""
        print(
            f"{'gen':<6} {'evals':<8} {'min_coll':<10} {'min_xor':<10} {'min_adj':<10}")
        print("-" * 46)

    def _evaluate_population(self, population):
        """Evaluates all invalid individuals in a population"""
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        if not invalid_ind:
            return 0

        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        return len(invalid_ind)

    def _select_with_elitism(self, population, elites, offspring, mu):
        """Performs selection while preserving elites"""
        non_elite_pop = [ind for ind in population if ind not in elites]
        combined_pool = non_elite_pop + offspring
        selected = self.toolbox.select(combined_pool, mu - len(elites))
        return elites + selected

    def _get_population_stats(self, population):
        """Extracts key statistics from population"""
        collisions = [ind.fitness.values[0]
                      for ind in population if ind.fitness.valid]
        xor_scores = [ind.fitness.values[1]
                      for ind in population if ind.fitness.valid]
        adj_scores = [ind.fitness.values[2]
                      for ind in population if ind.fitness.valid]

        return {
            'min_coll': np.min(collisions) if collisions else 0,
            'min_xor': np.min(xor_scores) if xor_scores else 0,
            'min_adj': np.min(adj_scores) if adj_scores else 0,
            'zero_count': sum(1 for c in collisions if c == 0)
        }

    def _print_generation_stats(self, gen, eval_count, stats):
        """Prints statistics for a single generation"""
        print(f"{gen:<6} {eval_count:<8} "
              f"{stats['min_coll']:<10.1f} {stats['min_xor']:<10.1f} "
              f"{stats['min_adj']:<10.1f}")

    def eu_mu_plus_lambda_custom(self, config: GAConfig = GAConfig()):
        """Custom implementation of deap function that can exit when solution is found"""
        population = self.toolbox.population(n=config.mu)

        # Print header
        self._print_generation_header()

        # Evaluate initial population
        self._evaluate_population(population)

        for gen in range(config.ngen):
            # Select elites
            k = int(config.mu * 0.02)
            elites = self.toolbox.elitism(population, k=k)

            # Generate offspring
            offspring = algorithms.varOr(
                population, self.toolbox, config.lambda_, config.cxpb, config.mutpb)

            # Evaluate offspring
            invalid_count = self._evaluate_population(offspring)

            # Select new population with elitism
            population[:] = self._select_with_elitism(
                population, elites, offspring, config.mu)

            # Get and print stats
            stats = self._get_population_stats(population)
            self._print_generation_stats(gen, invalid_count, stats)

            # Check for solution
            if stats['zero_count'] > 0:
                print("\nSolution found!\n\n")
                return True

        print("\nNo solution found.\n\n")
        return False
