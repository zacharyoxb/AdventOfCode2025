""" Genetic algorithm for placement """

from dataclasses import dataclass
import random
from typing import Optional

import numpy as np
from deap import algorithms, base, creator, tools

from ga_types import Present, Gene, PlacementArea, Plotter


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
            container_dims: Size of container (width, height)
            presents: List of Present objects to pack
            present_count: amount of each present to put in the population
        """

        # Store presents and container
        self.container_dims = container_dims
        self.presents = presents
        self.present_count = present_count
        self.toolbox = base.Toolbox()

        # get indexes for all presents we have to place
        self.required_present_indices = []
        for present_idx, count in enumerate(self.present_count):
            self.required_present_indices.extend([present_idx] * count)

        self._setup_deap_types()
        self.setup_deap()

    def _setup_deap_types(self):
        """ Setup DEAP types """
        if hasattr(creator, "Fitness"):
            del creator.Fitness
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("Fitness", base.Fitness, weights=(1.0,))
        creator.create("Individual", list,
                       fitness=creator.Fitness)

    def setup_deap(self):
        """ Sets up deap algorithm """

        # Register attributes not including idx
        self.toolbox.register("attr_orientation", random.randint, 0, 7)

        # Form genes / individuals / populations
        self.toolbox.register("gene", self.create_gene)
        self.toolbox.register("individual", self.create_individual)
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)

        # custom GA functions
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", self.two_point_crossover)
        self.toolbox.register("mutate", self.mutate,
                              orientpb=0.8, xpb=0.5, ypb=0.5)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("elitism", tools.selBest, k=10)

    def create_gene(
        self,
        present_idx: int,
        orientation: int,
        x: int,
        y: int
    ) -> tuple[int, int, int, int]:
        """ Creates gene of present_idx """
        return (
            present_idx,
            orientation,
            x,
            y
        )

    def create_individual(self) -> 'creator.Individual':
        """ Create individual with exact present counts """
        # Start with required indices
        present_indices = self.required_present_indices

        # get positions spread out evenly in area
        width, height = self.container_dims
        coords_needed = len(present_indices)

        # Create a linear index for all possible positions
        border = 1
        all_x = np.arange(border, width - border)
        all_y = np.arange(border, height - border)

        # Create meshgrid of all positions
        x_full, y_full = np.meshgrid(all_x, all_y)
        all_positions = np.column_stack([x_full.ravel(), y_full.ravel()])

        points = []

        # If we need fewer points than available, sample evenly
        if coords_needed < len(all_positions):
            # Create N evenly spaced indices
            indices = np.linspace(0, len(all_positions) -
                                  1, coords_needed, dtype=int)
            selected = all_positions[indices]
            points = [(int(x), int(y)) for x, y in selected]
        else:
            # If we need more points than available, repeat or use all
            points = [(int(x), int(y))
                      for x, y in all_positions][:coords_needed]
        # shuffle coordinates
        random.shuffle(points)

        orientation = self.toolbox.attr_orientation()

        # Create genes for each required present
        individual = []
        for present_idx, (x, y) in zip(present_indices, points):
            individual.append(self.create_gene(present_idx, orientation, x, y))

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

    def evaluate(self, individual: 'creator.Individual') -> tuple[float]:
        """ Evaluates placement, returns tuple of weights """
        area = PlacementArea(*self.container_dims, self.presents)

        # place all presents
        for gene_data in individual:
            gene = Gene(*gene_data)
            if not area.place_present(gene):
                break

        return (area.placed / len(individual),)

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
        print(f"{'gen':<6} {'evals':<8} {'best_score':<15}")
        print("-" * 27)

    def _evaluate_population(self, population):
        """ Evaluates all invalid individuals in a population """
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
        valid_i_val = [
            ind.fitness.values[0]
            for ind in population
            if ind.fitness.valid
        ]

        best_score = max(valid_i_val)

        return best_score

    def _print_generation_stats(self, gen, eval_count, best_score):
        """Prints statistics for a single generation"""
        print(f"{gen:<6} {eval_count:<8} {best_score:<15f}")

    def _plot_best(self, plotter: Plotter, population: 'creator.Population'):
        # get best
        best = min(population, key=lambda ind: ind.fitness.values[0])
        # get best individuals' plot
        area = PlacementArea(*self.container_dims, self.presents)

        genes = []

        for gene_data in best:
            gene = Gene(*gene_data)
            genes.append(gene)

        area.place_all_presents(genes)
        plotter.update(area.area)

    def _catastrophic_restart(self, population):
        """ Refreshes population to prevent plateaus """
        # Keep best 5%
        keep_n = max(1, int(len(population) * 0.05))
        best = tools.selBest(population, keep_n)

        # Create new population
        new_pop = self.toolbox.population(n=len(population) - keep_n)

        # Return combined
        return best + new_pop

    def eu_mu_plus_lambda_custom(
            self,
            config: GAConfig = GAConfig(),
            plotter: Optional[Plotter] = None
    ) -> bool:
        """ Custom implementation of deap function that can exit when solution is found """
        population = self.toolbox.population(n=config.mu)

        # Print header
        self._print_generation_header()

        # Evaluate initial population
        self._evaluate_population(population)

        # Plot best layout
        if plotter:
            self._plot_best(plotter, population)

        for gen in range(config.ngen):
            # If divisible by 15, do catastrophic restart
            if gen > 0 and gen % 15 == 0:
                population = self._catastrophic_restart(population)
                self._evaluate_population(population)

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
            best_score = self._get_population_stats(population)
            self._print_generation_stats(gen, invalid_count, best_score)

            # Check for solution
            if best_score == 1:
                print("\nSolution found!\n\n")
                return True

            if gen % 5 == 0 and plotter:
                self._plot_best(plotter, population)

        print("\nNo solution found.\n\n")
        return False
