import random
from copy import deepcopy
from rastion_core.base_optimizer import BaseOptimizer


class GeneticAlgorithm(BaseOptimizer):
    """
    A simple Genetic Algorithm solver for discrete/combinatorial problems.
    Expects the problem to implement random_solution() and evaluate_solution().
    """

    def __init__(self, population_size=50, mutation_rate=0.01, crossover_rate=0.8,
                 max_generations=100, verbose=False):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.verbose = verbose

    def initialize_population(self, problem):
        """Create an initial population of random solutions."""
        population = []
        for _ in range(self.population_size):
            population.append(problem.random_solution())
        return population

    def evaluate_population(self, population, problem):
        """Evaluate each individual's fitness (objective)."""
        scored_pop = [(sol, problem.evaluate_solution(sol)) for sol in population]
        return scored_pop

    def select_parents(self, scored_pop):
        """
        Simple tournament selection:
        Randomly pick k=3 individuals and select the two best as parents.
        """
        candidates = random.sample(scored_pop, 3)
        # sort by cost, ascending
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0], candidates[1][0]

    def crossover(self, parent1, parent2):
        """
        Simple crossover for permutations: one-cut approach (not strictly TSP-friendly).
        In real usage, you'd use PMX or OX for TSP, but this is for illustration.
        """
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)

        size = len(parent1)
        cx_point = random.randint(1, size - 2)
        child1 = parent1[:cx_point] + [gene for gene in parent2 if gene not in parent1[:cx_point]]
        child2 = parent2[:cx_point] + [gene for gene in parent1 if gene not in parent2[:cx_point]]
        return child1, child2

    def mutate(self, solution):
        """
        Swap mutation for permutations: pick two indices and swap.
        """
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(solution)), 2)
            solution[i], solution[j] = solution[j], solution[i]

    def optimize(self, problem, **kwargs):
        """
        Main GA loop.
        Returns (best_solution, best_cost).
        """
        population = self.initialize_population(problem)
        best_solution = None
        best_cost = float("inf")

        for gen in range(self.max_generations):
            scored_pop = self.evaluate_population(population, problem)

            # Track best in current population
            for sol, cost in scored_pop:
                if cost < best_cost:
                    best_solution, best_cost = sol, cost

            if self.verbose:
                print(f"Generation {gen} | Best cost so far: {best_cost}")

            # Create new population
            new_pop = []
            while len(new_pop) < self.population_size:
                p1, p2 = self.select_parents(scored_pop)
                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1)
                self.mutate(c2)
                new_pop.append(c1)
                new_pop.append(c2)

            population = new_pop[:self.population_size]

        return best_solution, best_cost

