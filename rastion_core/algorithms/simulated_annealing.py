# rastion_core/algorithms/simulated_annealing.py

import math
import random
from copy import deepcopy
from rastion_core.base_optimizer import BaseOptimizer

class SimulatedAnnealing(BaseOptimizer):
    """
    A general-purpose Simulated Annealing solver.
    Works for both discrete or continuous problems,
    but you must define how to perturb the solution.
    For discrete problems (like TSP, Knapsack), we can do small flips or swaps.
    For continuous, do random perturbation in the real space.
    """

    def __init__(self, initial_temp=100, cooling_rate=0.99, max_iters=1000,
                 neighbor_func=None, verbose=False):
        """
        :param initial_temp: starting temperature
        :param cooling_rate: each iteration T <- T * cooling_rate
        :param max_iters: total number of iterations
        :param neighbor_func: a function that given a solution returns a neighbor
        :param verbose: prints progress if True
        """
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iters = max_iters
        self.neighbor_func = neighbor_func
        self.verbose = verbose

    def default_neighbor_discrete(self, solution):
        """ For discrete solutions (like 0/1, or permutations), do small random change. """
        sol = deepcopy(solution)
        i = random.randrange(len(sol))
        # If it's 0/1, flip bit
        if isinstance(sol[i], int):
            sol[i] = 1 - sol[i]
        else:
            # or for permutations, swap with another index
            j = random.randrange(len(sol))
            sol[i], sol[j] = sol[j], sol[i]
        return sol

    def default_neighbor_continuous(self, solution):
        """ For continuous solutions, do a small random perturbation. """
        sol = solution.copy()  # if it's a list or np.array
        i = random.randrange(len(sol))
        step = random.uniform(-0.5, 0.5)
        sol[i] += step
        return sol

    def optimize(self, problem, initial_solution=None, **kwargs):
        # if no neighbor_func is provided, guess based on problem type
        if not self.neighbor_func:
            # If the problem is 0/1 or permutations, you might try default_neighbor_discrete
            # If it's continuous, you might do default_neighbor_continuous
            # We'll do a naive check if the solution is feasible or we guess from random
            pass

        if initial_solution is None:
            current_sol = problem.random_solution()
        else:
            current_sol = deepcopy(initial_solution)

        current_cost = problem.evaluate_solution(current_sol)
        best_sol = deepcopy(current_sol)
        best_cost = current_cost

        T = self.initial_temp

        for i in range(self.max_iters):
            if not self.neighbor_func:
                # guess from the type of the solution
                if isinstance(current_sol[0], float):
                    neighbor = self.default_neighbor_continuous(current_sol)
                else:
                    neighbor = self.default_neighbor_discrete(current_sol)
            else:
                neighbor = self.neighbor_func(current_sol)

            neighbor_cost = problem.evaluate_solution(neighbor)
            delta = neighbor_cost - current_cost  # we minimize cost
            if delta < 0:
                # accept better solution
                current_sol = neighbor
                current_cost = neighbor_cost
                if current_cost < best_cost:
                    best_sol = deepcopy(current_sol)
                    best_cost = current_cost
            else:
                # accept worse solution w.p. exp(-delta / T)
                acceptance_prob = math.exp(-delta / T) if T > 1e-9 else 0
                if random.random() < acceptance_prob:
                    current_sol = neighbor
                    current_cost = neighbor_cost

            T *= self.cooling_rate
            if self.verbose:
                print(f"Iter {i}, cost={current_cost}, best_cost={best_cost}, T={T}")

        return best_sol, best_cost
