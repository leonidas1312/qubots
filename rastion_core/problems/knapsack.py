# rastion_core/problems/knapsack.py

import random
from rastion_core.base_problem import BaseProblem

class KnapsackProblem(BaseProblem):
    """
    0-1 Knapsack:
      - We have n items, each with (value, weight).
      - We have a maximum capacity W.
      - A solution is a binary list of length n (0 or 1 for each item).
    """

    def __init__(self, items, capacity):
        """
        :param items: list of (value, weight) for each item
        :param capacity: max capacity of knapsack
        """
        self.items = items
        self.capacity = capacity
        self.n = len(items)

    def evaluate_solution(self, solution) -> float:
        """
        Return negative of total value if solution is feasible, else add a penalty.
        By default, we return a 'cost' we want to MINIMIZE, so we can either:
         1) return negative total value (so a higher total value => lower cost)
         2) or return something like: cost = capacity_violation + (some factor).
        We'll do negative total value + large penalty if overweight.
        """
        total_value = 0
        total_weight = 0
        for i, bit in enumerate(solution):
            if bit == 1:
                val, wt = self.items[i]
                total_value += val
                total_weight += wt
        if total_weight > self.capacity:
            # big penalty if we exceed capacity
            penalty = 1000 * (total_weight - self.capacity)
            return -total_value + penalty
        else:
            # We want to minimize cost => cost = negative of total value
            return -total_value

    def is_feasible(self, solution) -> bool:
        """
        Check if total weight <= capacity.
        """
        total_weight = 0
        for i, bit in enumerate(solution):
            if bit == 1:
                _, wt = self.items[i]
                total_weight += wt
        return (total_weight <= self.capacity)

    def random_solution(self):
        """
        Return a random 0/1 vector of length n.
        """
        sol = [random.randint(0, 1) for _ in range(self.n)]
        return sol
