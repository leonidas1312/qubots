# rastion_core/problems/mixed_integer.py

import random
from rastion_core.base_problem import BaseProblem

class MixedIntegerProblem(BaseProblem):
    """
    A generic stub for a mixed-integer problem.
    Some variables are integer/binary, some are continuous.
    """

    def __init__(self, int_dims, cont_dims, evaluate_func):
        """
        :param int_dims: number of integer (or binary) variables
        :param cont_dims: number of continuous variables
        :param evaluate_func: a callable to evaluate solution
        """
        self.int_dims = int_dims
        self.cont_dims = cont_dims
        self.evaluate_func = evaluate_func  # user-provided

    def evaluate_solution(self, solution) -> float:
        # solution might be a list [int1, int2, ..., float1, float2, ...]
        return self.evaluate_func(solution)

    def random_solution(self):
        sol = []
        # integer part in [0..9] for example
        for _ in range(self.int_dims):
            sol.append(random.randint(0, 9))
        # continuous part in [-10,10]
        for _ in range(self.cont_dims):
            sol.append(random.uniform(-10,10))
        return sol
