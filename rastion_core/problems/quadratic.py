# rastion_core/problems/quadratic.py

import random
import numpy as np
from rastion_core.base_problem import BaseProblem

class QuadraticProblem(BaseProblem):
    """
    A basic continuous quadratic problem:
      f(x) = x^T Q x + b^T x + c
    For simplicity, we'll only store Q, assume b=0, c=0 for now.
    """

    def __init__(self, dimension, Q=None):
        """
        :param dimension: number of variables
        :param Q: optional NxN matrix for the quadratic form
        """
        self.dimension = dimension
        if Q is None:
            # default to Identity
            self.Q = np.eye(dimension)
        else:
            self.Q = np.array(Q, dtype=float)

    def evaluate_solution(self, x) -> float:
        x = np.array(x, dtype=float)
        return float(x @ self.Q @ x)

    def gradient(self, x):
        """
        grad f(x) = (Q + Q^T) x
        If Q is symmetric, grad f(x) = 2 Q x.
        We'll just do (Q + Q.T) * x for general Q.
        """
        Qsym = self.Q + self.Q.T
        return Qsym @ x

    def random_solution(self):
        # random x in [-10, 10]
        return np.random.uniform(-10, 10, size=self.dimension)
