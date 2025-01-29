# rastion_core/problems/qubo_problem.py

import random
from rastion_core.base_problem import BaseProblem

class QUBOProblem(BaseProblem):
    """
    QUBO: Minimize x^T Q x  for x in {0, 1}^n.
    Q can be a NxN matrix or a dict of ((i,j), coefficient).
    """

    def __init__(self, Q):
        """
        :param Q: NxN 2D list or a dict of ((i, j), value).
        """
        if isinstance(Q, dict):
            self.Q = Q
            # dimension from largest index
            self.n = max(max(i,j) for i,j in Q.keys()) + 1
            self.is_dict = True
        else:
            self.Q = Q
            self.n = len(Q)
            self.is_dict = False

    def evaluate_solution(self, x) -> float:
        cost = 0
        if self.is_dict:
            for (i, j), val in self.Q.items():
                cost += x[i] * val * x[j]
        else:
            for i in range(self.n):
                for j in range(self.n):
                    cost += x[i] * self.Q[i][j] * x[j]
        return cost

    def random_solution(self):
        return [random.randint(0, 1) for _ in range(self.n)]
