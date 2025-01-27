from abc import ABC, abstractmethod

class BaseProblem(ABC):
    """
    Generic base class for an optimization problem.
    It defines the core interface all problems should have.
    """

    @abstractmethod
    def evaluate_solution(self, solution) -> float:
        """
        Given a candidate solution, return its objective value (a float).
        By convention, 'lower' is better (unless otherwise stated).
        """
        pass

    def is_feasible(self, solution) -> bool:
        """
        Check if the solution is valid under problem constraints.
        Default = True, override if needed.
        """
        return True

    def random_solution(self):
        """
        Generate a random feasible solution (if applicable).
        Raise NotImplementedError by default.
        """
        raise NotImplementedError("random_solution() not implemented for this problem.")

