# rastion_core/base_optimizer.py

from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """
    Base class for any Rastion optimizer.
    Defines a minimal interface so all optimizers
    have a consistent `optimize(problem, ...)` method.
    """

    @abstractmethod
    def optimize(self, problem, **kwargs):
        """
        Run the optimization on the given problem.
        Return a tuple (best_solution, best_value).
        Child classes must implement their own logic.
        """
        pass
