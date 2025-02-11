from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """
    Base class for any Rastion optimizer.
    Defines a minimal interface so all optimizers
    have a consistent `optimize(problem, initial_solution, ...)` method.
    """

    @abstractmethod
    def optimize(self, problem, initial_solution=None, **kwargs):
        """
        Run the optimization on the given problem.

        If `initial_solution` is provided, use it.
        Otherwise, try using `problem.random_solution()`.

        Return a tuple (best_solution, best_value).
        Child classes must implement their own logic.
        """
        pass
