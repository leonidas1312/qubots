"""
Specialized base classes for different types of optimization algorithms.
These classes extend BaseOptimizer with algorithm-family-specific functionality.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from .base_optimizer import BaseOptimizer, OptimizerMetadata, OptimizerType, OptimizerFamily, OptimizationResult
from .base_problem import BaseProblem

class PopulationBasedOptimizer(BaseOptimizer):
    """
    Base class for population-based optimization algorithms.
    Provides common functionality for evolutionary algorithms, swarm intelligence, etc.
    """

    def __init__(self, population_size: int = 50, **kwargs):
        """
        Initialize population-based optimizer.

        Args:
            population_size: Size of the population
            **kwargs: Additional parameters
        """
        super().__init__(population_size=population_size, **kwargs)
        self._population_size = population_size
        self._population = []
        self._fitness_values = []
        self._generation = 0

    @abstractmethod
    def initialize_population(self, problem: BaseProblem) -> List[Any]:
        """Initialize the population for the given problem."""
        pass

    @abstractmethod
    def evaluate_population(self, problem: BaseProblem, population: List[Any]) -> List[float]:
        """Evaluate fitness for entire population."""
        pass

    @abstractmethod
    def select_parents(self, population: List[Any], fitness: List[float]) -> List[Any]:
        """Select parents for reproduction."""
        pass

    @abstractmethod
    def reproduce(self, parents: List[Any]) -> List[Any]:
        """Create offspring from parents."""
        pass

    @abstractmethod
    def mutate(self, individual: Any) -> Any:
        """Mutate an individual."""
        pass

    def get_best_individual(self) -> Tuple[Any, float]:
        """Get the best individual from current population."""
        if not self._fitness_values:
            return None, float('inf')

        best_idx = np.argmin(self._fitness_values)  # Assuming minimization
        return self._population[best_idx], self._fitness_values[best_idx]

    def get_population_statistics(self) -> Dict[str, float]:
        """Get statistics about current population."""
        if not self._fitness_values:
            return {}

        fitness_array = np.array(self._fitness_values)
        return {
            "best_fitness": np.min(fitness_array),
            "worst_fitness": np.max(fitness_array),
            "mean_fitness": np.mean(fitness_array),
            "std_fitness": np.std(fitness_array),
            "generation": self._generation
        }

    @property
    def population_size(self) -> int:
        """Get population size."""
        return self._population_size

    @property
    def current_population(self) -> List[Any]:
        """Get current population."""
        return self._population.copy()

    @property
    def current_generation(self) -> int:
        """Get current generation number."""
        return self._generation

class LocalSearchOptimizer(BaseOptimizer):
    """
    Base class for local search optimization algorithms.
    Provides common functionality for hill climbing, simulated annealing, etc.
    """

    def __init__(self, **kwargs):
        """Initialize local search optimizer."""
        super().__init__(**kwargs)
        self._current_solution = None
        self._current_value = None
        self._iteration = 0

    @abstractmethod
    def get_initial_solution(self, problem: BaseProblem) -> Any:
        """Get initial solution for the search."""
        pass

    @abstractmethod
    def get_neighbor(self, solution: Any, problem: BaseProblem) -> Any:
        """Generate a neighboring solution."""
        pass

    @abstractmethod
    def accept_neighbor(self, current_value: float, neighbor_value: float,
                       iteration: int) -> bool:
        """Decide whether to accept a neighboring solution."""
        pass

    def local_search_step(self, problem: BaseProblem) -> bool:
        """
        Perform one step of local search.

        Returns:
            True if improvement was made, False otherwise
        """
        if self._current_solution is None:
            self._current_solution = self.get_initial_solution(problem)
            self._current_value = problem.evaluate_solution(self._current_solution)
            if hasattr(self._current_value, 'objective_value'):
                self._current_value = self._current_value.objective_value

        # Generate neighbor
        neighbor = self.get_neighbor(self._current_solution, problem)
        neighbor_value = problem.evaluate_solution(neighbor)
        if hasattr(neighbor_value, 'objective_value'):
            neighbor_value = neighbor_value.objective_value

        # Decide whether to accept
        if self.accept_neighbor(self._current_value, neighbor_value, self._iteration):
            self._current_solution = neighbor
            self._current_value = neighbor_value
            self._iteration += 1
            return True

        self._iteration += 1
        return False

    @property
    def current_solution(self) -> Any:
        """Get current solution."""
        return self._current_solution

    @property
    def current_value(self) -> float:
        """Get current objective value."""
        return self._current_value

    @property
    def iteration(self) -> int:
        """Get current iteration number."""
        return self._iteration

class GradientBasedOptimizer(BaseOptimizer):
    """
    Base class for gradient-based optimization algorithms.
    Provides common functionality for gradient descent variants.
    """

    def __init__(self, learning_rate: float = 0.01, **kwargs):
        """
        Initialize gradient-based optimizer.

        Args:
            learning_rate: Learning rate for gradient updates
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self._learning_rate = learning_rate
        self._gradient_history = []

    @abstractmethod
    def compute_gradient(self, problem: BaseProblem, solution: List[float]) -> List[float]:
        """Compute gradient at given solution."""
        pass

    def numerical_gradient(self, problem: BaseProblem, solution: List[float],
                          epsilon: float = 1e-8) -> List[float]:
        """Compute numerical gradient using finite differences."""
        gradient = []
        for i in range(len(solution)):
            # Forward difference
            solution_plus = solution.copy()
            solution_plus[i] += epsilon

            solution_minus = solution.copy()
            solution_minus[i] -= epsilon

            f_plus = problem.evaluate_solution(solution_plus)
            f_minus = problem.evaluate_solution(solution_minus)

            if hasattr(f_plus, 'objective_value'):
                f_plus = f_plus.objective_value
            if hasattr(f_minus, 'objective_value'):
                f_minus = f_minus.objective_value

            grad_i = (f_plus - f_minus) / (2 * epsilon)
            gradient.append(grad_i)

        return gradient

    def gradient_step(self, solution: List[float], gradient: List[float]) -> List[float]:
        """Perform one gradient descent step."""
        new_solution = []
        for i in range(len(solution)):
            new_solution.append(solution[i] - self._learning_rate * gradient[i])
        return new_solution

    def apply_bounds(self, solution: List[float], bounds: Dict[str, Tuple[float, float]]) -> List[float]:
        """Apply variable bounds to solution."""
        bounded_solution = []
        var_names = sorted(bounds.keys())

        for i, var_name in enumerate(var_names):
            if i < len(solution):
                min_val, max_val = bounds[var_name]
                bounded_val = max(min_val, min(max_val, solution[i]))
                bounded_solution.append(bounded_val)

        return bounded_solution

    @property
    def learning_rate(self) -> float:
        """Get learning rate."""
        return self._learning_rate

    @property
    def gradient_history(self) -> List[List[float]]:
        """Get gradient history."""
        return self._gradient_history.copy()

class SwarmOptimizer(BaseOptimizer):
    """
    Base class for swarm intelligence algorithms.
    Provides common functionality for PSO, ACO, etc.
    """

    def __init__(self, swarm_size: int = 30, **kwargs):
        """
        Initialize swarm optimizer.

        Args:
            swarm_size: Number of particles/agents in swarm
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self._swarm_size = swarm_size
        self._swarm = []
        self._global_best_position = None
        self._global_best_value = float('inf')

    @abstractmethod
    def initialize_swarm(self, problem: BaseProblem) -> List[Dict[str, Any]]:
        """Initialize the swarm."""
        pass

    @abstractmethod
    def update_particle(self, particle: Dict[str, Any], problem: BaseProblem) -> Dict[str, Any]:
        """Update a single particle/agent."""
        pass

    def update_global_best(self, particle: Dict[str, Any]):
        """Update global best solution."""
        if 'best_value' in particle and particle['best_value'] < self._global_best_value:
            self._global_best_value = particle['best_value']
            self._global_best_position = particle['best_position'].copy()

    @property
    def swarm_size(self) -> int:
        """Get swarm size."""
        return self._swarm_size

    @property
    def global_best_position(self) -> Any:
        """Get global best position."""
        return self._global_best_position

    @property
    def global_best_value(self) -> float:
        """Get global best value."""
        return self._global_best_value

class HybridOptimizer(BaseOptimizer):
    """
    Base class for hybrid optimization algorithms.
    Combines multiple optimization strategies.
    """

    def __init__(self, optimizers: List[BaseOptimizer], **kwargs):
        """
        Initialize hybrid optimizer.

        Args:
            optimizers: List of optimizers to combine
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self._optimizers = optimizers
        self._current_optimizer_idx = 0
        self._switch_criteria = {}

    def add_switch_criterion(self, name: str, criterion_func: callable):
        """Add criterion for switching between optimizers."""
        self._switch_criteria[name] = criterion_func

    def should_switch_optimizer(self, current_result: OptimizationResult) -> bool:
        """Check if optimizer should be switched."""
        for name, criterion in self._switch_criteria.items():
            if criterion(current_result):
                return True
        return False

    def switch_to_next_optimizer(self):
        """Switch to next optimizer in sequence."""
        self._current_optimizer_idx = (self._current_optimizer_idx + 1) % len(self._optimizers)

    @property
    def current_optimizer(self) -> BaseOptimizer:
        """Get currently active optimizer."""
        return self._optimizers[self._current_optimizer_idx]

    @property
    def optimizers(self) -> List[BaseOptimizer]:
        """Get list of all optimizers."""
        return self._optimizers.copy()
