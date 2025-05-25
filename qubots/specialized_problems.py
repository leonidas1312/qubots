"""
Specialized base classes for different types of optimization problems.
These classes extend BaseProblem with domain-specific functionality.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from .base_problem import BaseProblem, ProblemMetadata, ProblemType, ObjectiveType, DifficultyLevel

class ContinuousProblem(BaseProblem):
    """
    Base class for continuous optimization problems.
    Provides additional functionality for problems with continuous variables.
    """
    
    def __init__(self, dimension: int, bounds: Dict[str, Tuple[float, float]], 
                 metadata: Optional[ProblemMetadata] = None):
        """
        Initialize continuous problem.
        
        Args:
            dimension: Number of variables
            bounds: Variable bounds as {"var_name": (min, max)}
            metadata: Problem metadata
        """
        if metadata is None:
            metadata = self._get_default_metadata()
        
        metadata.problem_type = ProblemType.CONTINUOUS
        metadata.dimension = dimension
        metadata.variable_bounds = bounds
        
        super().__init__(metadata)
        self._dimension = dimension
        self._bounds = bounds
        
    def random_solution(self) -> List[float]:
        """Generate a random solution within bounds."""
        solution = []
        for var_name in sorted(self._bounds.keys()):
            min_val, max_val = self._bounds[var_name]
            solution.append(np.random.uniform(min_val, max_val))
        return solution
    
    def get_neighbor_solution(self, solution: List[float], step_size: float = 1.0) -> List[float]:
        """Generate a neighboring solution with Gaussian perturbation."""
        neighbor = []
        for i, (var_name, (min_val, max_val)) in enumerate(sorted(self._bounds.items())):
            # Add Gaussian noise scaled by step_size and variable range
            range_size = max_val - min_val
            noise = np.random.normal(0, step_size * range_size * 0.1)
            new_val = solution[i] + noise
            # Clip to bounds
            new_val = max(min_val, min(max_val, new_val))
            neighbor.append(new_val)
        return neighbor
    
    def distance_between_solutions(self, solution1: List[float], solution2: List[float]) -> float:
        """Calculate Euclidean distance between solutions."""
        return np.linalg.norm(np.array(solution1) - np.array(solution2))
    
    def validate_solution_format(self, solution: Any) -> bool:
        """Validate solution format for continuous problems."""
        if not isinstance(solution, (list, tuple, np.ndarray)):
            return False
        if len(solution) != self._dimension:
            return False
        return all(isinstance(x, (int, float)) for x in solution)
    
    @property
    def dimension(self) -> int:
        """Get problem dimension."""
        return self._dimension
    
    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get variable bounds."""
        return self._bounds.copy()

class DiscreteProblem(BaseProblem):
    """
    Base class for discrete optimization problems.
    Provides functionality for problems with discrete variables.
    """
    
    def __init__(self, variables: Dict[str, List[Any]], 
                 metadata: Optional[ProblemMetadata] = None):
        """
        Initialize discrete problem.
        
        Args:
            variables: Dictionary mapping variable names to possible values
            metadata: Problem metadata
        """
        if metadata is None:
            metadata = self._get_default_metadata()
            
        metadata.problem_type = ProblemType.DISCRETE
        metadata.dimension = len(variables)
        
        super().__init__(metadata)
        self._variables = variables
        self._variable_names = list(sorted(variables.keys()))
    
    def random_solution(self) -> Dict[str, Any]:
        """Generate a random solution."""
        solution = {}
        for var_name in self._variable_names:
            possible_values = self._variables[var_name]
            solution[var_name] = np.random.choice(possible_values)
        return solution
    
    def get_neighbor_solution(self, solution: Dict[str, Any], step_size: float = 1.0) -> Dict[str, Any]:
        """Generate a neighboring solution by changing one variable."""
        neighbor = solution.copy()
        # Choose random variable to change
        var_name = np.random.choice(self._variable_names)
        possible_values = self._variables[var_name]
        # Choose different value
        current_value = solution[var_name]
        other_values = [v for v in possible_values if v != current_value]
        if other_values:
            neighbor[var_name] = np.random.choice(other_values)
        return neighbor
    
    def validate_solution_format(self, solution: Any) -> bool:
        """Validate solution format for discrete problems."""
        if not isinstance(solution, dict):
            return False
        if set(solution.keys()) != set(self._variable_names):
            return False
        for var_name, value in solution.items():
            if value not in self._variables[var_name]:
                return False
        return True
    
    @property
    def variables(self) -> Dict[str, List[Any]]:
        """Get variable definitions."""
        return self._variables.copy()

class CombinatorialProblem(BaseProblem):
    """
    Base class for combinatorial optimization problems.
    Provides functionality for problems involving permutations, selections, etc.
    """
    
    def __init__(self, elements: List[Any], metadata: Optional[ProblemMetadata] = None):
        """
        Initialize combinatorial problem.
        
        Args:
            elements: List of elements to be arranged/selected
            metadata: Problem metadata
        """
        if metadata is None:
            metadata = self._get_default_metadata()
            
        metadata.problem_type = ProblemType.COMBINATORIAL
        metadata.dimension = len(elements)
        
        super().__init__(metadata)
        self._elements = elements
        self._n_elements = len(elements)
    
    def random_permutation(self) -> List[Any]:
        """Generate a random permutation of elements."""
        return np.random.permutation(self._elements).tolist()
    
    def random_selection(self, k: int) -> List[Any]:
        """Generate a random selection of k elements."""
        return np.random.choice(self._elements, size=k, replace=False).tolist()
    
    def get_neighbor_permutation(self, solution: List[Any], step_size: float = 1.0) -> List[Any]:
        """Generate neighboring permutation using 2-opt swap."""
        neighbor = solution.copy()
        n = len(neighbor)
        if n < 2:
            return neighbor
        
        # Perform random swaps based on step_size
        num_swaps = max(1, int(step_size))
        for _ in range(num_swaps):
            i, j = np.random.choice(n, size=2, replace=False)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        return neighbor
    
    def distance_between_permutations(self, perm1: List[Any], perm2: List[Any]) -> float:
        """Calculate distance between permutations (number of different positions)."""
        if len(perm1) != len(perm2):
            return float('inf')
        return sum(1 for a, b in zip(perm1, perm2) if a != b)
    
    @property
    def elements(self) -> List[Any]:
        """Get problem elements."""
        return self._elements.copy()
    
    @property
    def n_elements(self) -> int:
        """Get number of elements."""
        return self._n_elements

class ConstrainedProblem(BaseProblem):
    """
    Base class for constrained optimization problems.
    Provides functionality for handling constraints.
    """
    
    def __init__(self, metadata: Optional[ProblemMetadata] = None):
        """Initialize constrained problem."""
        if metadata is None:
            metadata = self._get_default_metadata()
        
        super().__init__(metadata)
        self._constraints = []
        self._constraint_names = []
    
    def add_constraint(self, constraint_func: callable, name: str, 
                      constraint_type: str = "inequality"):
        """
        Add a constraint to the problem.
        
        Args:
            constraint_func: Function that returns constraint violation (0 = satisfied)
            name: Constraint name
            constraint_type: "equality" or "inequality"
        """
        self._constraints.append({
            "function": constraint_func,
            "name": name,
            "type": constraint_type
        })
        self._constraint_names.append(name)
        self._metadata.constraints_count = len(self._constraints)
    
    def evaluate_constraints(self, solution: Any) -> Dict[str, float]:
        """
        Evaluate all constraints for a solution.
        
        Args:
            solution: Solution to evaluate
            
        Returns:
            Dictionary mapping constraint names to violation values
        """
        violations = {}
        for constraint in self._constraints:
            try:
                violation = constraint["function"](solution)
                violations[constraint["name"]] = float(violation)
            except Exception as e:
                violations[constraint["name"]] = float('inf')
        return violations
    
    def is_feasible(self, solution: Any) -> bool:
        """Check if solution satisfies all constraints."""
        violations = self.evaluate_constraints(solution)
        return all(v <= 1e-6 for v in violations.values())  # Small tolerance for numerical errors
    
    def get_constraint_violation(self, solution: Any) -> float:
        """Get total constraint violation."""
        violations = self.evaluate_constraints(solution)
        return sum(max(0, v) for v in violations.values())
    
    def get_constraints(self) -> List[str]:
        """Get list of constraint descriptions."""
        return [f"{c['name']} ({c['type']})" for c in self._constraints]
    
    @property
    def constraint_count(self) -> int:
        """Get number of constraints."""
        return len(self._constraints)

class MultiObjectiveProblem(BaseProblem):
    """
    Base class for multi-objective optimization problems.
    """
    
    def __init__(self, n_objectives: int, metadata: Optional[ProblemMetadata] = None):
        """
        Initialize multi-objective problem.
        
        Args:
            n_objectives: Number of objectives
            metadata: Problem metadata
        """
        if metadata is None:
            metadata = self._get_default_metadata()
            
        metadata.problem_type = ProblemType.MULTI_OBJECTIVE
        
        super().__init__(metadata)
        self._n_objectives = n_objectives
    
    @abstractmethod
    def evaluate_objectives(self, solution: Any) -> List[float]:
        """
        Evaluate all objectives for a solution.
        
        Args:
            solution: Solution to evaluate
            
        Returns:
            List of objective values
        """
        pass
    
    def evaluate_solution(self, solution: Any) -> List[float]:
        """Evaluate solution (returns list of objective values)."""
        return self.evaluate_objectives(solution)
    
    def dominates(self, solution1: Any, solution2: Any) -> bool:
        """Check if solution1 dominates solution2 (Pareto dominance)."""
        obj1 = self.evaluate_objectives(solution1)
        obj2 = self.evaluate_objectives(solution2)
        
        # Assuming minimization for all objectives
        better_in_all = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
        better_in_some = any(o1 < o2 for o1, o2 in zip(obj1, obj2))
        
        return better_in_all and better_in_some
    
    @property
    def n_objectives(self) -> int:
        """Get number of objectives."""
        return self._n_objectives
