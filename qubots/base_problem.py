"""
Base class for optimization problems with comprehensive metadata,
standardized interfaces, and advanced features for the Rastion ecosystem.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import uuid
from datetime import datetime

class ProblemType(Enum):
    """Enumeration of optimization problem types."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    COMBINATORIAL = "combinatorial"
    MIXED_INTEGER = "mixed_integer"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    MULTI_OBJECTIVE = "multi_objective"
    DYNAMIC = "dynamic"
    STOCHASTIC = "stochastic"

class ObjectiveType(Enum):
    """Enumeration of objective function types."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

class DifficultyLevel(Enum):
    """Problem difficulty levels for categorization."""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    RESEARCH = 5

@dataclass
class ProblemMetadata:
    """Comprehensive metadata for optimization problems."""
    name: str
    description: str
    problem_type: ProblemType
    objective_type: ObjectiveType = ObjectiveType.MINIMIZE
    difficulty_level: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    domain: str = "general"
    tags: Set[str] = field(default_factory=set)
    author: str = ""
    version: str = "1.0.0"
    license: str = "MIT"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Problem characteristics
    dimension: Optional[int] = None
    variable_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    constraints_count: int = 0

    # Performance characteristics
    evaluation_complexity: str = "O(n)"  # Big O notation
    memory_complexity: str = "O(n)"
    typical_runtime_ms: Optional[float] = None

    # Benchmarking info
    known_optimal: Optional[float] = None
    benchmark_instances: List[str] = field(default_factory=list)
    reference_papers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "problem_type": self.problem_type.value,
            "objective_type": self.objective_type.value,
            "difficulty_level": self.difficulty_level.value,
            "domain": self.domain,
            "tags": list(self.tags),
            "author": self.author,
            "version": self.version,
            "license": self.license,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "dimension": self.dimension,
            "variable_bounds": self.variable_bounds,
            "constraints_count": self.constraints_count,
            "evaluation_complexity": self.evaluation_complexity,
            "memory_complexity": self.memory_complexity,
            "typical_runtime_ms": self.typical_runtime_ms,
            "known_optimal": self.known_optimal,
            "benchmark_instances": self.benchmark_instances,
            "reference_papers": self.reference_papers
        }

@dataclass
class EvaluationResult:
    """Detailed result from solution evaluation."""
    objective_value: float
    is_feasible: bool = True
    constraint_violations: List[str] = field(default_factory=list)
    evaluation_time_ms: float = 0.0
    additional_metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the evaluation result."""
        if self.constraint_violations:
            self.is_feasible = False

class BaseProblem(ABC):
    """
    Enhanced base class for optimization problems with comprehensive metadata,
    standardized interfaces, and advanced features for the Rastion ecosystem.
    """

    def __init__(self, metadata: Optional[ProblemMetadata] = None):
        """
        Initialize the problem with metadata.

        Args:
            metadata: Problem metadata. If None, subclasses should provide default metadata.
        """
        self._metadata = metadata or self._get_default_metadata()
        self._evaluation_count = 0
        self._total_evaluation_time = 0.0
        self._best_known_solution = None
        self._best_known_value = float('inf') if self._metadata.objective_type == ObjectiveType.MINIMIZE else float('-inf')
        self._instance_id = str(uuid.uuid4())

    @abstractmethod
    def _get_default_metadata(self) -> ProblemMetadata:
        """
        Return default metadata for this problem type.
        Subclasses must implement this method.
        """
        pass

    @abstractmethod
    def evaluate_solution(self, solution: Any) -> Union[float, EvaluationResult]:
        """
        Evaluate a candidate solution and return its objective value.

        Args:
            solution: The candidate solution to evaluate

        Returns:
            Either a float (objective value) or EvaluationResult for detailed feedback
        """
        pass

    def evaluate_solution_detailed(self, solution: Any) -> EvaluationResult:
        """
        Evaluate a solution and return detailed results including timing and feasibility.

        Args:
            solution: The candidate solution to evaluate

        Returns:
            EvaluationResult with comprehensive evaluation information
        """
        start_time = time.perf_counter()

        # Call the main evaluation method
        result = self.evaluate_solution(solution)

        end_time = time.perf_counter()
        evaluation_time = (end_time - start_time) * 1000  # Convert to milliseconds

        # Update statistics
        self._evaluation_count += 1
        self._total_evaluation_time += evaluation_time

        # Handle different return types
        if isinstance(result, EvaluationResult):
            result.evaluation_time_ms = evaluation_time
            objective_value = result.objective_value
        else:
            # Legacy float return
            objective_value = float(result)
            result = EvaluationResult(
                objective_value=objective_value,
                is_feasible=self.is_feasible(solution),
                evaluation_time_ms=evaluation_time
            )

        # Update best known solution
        is_better = (
            (self._metadata.objective_type == ObjectiveType.MINIMIZE and objective_value < self._best_known_value) or
            (self._metadata.objective_type == ObjectiveType.MAXIMIZE and objective_value > self._best_known_value)
        )

        if is_better and result.is_feasible:
            self._best_known_value = objective_value
            self._best_known_solution = solution

        return result

    def is_feasible(self, solution: Any) -> bool:
        """
        Check if the solution is valid under problem constraints.
        Default implementation returns True. Override for constrained problems.

        Args:
            solution: The candidate solution to check

        Returns:
            True if solution is feasible, False otherwise
        """
        return True

    def random_solution(self) -> Any:
        """
        Generate a random feasible solution.
        Default implementation raises NotImplementedError.

        Returns:
            A random feasible solution

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("random_solution() not implemented for this problem.")

    def get_solution_space_info(self) -> Dict[str, Any]:
        """
        Get information about the solution space.

        Returns:
            Dictionary containing solution space characteristics
        """
        return {
            "type": self._metadata.problem_type.value,
            "dimension": self._metadata.dimension,
            "bounds": self._metadata.variable_bounds,
            "constraints_count": self._metadata.constraints_count
        }

    def validate_solution_format(self, solution: Any) -> bool:
        """
        Validate that a solution has the correct format for this problem.
        Default implementation returns True. Override for specific validation.

        Args:
            solution: The solution to validate

        Returns:
            True if format is valid, False otherwise
        """
        return True

    def get_neighbor_solution(self, solution: Any, step_size: float = 1.0) -> Any:
        """
        Generate a neighboring solution for local search algorithms.
        Default implementation raises NotImplementedError.

        Args:
            solution: Current solution
            step_size: Size of the neighborhood step

        Returns:
            A neighboring solution

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("get_neighbor_solution() not implemented for this problem.")

    def distance_between_solutions(self, solution1: Any, solution2: Any) -> float:
        """
        Calculate distance between two solutions.
        Default implementation raises NotImplementedError.

        Args:
            solution1: First solution
            solution2: Second solution

        Returns:
            Distance between solutions

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("distance_between_solutions() not implemented for this problem.")

    def get_bounds(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Get variable bounds for continuous problems.

        Returns:
            Dictionary mapping variable names to (min, max) bounds, or None
        """
        return self._metadata.variable_bounds

    def get_constraints(self) -> List[str]:
        """
        Get list of constraint descriptions.
        Default implementation returns empty list.

        Returns:
            List of constraint descriptions
        """
        return []

    # Metadata and statistics methods
    @property
    def metadata(self) -> ProblemMetadata:
        """Get problem metadata."""
        return self._metadata

    @property
    def evaluation_count(self) -> int:
        """Get number of solution evaluations performed."""
        return self._evaluation_count

    @property
    def average_evaluation_time_ms(self) -> float:
        """Get average evaluation time in milliseconds."""
        if self._evaluation_count == 0:
            return 0.0
        return self._total_evaluation_time / self._evaluation_count

    @property
    def best_known_solution(self) -> Optional[Any]:
        """Get the best solution found so far."""
        return self._best_known_solution

    @property
    def best_known_value(self) -> float:
        """Get the best objective value found so far."""
        return self._best_known_value

    @property
    def instance_id(self) -> str:
        """Get unique instance identifier."""
        return self._instance_id

    def reset_statistics(self):
        """Reset evaluation statistics."""
        self._evaluation_count = 0
        self._total_evaluation_time = 0.0
        self._best_known_solution = None
        self._best_known_value = float('inf') if self._metadata.objective_type == ObjectiveType.MINIMIZE else float('-inf')

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about this problem instance.

        Returns:
            Dictionary containing evaluation statistics
        """
        return {
            "instance_id": self._instance_id,
            "evaluation_count": self._evaluation_count,
            "total_evaluation_time_ms": self._total_evaluation_time,
            "average_evaluation_time_ms": self.average_evaluation_time_ms,
            "best_known_value": self._best_known_value,
            "has_best_solution": self._best_known_solution is not None,
            "metadata": self._metadata.to_dict()
        }

    def export_problem_definition(self) -> Dict[str, Any]:
        """
        Export complete problem definition for sharing or storage.

        Returns:
            Dictionary containing complete problem definition
        """
        return {
            "class_name": self.__class__.__name__,
            "module_name": self.__class__.__module__,
            "metadata": self._metadata.to_dict(),
            "statistics": self.get_statistics(),
            "solution_space_info": self.get_solution_space_info(),
            "constraints": self.get_constraints(),
            "export_timestamp": datetime.now().isoformat()
        }

    def __str__(self) -> str:
        """String representation of the problem."""
        return f"{self._metadata.name} ({self.__class__.__name__})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"{self.__class__.__name__}(name='{self._metadata.name}', "
                f"type={self._metadata.problem_type.value}, "
                f"evaluations={self._evaluation_count})")

