"""
Base class for optimization algorithms with comprehensive metadata,
standardized interfaces, and advanced features for the Rastion ecosystem.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
from datetime import datetime
import threading
from .base_problem import BaseProblem, EvaluationResult

class OptimizerType(Enum):
    """Enumeration of optimizer categories."""
    EXACT = "exact"
    HEURISTIC = "heuristic"
    METAHEURISTIC = "metaheuristic"
    HYBRID = "hybrid"
    MACHINE_LEARNING = "machine_learning"
    QUANTUM = "quantum"

class OptimizerFamily(Enum):
    """Enumeration of optimizer families."""
    GRADIENT_BASED = "gradient_based"
    EVOLUTIONARY = "evolutionary"
    SWARM_INTELLIGENCE = "swarm_intelligence"
    LOCAL_SEARCH = "local_search"
    GLOBAL_SEARCH = "global_search"
    POPULATION_BASED = "population_based"
    SINGLE_SOLUTION = "single_solution"
    TREE_SEARCH = "tree_search"
    CONSTRAINT_PROGRAMMING = "constraint_programming"
    LINEAR_PROGRAMMING = "linear_programming"
    REINFORCEMENT_LEARNING = "reinforcement_learning"

@dataclass
class OptimizerMetadata:
    """Comprehensive metadata for optimization algorithms."""
    name: str
    description: str
    optimizer_type: OptimizerType
    optimizer_family: OptimizerFamily
    author: str = ""
    version: str = "1.0.0"
    license: str = "MIT"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Algorithm characteristics
    is_deterministic: bool = True
    supports_constraints: bool = False
    supports_multi_objective: bool = False
    supports_continuous: bool = True
    supports_discrete: bool = True
    supports_mixed_integer: bool = False

    # Performance characteristics
    time_complexity: str = "O(n)"
    space_complexity: str = "O(n)"
    convergence_guaranteed: bool = False
    parallel_capable: bool = False

    # Parameter information
    required_parameters: List[str] = field(default_factory=list)
    optional_parameters: List[str] = field(default_factory=list)
    parameter_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Benchmarking info
    typical_problems: List[str] = field(default_factory=list)
    benchmark_results: Dict[str, float] = field(default_factory=dict)
    reference_papers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "optimizer_type": self.optimizer_type.value,
            "optimizer_family": self.optimizer_family.value,
            "author": self.author,
            "version": self.version,
            "license": self.license,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_deterministic": self.is_deterministic,
            "supports_constraints": self.supports_constraints,
            "supports_multi_objective": self.supports_multi_objective,
            "supports_continuous": self.supports_continuous,
            "supports_discrete": self.supports_discrete,
            "supports_mixed_integer": self.supports_mixed_integer,
            "time_complexity": self.time_complexity,
            "space_complexity": self.space_complexity,
            "convergence_guaranteed": self.convergence_guaranteed,
            "parallel_capable": self.parallel_capable,
            "required_parameters": self.required_parameters,
            "optional_parameters": self.optional_parameters,
            "parameter_ranges": self.parameter_ranges,
            "typical_problems": self.typical_problems,
            "benchmark_results": self.benchmark_results,
            "reference_papers": self.reference_papers
        }

@dataclass
class OptimizationResult:
    """Comprehensive result from optimization run."""
    best_solution: Any
    best_value: float
    is_feasible: bool = True

    # Optimization statistics
    iterations: int = 0
    evaluations: int = 0
    runtime_seconds: float = 0.0
    convergence_achieved: bool = False
    termination_reason: str = "max_iterations"

    # Detailed tracking
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    parameter_values: Dict[str, Any] = field(default_factory=dict)
    additional_metrics: Dict[str, float] = field(default_factory=dict)

    # Problem and optimizer info
    problem_metadata: Optional[Dict[str, Any]] = None
    optimizer_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "best_value": self.best_value,
            "is_feasible": self.is_feasible,
            "iterations": self.iterations,
            "evaluations": self.evaluations,
            "runtime_seconds": self.runtime_seconds,
            "convergence_achieved": self.convergence_achieved,
            "termination_reason": self.termination_reason,
            "parameter_values": self.parameter_values,
            "additional_metrics": self.additional_metrics,
            "problem_metadata": self.problem_metadata,
            "optimizer_metadata": self.optimizer_metadata,
            "history_length": len(self.optimization_history)
        }

class BaseOptimizer(ABC):
    """
    Enhanced base class for optimization algorithms with comprehensive metadata,
    standardized interfaces, and advanced features for the Rastion ecosystem.
    """

    def __init__(self, metadata: Optional[OptimizerMetadata] = None, **parameters):
        """
        Initialize the optimizer with metadata and parameters.

        Args:
            metadata: Optimizer metadata. If None, subclasses should provide default metadata.
            **parameters: Algorithm-specific parameters
        """
        self._metadata = metadata or self._get_default_metadata()
        self._parameters = parameters
        self._run_count = 0
        self._total_runtime = 0.0
        self._instance_id = str(uuid.uuid4())
        self._is_running = False
        self._should_stop = False
        self._current_result = None
        self._progress_callback = None

        # Validate parameters
        self._validate_parameters()

    @abstractmethod
    def _get_default_metadata(self) -> OptimizerMetadata:
        """
        Return default metadata for this optimizer.
        Subclasses must implement this method.
        """
        pass

    @abstractmethod
    def _optimize_implementation(self, problem: BaseProblem, initial_solution: Optional[Any] = None) -> OptimizationResult:
        """
        Core optimization implementation.
        Subclasses must implement this method.

        Args:
            problem: The optimization problem to solve
            initial_solution: Optional initial solution

        Returns:
            OptimizationResult with comprehensive optimization information
        """
        pass

    def optimize(self, problem: BaseProblem, initial_solution: Optional[Any] = None,
                 progress_callback: Optional[Callable] = None,
                 log_callback: Optional[Callable] = None, **kwargs) -> OptimizationResult:
        """
        Main optimization interface with comprehensive tracking and control.

        Args:
            problem: The optimization problem to solve
            initial_solution: Optional initial solution
            progress_callback: Optional callback for progress updates
            log_callback: Optional callback for real-time logging (level, message, source)
            **kwargs: Additional parameters that override instance parameters

        Returns:
            OptimizationResult with comprehensive optimization information
        """
        # Merge kwargs with instance parameters
        merged_params = {**self._parameters, **kwargs}
        original_params = self._parameters.copy()
        self._parameters.update(kwargs)

        try:
            # Setup
            self._is_running = True
            self._should_stop = False
            self._progress_callback = progress_callback
            self._log_callback = log_callback
            start_time = time.time()

            # Log optimization start
            if log_callback:
                log_callback('info', f'Starting {self._metadata.name} optimization', 'optimizer')
                log_callback('info', f'Problem: {problem.metadata.name}', 'optimizer')
                if merged_params:
                    log_callback('debug', f'Parameters: {merged_params}', 'optimizer')

            # Validate compatibility
            self._validate_problem_compatibility(problem)

            # Run optimization
            result = self._optimize_implementation(problem, initial_solution)

            # Finalize result
            end_time = time.time()
            result.runtime_seconds = end_time - start_time
            result.parameter_values = merged_params.copy()
            result.problem_metadata = problem.metadata.to_dict()
            result.optimizer_metadata = self._metadata.to_dict()

            # Update statistics
            self._run_count += 1
            self._total_runtime += result.runtime_seconds
            self._current_result = result

            # Log completion
            if log_callback:
                log_callback('info', f'Optimization completed in {result.runtime_seconds:.3f}s', 'optimizer')
                log_callback('info', f'Best value: {result.best_value:.6f}', 'optimizer')
                if hasattr(result, 'iterations'):
                    log_callback('info', f'Total iterations: {result.iterations}', 'optimizer')

            return result

        finally:
            # Cleanup
            self._is_running = False
            self._should_stop = False
            self._progress_callback = None
            self._log_callback = None
            self._parameters = original_params

    def optimize_legacy(self, problem: BaseProblem, initial_solution: Optional[Any] = None, **kwargs) -> Tuple[Any, float]:
        """
        Legacy optimization interface for backward compatibility.

        Args:
            problem: The optimization problem to solve
            initial_solution: Optional initial solution
            **kwargs: Additional parameters

        Returns:
            Tuple of (best_solution, best_value)
        """
        result = self.optimize(problem, initial_solution, **kwargs)
        return result.best_solution, result.best_value

    def stop_optimization(self):
        """Request optimization to stop gracefully."""
        self._should_stop = True

    def is_running(self) -> bool:
        """Check if optimization is currently running."""
        return self._is_running

    def should_stop(self) -> bool:
        """Check if optimization should stop."""
        return self._should_stop

    def report_progress(self, iteration: int, best_value: float, **metrics):
        """Report progress to callback if available."""
        if self._progress_callback:
            progress_data = {
                "iteration": iteration,
                "best_value": best_value,
                "optimizer_id": self._instance_id,
                **metrics
            }
            self._progress_callback(progress_data)

    def log_message(self, level: str, message: str, **context):
        """Log a message during optimization if callback is available."""
        if hasattr(self, '_log_callback') and self._log_callback:
            self._log_callback(level, message, 'optimizer', **context)

    def set_log_callback(self, log_callback):
        """Set logging callback for real-time log streaming."""
        self._log_callback = log_callback

    def _validate_parameters(self):
        """Validate optimizer parameters against metadata requirements."""
        required = set(self._metadata.required_parameters)
        provided = set(self._parameters.keys())
        optional = set(self._metadata.optional_parameters)
        missing = required - provided - optional
        if missing:
            raise ValueError(f"Missing required parameters: {missing} (optional: {optional})")

        # Validate parameter ranges
        for param, value in self._parameters.items():
            if param in self._metadata.parameter_ranges:
                min_val, max_val = self._metadata.parameter_ranges[param]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"Parameter {param}={value} outside valid range [{min_val}, {max_val}]")

    def _validate_problem_compatibility(self, problem: BaseProblem):
        """Validate that this optimizer can handle the given problem."""
        problem_type = problem.metadata.problem_type

        # Check basic compatibility
        if problem_type.value == "continuous" and not self._metadata.supports_continuous:
            raise ValueError(f"Optimizer {self._metadata.name} does not support continuous problems")
        elif problem_type.value == "discrete" and not self._metadata.supports_discrete:
            raise ValueError(f"Optimizer {self._metadata.name} does not support discrete problems")
        elif problem_type.value == "mixed_integer" and not self._metadata.supports_mixed_integer:
            raise ValueError(f"Optimizer {self._metadata.name} does not support mixed-integer problems")

        # Check constraint support
        if problem.metadata.constraints_count > 0 and not self._metadata.supports_constraints:
            raise ValueError(f"Optimizer {self._metadata.name} does not support constrained problems")

    def get_parameter_info(self) -> Dict[str, Any]:
        """Get information about optimizer parameters."""
        return {
            "required_parameters": self._metadata.required_parameters,
            "optional_parameters": self._metadata.optional_parameters,
            "parameter_ranges": self._metadata.parameter_ranges,
            "current_values": self._parameters.copy()
        }

    def set_parameters(self, **parameters):
        """Update optimizer parameters."""
        self._parameters.update(parameters)
        self._validate_parameters()

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a specific parameter value."""
        return self._parameters.get(name, default)

    # Metadata and statistics methods
    @property
    def metadata(self) -> OptimizerMetadata:
        """Get optimizer metadata."""
        return self._metadata

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return self._parameters.copy()

    @property
    def run_count(self) -> int:
        """Get number of optimization runs performed."""
        return self._run_count

    @property
    def average_runtime_seconds(self) -> float:
        """Get average runtime in seconds."""
        if self._run_count == 0:
            return 0.0
        return self._total_runtime / self._run_count

    @property
    def instance_id(self) -> str:
        """Get unique instance identifier."""
        return self._instance_id

    @property
    def current_result(self) -> Optional[OptimizationResult]:
        """Get the most recent optimization result."""
        return self._current_result

    def reset_statistics(self):
        """Reset optimization statistics."""
        self._run_count = 0
        self._total_runtime = 0.0
        self._current_result = None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about this optimizer instance.

        Returns:
            Dictionary containing optimization statistics
        """
        return {
            "instance_id": self._instance_id,
            "run_count": self._run_count,
            "total_runtime_seconds": self._total_runtime,
            "average_runtime_seconds": self.average_runtime_seconds,
            "is_running": self._is_running,
            "metadata": self._metadata.to_dict(),
            "current_parameters": self._parameters.copy(),
            "has_current_result": self._current_result is not None
        }

    def export_optimizer_definition(self) -> Dict[str, Any]:
        """
        Export complete optimizer definition for sharing or storage.

        Returns:
            Dictionary containing complete optimizer definition
        """
        return {
            "class_name": self.__class__.__name__,
            "module_name": self.__class__.__module__,
            "metadata": self._metadata.to_dict(),
            "parameters": self._parameters.copy(),
            "statistics": self.get_statistics(),
            "parameter_info": self.get_parameter_info(),
            "export_timestamp": datetime.now().isoformat()
        }

    def __str__(self) -> str:
        """String representation of the optimizer."""
        return f"{self._metadata.name} ({self.__class__.__name__})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"{self.__class__.__name__}(name='{self._metadata.name}', "
                f"type={self._metadata.optimizer_type.value}, "
                f"runs={self._run_count})")