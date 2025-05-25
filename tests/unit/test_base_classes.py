"""
Unit tests for qubots base classes.

Tests the core functionality of BaseProblem and BaseOptimizer classes,
ensuring proper interface implementation and error handling.
"""

import pytest
import time
import numpy as np
from typing import Any, Optional

from qubots import (
    BaseProblem, BaseOptimizer,
    ProblemMetadata, OptimizerMetadata,
    ProblemType, ObjectiveType, DifficultyLevel,
    OptimizerType, OptimizerFamily,
    OptimizationResult
)


class SimpleProblem(BaseProblem):
    """Simple test problem for unit testing."""
    
    def __init__(self, target_value: float = 0.0):
        self.target_value = target_value
        metadata = ProblemMetadata(
            name="Simple Test Problem",
            description="Minimize (x - target)^2",
            problem_type=ProblemType.CONTINUOUS,
            objective_type=ObjectiveType.MINIMIZE,
            difficulty_level=DifficultyLevel.EASY,
            author="Test Suite",
            version="1.0.0"
        )
        super().__init__(metadata)
    
    def evaluate_solution(self, solution: float) -> float:
        """Evaluate solution as (x - target)^2."""
        return (solution - self.target_value) ** 2
    
    def get_random_solution(self) -> float:
        """Generate random solution between -10 and 10."""
        return np.random.uniform(-10, 10)
    
    def is_feasible(self, solution: float) -> bool:
        """All real numbers are feasible."""
        return isinstance(solution, (int, float))


class SimpleOptimizer(BaseOptimizer):
    """Simple test optimizer for unit testing."""
    
    def __init__(self, max_iterations: int = 100):
        self.max_iterations = max_iterations
        metadata = OptimizerMetadata(
            name="Simple Test Optimizer",
            description="Random search optimizer for testing",
            optimizer_type=OptimizerType.HEURISTIC,
            optimizer_family=OptimizerFamily.RANDOM_SEARCH,
            author="Test Suite",
            version="1.0.0"
        )
        super().__init__(metadata, max_iterations=max_iterations)
    
    def _optimize_implementation(self, problem: BaseProblem, initial_solution: Optional[Any] = None) -> OptimizationResult:
        """Simple random search implementation."""
        start_time = time.time()
        
        best_solution = initial_solution if initial_solution is not None else problem.get_random_solution()
        best_value = problem.evaluate_solution(best_solution)
        
        for iteration in range(self.max_iterations):
            candidate = problem.get_random_solution()
            candidate_value = problem.evaluate_solution(candidate)
            
            # Assuming minimization
            if candidate_value < best_value:
                best_solution = candidate
                best_value = candidate_value
        
        end_time = time.time()
        
        return OptimizationResult(
            best_solution=best_solution,
            best_value=best_value,
            iterations=self.max_iterations,
            evaluations=self.max_iterations + 1,
            runtime_seconds=end_time - start_time,
            convergence_achieved=True,
            termination_reason="Maximum iterations reached"
        )


class TestBaseProblem:
    """Test suite for BaseProblem class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.problem = SimpleProblem(target_value=5.0)
    
    def test_problem_initialization(self):
        """Test problem initialization."""
        assert self.problem.metadata.name == "Simple Test Problem"
        assert self.problem.metadata.problem_type == ProblemType.CONTINUOUS
        assert self.problem.metadata.objective_type == ObjectiveType.MINIMIZE
        assert self.problem.target_value == 5.0
    
    def test_metadata_validation(self):
        """Test metadata validation."""
        metadata = self.problem.metadata
        assert hasattr(metadata, 'name')
        assert hasattr(metadata, 'description')
        assert hasattr(metadata, 'problem_type')
        assert hasattr(metadata, 'objective_type')
        assert hasattr(metadata, 'author')
        assert hasattr(metadata, 'version')
    
    def test_solution_evaluation(self):
        """Test solution evaluation."""
        # Test exact target
        result = self.problem.evaluate_solution(5.0)
        assert result == 0.0
        
        # Test off-target
        result = self.problem.evaluate_solution(6.0)
        assert result == 1.0
        
        # Test negative values
        result = self.problem.evaluate_solution(3.0)
        assert result == 4.0
    
    def test_random_solution_generation(self):
        """Test random solution generation."""
        for _ in range(10):
            solution = self.problem.get_random_solution()
            assert isinstance(solution, (int, float))
            assert -10 <= solution <= 10
    
    def test_feasibility_checking(self):
        """Test feasibility checking."""
        assert self.problem.is_feasible(5.0)
        assert self.problem.is_feasible(-3.14)
        assert self.problem.is_feasible(0)
        assert not self.problem.is_feasible("invalid")
        assert not self.problem.is_feasible(None)
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # This should fail because abstract methods aren't implemented
            class IncompleteProblem(BaseProblem):
                def __init__(self):
                    metadata = ProblemMetadata(name="Incomplete", description="Test")
                    super().__init__(metadata)
            
            IncompleteProblem()


class TestBaseOptimizer:
    """Test suite for BaseOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = SimpleOptimizer(max_iterations=50)
        self.problem = SimpleProblem(target_value=3.0)
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.metadata.name == "Simple Test Optimizer"
        assert self.optimizer.metadata.optimizer_type == OptimizerType.HEURISTIC
        assert self.optimizer.metadata.optimizer_family == OptimizerFamily.RANDOM_SEARCH
        assert self.optimizer.max_iterations == 50
    
    def test_parameter_storage(self):
        """Test parameter storage."""
        assert hasattr(self.optimizer, 'parameters')
        assert 'max_iterations' in self.optimizer.parameters
        assert self.optimizer.parameters['max_iterations'] == 50
    
    def test_optimization_execution(self):
        """Test optimization execution."""
        result = self.optimizer.optimize(self.problem)
        
        # Check result structure
        assert isinstance(result, OptimizationResult)
        assert hasattr(result, 'best_solution')
        assert hasattr(result, 'best_value')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'evaluations')
        assert hasattr(result, 'runtime_seconds')
        
        # Check result values
        assert result.best_solution is not None
        assert isinstance(result.best_value, (int, float))
        assert result.iterations == 50
        assert result.evaluations == 51  # 50 iterations + 1 initial
        assert result.runtime_seconds > 0
    
    def test_optimization_with_initial_solution(self):
        """Test optimization with initial solution."""
        initial_solution = 2.9  # Close to target of 3.0
        result = self.optimizer.optimize(self.problem, initial_solution=initial_solution)
        
        assert result.best_solution is not None
        assert result.best_value is not None
        # Should find a solution close to target
        assert abs(result.best_solution - 3.0) < 2.0  # Reasonable tolerance
    
    def test_optimization_result_validation(self):
        """Test optimization result validation."""
        result = self.optimizer.optimize(self.problem)
        
        # Validate solution feasibility
        assert self.problem.is_feasible(result.best_solution)
        
        # Validate evaluation consistency
        evaluated_value = self.problem.evaluate_solution(result.best_solution)
        assert abs(evaluated_value - result.best_value) < 1e-10
    
    def test_multiple_runs_consistency(self):
        """Test that multiple runs produce valid results."""
        results = []
        for _ in range(5):
            result = self.optimizer.optimize(self.problem)
            results.append(result)
        
        # All results should be valid
        for result in results:
            assert result.best_solution is not None
            assert result.best_value is not None
            assert self.problem.is_feasible(result.best_solution)
        
        # Results should vary (randomness)
        solutions = [r.best_solution for r in results]
        assert len(set(solutions)) > 1  # Should have some variation
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # This should fail because abstract methods aren't implemented
            class IncompleteOptimizer(BaseOptimizer):
                def __init__(self):
                    metadata = OptimizerMetadata(name="Incomplete", description="Test")
                    super().__init__(metadata)
            
            IncompleteOptimizer()


class TestOptimizationResult:
    """Test suite for OptimizationResult class."""
    
    def test_result_creation(self):
        """Test optimization result creation."""
        result = OptimizationResult(
            best_solution=[1, 2, 3],
            best_value=42.0,
            iterations=100,
            evaluations=150,
            runtime_seconds=1.5,
            convergence_achieved=True,
            termination_reason="Maximum iterations"
        )
        
        assert result.best_solution == [1, 2, 3]
        assert result.best_value == 42.0
        assert result.iterations == 100
        assert result.evaluations == 150
        assert result.runtime_seconds == 1.5
        assert result.convergence_achieved is True
        assert result.termination_reason == "Maximum iterations"
    
    def test_result_with_additional_metrics(self):
        """Test result with additional metrics."""
        additional_metrics = {
            "population_size": 50,
            "mutation_rate": 0.1,
            "custom_metric": "test_value"
        }
        
        result = OptimizationResult(
            best_solution=None,
            best_value=0.0,
            iterations=0,
            evaluations=0,
            runtime_seconds=0.0,
            convergence_achieved=False,
            termination_reason="Test",
            additional_metrics=additional_metrics
        )
        
        assert result.additional_metrics == additional_metrics
        assert result.additional_metrics["population_size"] == 50


class TestMetadataClasses:
    """Test suite for metadata classes."""
    
    def test_problem_metadata(self):
        """Test ProblemMetadata class."""
        metadata = ProblemMetadata(
            name="Test Problem",
            description="A test problem",
            problem_type=ProblemType.DISCRETE,
            objective_type=ObjectiveType.MAXIMIZE,
            difficulty_level=DifficultyLevel.ADVANCED,
            domain="test_domain",
            tags={"test", "optimization"},
            author="Test Author",
            version="2.0.0",
            dimension=100
        )
        
        assert metadata.name == "Test Problem"
        assert metadata.description == "A test problem"
        assert metadata.problem_type == ProblemType.DISCRETE
        assert metadata.objective_type == ObjectiveType.MAXIMIZE
        assert metadata.difficulty_level == DifficultyLevel.ADVANCED
        assert metadata.domain == "test_domain"
        assert metadata.tags == {"test", "optimization"}
        assert metadata.author == "Test Author"
        assert metadata.version == "2.0.0"
        assert metadata.dimension == 100
    
    def test_optimizer_metadata(self):
        """Test OptimizerMetadata class."""
        metadata = OptimizerMetadata(
            name="Test Optimizer",
            description="A test optimizer",
            optimizer_type=OptimizerType.METAHEURISTIC,
            optimizer_family=OptimizerFamily.EVOLUTIONARY,
            author="Test Author",
            version="1.5.0",
            supports_constraints=True,
            supports_multi_objective=False,
            typical_problems=["tsp", "vrp"],
            required_parameters=["population_size"],
            optional_parameters=["mutation_rate", "crossover_rate"]
        )
        
        assert metadata.name == "Test Optimizer"
        assert metadata.description == "A test optimizer"
        assert metadata.optimizer_type == OptimizerType.METAHEURISTIC
        assert metadata.optimizer_family == OptimizerFamily.EVOLUTIONARY
        assert metadata.author == "Test Author"
        assert metadata.version == "1.5.0"
        assert metadata.supports_constraints is True
        assert metadata.supports_multi_objective is False
        assert metadata.typical_problems == ["tsp", "vrp"]
        assert metadata.required_parameters == ["population_size"]
        assert metadata.optional_parameters == ["mutation_rate", "crossover_rate"]


class TestIntegration:
    """Integration tests for problem-optimizer combinations."""
    
    def test_problem_optimizer_compatibility(self):
        """Test that problems and optimizers work together."""
        problem = SimpleProblem(target_value=0.0)
        optimizer = SimpleOptimizer(max_iterations=20)
        
        result = optimizer.optimize(problem)
        
        # Should find solution close to target
        assert abs(result.best_value) < 10.0  # Should be reasonably close
        assert problem.is_feasible(result.best_solution)
    
    def test_different_problem_types(self):
        """Test optimizer with different problem configurations."""
        targets = [-5.0, 0.0, 5.0, 10.0]
        optimizer = SimpleOptimizer(max_iterations=30)
        
        for target in targets:
            problem = SimpleProblem(target_value=target)
            result = optimizer.optimize(problem)
            
            assert result.best_solution is not None
            assert problem.is_feasible(result.best_solution)
            # Should find reasonable solution
            assert abs(result.best_solution - target) < 5.0


if __name__ == "__main__":
    pytest.main([__file__])
