"""
Enhanced Qubots Library Demo
============================

This example demonstrates the new advanced features of the qubots library v2.0,
including enhanced base classes, specialized problem types, comprehensive benchmarking,
and the registry system.
"""

import numpy as np
from typing import List, Dict, Any

# Import enhanced qubots components
from qubots.base_problem import BaseProblem, ProblemMetadata, ProblemType, ObjectiveType, DifficultyLevel
from qubots.base_optimizer import BaseOptimizer, OptimizerMetadata, OptimizerType, OptimizerFamily
from qubots.specialized_problems import ContinuousProblem, DiscreteProblem
from qubots.specialized_optimizers import PopulationBasedOptimizer, LocalSearchOptimizer
from qubots.benchmarking import BenchmarkSuite, BenchmarkType
from qubots.registry import get_global_registry

# Example 1: Enhanced Problem with Rich Metadata
class RosenbrockProblem(ContinuousProblem):
    """Enhanced Rosenbrock function with comprehensive metadata."""
    
    def __init__(self, dimension: int = 2):
        # Create bounds for all variables
        bounds = {f"x{i}": (-5.0, 5.0) for i in range(dimension)}
        
        # Create comprehensive metadata
        metadata = ProblemMetadata(
            name="Rosenbrock Function",
            description="Classic optimization benchmark function with global minimum at (1,1,...,1)",
            problem_type=ProblemType.CONTINUOUS,
            objective_type=ObjectiveType.MINIMIZE,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            domain="mathematical_optimization",
            tags={"benchmark", "continuous", "non-convex", "multimodal"},
            author="Rastion",
            version="0.0.1",
            dimension=dimension,
            known_optimal=0.0,
            evaluation_complexity="O(n)",
            reference_papers=["Rosenbrock, H.H. (1960). An automatic method for finding the greatest or least value of a function."]
        )
        
        super().__init__(dimension, bounds, metadata)
    
    def _get_default_metadata(self) -> ProblemMetadata:
        return self._metadata
    
    def evaluate_solution(self, solution: List[float]) -> float:
        """Evaluate Rosenbrock function."""
        if len(solution) != self.dimension:
            raise ValueError(f"Solution must have {self.dimension} dimensions")
        
        total = 0.0
        for i in range(len(solution) - 1):
            total += 100 * (solution[i+1] - solution[i]**2)**2 + (1 - solution[i])**2
        
        return total

# Example 2: Enhanced Optimizer with Rich Metadata
class EnhancedGeneticAlgorithm(PopulationBasedOptimizer):
    """Enhanced Genetic Algorithm with comprehensive metadata."""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1, 
                 crossover_rate: float = 0.8, max_generations: int = 100):
        
        # Create comprehensive metadata
        metadata = OptimizerMetadata(
            name="Enhanced Genetic Algorithm",
            description="Population-based evolutionary algorithm with advanced features",
            optimizer_type=OptimizerType.METAHEURISTIC,
            optimizer_family=OptimizerFamily.EVOLUTIONARY,
            author="Rastion Team",
            version="0.0.1",
            is_deterministic=False,
            supports_constraints=False,
            supports_continuous=True,
            supports_discrete=True,
            time_complexity="O(g * p * n)",
            convergence_guaranteed=False,
            parallel_capable=True,
            required_parameters=["population_size"],
            optional_parameters=["mutation_rate", "crossover_rate", "max_generations"],
            parameter_ranges={
                "mutation_rate": (0.0, 1.0),
                "crossover_rate": (0.0, 1.0),
                "population_size": (10, 1000)
            }
        )
        
        super().__init__(
            population_size=population_size,
            metadata=metadata,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            max_generations=max_generations
        )
    
    def _get_default_metadata(self) -> OptimizerMetadata:
        return self._metadata
    
    def initialize_population(self, problem: BaseProblem) -> List[Any]:
        """Initialize random population."""
        population = []
        for _ in range(self.population_size):
            solution = problem.random_solution()
            population.append(solution)
        return population
    
    def evaluate_population(self, problem: BaseProblem, population: List[Any]) -> List[float]:
        """Evaluate entire population."""
        fitness_values = []
        for individual in population:
            fitness = problem.evaluate_solution(individual)
            if hasattr(fitness, 'objective_value'):
                fitness = fitness.objective_value
            fitness_values.append(fitness)
        return fitness_values
    
    def select_parents(self, population: List[Any], fitness: List[float]) -> List[Any]:
        """Tournament selection."""
        parents = []
        for _ in range(len(population)):
            # Tournament selection with size 3
            tournament_indices = np.random.choice(len(population), size=3, replace=False)
            tournament_fitness = [fitness[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            parents.append(population[winner_idx])
        return parents
    
    def reproduce(self, parents: List[Any]) -> List[Any]:
        """Create offspring through crossover."""
        offspring = []
        crossover_rate = self.get_parameter("crossover_rate", 0.8)
        
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % len(parents)]
            
            if np.random.random() < crossover_rate:
                # Simple arithmetic crossover
                alpha = np.random.random()
                child1 = [alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)]
                child2 = [(1 - alpha) * p1 + alpha * p2 for p1, p2 in zip(parent1, parent2)]
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])
        
        return offspring[:len(parents)]
    
    def mutate(self, individual: Any) -> Any:
        """Gaussian mutation."""
        mutation_rate = self.get_parameter("mutation_rate", 0.1)
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[i] += np.random.normal(0, 0.1)
                # Simple bounds checking (assuming [-5, 5])
                mutated[i] = max(-5.0, min(5.0, mutated[i]))
        
        return mutated
    
    def _optimize_implementation(self, problem: BaseProblem, initial_solution=None):
        """Core GA implementation."""
        from qubots.base_optimizer import OptimizationResult
        
        # Initialize
        self._population = self.initialize_population(problem)
        self._fitness_values = self.evaluate_population(problem, self._population)
        
        max_generations = self.get_parameter("max_generations", 100)
        history = []
        
        for generation in range(max_generations):
            # Selection, reproduction, mutation
            parents = self.select_parents(self._population, self._fitness_values)
            offspring = self.reproduce(parents)
            
            # Mutate offspring
            for i in range(len(offspring)):
                offspring[i] = self.mutate(offspring[i])
            
            # Evaluate new population
            self._population = offspring
            self._fitness_values = self.evaluate_population(problem, self._population)
            self._generation = generation
            
            # Track progress
            best_fitness = min(self._fitness_values)
            best_idx = self._fitness_values.index(best_fitness)
            best_solution = self._population[best_idx]
            
            history.append({
                "generation": generation,
                "best_fitness": best_fitness,
                "mean_fitness": np.mean(self._fitness_values),
                "std_fitness": np.std(self._fitness_values)
            })
            
            # Report progress
            self.report_progress(generation, best_fitness, mean_fitness=np.mean(self._fitness_values))
            
            # Check for early stopping
            if self.should_stop():
                break
        
        # Return comprehensive result
        return OptimizationResult(
            best_solution=best_solution,
            best_value=best_fitness,
            is_feasible=True,
            iterations=generation + 1,
            evaluations=len(self._population) * (generation + 1),
            convergence_achieved=False,
            termination_reason="max_generations" if generation == max_generations - 1 else "user_stop",
            optimization_history=history
        )

def demo_enhanced_qubots():
    """Demonstrate enhanced qubots functionality."""
    print("=== Enhanced Qubots Library Demo ===\n")
    
    # 1. Create enhanced problem and optimizer
    print("1. Creating enhanced problem and optimizer...")
    problem = RosenbrockProblem(dimension=5)
    optimizer = EnhancedGeneticAlgorithm(population_size=50, max_generations=100)
    
    print(f"Problem: {problem}")
    print(f"Optimizer: {optimizer}")
    print(f"Problem metadata: {problem.metadata.name} (v{problem.metadata.version})")
    print(f"Optimizer metadata: {optimizer.metadata.name} (v{optimizer.metadata.version})\n")
    
    # 2. Run optimization with enhanced tracking
    print("2. Running optimization with enhanced tracking...")
    
    def progress_callback(data):
        if data["iteration"] % 10 == 0:
            print(f"  Generation {data['iteration']}: Best = {data['best_value']:.6f}")
    
    result = optimizer.optimize(problem, progress_callback=progress_callback)
    
    print(f"\nOptimization completed!")
    print(f"Best solution: {result.best_solution}")
    print(f"Best value: {result.best_value:.6f}")
    print(f"Iterations: {result.iterations}")
    print(f"Evaluations: {result.evaluations}")
    print(f"Runtime: {result.runtime_seconds:.2f} seconds")
    print(f"Termination reason: {result.termination_reason}\n")
    
    # 3. Demonstrate benchmarking
    print("3. Setting up comprehensive benchmarking...")
    benchmark_suite = BenchmarkSuite("Enhanced Qubots Demo", "Demonstration of enhanced benchmarking")
    
    # Add problems and optimizers
    benchmark_suite.add_problem("rosenbrock_2d", RosenbrockProblem(2))
    benchmark_suite.add_problem("rosenbrock_5d", RosenbrockProblem(5))
    
    benchmark_suite.add_optimizer("ga_small", EnhancedGeneticAlgorithm(population_size=20, max_generations=30))
    benchmark_suite.add_optimizer("ga_large", EnhancedGeneticAlgorithm(population_size=50, max_generations=30))
    
    # Run benchmark
    print("Running benchmark (this may take a moment)...")
    benchmark_result = benchmark_suite.run_benchmark("rosenbrock_2d", "ga_small", num_runs=3)
    
    print(f"Benchmark completed!")
    print(f"Best value: {benchmark_result.metrics.best_value:.6f}")
    print(f"Mean value: {benchmark_result.metrics.mean_value:.6f}")
    print(f"Success rate: {benchmark_result.metrics.success_rate:.1f}%")
    print(f"Mean runtime: {benchmark_result.metrics.mean_runtime_seconds:.2f}s\n")
    
    # 4. Demonstrate registry functionality
    print("4. Demonstrating registry functionality...")
    registry = get_global_registry()
    
    # Register our problem and optimizer
    problem_id = registry.register_problem(problem)
    optimizer_id = registry.register_optimizer(optimizer)
    
    print(f"Registered problem with ID: {problem_id}")
    print(f"Registered optimizer with ID: {optimizer_id}")
    
    # Search registry
    search_results = registry.search("rosenbrock")
    print(f"Found {len(search_results)} entries matching 'rosenbrock'")
    
    # Get statistics
    stats = registry.get_statistics()
    print(f"Registry statistics: {stats['total_entries']} total entries")
    print(f"  - Problems: {stats['problems']}")
    print(f"  - Optimizers: {stats['optimizers']}")
    print(f"  - Unique authors: {stats['unique_authors']}\n")
    
    print("=== Demo completed successfully! ===")

if __name__ == "__main__":
    demo_enhanced_qubots()
