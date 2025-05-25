# Creating Custom Optimizers

This tutorial teaches you how to create sophisticated custom optimization algorithms using the qubots framework. You'll learn to implement various optimization strategies, handle different problem types, and create production-ready optimizers.

## 🎯 Learning Objectives

After completing this tutorial, you will be able to:
- Design and implement custom optimization algorithms
- Handle different problem types (continuous, discrete, combinatorial)
- Implement advanced optimization strategies (genetic algorithms, simulated annealing, etc.)
- Add proper error handling and validation
- Create configurable and reusable optimizers
- Benchmark and compare your algorithms

## 📋 Prerequisites

- Completed the [Getting Started Tutorial](getting_started.md)
- Understanding of optimization concepts (local search, metaheuristics)
- Familiarity with NumPy and basic algorithms

## 🏗️ Optimizer Architecture

### Core Components

Every qubots optimizer must implement:

1. **Metadata Definition**: Describe your optimizer
2. **Parameter Handling**: Accept and validate configuration
3. **Optimization Logic**: The core algorithm implementation
4. **Result Generation**: Return structured results

### Base Class Structure

```python
from qubots import BaseOptimizer, OptimizerMetadata, OptimizationResult

class MyOptimizer(BaseOptimizer):
    def __init__(self, **params):
        # Define metadata
        metadata = OptimizerMetadata(...)
        super().__init__(metadata, **params)
    
    def _optimize_implementation(self, problem, initial_solution=None):
        # Your optimization logic here
        return OptimizationResult(...)
```

## 🧬 Example 1: Genetic Algorithm

Let's implement a genetic algorithm for combinatorial optimization:

```python
import numpy as np
import time
from typing import List, Any, Optional
from qubots import BaseOptimizer, OptimizerMetadata, OptimizationResult
from qubots import OptimizerType, OptimizerFamily

class GeneticAlgorithm(BaseOptimizer):
    """
    Genetic Algorithm implementation for combinatorial optimization.
    
    Uses tournament selection, crossover, and mutation to evolve
    a population of solutions toward better fitness.
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 max_generations: int = 200,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 3,
                 elitism_count: int = 2):
        """
        Initialize Genetic Algorithm.
        
        Args:
            population_size: Number of individuals in population
            max_generations: Maximum number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            tournament_size: Size of tournament for selection
            elitism_count: Number of best individuals to preserve
        """
        
        # Validate parameters
        if population_size < 2:
            raise ValueError("Population size must be at least 2")
        if not 0 <= crossover_rate <= 1:
            raise ValueError("Crossover rate must be between 0 and 1")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        
        # Store parameters
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        
        # Define metadata
        metadata = OptimizerMetadata(
            name="Genetic Algorithm",
            description="Evolutionary algorithm using selection, crossover, and mutation",
            optimizer_type=OptimizerType.METAHEURISTIC,
            optimizer_family=OptimizerFamily.EVOLUTIONARY,
            author="Custom Optimizer Tutorial",
            version="1.0.0",
            supports_constraints=True,
            supports_multi_objective=False,
            typical_problems=["combinatorial", "discrete", "permutation"],
            required_parameters=["population_size", "max_generations"],
            optional_parameters=["crossover_rate", "mutation_rate", "tournament_size", "elitism_count"]
        )
        
        super().__init__(metadata, 
                        population_size=population_size,
                        max_generations=max_generations,
                        crossover_rate=crossover_rate,
                        mutation_rate=mutation_rate,
                        tournament_size=tournament_size,
                        elitism_count=elitism_count)
    
    def _optimize_implementation(self, problem, initial_solution=None):
        """
        Run genetic algorithm optimization.
        
        Args:
            problem: Problem instance to optimize
            initial_solution: Optional initial solution
            
        Returns:
            OptimizationResult with best solution found
        """
        start_time = time.time()
        
        # Initialize population
        population = self._initialize_population(problem, initial_solution)
        fitness_values = [problem.evaluate_solution(ind) for ind in population]
        
        # Track best solution
        best_idx = np.argmin(fitness_values)  # Assuming minimization
        best_solution = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        
        generation_count = 0
        evaluations = len(population)
        
        # Evolution loop
        for generation in range(self.max_generations):
            generation_count = generation + 1
            
            # Create new population
            new_population = []
            
            # Elitism: preserve best individuals
            elite_indices = np.argsort(fitness_values)[:self.elitism_count]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_values)
                parent2 = self._tournament_selection(population, fitness_values)
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2, problem)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if np.random.random() < self.mutation_rate:
                    child1 = self._mutate(child1, problem)
                if np.random.random() < self.mutation_rate:
                    child2 = self._mutate(child2, problem)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            new_population = new_population[:self.population_size]
            
            # Evaluate new population
            new_fitness = [problem.evaluate_solution(ind) for ind in new_population]
            evaluations += len(new_population)
            
            # Update best solution
            current_best_idx = np.argmin(new_fitness)
            current_best_fitness = new_fitness[current_best_idx]
            
            if current_best_fitness < best_fitness:
                best_solution = new_population[current_best_idx].copy()
                best_fitness = current_best_fitness
            
            # Update population
            population = new_population
            fitness_values = new_fitness
            
            # Optional: early stopping criteria
            if self._should_terminate(generation, best_fitness):
                break
        
        end_time = time.time()
        
        return OptimizationResult(
            best_solution=best_solution,
            best_value=best_fitness,
            iterations=generation_count,
            evaluations=evaluations,
            runtime_seconds=end_time - start_time,
            convergence_achieved=generation_count < self.max_generations,
            termination_reason="Maximum generations reached" if generation_count >= self.max_generations else "Early termination",
            additional_metrics={
                "final_population_diversity": self._calculate_diversity(population),
                "generations_completed": generation_count,
                "average_fitness": np.mean(fitness_values),
                "fitness_std": np.std(fitness_values)
            }
        )
    
    def _initialize_population(self, problem, initial_solution=None):
        """Initialize the population with random solutions."""
        population = []
        
        # Add initial solution if provided
        if initial_solution is not None and problem.is_feasible(initial_solution):
            population.append(initial_solution)
        
        # Fill rest with random solutions
        while len(population) < self.population_size:
            solution = problem.get_random_solution()
            if problem.is_feasible(solution):
                population.append(solution)
        
        return population
    
    def _tournament_selection(self, population, fitness_values):
        """Select individual using tournament selection."""
        tournament_indices = np.random.choice(
            len(population), 
            size=min(self.tournament_size, len(population)), 
            replace=False
        )
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1, parent2, problem):
        """
        Perform crossover operation.
        This is a generic implementation - override for specific problem types.
        """
        # For list-based solutions, use order crossover
        if isinstance(parent1, list) and isinstance(parent2, list):
            return self._order_crossover(parent1, parent2)
        
        # For array-based solutions, use uniform crossover
        elif hasattr(parent1, '__len__'):
            return self._uniform_crossover(parent1, parent2)
        
        # For scalar solutions, use arithmetic crossover
        else:
            alpha = np.random.random()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
            return child1, child2
    
    def _order_crossover(self, parent1, parent2):
        """Order crossover for permutation problems."""
        size = len(parent1)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        
        child1 = [None] * size
        child2 = [None] * size
        
        # Copy segments
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        # Fill remaining positions
        self._fill_remaining_positions(child1, parent2, start, end)
        self._fill_remaining_positions(child2, parent1, start, end)
        
        return child1, child2
    
    def _fill_remaining_positions(self, child, parent, start, end):
        """Helper for order crossover."""
        child_set = set(child[start:end])
        parent_filtered = [x for x in parent if x not in child_set]
        
        j = 0
        for i in range(len(child)):
            if child[i] is None:
                child[i] = parent_filtered[j]
                j += 1
    
    def _uniform_crossover(self, parent1, parent2):
        """Uniform crossover for array-based solutions."""
        mask = np.random.random(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2
    
    def _mutate(self, individual, problem):
        """
        Perform mutation operation.
        This is a generic implementation - override for specific problem types.
        """
        if isinstance(individual, list):
            return self._swap_mutation(individual)
        elif hasattr(individual, '__len__'):
            return self._gaussian_mutation(individual)
        else:
            # Scalar mutation
            return individual + np.random.normal(0, 0.1)
    
    def _swap_mutation(self, individual):
        """Swap mutation for permutation problems."""
        mutated = individual.copy()
        if len(mutated) > 1:
            i, j = np.random.choice(len(mutated), 2, replace=False)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    def _gaussian_mutation(self, individual):
        """Gaussian mutation for continuous problems."""
        mutated = individual.copy()
        mutation_strength = 0.1
        mutated += np.random.normal(0, mutation_strength, size=len(individual))
        return mutated
    
    def _should_terminate(self, generation, best_fitness):
        """Check if early termination criteria are met."""
        # Implement custom termination criteria here
        # For example: no improvement for X generations
        return False
    
    def _calculate_diversity(self, population):
        """Calculate population diversity metric."""
        if len(population) < 2:
            return 0.0
        
        # Simple diversity measure: average pairwise distance
        total_distance = 0
        count = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # Simple distance measure
                if isinstance(population[i], list):
                    distance = sum(1 for a, b in zip(population[i], population[j]) if a != b)
                else:
                    distance = np.linalg.norm(np.array(population[i]) - np.array(population[j]))
                
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0

# Example usage
if __name__ == "__main__":
    # Test with a simple problem
    from qubots import BaseProblem, ProblemMetadata, ProblemType, ObjectiveType
    
    class TestProblem(BaseProblem):
        def __init__(self):
            metadata = ProblemMetadata(
                name="Test Problem",
                description="Simple test for genetic algorithm",
                problem_type=ProblemType.DISCRETE,
                objective_type=ObjectiveType.MINIMIZE
            )
            super().__init__(metadata)
        
        def evaluate_solution(self, solution):
            # Minimize sum of squares
            return sum(x**2 for x in solution)
        
        def get_random_solution(self):
            return [np.random.randint(-10, 11) for _ in range(5)]
        
        def is_feasible(self, solution):
            return len(solution) == 5 and all(-10 <= x <= 10 for x in solution)
    
    # Test the genetic algorithm
    problem = TestProblem()
    optimizer = GeneticAlgorithm(
        population_size=50,
        max_generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    result = optimizer.optimize(problem)
    print(f"Best solution: {result.best_solution}")
    print(f"Best value: {result.best_value}")
    print(f"Generations: {result.iterations}")
    print(f"Runtime: {result.runtime_seconds:.3f}s")
```

## 🌡️ Example 2: Simulated Annealing

Now let's implement simulated annealing for continuous optimization:

```python
import math
import numpy as np
from qubots import BaseOptimizer, OptimizerMetadata, OptimizationResult

class SimulatedAnnealing(BaseOptimizer):
    """
    Simulated Annealing algorithm for continuous optimization.
    
    Uses temperature-based acceptance probability to escape local optima.
    """
    
    def __init__(self,
                 initial_temperature: float = 100.0,
                 final_temperature: float = 0.01,
                 cooling_rate: float = 0.95,
                 max_iterations: int = 10000,
                 step_size: float = 1.0):
        
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.step_size = step_size
        
        metadata = OptimizerMetadata(
            name="Simulated Annealing",
            description="Temperature-based metaheuristic for global optimization",
            optimizer_type=OptimizerType.METAHEURISTIC,
            optimizer_family=OptimizerFamily.LOCAL_SEARCH,
            author="Custom Optimizer Tutorial",
            version="1.0.0"
        )
        
        super().__init__(metadata, **locals())
    
    def _optimize_implementation(self, problem, initial_solution=None):
        start_time = time.time()
        
        # Initialize solution
        if initial_solution is not None and problem.is_feasible(initial_solution):
            current_solution = initial_solution
        else:
            current_solution = problem.get_random_solution()
        
        current_value = problem.evaluate_solution(current_solution)
        
        # Track best solution
        best_solution = current_solution.copy() if hasattr(current_solution, 'copy') else current_solution
        best_value = current_value
        
        temperature = self.initial_temperature
        evaluations = 1
        
        for iteration in range(self.max_iterations):
            # Generate neighbor solution
            neighbor = self._generate_neighbor(current_solution, problem)
            neighbor_value = problem.evaluate_solution(neighbor)
            evaluations += 1
            
            # Calculate acceptance probability
            if neighbor_value < current_value:
                # Better solution - always accept
                accept = True
            else:
                # Worse solution - accept with probability
                delta = neighbor_value - current_value
                probability = math.exp(-delta / temperature)
                accept = np.random.random() < probability
            
            # Update current solution
            if accept:
                current_solution = neighbor
                current_value = neighbor_value
                
                # Update best solution
                if current_value < best_value:
                    best_solution = current_solution.copy() if hasattr(current_solution, 'copy') else current_solution
                    best_value = current_value
            
            # Cool down temperature
            temperature *= self.cooling_rate
            
            # Check termination
            if temperature < self.final_temperature:
                break
        
        end_time = time.time()
        
        return OptimizationResult(
            best_solution=best_solution,
            best_value=best_value,
            iterations=iteration + 1,
            evaluations=evaluations,
            runtime_seconds=end_time - start_time,
            convergence_achieved=temperature < self.final_temperature,
            termination_reason="Temperature threshold reached" if temperature < self.final_temperature else "Maximum iterations",
            additional_metrics={
                "final_temperature": temperature,
                "acceptance_ratio": iteration / evaluations if evaluations > 0 else 0
            }
        )
    
    def _generate_neighbor(self, solution, problem):
        """Generate a neighbor solution."""
        if isinstance(solution, (list, np.ndarray)):
            # For array-like solutions
            neighbor = np.array(solution) + np.random.normal(0, self.step_size, size=len(solution))
            return neighbor.tolist() if isinstance(solution, list) else neighbor
        else:
            # For scalar solutions
            return solution + np.random.normal(0, self.step_size)
```

## 🎯 Best Practices for Custom Optimizers

### 1. Parameter Validation
```python
def __init__(self, param1, param2, **kwargs):
    # Validate parameters
    if param1 <= 0:
        raise ValueError("param1 must be positive")
    if not 0 <= param2 <= 1:
        raise ValueError("param2 must be between 0 and 1")
```

### 2. Comprehensive Metadata
```python
metadata = OptimizerMetadata(
    name="My Algorithm",
    description="Detailed description of what it does",
    optimizer_type=OptimizerType.METAHEURISTIC,
    optimizer_family=OptimizerFamily.EVOLUTIONARY,
    author="Your Name",
    version="1.0.0",
    supports_constraints=True,
    supports_multi_objective=False,
    typical_problems=["routing", "scheduling"],
    required_parameters=["population_size"],
    optional_parameters=["mutation_rate", "crossover_rate"]
)
```

### 3. Robust Error Handling
```python
def _optimize_implementation(self, problem, initial_solution=None):
    try:
        # Validation
        if not hasattr(problem, 'evaluate_solution'):
            raise ValueError("Problem must have evaluate_solution method")
        
        # Your optimization logic
        
    except Exception as e:
        return OptimizationResult(
            best_solution=None,
            best_value=float('inf'),
            iterations=0,
            evaluations=0,
            runtime_seconds=0,
            convergence_achieved=False,
            termination_reason=f"Error: {str(e)}"
        )
```

### 4. Progress Tracking
```python
def _optimize_implementation(self, problem, initial_solution=None):
    # Track progress
    progress_callback = self.parameters.get('progress_callback')
    
    for iteration in range(max_iterations):
        # Optimization logic
        
        # Report progress
        if progress_callback and iteration % 100 == 0:
            progress_callback(iteration, max_iterations, best_value)
```

## 🧪 Testing Your Optimizer

Create comprehensive tests for your optimizer:

```python
import unittest
from qubots import BaseProblem, ProblemMetadata

class TestMyOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = MyOptimizer()
        self.simple_problem = SimpleTestProblem()
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertIsNotNone(self.optimizer.metadata)
        self.assertEqual(self.optimizer.metadata.name, "My Algorithm")
    
    def test_optimization_runs(self):
        """Test that optimization completes without errors."""
        result = self.optimizer.optimize(self.simple_problem)
        self.assertIsNotNone(result.best_solution)
        self.assertIsInstance(result.best_value, (int, float))
        self.assertGreaterEqual(result.runtime_seconds, 0)
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        with self.assertRaises(ValueError):
            MyOptimizer(invalid_param=-1)
    
    def test_convergence(self):
        """Test convergence on known problem."""
        # Use a problem with known optimal solution
        result = self.optimizer.optimize(self.simple_problem)
        # Assert that result is reasonable
        self.assertLess(result.best_value, 1000)  # Sanity check

if __name__ == '__main__':
    unittest.main()
```

## 📊 Benchmarking Your Optimizer

Compare your optimizer against others:

```python
from qubots import BenchmarkSuite

# Create benchmark suite
suite = BenchmarkSuite()

# Add optimizers to compare
suite.add_optimizer("My Algorithm", MyOptimizer())
suite.add_optimizer("Random Search", RandomSearchOptimizer())
suite.add_optimizer("Genetic Algorithm", GeneticAlgorithm())

# Run benchmarks
results = suite.run_benchmarks(problem, num_runs=10)

# Generate report
suite.generate_report(results, "my_optimizer_benchmark.html")
```

## 🚀 Next Steps

Now that you can create custom optimizers:

1. **Implement Domain-Specific Optimizers**: Create optimizers tailored to specific problem types
2. **Add Advanced Features**: Multi-objective optimization, constraint handling, parallel processing
3. **Share with Community**: Upload your optimizers to the Rastion platform
4. **Contribute to Qubots**: Submit your optimizers as contributions to the main library

## 📚 Further Reading

- [Metaheuristic Algorithms](https://en.wikipedia.org/wiki/Metaheuristic)
- [Evolutionary Computation](https://en.wikipedia.org/wiki/Evolutionary_computation)
- [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing)
- [Benchmarking Guide](../guides/benchmarking.md)

Happy optimizing! 🎯
