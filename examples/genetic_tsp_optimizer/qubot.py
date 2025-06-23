"""
Genetic Algorithm TSP Optimizer for Qubots Framework

This module implements a high-quality genetic algorithm specifically designed
for the Traveling Salesman Problem. It uses order crossover (OX), inversion
mutation, and tournament selection to find optimal or near-optimal tours.

Compatible with Rastion platform workflow automation and dataset-aware problems.
"""

import random
import time
from typing import List, Tuple, Optional, Dict, Any
from qubots import (
    BaseOptimizer, OptimizerMetadata, OptimizerType, OptimizerFamily,
    OptimizationResult, BaseProblem
)


class GeneticTSPOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimizer specifically designed for TSP problems.
    
    Features:
    - Order Crossover (OX) for permutation problems
    - Inversion mutation to maintain tour validity
    - Tournament selection for parent selection
    - Elitism to preserve best solutions
    - Adaptive parameters based on problem size
    - Compatible with dataset-aware TSP problems
    """
    
    def __init__(self,
                 population_size: int = 100,
                 generations: int = 500,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 3,
                 elitism_count: int = 2,
                 adaptive_params: bool = True,
                 early_stopping: bool = True,
                 stagnation_limit: int = 50,
                 **kwargs):
        """
        Initialize genetic algorithm optimizer for TSP.
        
        Args:
            population_size: Number of individuals in population
            generations: Maximum number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            tournament_size: Size of tournament for selection
            elitism_count: Number of best individuals to preserve
            adaptive_params: Whether to adapt parameters based on problem size
            early_stopping: Whether to stop early if no improvement
            stagnation_limit: Generations without improvement before stopping
            **kwargs: Additional parameters
        """
        # Create metadata
        metadata = OptimizerMetadata(
            name="Genetic TSP Optimizer",
            description="Genetic algorithm with order crossover for TSP problems",
            optimizer_type=OptimizerType.METAHEURISTIC,
            optimizer_family=OptimizerFamily.EVOLUTIONARY,
            author="Qubots Framework",
            version="1.0.0",
            is_deterministic=False,
            supports_constraints=False,
            supports_continuous=False,
            supports_discrete=True,
            time_complexity="O(generations * population_size * n)",
            space_complexity="O(population_size * n)",
            convergence_guaranteed=False,
            parallel_capable=False
        )
        
        super().__init__(metadata)
        
        # Algorithm parameters
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.adaptive_params = adaptive_params
        self.early_stopping = early_stopping
        self.stagnation_limit = stagnation_limit
        
        # Runtime state
        self.population = []
        self.fitness_values = []
        self.best_individual = None
        self.best_fitness = float('inf')
        self.generation_count = 0
        self.stagnation_count = 0
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def _get_default_metadata(self) -> OptimizerMetadata:
        """Return default metadata for genetic TSP optimizer."""
        return OptimizerMetadata(
            name="Genetic TSP Optimizer",
            description="Genetic algorithm for TSP problems with order crossover and adaptive parameters",
            optimizer_type=OptimizerType.METAHEURISTIC,
            optimizer_family=OptimizerFamily.EVOLUTIONARY,
            author="Qubots Framework",
            version="1.0.0",
            is_deterministic=False,
            supports_constraints=False,
            supports_continuous=False,
            supports_discrete=True,
            time_complexity="O(generations * population_size * n)",
            space_complexity="O(population_size * n)",
            convergence_guaranteed=False,
            parallel_capable=False
        )
    
    def _optimize_implementation(self, problem: BaseProblem, initial_solution: Optional[Any] = None) -> OptimizationResult:
        """
        Core genetic algorithm implementation for TSP optimization.

        Args:
            problem: TSP problem instance
            initial_solution: Optional initial solution (not used in genetic algorithm)

        Returns:
            OptimizationResult with best tour found
        """
        start_time = time.time()

        # Adapt parameters based on problem size if enabled
        if self.adaptive_params:
            self._adapt_parameters(problem)

        # Initialize population
        self._initialize_population(problem)

        # Evolution loop
        for generation in range(self.generations):
            self.generation_count = generation

            # Check for early termination request
            if self.should_stop():
                self.log_message('info', f'Optimization stopped early at generation {generation}')
                break

            # Evaluate population
            self._evaluate_population(problem)

            # Check for improvement
            current_best = min(self.fitness_values)
            if current_best < self.best_fitness:
                self.best_fitness = current_best
                best_idx = self.fitness_values.index(current_best)
                self.best_individual = self.population[best_idx].copy()
                self.stagnation_count = 0

                # Log improvement
                self.log_message('info', f'New best solution found at generation {generation}: {self.best_fitness:.2f}')
            else:
                self.stagnation_count += 1

            # Report progress every 10 generations
            if generation % 10 == 0:
                self.report_progress(
                    iteration=generation,
                    best_value=self.best_fitness,
                    current_value=current_best,
                    stagnation_count=self.stagnation_count,
                    population_diversity=self._calculate_population_diversity()
                )

            # Early stopping check
            if self.early_stopping and self.stagnation_count >= self.stagnation_limit:
                self.log_message('info', f'Early stopping triggered after {self.stagnation_count} generations without improvement')
                break

            # Create next generation
            self._create_next_generation(problem)

        # Final evaluation
        self._evaluate_population(problem)

        end_time = time.time()
        runtime = end_time - start_time

        return OptimizationResult(
            best_solution=self.best_individual,
            best_value=self.best_fitness,
            iterations=self.generation_count + 1,
            runtime_seconds=runtime,
            convergence_achieved=self.stagnation_count >= self.stagnation_limit,
            termination_reason="early_stopping" if self.stagnation_count >= self.stagnation_limit else "max_generations",
            additional_metrics={
                "population_size": self.population_size,
                "final_generation": self.generation_count,
                "stagnation_count": self.stagnation_count,
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
                "early_stopped": self.stagnation_count >= self.stagnation_limit
            }
        )
    
    def _adapt_parameters(self, problem: BaseProblem):
        """Adapt algorithm parameters based on problem characteristics."""
        # Get problem size
        problem_info = getattr(problem, 'get_problem_info', lambda: {})()
        n_cities = problem_info.get('n_cities', getattr(problem, 'n_cities', 20))
        
        # Adapt population size
        if n_cities <= 20:
            self.population_size = max(50, self.population_size)
        elif n_cities <= 50:
            self.population_size = max(100, self.population_size)
        elif n_cities <= 100:
            self.population_size = max(150, self.population_size)
        else:
            self.population_size = max(200, self.population_size)
        
        # Adapt mutation rate based on problem size
        if n_cities <= 20:
            self.mutation_rate = min(0.15, self.mutation_rate)
        elif n_cities <= 50:
            self.mutation_rate = min(0.1, self.mutation_rate)
        else:
            self.mutation_rate = min(0.05, self.mutation_rate)
        
        # Adapt generations
        if n_cities <= 20:
            self.generations = max(200, self.generations)
        elif n_cities <= 50:
            self.generations = max(500, self.generations)
        else:
            self.generations = max(1000, self.generations)
    
    def _initialize_population(self, problem: BaseProblem):
        """Initialize population with random valid tours."""
        self.population = []
        
        # Get problem size
        n_cities = getattr(problem, 'n_cities', 20)
        
        for _ in range(self.population_size):
            # Create random tour
            tour = list(range(n_cities))
            random.shuffle(tour)
            self.population.append(tour)
        
        self.fitness_values = [float('inf')] * self.population_size
    
    def _evaluate_population(self, problem: BaseProblem):
        """Evaluate fitness of all individuals in population."""
        for i, individual in enumerate(self.population):
            result = problem.evaluate_solution_detailed(individual)
            self.fitness_values[i] = result.objective_value
    
    def _create_next_generation(self, problem: BaseProblem):
        """Create next generation using selection, crossover, and mutation."""
        new_population = []
        
        # Elitism: preserve best individuals
        if self.elitism_count > 0:
            elite_indices = sorted(range(len(self.fitness_values)), 
                                 key=lambda i: self.fitness_values[i])[:self.elitism_count]
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._order_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1 = self._inversion_mutation(child1)
            if random.random() < self.mutation_rate:
                child2 = self._inversion_mutation(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        self.population = new_population[:self.population_size]
        self.fitness_values = [float('inf')] * self.population_size
    
    def _tournament_selection(self) -> List[int]:
        """Select individual using tournament selection."""
        tournament_indices = random.sample(range(len(self.population)), 
                                         min(self.tournament_size, len(self.population)))
        
        best_idx = min(tournament_indices, key=lambda i: self.fitness_values[i])
        return self.population[best_idx].copy()
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Order Crossover (OX) for permutation problems.
        Preserves relative order of cities from parents.
        """
        size = len(parent1)
        
        # Choose random crossover points
        start = random.randint(0, size - 1)
        end = random.randint(start, size - 1)
        
        # Create children
        child1 = [-1] * size
        child2 = [-1] * size
        
        # Copy segments from parents
        child1[start:end+1] = parent1[start:end+1]
        child2[start:end+1] = parent2[start:end+1]
        
        # Fill remaining positions maintaining order
        self._fill_remaining_ox(child1, parent2, start, end)
        self._fill_remaining_ox(child2, parent1, start, end)
        
        return child1, child2
    
    def _fill_remaining_ox(self, child: List[int], parent: List[int], start: int, end: int):
        """Fill remaining positions in order crossover."""
        size = len(child)
        parent_idx = (end + 1) % size
        child_idx = (end + 1) % size
        
        while -1 in child:
            if parent[parent_idx] not in child:
                child[child_idx] = parent[parent_idx]
                child_idx = (child_idx + 1) % size
            parent_idx = (parent_idx + 1) % size
    
    def _inversion_mutation(self, individual: List[int]) -> List[int]:
        """
        Inversion mutation: reverse a random segment of the tour.
        Maintains tour validity while providing local search capability.
        """
        mutated = individual.copy()
        size = len(mutated)
        
        if size < 2:
            return mutated
        
        # Choose random segment to invert
        start = random.randint(0, size - 1)
        end = random.randint(start, size - 1)
        
        # Reverse the segment
        mutated[start:end+1] = reversed(mutated[start:end+1])
        
        return mutated

    def _calculate_population_diversity(self) -> float:
        """
        Calculate population diversity as average pairwise distance.

        Returns:
            Average diversity score (0.0 = no diversity, 1.0 = maximum diversity)
        """
        if len(self.population) < 2:
            return 0.0

        total_distance = 0.0
        comparisons = 0

        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Calculate Hamming distance between tours
                different_positions = sum(1 for a, b in zip(self.population[i], self.population[j]) if a != b)
                total_distance += different_positions / len(self.population[i])
                comparisons += 1

        return total_distance / comparisons if comparisons > 0 else 0.0

    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get comprehensive algorithm information."""
        return {
            "algorithm_name": "Genetic Algorithm",
            "problem_type": "TSP",
            "population_size": self.population_size,
            "generations": self.generations,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "tournament_size": self.tournament_size,
            "elitism_count": self.elitism_count,
            "adaptive_params": self.adaptive_params,
            "early_stopping": self.early_stopping,
            "stagnation_limit": self.stagnation_limit,
            "current_generation": self.generation_count,
            "current_best_fitness": self.best_fitness,
            "stagnation_count": self.stagnation_count
        }


# For backward compatibility
TSPOptimizer = GeneticTSPOptimizer
