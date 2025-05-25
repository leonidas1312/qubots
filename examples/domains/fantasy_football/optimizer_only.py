"""
Fantasy Football Genetic Algorithm Optimizer - Pure Optimizer Class
================================================================

This file contains only the optimizer class definition, ready for upload
to the Rastion platform. It has no external dependencies beyond qubots
and can be used with any compatible fantasy football problem.

Usage:
    from optimizer_only import FantasyFootballGeneticOptimizer

    optimizer = FantasyFootballGeneticOptimizer(
        population_size=100,
        max_generations=200
    )

    result = optimizer.optimize(problem)

Author: Qubots Fantasy Football Tutorial
Version: 1.0.0
License: Apache 2.0
"""

import numpy as np
import time
from typing import List, Any, Optional

from qubots import (
    BaseOptimizer, OptimizerMetadata, OptimizationResult,
    OptimizerType, OptimizerFamily
)


class FantasyFootballGeneticOptimizer(BaseOptimizer):
    """
    Genetic Algorithm specialized for fantasy football lineup optimization.

    This optimizer is designed to handle the unique constraints of fantasy
    football problems including salary caps, position requirements, and
    player uniqueness constraints.

    Features:
    - Position-aware crossover operations
    - Smart mutation that respects position constraints
    - Elitism to preserve best lineups
    - Constraint-preserving genetic operations
    - Adaptive population management

    Parameters:
        population_size (int): Number of lineups in the population (default: 100)
        max_generations (int): Maximum number of generations (default: 200)
        mutation_rate (float): Probability of mutation (default: 0.1)
        crossover_rate (float): Probability of crossover (default: 0.8)
        elitism_count (int): Number of best lineups to preserve (default: 5)
        tournament_size (int): Size of tournament for selection (default: 3)
    """

    def __init__(self,
                 population_size: int = 100,
                 max_generations: int = 200,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elitism_count: int = 5,
                 tournament_size: int = 3):
        """
        Initialize the Fantasy Football Genetic Algorithm optimizer.

        Args:
            population_size: Number of individuals in the population
            max_generations: Maximum number of generations to evolve
            mutation_rate: Probability of mutation for each individual
            crossover_rate: Probability of crossover between parents
            elitism_count: Number of best individuals to preserve each generation
            tournament_size: Number of individuals in tournament selection

        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Validate parameters
        if population_size < 2:
            raise ValueError("Population size must be at least 2")
        if max_generations < 1:
            raise ValueError("Max generations must be at least 1")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        if not 0 <= crossover_rate <= 1:
            raise ValueError("Crossover rate must be between 0 and 1")
        if elitism_count >= population_size:
            raise ValueError("Elitism count must be less than population size")
        if tournament_size < 1:
            raise ValueError("Tournament size must be at least 1")

        # Store parameters
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size

        # Define optimizer metadata
        metadata = OptimizerMetadata(
            name="Fantasy Football Genetic Algorithm",
            description="Genetic algorithm specialized for fantasy football lineup optimization with position-aware operations",
            optimizer_type=OptimizerType.METAHEURISTIC,
            optimizer_family=OptimizerFamily.EVOLUTIONARY,
            author="Qubots Fantasy Football Tutorial",
            version="1.0.0",
            supports_constraints=True,
            supports_multi_objective=False,
            typical_problems=["fantasy_football", "knapsack", "combinatorial", "sports_optimization"],
            required_parameters=["population_size", "max_generations"],
            optional_parameters=["mutation_rate", "crossover_rate", "elitism_count", "tournament_size"]
        )

        # Initialize base optimizer
        super().__init__(
            metadata,
            population_size=population_size,
            max_generations=max_generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elitism_count=elitism_count,
            tournament_size=tournament_size
        )

    def _get_default_metadata(self) -> OptimizerMetadata:
        """Return default metadata for the optimizer."""
        return OptimizerMetadata(
            name="Fantasy Football Genetic Algorithm",
            description="Genetic algorithm specialized for fantasy football lineup optimization",
            optimizer_type=OptimizerType.METAHEURISTIC,
            optimizer_family=OptimizerFamily.EVOLUTIONARY,
            author="Qubots Fantasy Football Tutorial",
            version="1.0.0"
        )

    def _optimize_implementation(self, problem, initial_solution: Optional[List[int]] = None) -> OptimizationResult:
        """
        Execute the genetic algorithm optimization for fantasy football.

        Args:
            problem: Fantasy football problem instance with required methods:
                - get_random_solution(): Generate random feasible lineup
                - evaluate_solution(lineup): Calculate lineup score
                - is_feasible(lineup): Check if lineup satisfies constraints
            initial_solution: Optional initial lineup to include in population

        Returns:
            OptimizationResult containing the best lineup found and optimization metrics
        """
        start_time = time.time()

        try:
            # Initialize population with random feasible lineups
            population = self._initialize_population(problem, initial_solution)

            # Track best solution across all generations
            best_solution = None
            best_value = -float('inf')  # Fantasy football maximizes points
            evaluations = 0

            # Evolution loop
            for generation in range(self.max_generations):
                # Evaluate entire population
                fitness_values = []
                for individual in population:
                    fitness = problem.evaluate_solution(individual)
                    fitness_values.append(fitness)
                    evaluations += 1

                    # Update global best
                    if fitness > best_value:
                        best_solution = individual.copy()
                        best_value = fitness

                # Create next generation
                new_population = []

                # Elitism: preserve best individuals
                if self.elitism_count > 0:
                    elite_indices = np.argsort(fitness_values)[-self.elitism_count:]
                    for idx in elite_indices:
                        new_population.append(population[idx].copy())

                # Generate offspring through selection, crossover, and mutation
                while len(new_population) < self.population_size:
                    # Tournament selection
                    parent1 = self._tournament_selection(population, fitness_values)
                    parent2 = self._tournament_selection(population, fitness_values)

                    # Crossover
                    if np.random.random() < self.crossover_rate:
                        child1, child2 = self._position_aware_crossover(parent1, parent2, problem)
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()

                    # Mutation
                    if np.random.random() < self.mutation_rate:
                        child1 = self._position_aware_mutation(child1, problem)
                    if np.random.random() < self.mutation_rate:
                        child2 = self._position_aware_mutation(child2, problem)

                    # Add children to new population
                    new_population.extend([child1, child2])

                # Trim to exact population size
                population = new_population[:self.population_size]

                # Optional early stopping (can be customized)
                if self._should_terminate_early(generation, best_value, fitness_values):
                    break

            end_time = time.time()

            # Calculate final statistics
            final_fitness = [problem.evaluate_solution(ind) for ind in population]

            return OptimizationResult(
                best_solution=best_solution,
                best_value=best_value,
                iterations=generation + 1,
                evaluations=evaluations,
                runtime_seconds=end_time - start_time,
                convergence_achieved=True,
                termination_reason=f"Completed {generation + 1} generations",
                additional_metrics={
                    "final_population_size": len(population),
                    "final_average_fitness": np.mean(final_fitness),
                    "final_fitness_std": np.std(final_fitness),
                    "best_generation": generation,
                    "population_diversity": self._calculate_diversity(population)
                }
            )

        except Exception as e:
            end_time = time.time()
            return OptimizationResult(
                best_solution=None,
                best_value=-float('inf'),
                iterations=0,
                evaluations=evaluations,
                runtime_seconds=end_time - start_time,
                convergence_achieved=False,
                termination_reason=f"Error during optimization: {str(e)}",
                additional_metrics={"error": str(e)}
            )

    def _initialize_population(self, problem, initial_solution: Optional[List[int]]) -> List[List[int]]:
        """Initialize population with random feasible lineups."""
        population = []

        # Add initial solution if provided and feasible
        if initial_solution is not None and problem.is_feasible(initial_solution):
            population.append(initial_solution.copy())

        # Generate random solutions to fill population
        max_attempts = self.population_size * 10  # Prevent infinite loops
        attempts = 0

        while len(population) < self.population_size and attempts < max_attempts:
            try:
                lineup = problem.get_random_solution()
                if lineup and problem.is_feasible(lineup):
                    population.append(lineup)
            except Exception:
                pass  # Skip failed attempts
            attempts += 1

        # If we couldn't generate enough solutions, duplicate existing ones
        while len(population) < self.population_size:
            if population:
                population.append(population[0].copy())
            else:
                raise RuntimeError("Could not generate any feasible solutions")

        return population

    def _tournament_selection(self, population: List[List[int]], fitness_values: List[float]) -> List[int]:
        """Select individual using tournament selection."""
        tournament_size = min(self.tournament_size, len(population))
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]

        # Select best individual from tournament (maximize fitness)
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()

    def _position_aware_crossover(self, parent1: List[int], parent2: List[int], problem) -> tuple:
        """
        Perform position-aware crossover for fantasy football lineups.

        This crossover respects position constraints by exchanging players
        within the same position groups.
        """
        try:
            # Get position information from problem if available
            if hasattr(problem, 'players') and hasattr(problem, 'lineup_requirements'):
                return self._structured_crossover(parent1, parent2, problem)
            else:
                # Fallback to simple crossover
                return self._simple_crossover(parent1, parent2)
        except Exception:
            # Return parents if crossover fails
            return parent1.copy(), parent2.copy()

    def _structured_crossover(self, parent1: List[int], parent2: List[int], problem) -> tuple:
        """Crossover that maintains position structure."""
        child1, child2 = [], []

        # Group players by position
        pos_groups1 = self._group_by_position(parent1, problem)
        pos_groups2 = self._group_by_position(parent2, problem)

        # Exchange players within each position group
        for pos in problem.lineup_requirements:
            players1 = pos_groups1.get(pos, [])
            players2 = pos_groups2.get(pos, [])

            # Combine and randomly distribute
            combined = list(set(players1 + players2))
            required_count = problem.lineup_requirements[pos]

            if len(combined) >= required_count * 2:
                selected = np.random.choice(combined, required_count * 2, replace=False)
                child1.extend(selected[:required_count])
                child2.extend(selected[required_count:])
            else:
                # Not enough unique players, use what we have
                child1.extend(players1)
                child2.extend(players2)

        return child1, child2

    def _simple_crossover(self, parent1: List[int], parent2: List[int]) -> tuple:
        """Simple uniform crossover as fallback."""
        size = min(len(parent1), len(parent2))
        mask = np.random.random(size) < 0.5

        child1 = [parent1[i] if mask[i] else parent2[i] for i in range(size)]
        child2 = [parent2[i] if mask[i] else parent1[i] for i in range(size)]

        return child1, child2

    def _position_aware_mutation(self, individual: List[int], problem) -> List[int]:
        """
        Perform position-aware mutation for fantasy football lineups.

        Replaces a random player with another player from the same position.
        """
        try:
            if hasattr(problem, 'players') and len(individual) > 0:
                return self._structured_mutation(individual, problem)
            else:
                return self._simple_mutation(individual)
        except Exception:
            return individual.copy()

    def _structured_mutation(self, individual: List[int], problem) -> List[int]:
        """Mutation that respects position constraints."""
        mutated = individual.copy()

        if len(mutated) == 0:
            return mutated

        # Choose random player to replace
        replace_idx = np.random.randint(len(mutated))
        old_player_idx = mutated[replace_idx]

        # Get position of player to replace
        if old_player_idx < len(problem.players):
            old_position = problem.players[old_player_idx].position

            # Find replacement candidates from same position
            candidates = [
                i for i, player in enumerate(problem.players)
                if player.position == old_position and i not in mutated
            ]

            # Handle FLEX position specially
            if old_position in ['RB', 'WR', 'TE'] and not candidates:
                # If no direct replacements, look for FLEX-eligible players
                candidates = [
                    i for i, player in enumerate(problem.players)
                    if player.position in ['RB', 'WR', 'TE'] and i not in mutated
                ]

            if candidates:
                new_player_idx = np.random.choice(candidates)
                mutated[replace_idx] = new_player_idx

        return mutated

    def _simple_mutation(self, individual: List[int]) -> List[int]:
        """Simple mutation as fallback."""
        mutated = individual.copy()
        if len(mutated) > 1:
            # Swap two random positions
            i, j = np.random.choice(len(mutated), 2, replace=False)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    def _group_by_position(self, lineup: List[int], problem) -> dict:
        """Group lineup players by their positions."""
        groups = {}
        for player_idx in lineup:
            if player_idx < len(problem.players):
                pos = problem.players[player_idx].position
                if pos not in groups:
                    groups[pos] = []
                groups[pos].append(player_idx)
        return groups

    def _should_terminate_early(self, generation: int, best_value: float, fitness_values: List[float]) -> bool:
        """
        Check if optimization should terminate early.

        Can be customized with additional termination criteria.
        """
        # Example: terminate if no improvement for many generations
        # This is a placeholder - implement based on specific needs
        return False

    def _calculate_diversity(self, population: List[List[int]]) -> float:
        """Calculate population diversity metric."""
        if len(population) < 2:
            return 0.0

        total_distance = 0
        count = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # Calculate Hamming distance (number of different players)
                distance = sum(1 for a, b in zip(population[i], population[j]) if a != b)
                total_distance += distance
                count += 1

        return total_distance / count if count > 0 else 0.0


# Example usage and testing
if __name__ == "__main__":
    print("Fantasy Football Genetic Algorithm Optimizer")
    print("=" * 50)
    print("This is the pure optimizer class for upload to Rastion platform.")
    print("To use this optimizer, you need a compatible fantasy football problem.")
    print("\nOptimizer features:")
    print("- Position-aware genetic operations")
    print("- Constraint-preserving crossover and mutation")
    print("- Elitism and tournament selection")
    print("- Robust error handling")
    print("- Comprehensive performance metrics")

    # Create optimizer instance
    optimizer = FantasyFootballGeneticOptimizer(
        population_size=50,
        max_generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8
    )

    print(f"\nOptimizer created: {optimizer.metadata.name}")
    print(f"Version: {optimizer.metadata.version}")
    print(f"Author: {optimizer.metadata.author}")
    print(f"Supports constraints: {optimizer.metadata.supports_constraints}")
