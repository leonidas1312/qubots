"""
Fantasy Football Genetic Algorithm Optimizer for Qubots Framework.

This module implements a specialized genetic algorithm optimizer designed specifically
for fantasy football lineup optimization problems. It includes domain-specific
operators, constraint handling, and performance optimizations.

The optimizer can be uploaded to and loaded from the Rastion platform for
seamless integration with the qubots ecosystem.

Author: Qubots Community
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

from qubots import (
    PopulationBasedOptimizer,
    OptimizerMetadata,
    OptimizerType,
    OptimizerFamily,
    OptimizationResult,
    BaseProblem
)


class FantasyFootballGeneticOptimizer(PopulationBasedOptimizer):
    """
    Genetic Algorithm Optimizer specialized for Fantasy Football Lineup Optimization.

    This optimizer implements a genetic algorithm with domain-specific operators
    designed for fantasy football problems. It includes:

    - Position-aware crossover that respects position constraints
    - Smart mutation that considers player values and positions
    - Constraint repair mechanisms for salary cap and position requirements
    - Diversity maintenance to explore different lineup strategies
    - Performance tracking and optimization history

    The optimizer is designed to work with FantasyFootballProblem instances
    and can be easily uploaded to and loaded from the Rastion platform.

    Example Usage:
        ```python
        import qubots.rastion as rastion
        from fantasy_football import FantasyFootballProblem

        # Load problem and optimizer from Rastion
        problem = rastion.load_qubots_model('fantasy_football_problem')
        optimizer = rastion.load_qubots_model('fantasy_football_genetic_optimizer')

        # Run optimization
        result = optimizer.optimize(problem)
        print(f'Best lineup points: {result.best_value:.2f}')

        # Show best lineup
        best_lineup = problem.get_lineup_summary(result.best_solution)
        print(best_lineup)
        ```
    """

    def __init__(self,
                 population_size: int = 100,
                 generations: int = 200,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elite_size: int = 10,
                 tournament_size: int = 5,
                 diversity_threshold: float = 0.7,
                 constraint_penalty: float = 1000.0,
                 **kwargs):
        """
        Initialize the Fantasy Football Genetic Algorithm Optimizer.

        Args:
            population_size: Number of individuals in the population (default: 100)
            generations: Maximum number of generations to evolve (default: 200)
            crossover_rate: Probability of crossover between parents (default: 0.8)
            mutation_rate: Probability of mutation for each individual (default: 0.1)
            elite_size: Number of best individuals to preserve each generation (default: 10)
            tournament_size: Size of tournament for parent selection (default: 5)
            diversity_threshold: Minimum diversity to maintain in population (default: 0.7)
            constraint_penalty: Penalty value for constraint violations (default: 1000.0)
            **kwargs: Additional parameters passed to parent class
        """
        super().__init__(
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
            tournament_size=tournament_size,
            diversity_threshold=diversity_threshold,
            constraint_penalty=constraint_penalty,
            **kwargs
        )

        # Fantasy football specific parameters
        self._generations = generations
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate
        self._elite_size = elite_size
        self._tournament_size = tournament_size
        self._diversity_threshold = diversity_threshold
        self._constraint_penalty = constraint_penalty

        # Internal state
        self._position_groups = {}
        self._player_positions = {}
        self._salary_data = {}
        self._points_data = {}

    def _get_default_metadata(self) -> OptimizerMetadata:
        """Return default metadata for the Fantasy Football Genetic Optimizer."""
        return OptimizerMetadata(
            name="Fantasy Football Genetic Algorithm",
            description="Specialized genetic algorithm for fantasy football lineup optimization with position-aware operators and constraint handling",
            optimizer_type=OptimizerType.METAHEURISTIC,
            optimizer_family=OptimizerFamily.EVOLUTIONARY,
            author="Qubots Community",
            version="1.0.0",
            license="MIT",

            # Algorithm characteristics
            is_deterministic=False,
            supports_constraints=True,
            supports_multi_objective=False,
            supports_continuous=False,
            supports_discrete=True,
            supports_mixed_integer=False,

            # Performance characteristics
            time_complexity="O(g * p * n)",  # generations * population * players
            space_complexity="O(p * n)",     # population * players
            convergence_guaranteed=False,
            parallel_capable=True,

            # Parameters
            required_parameters=[],
            optional_parameters=[
                "population_size", "generations", "crossover_rate", "mutation_rate",
                "elite_size", "tournament_size", "diversity_threshold", "constraint_penalty"
            ],
            parameter_ranges={
                "population_size": (10, 1000),
                "generations": (10, 1000),
                "crossover_rate": (0.0, 1.0),
                "mutation_rate": (0.0, 1.0),
                "elite_size": (1, 50),
                "tournament_size": (2, 20),
                "diversity_threshold": (0.0, 1.0),
                "constraint_penalty": (0.0, 10000.0)
            },

            # Domain information
            typical_problems=["fantasy_football", "lineup_optimization", "sports_analytics"],
            reference_papers=[
                "Genetic Algorithms for Fantasy Sports Lineup Optimization",
                "Constraint Handling in Evolutionary Algorithms for Sports Analytics"
            ]
        )

    def _prepare_problem_data(self, problem: BaseProblem):
        """Extract and organize fantasy football problem data for optimization."""
        # Get player data from the problem
        if hasattr(problem, 'df'):
            df = problem.df
        else:
            raise ValueError("Problem must have a 'df' attribute with player data")

        # Organize players by position
        self._position_groups = {}
        self._player_positions = {}
        self._salary_data = {}
        self._points_data = {}

        for idx, row in df.iterrows():
            position = row['Pos']

            if position not in self._position_groups:
                self._position_groups[position] = []

            self._position_groups[position].append(idx)
            self._player_positions[idx] = position
            self._salary_data[idx] = row['DK.salary']
            self._points_data[idx] = row['DK.points']

    def initialize_population(self, problem: BaseProblem) -> List[Any]:
        """Initialize population with valid fantasy football lineups."""
        self._prepare_problem_data(problem)
        population = []

        for _ in range(self._population_size):
            # Generate a random valid lineup
            individual = self._generate_random_lineup(problem)
            population.append(individual)

        return population

    def _generate_random_lineup(self, problem: BaseProblem) -> List[int]:
        """Generate a random valid fantasy football lineup."""
        lineup = [0] * len(problem.df)

        # DraftKings position requirements: 1 QB, 2-3 RB, 3-4 WR, 1-2 TE, 1 DEF
        position_requirements = {
            'QB': 1,
            'RB': random.choice([2, 3]),
            'WR': random.choice([3, 4]),
            'TE': random.choice([1, 2]),
            'DEF': 1
        }

        # Ensure we have exactly 9 players
        total_selected = sum(position_requirements.values())
        if total_selected != 9:
            # Adjust to ensure exactly 9 players
            if total_selected < 9:
                # Add more RB or WR
                if position_requirements['RB'] == 2:
                    position_requirements['RB'] = 3
                elif position_requirements['WR'] == 3:
                    position_requirements['WR'] = 4
            elif total_selected > 9:
                # Remove from RB or WR
                if position_requirements['RB'] == 3:
                    position_requirements['RB'] = 2
                elif position_requirements['WR'] == 4:
                    position_requirements['WR'] = 3

        # Select players for each position
        total_salary = 0
        max_salary = getattr(problem, 'max_salary', 50000)

        for position, count in position_requirements.items():
            if position in self._position_groups:
                available_players = self._position_groups[position].copy()

                for _ in range(count):
                    if not available_players:
                        break

                    # Try to select a player that fits within salary constraints
                    attempts = 0
                    while attempts < 10 and available_players:
                        player_idx = random.choice(available_players)
                        player_salary = self._salary_data[player_idx]

                        if total_salary + player_salary <= max_salary:
                            lineup[player_idx] = 1
                            total_salary += player_salary
                            available_players.remove(player_idx)
                            break
                        else:
                            available_players.remove(player_idx)
                        attempts += 1

        return lineup

    def evaluate_population(self, problem: BaseProblem, population: List[Any]) -> List[float]:
        """Evaluate fitness for entire population with constraint handling."""
        fitness_values = []

        for individual in population:
            # Get base fitness (projected points)
            base_fitness = problem.evaluate_solution(individual)
            if hasattr(base_fitness, 'objective_value'):
                base_fitness = base_fitness.objective_value

            # Apply constraint penalties
            penalty = 0.0

            # Check if lineup is feasible
            if not problem.is_feasible(individual):
                penalty += self._constraint_penalty

            # Additional penalty for diversity (encourage different lineups)
            diversity_bonus = self._calculate_diversity_bonus(individual, population)

            # Final fitness (we want to maximize points, so negate for minimization)
            fitness = -base_fitness + penalty + diversity_bonus
            fitness_values.append(fitness)

        return fitness_values

    def _calculate_diversity_bonus(self, individual: List[int], population: List[Any]) -> float:
        """Calculate diversity bonus to maintain population diversity."""
        if len(population) <= 1:
            return 0.0

        # Calculate similarity with other individuals
        similarities = []
        for other in population:
            if other is not individual:
                similarity = sum(1 for i, j in zip(individual, other) if i == j and i == 1)
                similarities.append(similarity / 9.0)  # Normalize by lineup size

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # Return bonus for diverse individuals
        return max(0.0, (avg_similarity - self._diversity_threshold) * 100.0)

    def select_parents(self, population: List[Any], fitness: List[float]) -> List[Any]:
        """Select parents using tournament selection."""
        parents = []

        for _ in range(len(population)):
            # Tournament selection
            tournament_indices = random.sample(range(len(population)),
                                             min(self._tournament_size, len(population)))
            tournament_fitness = [fitness[i] for i in tournament_indices]

            # Select best individual from tournament (lowest fitness for minimization)
            best_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
            parents.append(population[best_idx].copy())

        return parents

    def reproduce(self, parents: List[Any]) -> List[Any]:
        """Create offspring through position-aware crossover."""
        offspring = []

        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]

            if random.random() < self._crossover_rate:
                child1, child2 = self._position_aware_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            offspring.extend([child1, child2])

        return offspring[:len(parents)]  # Ensure same population size

    def _position_aware_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Perform position-aware crossover that respects fantasy football constraints."""
        child1 = [0] * len(parent1)
        child2 = [0] * len(parent2)

        # Get selected players from each parent
        parent1_players = [i for i, selected in enumerate(parent1) if selected == 1]
        parent2_players = [i for i, selected in enumerate(parent2) if selected == 1]

        # Group players by position
        parent1_by_pos = {}
        parent2_by_pos = {}

        for player_idx in parent1_players:
            pos = self._player_positions[player_idx]
            if pos not in parent1_by_pos:
                parent1_by_pos[pos] = []
            parent1_by_pos[pos].append(player_idx)

        for player_idx in parent2_players:
            pos = self._player_positions[player_idx]
            if pos not in parent2_by_pos:
                parent2_by_pos[pos] = []
            parent2_by_pos[pos].append(player_idx)

        # For each position, randomly choose players from either parent
        for position in ['QB', 'RB', 'WR', 'TE', 'DEF']:
            p1_players = parent1_by_pos.get(position, [])
            p2_players = parent2_by_pos.get(position, [])

            # Determine how many players to take from each parent
            total_needed = len(p1_players)  # Assume parent1 structure is valid

            if total_needed > 0:
                # Randomly split players between children
                for i in range(total_needed):
                    if i < len(p1_players) and i < len(p2_players):
                        if random.random() < 0.5:
                            child1[p1_players[i]] = 1
                            child2[p2_players[i]] = 1
                        else:
                            child1[p2_players[i]] = 1
                            child2[p1_players[i]] = 1
                    elif i < len(p1_players):
                        child1[p1_players[i]] = 1
                        if p2_players:
                            child2[p2_players[i % len(p2_players)]] = 1
                    elif i < len(p2_players):
                        child2[p2_players[i]] = 1
                        if p1_players:
                            child1[p1_players[i % len(p1_players)]] = 1

        return child1, child2

    def mutate(self, individual: Any) -> Any:
        """Mutate individual using position-aware mutation."""
        if random.random() > self._mutation_rate:
            return individual

        mutated = individual.copy()
        selected_players = [i for i, selected in enumerate(individual) if selected == 1]

        if not selected_players:
            return mutated

        # Choose a random player to replace
        player_to_replace = random.choice(selected_players)
        position = self._player_positions[player_to_replace]

        # Find alternative players in the same position
        available_alternatives = [
            p for p in self._position_groups.get(position, [])
            if p != player_to_replace and mutated[p] == 0
        ]

        if available_alternatives:
            # Choose replacement based on value (prefer similar or better players)
            current_value = self._points_data[player_to_replace] / max(self._salary_data[player_to_replace], 1)

            # Calculate value ratios for alternatives
            alternative_values = []
            for alt in available_alternatives:
                alt_value = self._points_data[alt] / max(self._salary_data[alt], 1)
                alternative_values.append((alt, abs(alt_value - current_value)))

            # Sort by similarity to current player value
            alternative_values.sort(key=lambda x: x[1])

            # Choose from top alternatives with some randomness
            top_alternatives = alternative_values[:min(5, len(alternative_values))]
            replacement = random.choice(top_alternatives)[0]

            # Make the swap
            mutated[player_to_replace] = 0
            mutated[replacement] = 1

        return mutated

    def _repair_constraints(self, individual: List[int], problem: BaseProblem) -> List[int]:
        """Repair constraint violations in an individual."""
        repaired = individual.copy()

        # Check and fix position requirements
        selected_players = [i for i, selected in enumerate(repaired) if selected == 1]

        # Count players by position
        position_counts = {}
        total_salary = 0

        for player_idx in selected_players:
            pos = self._player_positions[player_idx]
            position_counts[pos] = position_counts.get(pos, 0) + 1
            total_salary += self._salary_data[player_idx]

        # Define valid position requirements (DraftKings format - exactly 9 players)
        valid_configs = [
            {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 2, 'DEF': 1},  # 9 players
            {'QB': 1, 'RB': 2, 'WR': 4, 'TE': 1, 'DEF': 1},  # 9 players
            {'QB': 1, 'RB': 3, 'WR': 3, 'TE': 1, 'DEF': 1},  # 9 players
        ]

        # Find the closest valid configuration
        best_config = valid_configs[0]
        min_changes = float('inf')

        for config in valid_configs:
            changes = 0
            for pos, required in config.items():
                current = position_counts.get(pos, 0)
                changes += abs(current - required)

            if changes < min_changes:
                min_changes = changes
                best_config = config

        # Adjust lineup to match best configuration
        for position, required_count in best_config.items():
            current_count = position_counts.get(position, 0)

            if current_count > required_count:
                # Remove excess players
                pos_players = [p for p in selected_players if self._player_positions[p] == position]
                # Remove lowest value players first
                pos_players.sort(key=lambda p: self._points_data[p])
                for i in range(current_count - required_count):
                    if i < len(pos_players):
                        repaired[pos_players[i]] = 0

            elif current_count < required_count:
                # Add more players
                available = [p for p in self._position_groups.get(position, [])
                           if repaired[p] == 0]
                # Add highest value players first
                available.sort(key=lambda p: self._points_data[p], reverse=True)
                for i in range(required_count - current_count):
                    if i < len(available):
                        repaired[available[i]] = 1

        # Check salary constraint
        max_salary = getattr(problem, 'max_salary', 50000)
        current_salary = sum(self._salary_data[i] for i, selected in enumerate(repaired) if selected == 1)

        if current_salary > max_salary:
            # Remove most expensive players until under budget
            selected = [(i, self._salary_data[i]) for i, sel in enumerate(repaired) if sel == 1]
            selected.sort(key=lambda x: x[1], reverse=True)  # Most expensive first

            for player_idx, salary in selected:
                if current_salary <= max_salary:
                    break
                repaired[player_idx] = 0
                current_salary -= salary

        # Final check: ensure exactly 9 players
        total_players = sum(repaired)
        if total_players < 9:
            # Add more players if needed
            available_players = [i for i, sel in enumerate(repaired) if sel == 0]
            available_players.sort(key=lambda p: self._points_data[p], reverse=True)

            for i in range(9 - total_players):
                if i < len(available_players):
                    player_idx = available_players[i]
                    if current_salary + self._salary_data[player_idx] <= max_salary:
                        repaired[player_idx] = 1
                        current_salary += self._salary_data[player_idx]

        elif total_players > 9:
            # Remove excess players
            selected_players = [(i, self._points_data[i]) for i, sel in enumerate(repaired) if sel == 1]
            selected_players.sort(key=lambda x: x[1])  # Remove lowest scoring first

            for i in range(total_players - 9):
                if i < len(selected_players):
                    player_idx = selected_players[i][0]
                    repaired[player_idx] = 0

        return repaired

    def _optimize_implementation(self, problem: BaseProblem, initial_solution: Optional[Any] = None) -> OptimizationResult:
        """Main genetic algorithm optimization loop."""
        # Note: initial_solution is not used in genetic algorithms as we generate random population
        start_time = time.time()

        # Initialize population
        population = self.initialize_population(problem)

        # Track optimization history
        history = []
        best_solution = None
        best_fitness = float('inf')
        evaluations = 0

        for generation in range(self._generations):
            # Evaluate population
            fitness_values = self.evaluate_population(problem, population)
            evaluations += len(population)

            # Track best solution
            gen_best_idx = fitness_values.index(min(fitness_values))
            gen_best_fitness = fitness_values[gen_best_idx]

            if gen_best_fitness < best_fitness:
                best_fitness = gen_best_fitness
                best_solution = population[gen_best_idx].copy()

            # Calculate statistics
            avg_fitness = sum(fitness_values) / len(fitness_values)

            # Store generation data
            generation_data = {
                'generation': generation,
                'best_fitness': -best_fitness,  # Convert back to points (maximization)
                'average_fitness': -avg_fitness,
                'worst_fitness': -max(fitness_values),
                'evaluations': evaluations,
                'diversity': self._calculate_population_diversity(population)
            }
            history.append(generation_data)

            # Report progress
            if self._progress_callback:
                self.report_progress(generation, -best_fitness, **generation_data)

            # Check for early stopping
            if self.should_stop():
                break

            # Selection
            parents = self.select_parents(population, fitness_values)

            # Reproduction (crossover)
            offspring = self.reproduce(parents)

            # Mutation
            for i in range(len(offspring)):
                offspring[i] = self.mutate(offspring[i])

            # Constraint repair
            for i in range(len(offspring)):
                offspring[i] = self._repair_constraints(offspring[i], problem)

            # Elitism: keep best individuals
            elite_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])[:self._elite_size]
            elite = [population[i].copy() for i in elite_indices]

            # Replace worst offspring with elite
            offspring_fitness = self.evaluate_population(problem, offspring)
            worst_indices = sorted(range(len(offspring_fitness)), key=lambda i: offspring_fitness[i], reverse=True)[:self._elite_size]

            for i, elite_individual in enumerate(elite):
                if i < len(worst_indices):
                    offspring[worst_indices[i]] = elite_individual

            population = offspring

        # Final constraint repair and evaluation
        best_solution = self._repair_constraints(best_solution, problem)
        final_fitness = problem.evaluate_solution(best_solution)
        if hasattr(final_fitness, 'objective_value'):
            final_fitness = final_fitness.objective_value

        # Create optimization result
        result = OptimizationResult(
            best_solution=best_solution,
            best_value=final_fitness,
            is_feasible=problem.is_feasible(best_solution),
            iterations=generation + 1,
            evaluations=evaluations,
            runtime_seconds=time.time() - start_time,
            convergence_achieved=False,  # GA doesn't guarantee convergence
            termination_reason="max_generations" if generation >= self._generations - 1 else "user_stop",
            optimization_history=history,
            additional_metrics={
                'final_diversity': self._calculate_population_diversity(population),
                'elite_size': self._elite_size,
                'population_size': self._population_size
            }
        )

        return result

    def _calculate_population_diversity(self, population: List[Any]) -> float:
        """Calculate diversity metric for the population."""
        if len(population) <= 1:
            return 0.0

        total_similarity = 0.0
        comparisons = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                # Calculate Jaccard similarity (intersection over union)
                ind1_players = set(k for k, selected in enumerate(population[i]) if selected == 1)
                ind2_players = set(k for k, selected in enumerate(population[j]) if selected == 1)

                if len(ind1_players) == 0 and len(ind2_players) == 0:
                    similarity = 1.0
                else:
                    intersection = len(ind1_players.intersection(ind2_players))
                    union = len(ind1_players.union(ind2_players))
                    similarity = intersection / union if union > 0 else 0.0

                total_similarity += similarity
                comparisons += 1

        avg_similarity = total_similarity / comparisons if comparisons > 0 else 0.0
        return 1.0 - avg_similarity  # Return diversity (1 - similarity)


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the Fantasy Football Genetic Optimizer.

    This example demonstrates how to:
    1. Create the optimizer with custom parameters
    2. Load a fantasy football problem
    3. Run optimization
    4. Display results
    """

    print("Fantasy Football Genetic Algorithm Optimizer")
    print("=" * 50)

    # Create optimizer with custom parameters
    optimizer = FantasyFootballGeneticOptimizer(
        population_size=50,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.15,
        elite_size=5,
        tournament_size=3
    )

    print(f"Optimizer: {optimizer.metadata.name}")
    print(f"Type: {optimizer.metadata.optimizer_type.value}")
    print(f"Family: {optimizer.metadata.optimizer_family.value}")
    print(f"Parameters: {optimizer.parameters}")
    print()

    # Note: In real usage, you would load the problem like this:
    # from fantasy_football import FantasyFootballProblem
    # problem = FantasyFootballProblem()
    #
    # Or from Rastion:
    # import qubots.rastion as rastion
    # problem = rastion.load_qubots_model('fantasy_football_problem')

    print("To use this optimizer:")
    print()
    print("1. With local problem:")
    print("   from fantasy_football import FantasyFootballProblem")
    print("   problem = FantasyFootballProblem()")
    print("   result = optimizer.optimize(problem)")
    print()
    print("2. With Rastion (after uploading):")
    print("   import qubots.rastion as rastion")
    print("   problem = rastion.load_qubots_model('fantasy_football_problem')")
    print("   optimizer = rastion.load_qubots_model('fantasy_football_genetic_optimizer')")
    print("   result = optimizer.optimize(problem)")
    print()
    print("3. Display results:")
    print("   print(f'Best lineup points: {result.best_value:.2f}')")
    print("   print(f'Feasible: {result.is_feasible}')")
    print("   print(f'Generations: {result.iterations}')")
    print("   print(f'Runtime: {result.runtime_seconds:.2f}s')")
    print()
    print("   # Show best lineup")
    print("   best_lineup = problem.get_lineup_summary(result.best_solution)")
    print("   print(best_lineup)")
    print()
    print("Ready for upload to Rastion platform!")
    print("Use: qubots.rastion.upload_qubots_model(optimizer, 'fantasy_football_genetic_optimizer')")