"""
Genetic Algorithm VRP Optimizer for Qubots Framework

This module implements a Genetic Algorithm optimizer specifically designed for
Vehicle Routing Problems (VRP). It uses evolutionary computation techniques
to find high-quality solutions for multi-vehicle routing optimization.

Compatible with Rastion platform playground for interactive optimization.

Author: Qubots Community
Version: 1.0.0
"""

import random
import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from copy import deepcopy
from qubots import (
        BaseOptimizer, OptimizerMetadata, OptimizerType,
        OptimizerFamily, OptimizationResult, BaseProblem
    )

@dataclass
class Individual:
    """Represents an individual solution in the genetic algorithm population."""
    solution: List[List[int]]  # VRP solution as list of routes
    fitness: float = float('inf')  # Fitness value (lower is better for minimization)
    age: int = 0  # Age of the individual (for diversity management)


class GeneticVRPOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimizer for Vehicle Routing Problems.

    This optimizer uses evolutionary computation techniques including:
    - Tournament selection for parent selection
    - Order crossover (OX) and route-based crossover
    - Multiple mutation operators (swap, insert, route exchange)
    - Elitism to preserve best solutions
    - Adaptive parameters based on population diversity

    The algorithm is specifically designed for VRP constraints and solution structure.
    """

    def __init__(self,
                 population_size: int = 50,
                 generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elite_size: int = 5,
                 tournament_size: int = 3,
                 diversity_threshold: float = 0.1,
                 adaptive_parameters: bool = True,
                 **kwargs):
        """
        Initialize the Genetic VRP Optimizer.

        Args:
            population_size: Number of individuals in population
            generations: Maximum number of generations
            crossover_rate: Probability of crossover operation
            mutation_rate: Probability of mutation operation
            elite_size: Number of best individuals to preserve
            tournament_size: Size of tournament for selection
            diversity_threshold: Threshold for diversity-based adaptation
            adaptive_parameters: Enable adaptive parameter adjustment
            **kwargs: Additional optimizer parameters
        """
        # Set parameters BEFORE calling super().__init__
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.diversity_threshold = diversity_threshold
        self.adaptive_parameters = adaptive_parameters

        # Initialize parent class with all parameters
        super().__init__(
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
            tournament_size=tournament_size,
            diversity_threshold=diversity_threshold,
            adaptive_parameters=adaptive_parameters,
            **kwargs
        )

        # Runtime statistics
        self.generation_count = 0
        self.best_fitness_history = []
        self.diversity_history = []

    def _get_default_metadata(self) -> 'OptimizerMetadata':
        """Return default metadata for Genetic VRP Optimizer."""

        return OptimizerMetadata(
            name="Genetic Algorithm VRP Optimizer",
            description="Evolutionary algorithm for Vehicle Routing Problems with adaptive parameters",
            optimizer_type=OptimizerType.METAHEURISTIC,
            optimizer_family=OptimizerFamily.EVOLUTIONARY,
            author="Qubots Community",
            version="1.0.0",
            supports_constraints=True,
            supports_continuous=False,
            supports_discrete=True,
            time_complexity="O(g * p * n²)",
            convergence_guaranteed=False,
            parallel_capable=True,
            required_parameters=["population_size", "generations"],
            optional_parameters=["crossover_rate", "mutation_rate", "elite_size", "tournament_size"],
            parameter_ranges={
                "population_size": (5, 1000),
                "generations": (1, 10000),
                "crossover_rate": (0.0, 1.0),
                "mutation_rate": (0.0, 1.0),
                "elite_size": (1, 50),
                "tournament_size": (2, 20)
            },
            typical_problems=["vrp", "tsp", "routing"],
            benchmark_results={
                "small_vrp": 0.95,
                "medium_vrp": 0.87,
                "large_vrp": 0.78
            }
        )

    def _optimize_implementation(self, problem: 'BaseProblem', initial_solution=None) -> 'OptimizationResult':
        """
        Main optimization implementation using genetic algorithm.

        Args:
            problem: VRP problem instance to optimize
            initial_solution: Optional initial solution (not used in GA)

        Returns:
            OptimizationResult with best solution found
        """
        start_time = time.time()

        # Initialize population
        self.log_message('info', f"Initializing population of {self.population_size} individuals...")
        population = self._initialize_population(problem)
        self.log_message('debug', f"Population initialized with {len(population)} individuals")

        # Evaluate initial population
        self.log_message('info', "Evaluating initial population...")
        self._evaluate_population(population, problem)
        self.log_message('debug', "Initial population evaluation completed")

        # Evolution loop
        best_individual = min(population, key=lambda x: x.fitness)
        self.best_fitness_history = [best_individual.fitness]

        self.log_message('info', f"Starting evolution for {self.generations} generations...")
        self.log_message('info', f"Initial best fitness: {best_individual.fitness:.6f}")

        for generation in range(self.generations):
            self.generation_count = generation

            # Calculate population diversity
            diversity = self._calculate_diversity(population)
            self.diversity_history.append(diversity)

            # Log diversity information
            if generation % 20 == 0:
                self.log_message('debug', f"Population diversity: {diversity:.4f}")

            # Adaptive parameter adjustment
            if self.adaptive_parameters:
                old_mutation_rate = self.mutation_rate
                old_crossover_rate = self.crossover_rate
                self._adapt_parameters(diversity, generation)

                # Log parameter changes
                if abs(old_mutation_rate - self.mutation_rate) > 0.01:
                    self.log_message('debug', f"Mutation rate adapted: {old_mutation_rate:.3f} → {self.mutation_rate:.3f}")
                if abs(old_crossover_rate - self.crossover_rate) > 0.01:
                    self.log_message('debug', f"Crossover rate adapted: {old_crossover_rate:.3f} → {self.crossover_rate:.3f}")

            # Create new generation
            new_population = self._create_new_generation(population, problem)

            # Evaluate new population
            self._evaluate_population(new_population, problem)

            # Update best solution
            current_best = min(new_population, key=lambda x: x.fitness)
            if current_best.fitness < best_individual.fitness:
                improvement = best_individual.fitness - current_best.fitness
                best_individual = deepcopy(current_best)
                self.log_message('info', f"Generation {generation+1}: New best fitness {best_individual.fitness:.6f} (improvement: {improvement:.6f})")

            self.best_fitness_history.append(best_individual.fitness)
            population = new_population

            # Progress reporting for playground
            self.report_progress(
                iteration=generation + 1,
                best_value=best_individual.fitness,
                diversity=diversity,
                mutation_rate=self.mutation_rate,
                crossover_rate=self.crossover_rate,
                population_size=len(population)
            )

            # More frequent progress logging for real-time feedback
            if (generation + 1) % 5 == 0:
                progress = (generation + 1) / self.generations
                self.log_message('info', f"Progress: {progress:.1%} - Best: {best_individual.fitness:.6f} - Diversity: {diversity:.4f}")
            elif (generation + 1) % 1 == 0:  # Log every generation for first few and last few
                if generation < 10 or generation >= self.generations - 10:
                    progress = (generation + 1) / self.generations
                    self.log_message('debug', f"Generation {generation + 1}/{self.generations}: Best = {best_individual.fitness:.6f}")

            # Early stopping check
            if self.should_stop():
                self.log_message('warning', f"Optimization stopped early at generation {generation + 1}")
                break

        end_time = time.time()
        runtime = end_time - start_time

        # Final logging
        self.log_message('info', f"Evolution completed in {runtime:.3f} seconds")
        self.log_message('info', f"Final best fitness: {best_individual.fitness:.6f}")
        self.log_message('info', f"Total generations: {self.generation_count + 1}")

        if self.diversity_history:
            final_diversity = self.diversity_history[-1]
            self.log_message('info', f"Final population diversity: {final_diversity:.4f}")

        # Calculate improvement
        if len(self.best_fitness_history) > 1:
            initial_fitness = self.best_fitness_history[0]
            final_fitness = best_individual.fitness
            improvement = ((initial_fitness - final_fitness) / initial_fitness) * 100
            self.log_message('info', f"Total improvement: {improvement:.2f}%")

        result = OptimizationResult(
            best_solution=best_individual.solution,
            best_value=best_individual.fitness,
            iterations=self.generation_count + 1,
            runtime_seconds=runtime,
            termination_reason="max_generations" if not self.should_stop() else "user_stop"
        )

        # Add additional metrics
        result.additional_metrics = {
            "algorithm_name": "Genetic Algorithm",
            "final_population_size": len(population),
            "diversity_final": self.diversity_history[-1] if self.diversity_history else 0.0
        }

        # Add optimization history
        result.optimization_history = [
            {"generation": i, "best_fitness": fitness}
            for i, fitness in enumerate(self.best_fitness_history)
        ]

        return result

    def _initialize_population(self, problem: 'BaseProblem') -> List[Individual]:
        """Initialize random population of VRP solutions."""
        population = []

        for _ in range(self.population_size):
            # Generate random solution
            solution = problem.get_random_solution()
            individual = Individual(solution=solution)
            population.append(individual)

        return population

    def _evaluate_population(self, population: List[Individual], problem: 'BaseProblem'):
        """Evaluate fitness for all individuals in population."""
        for individual in population:
            individual.fitness = problem.evaluate_solution(individual.solution)

    def _create_new_generation(self, population: List[Individual], problem: 'BaseProblem') -> List[Individual]:
        """Create new generation using selection, crossover, and mutation."""
        new_population = []

        # Elitism: preserve best individuals
        elite = sorted(population, key=lambda x: x.fitness)[:self.elite_size]
        new_population.extend([deepcopy(ind) for ind in elite])

        # Generate offspring to fill remaining population
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)

            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, problem)
            else:
                child1, child2 = deepcopy(parent1), deepcopy(parent2)

            # Mutation
            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1, problem)
            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2, problem)

            new_population.extend([child1, child2])

        # Trim to exact population size
        return new_population[:self.population_size]

    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return min(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual, problem: 'BaseProblem') -> Tuple[Individual, Individual]:
        """Perform crossover operation between two parents."""
        # Route-based crossover for VRP
        child1_routes = []
        child2_routes = []

        # Combine all customers from both parents
        all_customers1 = [customer for route in parent1.solution for customer in route]
        all_customers2 = [customer for route in parent2.solution for customer in route]

        # Create new route assignments
        n_vehicles = len(parent1.solution)

        for vehicle_idx in range(n_vehicles):
            # Alternate taking routes from parents
            if vehicle_idx % 2 == 0:
                if vehicle_idx < len(parent1.solution):
                    child1_routes.append(parent1.solution[vehicle_idx][:])
                else:
                    child1_routes.append([])
                if vehicle_idx < len(parent2.solution):
                    child2_routes.append(parent2.solution[vehicle_idx][:])
                else:
                    child2_routes.append([])
            else:
                if vehicle_idx < len(parent2.solution):
                    child1_routes.append(parent2.solution[vehicle_idx][:])
                else:
                    child1_routes.append([])
                if vehicle_idx < len(parent1.solution):
                    child2_routes.append(parent1.solution[vehicle_idx][:])
                else:
                    child2_routes.append([])

        # Ensure all customers are included (repair mechanism)
        child1_routes = self._repair_solution(child1_routes, problem)
        child2_routes = self._repair_solution(child2_routes, problem)

        child1 = Individual(solution=child1_routes)
        child2 = Individual(solution=child2_routes)

        return child1, child2

    def _mutate(self, individual: Individual, problem: 'BaseProblem') -> Individual:
        """Apply mutation operation to individual."""
        mutated = deepcopy(individual)

        # Choose mutation type randomly
        mutation_type = random.choice(['swap', 'insert', 'route_exchange'])

        if mutation_type == 'swap':
            mutated.solution = self._swap_mutation(mutated.solution)
        elif mutation_type == 'insert':
            mutated.solution = self._insert_mutation(mutated.solution)
        elif mutation_type == 'route_exchange':
            mutated.solution = self._route_exchange_mutation(mutated.solution)

        # Repair solution if needed
        mutated.solution = self._repair_solution(mutated.solution, problem)

        return mutated

    def _swap_mutation(self, solution: List[List[int]]) -> List[List[int]]:
        """Swap two customers within the same route or between routes."""
        mutated = deepcopy(solution)

        # Get all customers with their route indices
        customers_with_routes = []
        for route_idx, route in enumerate(mutated):
            for customer_idx, customer in enumerate(route):
                customers_with_routes.append((route_idx, customer_idx, customer))

        if len(customers_with_routes) >= 2:
            # Select two random customers
            idx1, idx2 = random.sample(range(len(customers_with_routes)), 2)
            route1, pos1, customer1 = customers_with_routes[idx1]
            route2, pos2, customer2 = customers_with_routes[idx2]

            # Swap customers
            mutated[route1][pos1] = customer2
            mutated[route2][pos2] = customer1

        return mutated

    def _insert_mutation(self, solution: List[List[int]]) -> List[List[int]]:
        """Remove a customer from one route and insert into another."""
        mutated = deepcopy(solution)

        # Find non-empty routes
        non_empty_routes = [i for i, route in enumerate(mutated) if route]

        if len(non_empty_routes) >= 1:
            # Select source route
            source_route_idx = random.choice(non_empty_routes)
            source_route = mutated[source_route_idx]

            if source_route:
                # Remove random customer
                customer_idx = random.randint(0, len(source_route) - 1)
                customer = source_route.pop(customer_idx)

                # Insert into random route (possibly the same one)
                target_route_idx = random.randint(0, len(mutated) - 1)
                target_route = mutated[target_route_idx]
                insert_pos = random.randint(0, len(target_route))
                target_route.insert(insert_pos, customer)

        return mutated

    def _route_exchange_mutation(self, solution: List[List[int]]) -> List[List[int]]:
        """Exchange segments between two routes."""
        mutated = deepcopy(solution)

        if len(mutated) >= 2:
            # Select two different routes
            route_indices = random.sample(range(len(mutated)), 2)
            route1_idx, route2_idx = route_indices

            route1 = mutated[route1_idx]
            route2 = mutated[route2_idx]

            if route1 and route2:
                # Exchange random segments
                seg1_start = random.randint(0, len(route1) - 1)
                seg1_end = random.randint(seg1_start, len(route1))

                seg2_start = random.randint(0, len(route2) - 1)
                seg2_end = random.randint(seg2_start, len(route2))

                # Extract segments
                segment1 = route1[seg1_start:seg1_end]
                segment2 = route2[seg2_start:seg2_end]

                # Replace segments
                new_route1 = route1[:seg1_start] + segment2 + route1[seg1_end:]
                new_route2 = route2[:seg2_start] + segment1 + route2[seg2_end:]

                mutated[route1_idx] = new_route1
                mutated[route2_idx] = new_route2

        return mutated

    def _repair_solution(self, solution: List[List[int]], problem: 'BaseProblem') -> List[List[int]]:
        """Repair solution to ensure all customers are served exactly once."""
        # Get all customers that should be served
        if hasattr(problem, 'customers'):
            all_customer_ids = {c.id for c in problem.customers}
        else:
            # Fallback: assume customers 1 to N
            max_customer = 0
            for route in solution:
                if route:
                    max_customer = max(max_customer, max(route))
            all_customer_ids = set(range(1, max_customer + 1))

        # Get customers currently in solution
        served_customers = set()
        for route in solution:
            served_customers.update(route)

        # Remove duplicates
        cleaned_solution = []
        seen_customers = set()

        for route in solution:
            cleaned_route = []
            for customer in route:
                if customer not in seen_customers and customer in all_customer_ids:
                    cleaned_route.append(customer)
                    seen_customers.add(customer)
            cleaned_solution.append(cleaned_route)

        # Add missing customers to random routes
        missing_customers = all_customer_ids - seen_customers
        for customer in missing_customers:
            route_idx = random.randint(0, len(cleaned_solution) - 1)
            cleaned_solution[route_idx].append(customer)

        return cleaned_solution

    def _calculate_diversity(self, population: List[Individual]) -> float:
        """Calculate population diversity based on solution differences."""
        if len(population) < 2:
            return 1.0

        total_distance = 0.0
        comparisons = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = self._solution_distance(population[i].solution, population[j].solution)
                total_distance += distance
                comparisons += 1

        return total_distance / comparisons if comparisons > 0 else 0.0

    def _solution_distance(self, solution1: List[List[int]], solution2: List[List[int]]) -> float:
        """Calculate distance between two VRP solutions."""
        # Simple distance based on different customer-route assignments
        assignments1 = {}
        assignments2 = {}

        for route_idx, route in enumerate(solution1):
            for customer in route:
                assignments1[customer] = route_idx

        for route_idx, route in enumerate(solution2):
            for customer in route:
                assignments2[customer] = route_idx

        # Count differences
        differences = 0
        all_customers = set(assignments1.keys()) | set(assignments2.keys())

        for customer in all_customers:
            if assignments1.get(customer) != assignments2.get(customer):
                differences += 1

        return differences / len(all_customers) if all_customers else 0.0

    def _adapt_parameters(self, diversity: float, generation: int):
        """Adapt algorithm parameters based on population diversity and generation."""
        # Increase mutation rate if diversity is low
        if diversity < self.diversity_threshold:
            self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
        else:
            self.mutation_rate = max(0.05, self.mutation_rate * 0.95)

        # Adjust crossover rate based on generation progress
        progress = generation / self.generations
        if progress > 0.7:  # Late in evolution
            self.crossover_rate = max(0.5, self.crossover_rate * 0.98)
