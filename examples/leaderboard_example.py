#!/usr/bin/env python3
"""
Qubots Leaderboard Example

This example demonstrates how to:
1. Get standardized benchmark problems
2. Run optimization on a standardized problem
3. Submit results to the leaderboard
4. View leaderboard rankings

Prerequisites:
- Qubots framework installed
- Rastion platform access
- Valid authentication credentials
"""

import qubots
from qubots import (
    get_standardized_problems,
    submit_to_leaderboard,
    get_problem_leaderboard,
    LeaderboardClient,
    StandardizedBenchmarkRegistry
)


class SimpleGeneticTSP(qubots.BaseOptimizer):
    """Simple genetic algorithm for TSP - example implementation."""
    
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1):
        super().__init__()
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
    
    def optimize(self, problem, **kwargs):
        """Run genetic algorithm optimization."""
        import random
        import time
        
        start_time = time.time()
        
        # Initialize random population
        n_cities = problem.metadata.dimension
        population = []
        for _ in range(self.population_size):
            tour = list(range(n_cities))
            random.shuffle(tour)
            population.append(tour)
        
        best_solution = None
        best_value = float('inf')
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate population
            fitness_scores = []
            for individual in population:
                value = problem.evaluate_solution(individual)
                fitness_scores.append(value)
                
                if value < best_value:
                    best_value = value
                    best_solution = individual.copy()
            
            # Selection and reproduction (simplified)
            new_population = []
            for _ in range(self.population_size):
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                child = self._crossover(parent1, parent2)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        runtime = time.time() - start_time
        
        return qubots.OptimizationResult(
            best_solution=best_solution,
            best_value=best_value,
            is_feasible=True,
            runtime_seconds=runtime,
            iterations=self.generations,
            termination_reason="max_generations"
        )
    
    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Tournament selection."""
        import random
        
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()
    
    def _crossover(self, parent1, parent2):
        """Order crossover (OX)."""
        import random
        
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [-1] * size
        child[start:end] = parent1[start:end]
        
        pointer = end
        for city in parent2[end:] + parent2[:end]:
            if city not in child:
                child[pointer % size] = city
                pointer += 1
        
        return child
    
    def _mutate(self, individual):
        """Swap mutation."""
        import random
        
        individual = individual.copy()
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
        return individual


def main():
    """Main example function."""
    print("🏆 Qubots Leaderboard Example")
    print("=" * 50)
    
    # 1. Get available standardized problems
    print("\n1. Getting standardized problems...")
    try:
        problems = get_standardized_problems(problem_type="tsp")
        print(f"Found {len(problems)} TSP problems:")
        for problem in problems[:3]:  # Show first 3
            print(f"  - {problem.name}: {problem.description}")
    except Exception as e:
        print(f"Error getting problems: {e}")
        print("Note: This requires connection to Rastion platform")
        
        # Use local standardized problems instead
        print("\nUsing local standardized problems...")
        registry = StandardizedBenchmarkRegistry()
        specs = registry.get_benchmark_specs()
        tsp_specs = [s for s in specs if s.problem_type == "tsp"]
        
        if not tsp_specs:
            print("No TSP problems available")
            return
        
        # Create a local problem instance
        problem_spec = tsp_specs[0]  # Berlin52
        problem = registry.create_problem(problem_spec)
        print(f"Using local problem: {problem_spec.name}")
    
    # 2. Create and configure optimizer
    print("\n2. Creating genetic algorithm optimizer...")
    optimizer = SimpleGeneticTSP(
        population_size=30,
        generations=50,
        mutation_rate=0.1
    )
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Config: pop_size={optimizer.population_size}, "
          f"generations={optimizer.generations}, "
          f"mutation_rate={optimizer.mutation_rate}")
    
    # 3. Run optimization
    print("\n3. Running optimization...")
    try:
        if 'problem' not in locals():
            # Use first problem from API
            problem_spec = problems[0]
            # Note: In real implementation, you'd load the actual problem instance
            print(f"Would optimize: {problem_spec.name}")
            print("(Skipping actual optimization - requires problem loading)")
            return
        else:
            # Use local problem
            result = optimizer.optimize(problem)
            print(f"Optimization completed!")
            print(f"Best value: {result.best_value:.2f}")
            print(f"Runtime: {result.runtime_seconds:.2f} seconds")
            print(f"Iterations: {result.iterations}")
    except Exception as e:
        print(f"Error during optimization: {e}")
        return
    
    # 4. Submit to leaderboard (if connected to platform)
    print("\n4. Submitting to leaderboard...")
    try:
        submission = submit_to_leaderboard(
            result=result,
            problem_id=1,  # Assuming Berlin52 has ID 1
            solver_name="SimpleGeneticTSP",
            solver_repository="examples/simple-genetic-tsp",
            solver_config={
                "population_size": optimizer.population_size,
                "generations": optimizer.generations,
                "mutation_rate": optimizer.mutation_rate
            },
            solver_version="1.0.0"
        )
        print(f"Submission successful! ID: {submission.get('id', 'unknown')}")
    except Exception as e:
        print(f"Submission failed: {e}")
        print("Note: This requires authentication and platform connection")
    
    # 5. View leaderboard
    print("\n5. Viewing leaderboard...")
    try:
        leaderboard = get_problem_leaderboard(problem_id=1, limit=10)
        print(f"Top 10 submissions for problem:")
        print(f"{'Rank':<6} {'Solver':<20} {'Value':<12} {'Runtime':<10}")
        print("-" * 50)
        for entry in leaderboard[:10]:
            print(f"{entry['rank_overall']:<6} "
                  f"{entry['solver_name'][:19]:<20} "
                  f"{entry['best_value']:<12.2f} "
                  f"{entry['runtime_seconds']:<10.2f}")
    except Exception as e:
        print(f"Error viewing leaderboard: {e}")
        print("Note: This requires platform connection")
    
    # 6. Advanced usage example
    print("\n6. Advanced leaderboard client usage...")
    try:
        client = LeaderboardClient()
        stats = client.get_leaderboard_stats()
        print(f"Platform statistics:")
        print(f"  Total problems: {stats['total_problems']}")
        print(f"  Total submissions: {stats['total_submissions']}")
        print(f"  Total solvers: {stats['total_solvers']}")
    except Exception as e:
        print(f"Error getting stats: {e}")
        print("Note: This requires platform connection")
    
    print("\n✅ Example completed!")
    print("\nNext steps:")
    print("1. Implement your own optimization algorithm")
    print("2. Test on local standardized problems")
    print("3. Submit to the live leaderboard")
    print("4. Compare with other solvers")
    print("5. Iterate and improve your algorithm")


if __name__ == "__main__":
    main()
