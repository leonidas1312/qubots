"""
Comprehensive Fantasy Football Optimization Example using Qubots and Rastion Platform

This example demonstrates how to:
1. Load a fantasy football benchmark problem from the Rastion platform
2. Set up problem parameters and constraints
3. Configure multiple optimization solvers (OR-Tools, genetic algorithm, heuristics)
4. Execute optimization to solve the fantasy football team selection problem
5. Display and interpret results with detailed analysis
6. Handle errors and edge cases

Author: Qubots Community
Version: 1.0.0
"""

import sys
import os
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import qubots framework
try:
    import qubots
    import qubots.rastion as rastion
    from qubots import (
        AutoOptimizer,
        BenchmarkSuite,
        BenchmarkType,
        OptimizationResult
    )

    # For creating custom optimizers if needed
    from qubots import (
        PopulationBasedOptimizer,
        OptimizerMetadata,
        OptimizerType,
        OptimizerFamily
    )
    QUBOTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: qubots not available: {e}")
    print("Running in local development mode...")
    QUBOTS_AVAILABLE = False

    # Create mock classes for local development
    class OptimizationResult:
        def __init__(self, best_solution=None, best_value=float('inf'), is_feasible=False,
                     iterations=0, evaluations=0, runtime_seconds=0.0, termination_reason=""):
            self.best_solution = best_solution
            self.best_value = best_value
            self.is_feasible = is_feasible
            self.iterations = iterations
            self.evaluations = evaluations
            self.runtime_seconds = runtime_seconds
            self.termination_reason = termination_reason

    class PopulationBasedOptimizer:
        def __init__(self, **kwargs):
            self._parameters = kwargs
            self._population_size = kwargs.get('population_size', 50)

    class OptimizerMetadata:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class OptimizerType:
        METAHEURISTIC = "metaheuristic"

    class OptimizerFamily:
        EVOLUTIONARY = "evolutionary"

def setup_authentication():
    """
    Set up authentication with Rastion platform.

    Note: In a real scenario, you would use your actual Gitea token.
    For this example, we'll handle the case where authentication might not be set up.
    """
    print("🔐 Setting up Rastion Authentication")
    print("=" * 50)

    try:
        # In practice, you would get this from environment variables or secure storage
        # rastion.authenticate("your_gitea_token_here")
        print("✅ Authentication configured (using demo mode)")
        print("   In production, use: rastion.authenticate('your_gitea_token')")
        print("   Get your token from: https://hub.rastion.com")
        return True
    except Exception as e:
        print(f"⚠️  Authentication not configured: {e}")
        print("   Continuing with local examples...")
        return False

def load_fantasy_football_problem():
    """
    Load the fantasy football problem from Rastion platform.

    Returns:
        Fantasy football problem instance or None if loading fails
    """
    print("\n🏈 Loading Fantasy Football Problem")
    print("=" * 50)

    if QUBOTS_AVAILABLE:
        try:
            # Load the fantasy football problem from Rastion platform
            print("📥 Loading fantasy_football_problem from Rastion...")
            problem = rastion.load_qubots_model("fantasy_football_problem")

            print(f"✅ Successfully loaded: {problem.metadata.name}")
            print(f"   Description: {problem.metadata.description}")
            print(f"   Problem Type: {problem.metadata.problem_type}")
            print(f"   Objective: {problem.metadata.objective_type}")
            print(f"   Domain: {problem.metadata.domain}")
            print(f"   Difficulty: {problem.metadata.difficulty_level}")
            print(f"   Number of players: {problem.n_players}")
            print(f"   Salary cap: ${problem.max_salary:,}")

            return problem

        except Exception as e:
            print(f"❌ Failed to load from Rastion: {e}")
            print("🔄 Falling back to local fantasy football problem...")
    else:
        print("🔄 Loading local fantasy football problem...")

    try:
        # Load local implementation
        from fantasy_football import FantasyFootballProblem

        problem = FantasyFootballProblem()
        print(f"✅ Successfully loaded local problem: {problem.metadata.name}")
        print(f"   Number of players: {problem.n_players}")
        print(f"   Salary cap: ${problem.max_salary:,}")

        return problem

    except Exception as local_e:
        print(f"❌ Failed to load local problem: {local_e}")
        return None

def configure_optimizers() -> Dict[str, Any]:
    """
    Configure multiple optimization solvers for comparison.

    Returns:
        Dictionary of configured optimizers
    """
    print("\n⚙️  Configuring Optimization Solvers")
    print("=" * 50)

    optimizers = {}

    if QUBOTS_AVAILABLE:
        # 1. Try to load OR-Tools optimizer from Rastion
        try:
            print("📥 Loading OR-Tools optimizer from Rastion...")
            ortools_optimizer = AutoOptimizer.from_repo("Rastion/ortools_optimizer")
            optimizers["OR-Tools"] = ortools_optimizer
            print("✅ OR-Tools optimizer loaded successfully")
        except Exception as e:
            print(f"⚠️  OR-Tools optimizer not available: {e}")

        # 2. Try to load Genetic Algorithm optimizer from Rastion
        try:
            print("📥 Loading Genetic Algorithm optimizer from Rastion...")
            ga_optimizer = AutoOptimizer.from_repo("Rastion/genetic_algorithm")
            optimizers["Genetic Algorithm"] = ga_optimizer
            print("✅ Genetic Algorithm optimizer loaded successfully")
        except Exception as e:
            print(f"⚠️  Genetic Algorithm optimizer not available: {e}")
    else:
        print("🔄 Running in local mode - using built-in optimizers...")

    # 3. Create a simple random search optimizer as fallback
    try:
        print("🔧 Creating Random Search optimizer...")
        random_optimizer = SimpleRandomSearchOptimizer(n_trials=1000)
        optimizers["Random Search"] = random_optimizer
        print("✅ Random Search optimizer created successfully")
    except Exception as e:
        print(f"❌ Failed to create Random Search optimizer: {e}")

    # 4. Create a genetic algorithm optimizer if not loaded from Rastion
    if "Genetic Algorithm" not in optimizers:
        try:
            print("🔧 Creating local Genetic Algorithm optimizer...")
            ga_optimizer = FantasyFootballGeneticOptimizer(
                population_size=50,
                max_generations=100,
                mutation_rate=0.1,
                crossover_rate=0.8
            )
            optimizers["Genetic Algorithm (Local)"] = ga_optimizer
            print("✅ Local Genetic Algorithm optimizer created successfully")
        except Exception as e:
            print(f"❌ Failed to create local GA optimizer: {e}")

    print(f"\n📊 Total optimizers configured: {len(optimizers)}")
    for name in optimizers.keys():
        print(f"   • {name}")

    return optimizers

def run_optimization(problem, optimizer, optimizer_name: str) -> Optional[OptimizationResult]:
    """
    Run optimization with proper error handling and timing.

    Args:
        problem: Fantasy football problem instance
        optimizer: Optimizer instance
        optimizer_name: Name of the optimizer for display

    Returns:
        OptimizationResult or None if optimization fails
    """
    print(f"\n🚀 Running {optimizer_name} Optimization")
    print("=" * 50)

    try:
        # Start timing
        start_time = time.time()

        # Run optimization
        print(f"⏳ Optimizing with {optimizer_name}...")
        result = optimizer.optimize(problem)

        # Calculate runtime
        runtime = time.time() - start_time

        # Validate result
        if result is None:
            print(f"❌ {optimizer_name} returned no result")
            return None

        # Handle different result formats
        if hasattr(result, 'best_solution') and hasattr(result, 'best_value'):
            # Standard OptimizationResult format
            solution = result.best_solution
            value = result.best_value
            is_feasible = getattr(result, 'is_feasible', True)
            iterations = getattr(result, 'iterations', 0)
            evaluations = getattr(result, 'evaluations', 0)
        elif isinstance(result, tuple) and len(result) == 2:
            # Simple tuple format (solution, value)
            solution, value = result
            is_feasible = problem.is_feasible(solution) if solution is not None else False
            iterations = 0
            evaluations = 0

            # Create proper OptimizationResult
            result = OptimizationResult(
                best_solution=solution,
                best_value=value,
                is_feasible=is_feasible,
                iterations=iterations,
                evaluations=evaluations,
                runtime_seconds=runtime,
                termination_reason="completed"
            )
        else:
            print(f"❌ {optimizer_name} returned unexpected result format: {type(result)}")
            return None

        # Validate solution
        if solution is None:
            print(f"❌ {optimizer_name} found no valid solution")
            return result

        # Check feasibility
        if not problem.is_feasible(solution):
            print(f"⚠️  {optimizer_name} solution is not feasible!")
            result.is_feasible = False

        # Display results
        print(f"✅ {optimizer_name} completed successfully!")
        print(f"   Runtime: {runtime:.2f} seconds")
        print(f"   Best value: {value:.2f} points")
        print(f"   Feasible: {'Yes' if result.is_feasible else 'No'}")
        if iterations > 0:
            print(f"   Iterations: {iterations}")
        if evaluations > 0:
            print(f"   Evaluations: {evaluations}")

        return result

    except Exception as e:
        print(f"❌ Error running {optimizer_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def display_lineup_analysis(problem, result: OptimizationResult, optimizer_name: str):
    """
    Display detailed analysis of the optimized lineup.

    Args:
        problem: Fantasy football problem instance
        result: Optimization result
        optimizer_name: Name of the optimizer
    """
    print(f"\n📋 {optimizer_name} - Lineup Analysis")
    print("=" * 50)

    if result is None or result.best_solution is None:
        print("❌ No solution to analyze")
        return

    try:
        # Get lineup summary
        lineup_df = problem.get_lineup_summary(result.best_solution)

        if lineup_df.empty:
            print("❌ No players selected in lineup")
            return

        # Display lineup table
        print("\n🏆 Selected Players:")
        print(lineup_df.to_string(index=False))

        # Calculate and display statistics
        total_salary = lineup_df['DK.salary'].sum()
        total_points = lineup_df['DK.points'].sum()
        salary_remaining = problem.max_salary - total_salary

        print(f"\n📊 Lineup Statistics:")
        print(f"   Total Players: {len(lineup_df)}")
        print(f"   Total Salary: ${total_salary:,}")
        print(f"   Salary Cap: ${problem.max_salary:,}")
        print(f"   Salary Remaining: ${salary_remaining:,}")
        print(f"   Total Projected Points: {total_points:.2f}")
        print(f"   Points per Dollar: {total_points/total_salary:.4f}")

        # Position breakdown
        print(f"\n🏈 Position Breakdown:")
        position_counts = lineup_df['Pos'].value_counts()
        for pos, count in position_counts.items():
            pos_points = lineup_df[lineup_df['Pos'] == pos]['DK.points'].sum()
            pos_salary = lineup_df[lineup_df['Pos'] == pos]['DK.salary'].sum()
            print(f"   {pos}: {count} players, {pos_points:.1f} pts, ${pos_salary:,}")

        # Team breakdown
        print(f"\n🏟️  Team Breakdown:")
        team_counts = lineup_df['Team'].value_counts()
        for team, count in team_counts.items():
            team_points = lineup_df[lineup_df['Team'] == team]['DK.points'].sum()
            print(f"   {team}: {count} players, {team_points:.1f} pts")

    except Exception as e:
        print(f"❌ Error analyzing lineup: {e}")

def compare_optimizers(results: Dict[str, OptimizationResult]):
    """
    Compare results from different optimizers.

    Args:
        results: Dictionary mapping optimizer names to their results
    """
    print(f"\n🔍 Optimizer Comparison")
    print("=" * 50)

    if not results:
        print("❌ No results to compare")
        return

    # Create comparison table
    comparison_data = []
    for name, result in results.items():
        if result is not None:
            comparison_data.append({
                'Optimizer': name,
                'Points': f"{result.best_value:.2f}" if result.best_value != float('inf') else "N/A",
                'Feasible': "Yes" if result.is_feasible else "No",
                'Runtime (s)': f"{result.runtime_seconds:.2f}",
                'Iterations': getattr(result, 'iterations', 0),
                'Evaluations': getattr(result, 'evaluations', 0)
            })

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        # Find best result
        valid_results = [(name, result) for name, result in results.items()
                        if result is not None and result.is_feasible and result.best_value != float('inf')]

        if valid_results:
            best_name, best_result = max(valid_results, key=lambda x: x[1].best_value)
            print(f"\n🏆 Best Optimizer: {best_name}")
            print(f"   Best Points: {best_result.best_value:.2f}")
            print(f"   Runtime: {best_result.runtime_seconds:.2f} seconds")
    else:
        print("❌ No valid results to compare")

# Custom optimizer implementations for fallback
class SimpleRandomSearchOptimizer:
    """Simple random search optimizer for fantasy football."""

    def __init__(self, n_trials: int = 1000):
        self.n_trials = n_trials

    def optimize(self, problem):
        """Run random search optimization."""
        best_solution = None
        best_value = -float('inf')  # Maximization problem

        for _ in range(self.n_trials):
            solution = problem.random_solution()
            if problem.is_feasible(solution):
                value = problem.evaluate_solution(solution)
                if value > best_value:
                    best_value = value
                    best_solution = solution

        return OptimizationResult(
            best_solution=best_solution,
            best_value=best_value,
            is_feasible=best_solution is not None,
            iterations=self.n_trials,
            evaluations=self.n_trials,
            runtime_seconds=0.0,
            termination_reason="max_trials"
        )

class FantasyFootballGeneticOptimizer(PopulationBasedOptimizer):
    """Genetic algorithm optimizer specifically for fantasy football."""

    def __init__(self, population_size: int = 50, max_generations: int = 100,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):

        if QUBOTS_AVAILABLE:
            metadata = OptimizerMetadata(
                name="Fantasy Football Genetic Algorithm",
                description="Genetic algorithm optimized for fantasy football lineup selection",
                optimizer_type=OptimizerType.METAHEURISTIC,
                optimizer_family=OptimizerFamily.EVOLUTIONARY,
                supports_discrete=True,
                supports_constraints=True
            )

            super().__init__(
                metadata=metadata,
                population_size=population_size,
                max_generations=max_generations,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate
            )
        else:
            # Simple initialization for local mode
            super().__init__(
                population_size=population_size,
                max_generations=max_generations,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate
            )

    def _get_default_metadata(self):
        """Return default metadata for the optimizer."""
        return OptimizerMetadata(
            name="Fantasy Football Genetic Algorithm",
            description="Genetic algorithm optimized for fantasy football lineup selection",
            optimizer_type=OptimizerType.METAHEURISTIC,
            optimizer_family=OptimizerFamily.EVOLUTIONARY,
            supports_discrete=True,
            supports_constraints=True
        )

    def initialize_population(self, problem):
        """Initialize population with random feasible solutions."""
        population = []
        attempts = 0
        max_attempts = self._population_size * 10

        while len(population) < self._population_size and attempts < max_attempts:
            solution = problem.random_solution()
            if problem.is_feasible(solution):
                population.append(solution)
            attempts += 1

        # Fill remaining with any random solutions if needed
        while len(population) < self._population_size:
            population.append(problem.random_solution())

        return population

    def evaluate_population(self, problem, population):
        """Evaluate fitness for entire population."""
        fitness = []
        for individual in population:
            if problem.is_feasible(individual):
                fitness.append(problem.evaluate_solution(individual))
            else:
                fitness.append(-1000)  # Penalty for infeasible solutions
        return fitness

    def select_parents(self, population, fitness):
        """Tournament selection."""
        parents = []
        for _ in range(len(population)):
            # Tournament selection with size 3
            tournament_indices = np.random.choice(len(population), 3, replace=False)
            tournament_fitness = [fitness[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])
        return parents

    def reproduce(self, parents):
        """Create offspring through crossover."""
        offspring = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % len(parents)]

            if np.random.random() < self._parameters.get('crossover_rate', 0.8):
                # Single-point crossover
                crossover_point = np.random.randint(1, len(parent1))
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1[:], parent2[:]])

        return offspring[:len(parents)]

    def mutate(self, individual):
        """Mutate individual by flipping bits."""
        mutated = individual[:]
        mutation_rate = self._parameters.get('mutation_rate', 0.1)

        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit

        return mutated

    def _optimize_implementation(self, problem, initial_solution=None):
        """Main optimization loop."""
        # Initialize population
        self._population = self.initialize_population(problem)
        self._fitness_values = self.evaluate_population(problem, self._population)

        best_solution = None
        best_fitness = -float('inf')

        for generation in range(self._parameters.get('max_generations', 100)):
            # Find current best
            current_best_idx = np.argmax(self._fitness_values)
            current_best_fitness = self._fitness_values[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = self._population[current_best_idx][:]

            # Selection and reproduction
            parents = self.select_parents(self._population, self._fitness_values)
            offspring = self.reproduce(parents)

            # Mutation
            for i in range(len(offspring)):
                offspring[i] = self.mutate(offspring[i])

            # Replace population
            self._population = offspring
            self._fitness_values = self.evaluate_population(problem, self._population)
            self._generation = generation

        return OptimizationResult(
            best_solution=best_solution,
            best_value=best_fitness,
            is_feasible=best_solution is not None and problem.is_feasible(best_solution),
            iterations=self._parameters.get('max_generations', 100),
            evaluations=self._parameters.get('max_generations', 100) * self._population_size,
            runtime_seconds=0.0,
            termination_reason="max_generations"
        )

def main():
    """Main function demonstrating fantasy football optimization."""
    print("🏈 Fantasy Football Optimization with Qubots")
    print("=" * 60)
    print("This example demonstrates comprehensive fantasy football optimization")
    print("using the qubots framework and Rastion platform integration.")
    print()

    # Step 1: Setup authentication
    auth_success = setup_authentication()

    # Step 2: Load fantasy football problem
    problem = load_fantasy_football_problem()
    if problem is None:
        print("❌ Failed to load fantasy football problem. Exiting.")
        return

    # Step 3: Configure optimizers
    optimizers = configure_optimizers()
    if not optimizers:
        print("❌ No optimizers available. Exiting.")
        return

    # Step 4: Run optimizations
    results = {}
    for name, optimizer in optimizers.items():
        result = run_optimization(problem, optimizer, name)
        if result is not None:
            results[name] = result
            display_lineup_analysis(problem, result, name)

    # Step 5: Compare results
    compare_optimizers(results)

    # Step 6: Summary
    print(f"\n🎯 Optimization Summary")
    print("=" * 50)
    print(f"✅ Successfully tested {len(results)} optimizers")
    print(f"📊 Problem size: {problem.n_players} players")
    print(f"💰 Salary cap: ${problem.max_salary:,}")

    if results:
        valid_results = [r for r in results.values() if r.is_feasible and r.best_value != float('inf')]
        if valid_results:
            best_points = max(r.best_value for r in valid_results)
            print(f"🏆 Best lineup points: {best_points:.2f}")

        total_runtime = sum(r.runtime_seconds for r in results.values())
        print(f"⏱️  Total runtime: {total_runtime:.2f} seconds")

    print("\n🚀 Fantasy football optimization completed!")
    print("   Try different optimizers and parameters to improve results.")
    print("   Consider uploading your own optimizers to the Rastion platform!")

if __name__ == "__main__":
    main()
