"""
Upload script for Fantasy Football Genetic Algorithm Optimizer to Rastion.

This script demonstrates how to upload the specialized fantasy football genetic
algorithm optimizer to the Rastion platform for sharing and reuse.

Usage:
    python upload_fantasy_football_genetic_optimizer.py
"""

import qubots.rastion as rastion
from fantasy_football_genetic_optimizer import FantasyFootballGeneticOptimizer


def main():
    """Upload the Fantasy Football Genetic Optimizer to Rastion."""

    print("Fantasy Football Genetic Algorithm Optimizer - Rastion Upload")
    print("=" * 60)
    print()

    # Create the optimizer instance
    print("Creating optimizer instance...")
    optimizer = FantasyFootballGeneticOptimizer(
        population_size=100,
        generations=200,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elite_size=10,
        tournament_size=5,
        diversity_threshold=0.7,
        constraint_penalty=1000.0
    )

    print(f"✓ Created: {optimizer.metadata.name}")
    print(f"  Type: {optimizer.metadata.optimizer_type.value}")
    print(f"  Family: {optimizer.metadata.optimizer_family.value}")
    print(f"  Version: {optimizer.metadata.version}")
    print()

    # Upload to Rastion
    print("Uploading to Rastion platform...")
    try:
        # Check if authenticated
        if not rastion.is_authenticated():
            print("⚠️  Not authenticated with Rastion.")
            print("   To upload, you need to authenticate first:")
            print("   1. Get your Gitea token from https://hub.rastion.com")
            print("   2. Run: rastion.authenticate('your_token_here')")
            print("\n   For now, we'll show what would be uploaded...")
            print("   Model ready for upload: fantasy_football_genetic_optimizer")
            return

        result = rastion.upload_model(
            model=optimizer,
            name="fantasy_football_genetic_optimizer",
            description="Specialized genetic algorithm for fantasy football lineup optimization with position-aware operators and constraint handling",
            requirements=["numpy", "pandas", "qubots"],
            private=False
        )

        print("✓ Upload successful!")
        print(f"  Model name: fantasy_football_genetic_optimizer")
        print(f"  Repository: {result}")
        print()

    except Exception as e:
        print(f"✗ Upload failed: {e}")
        print("Make sure you have proper Rastion credentials configured.")
        print()
        return

    # Demonstrate loading and usage
    print("Testing load from Rastion...")
    try:
        optimizer = rastion.load_qubots_model("fantasy_football_genetic_optimizer")
        print("✓ Successfully loaded from Rastion!")
        print(f"  Loaded: {optimizer.metadata.name}")
        print()
        print("  Running optimization...")
        problem = rastion.load_qubots_model("fantasy_football_problem")
        result = optimizer.optimize(problem)
        print(f"  Best lineup points: {result.best_value:.2f}")
        print(f"  Feasible: {result.is_feasible}")
        print(f"  Generations: {result.iterations}")
        print(f"  Runtime: {result.runtime_seconds:.2f}s")
        print()

    except Exception as e:
        print(f"✗ Load test failed: {e}")
        print()

    # Usage examples
    print("Usage Examples:")
    print("-" * 40)
    print()

    print("1. Load and use with local problem:")
    print("```python")
    print("import qubots.rastion as rastion")
    print("from fantasy_football import FantasyFootballProblem")
    print()
    print("# Load optimizer from Rastion")
    print("optimizer = rastion.load_qubots_model('fantasy_football_genetic_optimizer')")
    print()
    print("# Create or load problem")
    print("problem = FantasyFootballProblem(csv_file='your_data.csv')")
    print()
    print("# Run optimization")
    print("result = optimizer.optimize(problem)")
    print("print(f'Best lineup points: {result.best_value:.2f}')")
    print("```")
    print()

    print("2. Load both problem and optimizer from Rastion:")
    print("```python")
    print("import qubots.rastion as rastion")
    print()
    print("# Load both from Rastion")
    print("problem = rastion.load_qubots_model('fantasy_football_problem')")
    print("optimizer = rastion.load_qubots_model('fantasy_football_genetic_optimizer')")
    print()
    print("# Run optimization")
    print("result = optimizer.optimize(problem)")
    print()
    print("# Display results")
    print("print(f'Best lineup points: {result.best_value:.2f}')")
    print("print(f'Feasible: {result.is_feasible}')")
    print("print(f'Generations: {result.iterations}')")
    print("print(f'Runtime: {result.runtime_seconds:.2f}s')")
    print()
    print("# Show best lineup")
    print("best_lineup = problem.get_lineup_summary(result.best_solution)")
    print("print(best_lineup)")
    print("```")
    print()

    print("3. Custom parameters:")
    print("```python")
    print("# Load with custom parameters")
    print("optimizer = rastion.load_qubots_model('fantasy_football_genetic_optimizer')")
    print("optimizer.set_parameters(")
    print("    population_size=150,")
    print("    generations=300,")
    print("    mutation_rate=0.15")
    print(")")
    print()
    print("result = optimizer.optimize(problem)")
    print("```")
    print()

    print("✓ Fantasy Football Genetic Optimizer is now available on Rastion!")
    print("  Anyone can load and use it with:")
    print("  optimizer = rastion.load_qubots_model('fantasy_football_genetic_optimizer')")
    print()
    print("📋 Manual Upload Instructions:")
    print("  If you want to upload manually, use:")
    print("  1. rastion.authenticate('your_gitea_token')")
    print("  2. rastion.upload_model(optimizer, 'fantasy_football_genetic_optimizer', 'description')")


if __name__ == "__main__":
    main()
