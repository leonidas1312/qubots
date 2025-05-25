"""
Load Fantasy Football Problem and OR-Tools Optimizer from Rastion Platform.

This script demonstrates how to load a fantasy football problem and the OR-Tools
optimizer from the Rastion platform, then solve the optimization problem.
It showcases the seamless integration between local development and the
Rastion ecosystem.

Usage:
    python load_and_solve_from_rastion.py

Requirements:
    - qubots framework with Rastion integration
    - OR-Tools (pip install ortools)
    - Access to Rastion platform
    - Uploaded fantasy football problem and optimizer models

Author: Qubots Community
Version: 1.0.0
"""

import sys
import time
import pandas as pd
from typing import Optional, Dict, Any

# Import qubots and Rastion components
try:
    import qubots.rastion as rastion
    from qubots import OptimizationResult
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure qubots is installed with Rastion support")
    sys.exit(1)

# Check for OR-Tools
try:
    import ortools
    ORTOOLS_AVAILABLE = True
except ImportError:
    print("OR-Tools not available. Please install with: pip install ortools")
    ORTOOLS_AVAILABLE = False


def load_models_from_rastion():
    """Load fantasy football problem and optimizer from Rastion."""
    print("Loading Models from Rastion Platform")
    print("-" * 40)
    
    try:
        # Load fantasy football problem
        print("Loading fantasy football problem...")
        problem = rastion.load_qubots_model('fantasy_football_problem')
        print(f"✓ Problem loaded: {problem.metadata.name}")
        print(f"  Type: {problem.metadata.problem_type}")
        print(f"  Domain: {problem.metadata.domain}")
        print(f"  Version: {problem.metadata.version}")
        
        # Load OR-Tools optimizer
        print("\nLoading OR-Tools optimizer...")
        optimizer = rastion.load_qubots_model('fantasy_football_ortools_optimizer')
        print(f"✓ Optimizer loaded: {optimizer.metadata.name}")
        print(f"  Type: {optimizer.metadata.optimizer_type}")
        print(f"  Family: {optimizer.metadata.optimizer_family}")
        print(f"  Version: {optimizer.metadata.version}")
        
        return problem, optimizer
        
    except Exception as e:
        print(f"✗ Error loading models from Rastion: {e}")
        print("\nPossible solutions:")
        print("1. Make sure the models are uploaded to Rastion")
        print("2. Check your Rastion authentication")
        print("3. Verify model names are correct")
        print("4. Upload models using upload scripts first")
        return None, None


def solve_with_rastion_models(problem, optimizer, config: Optional[Dict[str, Any]] = None):
    """Solve the fantasy football problem using Rastion models."""
    print(f"\nSolving Fantasy Football Problem")
    print("-" * 40)
    
    if problem is None or optimizer is None:
        print("Cannot solve: models not loaded properly")
        return None
    
    try:
        # Display problem information
        print(f"Problem: {problem.metadata.name}")
        if hasattr(problem, 'n_players'):
            print(f"Number of players: {problem.n_players}")
        if hasattr(problem, 'max_salary'):
            print(f"Salary cap: ${problem.max_salary:,}")
        
        # Configure optimizer if config provided
        if config and hasattr(optimizer, 'update_config'):
            print(f"\nApplying custom configuration...")
            optimizer.update_config(**config)
            for key, value in config.items():
                print(f"  {key}: {value}")
        
        # Generate initial solution hint
        print(f"\nGenerating initial solution hint...")
        initial_solution = problem.random_solution()
        initial_points = problem.evaluate_solution(initial_solution)
        print(f"Initial solution points: {initial_points:.2f}")
        print(f"Initial solution feasible: {problem.is_feasible(initial_solution)}")
        
        # Run optimization
        print(f"\nRunning optimization...")
        start_time = time.time()
        result = optimizer.optimize(problem, initial_solution=initial_solution)
        end_time = time.time()
        
        # Display results
        print(f"\nOptimization Results:")
        print(f"{'='*50}")
        print(f"Best value: {result.best_value:.2f} points")
        print(f"Feasible: {result.is_feasible}")
        print(f"Optimal: {result.convergence_achieved}")
        print(f"Runtime: {result.runtime_seconds:.2f} seconds")
        print(f"Iterations: {result.iterations}")
        print(f"Termination: {result.termination_reason}")
        
        # Improvement over initial solution
        if result.is_feasible and initial_solution:
            improvement = result.best_value - initial_points
            improvement_pct = (improvement / initial_points) * 100 if initial_points > 0 else 0
            print(f"Improvement: +{improvement:.2f} points ({improvement_pct:.1f}%)")
        
        # Display lineup if feasible
        if result.is_feasible and result.best_solution:
            print(f"\nOptimal Lineup:")
            print(f"{'='*50}")
            if hasattr(problem, 'get_lineup_summary'):
                lineup = problem.get_lineup_summary(result.best_solution)
                print(lineup)
            else:
                # Fallback display
                selected_indices = [i for i, x in enumerate(result.best_solution) if x == 1]
                print(f"Selected player indices: {selected_indices}")
        
        # Display additional solver information
        if hasattr(result, 'additional_info') and result.additional_info:
            print(f"\nSolver Statistics:")
            print(f"{'='*50}")
            for key, value in result.additional_info.items():
                print(f"{key}: {value}")
        
        # Get optimizer-specific statistics if available
        if hasattr(optimizer, 'get_solver_statistics'):
            stats = optimizer.get_solver_statistics()
            print(f"\nDetailed Solver Statistics:")
            print(f"{'='*50}")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key}: {value}")
        
        return result
        
    except Exception as e:
        print(f"✗ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return None


def demonstrate_different_configurations(problem, optimizer):
    """Demonstrate solving with different optimizer configurations."""
    print(f"\n\nTesting Different Configurations")
    print("=" * 60)
    
    configurations = [
        {
            "name": "Quick Solve",
            "config": {"time_limit_seconds": 5, "num_search_workers": 1},
            "description": "Fast solution for quick results"
        },
        {
            "name": "Balanced Solve", 
            "config": {"time_limit_seconds": 15, "num_search_workers": 2},
            "description": "Good balance of speed and quality"
        },
        {
            "name": "Thorough Solve",
            "config": {"time_limit_seconds": 30, "num_search_workers": 4},
            "description": "Maximum quality optimization"
        }
    ]
    
    results = []
    
    for test_config in configurations:
        print(f"\n{test_config['name']}: {test_config['description']}")
        print("-" * 50)
        
        result = solve_with_rastion_models(problem, optimizer, test_config["config"])
        
        if result:
            results.append({
                "name": test_config["name"],
                "best_value": result.best_value,
                "runtime": result.runtime_seconds,
                "optimal": result.convergence_achieved,
                "feasible": result.is_feasible
            })
    
    # Summary comparison
    if results:
        print(f"\n\nConfiguration Comparison Summary")
        print("=" * 60)
        print(f"{'Configuration':<15} {'Best Value':<12} {'Runtime':<10} {'Optimal':<8} {'Feasible'}")
        print("-" * 60)
        for r in results:
            print(f"{r['name']:<15} {r['best_value']:<12.2f} {r['runtime']:<10.2f} {r['optimal']:<8} {r['feasible']}")
    
    return results


def main():
    """Main function to demonstrate Rastion integration."""
    print("Fantasy Football Optimization with Rastion Platform")
    print("=" * 60)
    
    if not ORTOOLS_AVAILABLE:
        print("ERROR: OR-Tools not available. Please install with: pip install ortools")
        return
    
    # Load models from Rastion
    problem, optimizer = load_models_from_rastion()
    
    if problem and optimizer:
        # Solve with default configuration
        result = solve_with_rastion_models(problem, optimizer)
        
        if result and result.is_feasible:
            # Demonstrate different configurations
            demonstrate_different_configurations(problem, optimizer)
            
            print(f"\n\nSuccess! Fantasy football optimization completed using Rastion models.")
            print("This demonstrates the power of the qubots ecosystem:")
            print("- Seamless model loading from Rastion platform")
            print("- High-performance OR-Tools optimization")
            print("- Comprehensive result analysis")
            print("- Easy configuration and experimentation")
        else:
            print(f"\nOptimization failed or no feasible solution found.")
    else:
        print(f"\nFailed to load models from Rastion.")
        print("Make sure to upload the models first using the upload scripts.")


if __name__ == "__main__":
    main()
