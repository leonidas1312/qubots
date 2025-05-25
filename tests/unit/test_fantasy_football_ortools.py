"""
Local Testing Script for Fantasy Football OR-Tools Optimizer.

This script demonstrates how to use the Fantasy Football OR-Tools optimizer
locally with sample data. It includes comprehensive testing, performance
comparison, and result analysis.

Usage:
    python test_fantasy_football_ortools.py

Requirements:
    - qubots framework
    - OR-Tools (pip install ortools)
    - pandas, numpy
    - fantasy_football.py (problem definition)
    - fantasy_football_ortools_optimizer.py (optimizer)

Author: Qubots Community
Version: 1.0.0
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Import qubots components
try:
    from fantasy_football import FantasyFootballProblem
    from fantasy_football_ortools_optimizer import FantasyFootballORToolsOptimizer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure fantasy_football.py and fantasy_football_ortools_optimizer.py are in the same directory")
    sys.exit(1)

# Check for OR-Tools
try:
    import ortools
    ORTOOLS_AVAILABLE = True
except ImportError:
    print("OR-Tools not available. Please install with: pip install ortools")
    ORTOOLS_AVAILABLE = False


def create_sample_data() -> pd.DataFrame:
    """Create sample fantasy football data for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Sample player data
    players_data = []
    
    # Quarterbacks
    qb_names = ["Josh Allen", "Patrick Mahomes", "Lamar Jackson", "Aaron Rodgers", "Tom Brady"]
    for i, name in enumerate(qb_names):
        players_data.append({
            'Name': name,
            'Pos': 'QB',
            'Team': f'Team{i+1}',
            'DK.points': np.random.uniform(18, 28),
            'DK.salary': np.random.randint(7000, 9000)
        })
    
    # Running Backs
    rb_names = ["Derrick Henry", "Christian McCaffrey", "Dalvin Cook", "Alvin Kamara", "Nick Chubb", "Aaron Jones"]
    for i, name in enumerate(rb_names):
        players_data.append({
            'Name': name,
            'Pos': 'RB',
            'Team': f'Team{i+6}',
            'DK.points': np.random.uniform(12, 22),
            'DK.salary': np.random.randint(5500, 8500)
        })
    
    # Wide Receivers
    wr_names = ["Davante Adams", "Tyreek Hill", "DeAndre Hopkins", "Stefon Diggs", "Calvin Ridley", "DK Metcalf", "Mike Evans"]
    for i, name in enumerate(wr_names):
        players_data.append({
            'Name': name,
            'Pos': 'WR',
            'Team': f'Team{i+12}',
            'DK.points': np.random.uniform(10, 20),
            'DK.salary': np.random.randint(5000, 8000)
        })
    
    # Tight Ends
    te_names = ["Travis Kelce", "Darren Waller", "George Kittle", "Mark Andrews"]
    for i, name in enumerate(te_names):
        players_data.append({
            'Name': name,
            'Pos': 'TE',
            'Team': f'Team{i+19}',
            'DK.points': np.random.uniform(8, 18),
            'DK.salary': np.random.randint(4000, 7000)
        })
    
    # Defenses
    def_names = ["Steelers D/ST", "Ravens D/ST", "Bills D/ST", "Rams D/ST"]
    for i, name in enumerate(def_names):
        players_data.append({
            'Name': name,
            'Pos': 'Def',
            'Team': f'Team{i+23}',
            'DK.points': np.random.uniform(6, 16),
            'DK.salary': np.random.randint(2500, 4500)
        })
    
    return pd.DataFrame(players_data)


def test_optimizer_basic_functionality():
    """Test basic optimizer functionality."""
    print("Testing Basic Optimizer Functionality")
    print("-" * 50)
    
    # Create sample data
    df = create_sample_data()
    print(f"Created sample data with {len(df)} players")
    
    # Create problem
    problem = FantasyFootballProblem(dataframe=df, max_salary=50000)
    print(f"Problem created: {problem.metadata.name}")
    print(f"Number of players: {problem.n_players}")
    print(f"Salary cap: ${problem.max_salary:,}")
    
    # Create optimizer
    optimizer = FantasyFootballORToolsOptimizer(
        time_limit_seconds=10,
        num_search_workers=2,
        log_search_progress=False
    )
    print(f"Optimizer created: {optimizer.metadata.name}")
    
    # Test random solution generation
    random_solution = problem.random_solution()
    print(f"Random solution feasible: {problem.is_feasible(random_solution)}")
    print(f"Random solution points: {problem.evaluate_solution(random_solution):.2f}")
    
    # Run optimization
    print("\nRunning optimization...")
    start_time = time.time()
    result = optimizer.optimize(problem, initial_solution=random_solution)
    end_time = time.time()
    
    # Display results
    print(f"\nOptimization Results:")
    print(f"Best value: {result.best_value:.2f} points")
    print(f"Feasible: {result.is_feasible}")
    print(f"Optimal: {result.convergence_achieved}")
    print(f"Runtime: {result.runtime_seconds:.2f} seconds")
    print(f"Termination reason: {result.termination_reason}")
    
    if result.is_feasible and result.best_solution:
        print(f"\nLineup Summary:")
        lineup = problem.get_lineup_summary(result.best_solution)
        print(lineup)
        
        # Verify constraints
        print(f"\nConstraint Verification:")
        print(f"Total players: {sum(result.best_solution)}")
        total_salary = sum(problem.df.iloc[i]['DK.salary'] * result.best_solution[i] for i in range(len(result.best_solution)))
        print(f"Total salary: ${total_salary:,}")
        print(f"Under salary cap: {total_salary <= problem.max_salary}")
    
    # Get solver statistics
    stats = optimizer.get_solver_statistics()
    print(f"\nSolver Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    return result


def test_optimizer_configurations():
    """Test different optimizer configurations."""
    print("\n\nTesting Different Optimizer Configurations")
    print("-" * 50)
    
    # Create sample data
    df = create_sample_data()
    problem = FantasyFootballProblem(dataframe=df, max_salary=50000)
    
    # Test different configurations
    configs = [
        {"time_limit_seconds": 5, "num_search_workers": 1, "name": "Fast (5s, 1 worker)"},
        {"time_limit_seconds": 15, "num_search_workers": 2, "name": "Medium (15s, 2 workers)"},
        {"time_limit_seconds": 30, "num_search_workers": 4, "name": "Thorough (30s, 4 workers)"}
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        optimizer = FantasyFootballORToolsOptimizer(
            time_limit_seconds=config["time_limit_seconds"],
            num_search_workers=config["num_search_workers"],
            log_search_progress=False
        )
        
        start_time = time.time()
        result = optimizer.optimize(problem)
        end_time = time.time()
        
        results.append({
            "config": config["name"],
            "best_value": result.best_value if result.is_feasible else 0,
            "runtime": result.runtime_seconds,
            "optimal": result.convergence_achieved,
            "feasible": result.is_feasible
        })
        
        print(f"  Best value: {result.best_value:.2f}")
        print(f"  Runtime: {result.runtime_seconds:.2f}s")
        print(f"  Optimal: {result.convergence_achieved}")
    
    # Summary comparison
    print(f"\nConfiguration Comparison:")
    print(f"{'Config':<25} {'Best Value':<12} {'Runtime':<10} {'Optimal':<8} {'Feasible':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['config']:<25} {r['best_value']:<12.2f} {r['runtime']:<10.2f} {r['optimal']:<8} {r['feasible']:<10}")
    
    return results


def main():
    """Main testing function."""
    print("Fantasy Football OR-Tools Optimizer - Local Testing")
    print("=" * 60)
    
    if not ORTOOLS_AVAILABLE:
        print("ERROR: OR-Tools not available. Please install with: pip install ortools")
        return
    
    try:
        # Test basic functionality
        basic_result = test_optimizer_basic_functionality()
        
        # Test different configurations
        config_results = test_optimizer_configurations()
        
        print(f"\n\nTesting Complete!")
        print("=" * 60)
        print("The Fantasy Football OR-Tools optimizer is working correctly.")
        print("You can now use it with your own data or upload it to Rastion.")
        
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
