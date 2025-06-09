#!/usr/bin/env python3
"""
Playground Consistency Test

This script compares local optimization results with playground execution results
to ensure consistency before uploading repositories to Rastion. It helps validate
that your optimization will work the same way in the Rastion playground as it
does locally.

Usage:
    python playground_consistency_test.py <problem_repo> <optimizer_repo>

Example:
    python playground_consistency_test.py tsp highs_tsp_solver
    python playground_consistency_test.py maxcut_problem ortools_maxcut_optimizer

Author: Qubots Community
Version: 1.0.0
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple
import statistics

# Import qubots modules
try:
    from qubots.playground_integration import execute_playground_optimization
    from qubots import BaseProblem, BaseOptimizer, OptimizationResult
except ImportError as e:
    print(f"Error importing qubots: {e}")
    print("Please install qubots: pip install qubots")
    sys.exit(1)


def load_local_models(problem_path: str, optimizer_path: str) -> Tuple[BaseProblem, BaseOptimizer]:
    """
    Load problem and optimizer models from local directories.
    
    Args:
        problem_path: Path to problem repository
        optimizer_path: Path to optimizer repository
        
    Returns:
        Tuple of (problem, optimizer) instances
    """
    import json
    import importlib.util
    
    def load_model_from_path(repo_path: str):
        repo_path = Path(repo_path)
        config_file = repo_path / "config.json"
        qubot_file = repo_path / "qubot.py"
        
        # Read configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        class_name = config.get("class_name")
        if not class_name:
            raise ValueError(f"class_name not specified in config.json for {repo_path}")
        
        # Load module
        spec = importlib.util.spec_from_file_location("qubot", qubot_file)
        module = importlib.util.module_from_spec(spec)
        
        # Add repo path to sys.path temporarily
        original_path = sys.path.copy()
        sys.path.insert(0, str(repo_path))
        
        try:
            spec.loader.exec_module(module)
            model_class = getattr(module, class_name)
            return model_class()
        finally:
            sys.path = original_path
    
    print("📁 Loading local models...")
    problem = load_model_from_path(problem_path)
    optimizer = load_model_from_path(optimizer_path)
    print(f"✅ Loaded {problem.__class__.__name__} and {optimizer.__class__.__name__}")
    
    return problem, optimizer


def run_local_optimization(problem: BaseProblem, optimizer: BaseOptimizer, 
                          iterations: int = 3) -> Dict[str, Any]:
    """
    Run optimization locally and collect results.
    
    Args:
        problem: Problem instance
        optimizer: Optimizer instance
        iterations: Number of iterations to run
        
    Returns:
        Dictionary containing results
    """
    print(f"🏠 Running local optimization ({iterations} iterations)...")
    
    results = {
        "method": "local",
        "costs": [],
        "times": [],
        "successful_runs": 0,
        "failed_runs": 0,
        "errors": []
    }
    
    for i in range(iterations):
        try:
            start_time = time.time()
            result = optimizer.optimize(problem)
            end_time = time.time()
            
            # Extract cost
            if isinstance(result, OptimizationResult):
                cost = result.best_value
            elif hasattr(result, 'cost'):
                cost = result.cost
            elif hasattr(result, 'objective_value'):
                cost = result.objective_value
            else:
                cost = problem.evaluate_solution(result)
            
            results["costs"].append(cost)
            results["times"].append(end_time - start_time)
            results["successful_runs"] += 1
            
            print(f"  ✅ Iteration {i+1}: cost={cost:.4f}, time={end_time - start_time:.2f}s")
            
        except Exception as e:
            results["failed_runs"] += 1
            results["errors"].append(f"Iteration {i+1}: {str(e)}")
            print(f"  ❌ Iteration {i+1} failed: {e}")
    
    return results


def run_playground_optimization(problem_path: str, optimizer_path: str, 
                               iterations: int = 3) -> Dict[str, Any]:
    """
    Run optimization using playground integration and collect results.
    
    Args:
        problem_path: Path to problem repository
        optimizer_path: Path to optimizer repository
        iterations: Number of iterations to run
        
    Returns:
        Dictionary containing results
    """
    print(f"🎮 Running playground optimization ({iterations} iterations)...")
    
    results = {
        "method": "playground",
        "costs": [],
        "times": [],
        "successful_runs": 0,
        "failed_runs": 0,
        "errors": []
    }
    
    for i in range(iterations):
        try:
            start_time = time.time()
            result = execute_playground_optimization(
                problem_dir=problem_path,
                optimizer_dir=optimizer_path
            )
            end_time = time.time()
            
            # Extract cost from playground result
            if isinstance(result, dict):
                cost = result.get('best_value') or result.get('cost') or result.get('objective_value')
                if cost is None:
                    raise ValueError("Could not extract cost from playground result")
            else:
                cost = float(result)
            
            results["costs"].append(cost)
            results["times"].append(end_time - start_time)
            results["successful_runs"] += 1
            
            print(f"  ✅ Iteration {i+1}: cost={cost:.4f}, time={end_time - start_time:.2f}s")
            
        except Exception as e:
            results["failed_runs"] += 1
            results["errors"].append(f"Iteration {i+1}: {str(e)}")
            print(f"  ❌ Iteration {i+1} failed: {e}")
    
    return results


def analyze_consistency(local_results: Dict[str, Any], 
                       playground_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze consistency between local and playground results.
    
    Args:
        local_results: Results from local optimization
        playground_results: Results from playground optimization
        
    Returns:
        Dictionary containing analysis
    """
    analysis = {
        "consistent": False,
        "cost_difference": None,
        "time_difference": None,
        "recommendation": "",
        "details": {}
    }
    
    # Check if both methods had successful runs
    if local_results["successful_runs"] == 0:
        analysis["recommendation"] = "❌ Local optimization failed - fix issues before testing playground"
        return analysis
    
    if playground_results["successful_runs"] == 0:
        analysis["recommendation"] = "❌ Playground optimization failed - check qubots compatibility"
        return analysis
    
    # Calculate statistics
    local_avg = statistics.mean(local_results["costs"])
    playground_avg = statistics.mean(playground_results["costs"])
    
    local_time_avg = statistics.mean(local_results["times"])
    playground_time_avg = statistics.mean(playground_results["times"])
    
    # Calculate differences
    cost_diff = abs(local_avg - playground_avg)
    relative_diff = cost_diff / max(abs(local_avg), abs(playground_avg), 1e-10)
    
    analysis["cost_difference"] = {
        "local_avg": local_avg,
        "playground_avg": playground_avg,
        "absolute_diff": cost_diff,
        "relative_diff": relative_diff
    }
    
    analysis["time_difference"] = {
        "local_avg": local_time_avg,
        "playground_avg": playground_time_avg,
        "difference": abs(local_time_avg - playground_time_avg)
    }
    
    # Determine consistency and recommendation
    if relative_diff < 0.01:  # Less than 1% difference
        analysis["consistent"] = True
        analysis["recommendation"] = "✅ EXCELLENT: Results are highly consistent - safe to upload to Rastion!"
    elif relative_diff < 0.05:  # Less than 5% difference
        analysis["consistent"] = True
        analysis["recommendation"] = "✅ GOOD: Results are reasonably consistent - should work well in Rastion"
    elif relative_diff < 0.1:  # Less than 10% difference
        analysis["consistent"] = False
        analysis["recommendation"] = "⚠️ FAIR: Some variation detected - test thoroughly in Rastion playground"
    else:
        analysis["consistent"] = False
        analysis["recommendation"] = "❌ POOR: Significant differences detected - investigate before uploading"
    
    return analysis


def print_results(local_results: Dict[str, Any], playground_results: Dict[str, Any], 
                 analysis: Dict[str, Any]):
    """Print comprehensive results."""
    print("\n" + "="*70)
    print("🎯 PLAYGROUND CONSISTENCY TEST RESULTS")
    print("="*70)
    
    # Local results
    print(f"\n🏠 LOCAL OPTIMIZATION:")
    print(f"   ✅ Successful: {local_results['successful_runs']}")
    print(f"   ❌ Failed: {local_results['failed_runs']}")
    if local_results["costs"]:
        avg_cost = statistics.mean(local_results["costs"])
        avg_time = statistics.mean(local_results["times"])
        print(f"   💰 Average cost: {avg_cost:.4f}")
        print(f"   ⏱️  Average time: {avg_time:.2f}s")
    
    # Playground results
    print(f"\n🎮 PLAYGROUND OPTIMIZATION:")
    print(f"   ✅ Successful: {playground_results['successful_runs']}")
    print(f"   ❌ Failed: {playground_results['failed_runs']}")
    if playground_results["costs"]:
        avg_cost = statistics.mean(playground_results["costs"])
        avg_time = statistics.mean(playground_results["times"])
        print(f"   💰 Average cost: {avg_cost:.4f}")
        print(f"   ⏱️  Average time: {avg_time:.2f}s")
    
    # Analysis
    print(f"\n📊 CONSISTENCY ANALYSIS:")
    if analysis["cost_difference"]:
        cost_diff = analysis["cost_difference"]
        print(f"   💰 Cost difference: {cost_diff['relative_diff']:.2%}")
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"   {analysis['recommendation']}")
    
    # Errors
    all_errors = local_results["errors"] + playground_results["errors"]
    if all_errors:
        print(f"\n❌ ERRORS:")
        for error in all_errors:
            print(f"   • {error}")
    
    print("="*70)


def main():
    """Main function to handle command line arguments and run consistency test."""
    parser = argparse.ArgumentParser(
        description="Test consistency between local and playground optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Test TSP problem with HiGHS solver
  python playground_consistency_test.py tsp highs_tsp_solver

  # Test MaxCut problem with OR-Tools optimizer
  python playground_consistency_test.py maxcut_problem ortools_maxcut_optimizer

  # Test VRP with genetic algorithm
  python playground_consistency_test.py vehicle_routing_problem genetic_vrp_optimizer
        """
    )

    parser.add_argument("problem_repo", help="Problem repository name or path")
    parser.add_argument("optimizer_repo", help="Optimizer repository name or path")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of test iterations (default: 3)")

    args = parser.parse_args()

    # Resolve repository paths
    def resolve_repo_path(repo_str):
        """Resolve repository string to actual path."""
        if Path(repo_str).exists():
            return str(Path(repo_str).resolve())

        # Check if it's in examples directory
        examples_path = Path(__file__).parent / repo_str
        if examples_path.exists():
            return str(examples_path.resolve())

        raise FileNotFoundError(f"Repository not found: {repo_str}")

    try:
        problem_path = resolve_repo_path(args.problem_repo)
        optimizer_path = resolve_repo_path(args.optimizer_repo)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Make sure the repository exists in the examples directory or provide a valid path")
        sys.exit(1)

    print("🎯 Qubots Playground Consistency Test")
    print("="*40)
    print(f"📁 Problem: {Path(problem_path).name}")
    print(f"🔧 Optimizer: {Path(optimizer_path).name}")
    print(f"🔄 Iterations: {args.iterations}")

    try:
        # Load local models
        problem, optimizer = load_local_models(problem_path, optimizer_path)

        # Run local optimization
        local_results = run_local_optimization(problem, optimizer, args.iterations)

        # Run playground optimization
        playground_results = run_playground_optimization(problem_path, optimizer_path, args.iterations)

        # Analyze consistency
        analysis = analyze_consistency(local_results, playground_results)

        # Print results
        print_results(local_results, playground_results, analysis)

        # Exit with appropriate code
        if analysis["consistent"]:
            print("\n🎉 Consistency test PASSED! You can safely upload to Rastion.")
            sys.exit(0)
        else:
            print("\n⚠️ Consistency test shows variations. Review before uploading.")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print(f"🔍 Error details: {traceback.format_exc()}")
        sys.exit(2)


if __name__ == "__main__":
    main()
