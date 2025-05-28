#!/usr/bin/env python3
"""
Load and Test Optimization

This script loads a problem and optimizer from the Rastion platform and tests
the optimization locally. It validates that the models work together correctly
and provides detailed feedback on the optimization process.

Usage:
    python load_and_test_optimization.py <problem_repo> <optimizer_repo> [options]

Example:
    python load_and_test_optimization.py vehicle_routing_problem genetic_vrp_optimizer
    python load_and_test_optimization.py user/my_problem user/my_optimizer --iterations 5

Author: Qubots Community
Version: 1.0.0
"""

import argparse
import sys
import time
import traceback
from typing import Optional, Dict, Any

# Import qubots modules
try:
    import qubots.rastion as rastion
    from qubots import BaseProblem, BaseOptimizer, OptimizationResult
except ImportError as e:
    print(f"Error importing qubots: {e}")
    print("Please install qubots: pip install qubots")
    sys.exit(1)


def load_model_safely(model_name: str, username: Optional[str] = None, model_type: str = "model") -> Any:
    """
    Safely load a model from Rastion with error handling.
    
    Args:
        model_name: Name of the model repository
        username: Repository owner (auto-detected if None)
        model_type: Type of model for error messages
        
    Returns:
        Loaded model instance
    """
    try:
        print(f"🔄 Loading {model_type}: {model_name}")
        if username:
            print(f"👤 Owner: {username}")
        
        model = rastion.load_qubots_model(model_name, username=username)
        
        print(f"✅ Successfully loaded {model_type}: {model.__class__.__name__}")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load {model_type} '{model_name}': {e}")
        print(f"💡 Make sure the repository exists and you have access to it")
        raise


def validate_compatibility(problem: BaseProblem, optimizer: BaseOptimizer) -> bool:
    """
    Validate that the problem and optimizer are compatible.
    
    Args:
        problem: Problem instance
        optimizer: Optimizer instance
        
    Returns:
        True if compatible, False otherwise
    """
    print("🔍 Validating problem-optimizer compatibility...")
    
    try:
        # Check if problem has required methods
        required_problem_methods = ["get_random_solution", "evaluate_solution"]
        for method in required_problem_methods:
            if not hasattr(problem, method):
                print(f"❌ Problem missing required method: {method}")
                return False
        
        # Check if optimizer has required methods
        required_optimizer_methods = ["optimize"]
        for method in required_optimizer_methods:
            if not hasattr(optimizer, method):
                print(f"❌ Optimizer missing required method: {method}")
                return False
        
        # Test basic problem functionality
        try:
            solution = problem.get_random_solution()
            cost = problem.evaluate_solution(solution)
            print(f"✅ Problem generates valid solutions (sample cost: {cost})")
        except Exception as e:
            print(f"❌ Problem failed basic functionality test: {e}")
            return False
        
        print("✅ Compatibility validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility validation failed: {e}")
        return False


def run_optimization_test(problem: BaseProblem, optimizer: BaseOptimizer, 
                         iterations: int = 1, verbose: bool = True) -> Dict[str, Any]:
    """
    Run optimization test and collect results.
    
    Args:
        problem: Problem instance
        optimizer: Optimizer instance
        iterations: Number of test iterations
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary containing test results
    """
    print(f"🚀 Running optimization test ({iterations} iteration{'s' if iterations != 1 else ''})...")
    
    results = {
        "iterations": iterations,
        "successful_runs": 0,
        "failed_runs": 0,
        "best_cost": float('inf'),
        "worst_cost": float('-inf'),
        "average_cost": 0.0,
        "total_time": 0.0,
        "average_time": 0.0,
        "errors": []
    }
    
    costs = []
    times = []
    
    for i in range(iterations):
        print(f"\n--- Iteration {i + 1}/{iterations} ---")
        
        try:
            # Run optimization
            start_time = time.time()
            result = optimizer.optimize(problem)
            end_time = time.time()
            
            iteration_time = end_time - start_time
            times.append(iteration_time)
            
            # Extract cost from result
            if isinstance(result, OptimizationResult):
                cost = result.best_value
                solution = result.best_solution
            else:
                # Handle different result formats
                if hasattr(result, 'cost'):
                    cost = result.cost
                elif hasattr(result, 'objective_value'):
                    cost = result.objective_value
                else:
                    # Assume result is the solution itself
                    solution = result
                    cost = problem.evaluate_solution(solution)
            
            costs.append(cost)
            results["successful_runs"] += 1
            
            if verbose:
                print(f"✅ Optimization completed successfully")
                print(f"💰 Cost: {cost}")
                print(f"⏱️  Time: {iteration_time:.2f} seconds")
                
                # Show solution summary if available
                if hasattr(problem, 'get_solution_summary'):
                    try:
                        summary = problem.get_solution_summary(solution)
                        print(f"📊 Solution summary: {summary}")
                    except:
                        pass
            
        except Exception as e:
            results["failed_runs"] += 1
            error_msg = f"Iteration {i + 1}: {str(e)}"
            results["errors"].append(error_msg)
            
            if verbose:
                print(f"❌ Optimization failed: {e}")
                print(f"🔍 Error details: {traceback.format_exc()}")
    
    # Calculate statistics
    if costs:
        results["best_cost"] = min(costs)
        results["worst_cost"] = max(costs)
        results["average_cost"] = sum(costs) / len(costs)
    
    if times:
        results["total_time"] = sum(times)
        results["average_time"] = sum(times) / len(times)
    
    return results


def print_test_summary(results: Dict[str, Any], problem_name: str, optimizer_name: str):
    """
    Print a comprehensive test summary.
    
    Args:
        results: Test results dictionary
        problem_name: Name of the problem
        optimizer_name: Name of the optimizer
    """
    print("\n" + "="*60)
    print("📋 OPTIMIZATION TEST SUMMARY")
    print("="*60)
    
    print(f"🎯 Problem: {problem_name}")
    print(f"🔧 Optimizer: {optimizer_name}")
    print(f"🔄 Total iterations: {results['iterations']}")
    print(f"✅ Successful runs: {results['successful_runs']}")
    print(f"❌ Failed runs: {results['failed_runs']}")
    
    if results['successful_runs'] > 0:
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"💰 Best cost: {results['best_cost']}")
        print(f"💸 Worst cost: {results['worst_cost']}")
        print(f"📈 Average cost: {results['average_cost']:.4f}")
        print(f"⏱️  Total time: {results['total_time']:.2f} seconds")
        print(f"⏱️  Average time: {results['average_time']:.2f} seconds")
        
        # Success rate
        success_rate = (results['successful_runs'] / results['iterations']) * 100
        print(f"📈 Success rate: {success_rate:.1f}%")
    
    if results['errors']:
        print(f"\n❌ ERRORS ENCOUNTERED:")
        for error in results['errors']:
            print(f"   • {error}")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if results['failed_runs'] == 0:
        print("🟢 EXCELLENT: All optimization runs completed successfully!")
    elif results['successful_runs'] > results['failed_runs']:
        print("🟡 GOOD: Most optimization runs completed successfully")
    elif results['successful_runs'] > 0:
        print("🟠 FAIR: Some optimization runs completed successfully")
    else:
        print("🔴 POOR: No optimization runs completed successfully")
    
    print("="*60)


def main():
    """Main function to handle command line arguments and run tests."""
    parser = argparse.ArgumentParser(
        description="Load and test optimization models from Rastion platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default settings
  python load_and_test_optimization.py vehicle_routing_problem genetic_vrp_optimizer
  
  # Test with specific user repositories
  python load_and_test_optimization.py user/my_problem user/my_optimizer
  
  # Run multiple iterations for statistical analysis
  python load_and_test_optimization.py maxcut_problem cplex_maxcut_optimizer --iterations 5
  
  # Quiet mode with minimal output
  python load_and_test_optimization.py my_problem my_optimizer --quiet
        """
    )
    
    parser.add_argument("problem_repo", help="Problem repository name (format: [username/]repo_name)")
    parser.add_argument("optimizer_repo", help="Optimizer repository name (format: [username/]repo_name)")
    parser.add_argument("--iterations", type=int, default=1, help="Number of optimization iterations to run (default: 1)")
    parser.add_argument("--quiet", action="store_true", help="Minimal output mode")
    parser.add_argument("--token", help="Rastion authentication token (can also be set via environment variable RASTION_TOKEN)")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Parse repository names
    def parse_repo_name(repo_str):
        if "/" in repo_str:
            username, repo_name = repo_str.split("/", 1)
            return repo_name, username
        else:
            return repo_str, None
    
    problem_name, problem_username = parse_repo_name(args.problem_repo)
    optimizer_name, optimizer_username = parse_repo_name(args.optimizer_repo)
    
    if verbose:
        print("🎯 Qubots Optimization Testing Tool")
        print("="*40)
    
    # Authentication (optional for public repositories)
    if args.token:
        if verbose:
            print("🔐 Authenticating with Rastion platform...")
        if not rastion.authenticate(args.token):
            print("❌ Authentication failed!")
            sys.exit(1)
        if verbose:
            print("✅ Authentication successful!")
    
    try:
        # Load models
        problem = load_model_safely(problem_name, problem_username, "problem")
        optimizer = load_model_safely(optimizer_name, optimizer_username, "optimizer")
        
        # Validate compatibility
        if not validate_compatibility(problem, optimizer):
            print("❌ Models are not compatible!")
            sys.exit(1)
        
        # Run optimization tests
        results = run_optimization_test(problem, optimizer, args.iterations, verbose)
        
        # Print summary
        print_test_summary(results, args.problem_repo, args.optimizer_repo)
        
        # Exit with appropriate code
        if results['failed_runs'] == 0:
            sys.exit(0)  # Success
        elif results['successful_runs'] > 0:
            sys.exit(1)  # Partial success
        else:
            sys.exit(2)  # Complete failure
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        if verbose:
            print(f"🔍 Error details: {traceback.format_exc()}")
        sys.exit(3)


if __name__ == "__main__":
    main()
