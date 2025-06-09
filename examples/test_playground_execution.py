#!/usr/bin/env python3
"""
Test Playground Execution

This script tests the playground execution functionality to ensure consistency
between local testing and Rastion container environments. It simulates the
container environment and validates that optimization results are reproducible.

Usage:
    python test_playground_execution.py <problem_repo> <optimizer_repo> [options]

Example:
    python test_playground_execution.py tsp highs_tsp_solver
    python test_playground_execution.py maxcut_problem ortools_maxcut_optimizer --iterations 3

Author: Qubots Community
Version: 1.0.0
"""

import argparse
import sys
import time
import traceback
import tempfile
import shutil
import subprocess
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json

# Import qubots modules
try:
    import qubots.rastion as rastion
    from qubots import BaseProblem, BaseOptimizer, OptimizationResult
    from qubots.playground_integration import execute_playground_optimization
except ImportError as e:
    print(f"Error importing qubots: {e}")
    print("Please install qubots: pip install qubots")
    sys.exit(1)


class PlaygroundTester:
    """
    Test playground execution functionality and compare with local execution.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.temp_dirs = []
        
    def log(self, message: str, level: str = "info"):
        """Log a message with appropriate formatting."""
        if not self.verbose:
            return
            
        icons = {
            "info": "ℹ️",
            "success": "✅", 
            "error": "❌",
            "warning": "⚠️",
            "debug": "🔍"
        }
        
        icon = icons.get(level, "📝")
        print(f"{icon} {message}")
    
    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                self.log(f"Cleaned up temporary directory: {temp_dir}", "debug")
    
    def create_isolated_environment(self, problem_path: str, optimizer_path: str) -> Tuple[Path, Path]:
        """
        Create an isolated environment that simulates the container environment.
        
        Args:
            problem_path: Path to problem repository
            optimizer_path: Path to optimizer repository
            
        Returns:
            Tuple of (problem_dir, optimizer_dir) in isolated environment
        """
        self.log("Creating isolated container simulation environment...")
        
        # Create temporary directory for isolation
        temp_base = Path(tempfile.mkdtemp(prefix="qubots_playground_test_"))
        self.temp_dirs.append(temp_base)
        
        # Copy repositories to isolated environment
        problem_dir = temp_base / "problem"
        optimizer_dir = temp_base / "optimizer"
        
        shutil.copytree(problem_path, problem_dir)
        shutil.copytree(optimizer_path, optimizer_dir)
        
        self.log(f"Copied problem to: {problem_dir}")
        self.log(f"Copied optimizer to: {optimizer_dir}")
        
        return problem_dir, optimizer_dir
    
    def install_requirements_in_isolation(self, repo_dir: Path) -> bool:
        """
        Install requirements for a repository in isolation (simulating container).
        
        Args:
            repo_dir: Path to repository directory
            
        Returns:
            True if successful, False otherwise
        """
        requirements_file = repo_dir / "requirements.txt"
        
        if not requirements_file.exists():
            self.log(f"No requirements.txt found in {repo_dir.name}", "warning")
            return True
            
        try:
            # Read requirements
            with open(requirements_file, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            if not requirements:
                self.log(f"Empty requirements.txt in {repo_dir.name}")
                return True
                
            self.log(f"Installing requirements for {repo_dir.name}: {requirements}")
            
            # Install requirements using pip (simulating container environment)
            cmd = [sys.executable, "-m", "pip", "install"] + requirements
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_dir)
            
            if result.returncode != 0:
                self.log(f"Failed to install requirements for {repo_dir.name}: {result.stderr}", "error")
                return False
                
            self.log(f"Successfully installed requirements for {repo_dir.name}", "success")
            return True
            
        except Exception as e:
            self.log(f"Error installing requirements for {repo_dir.name}: {e}", "error")
            return False
    
    def load_model_from_directory(self, repo_dir: Path) -> Any:
        """
        Load a qubots model from a directory (simulating container loading).
        
        Args:
            repo_dir: Path to repository directory
            
        Returns:
            Loaded model instance
        """
        config_file = repo_dir / "config.json"
        qubot_file = repo_dir / "qubot.py"
        
        if not config_file.exists():
            raise ValueError(f"config.json not found in {repo_dir}")
        if not qubot_file.exists():
            raise ValueError(f"qubot.py not found in {repo_dir}")
        
        # Read configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        class_name = config.get("class_name")
        if not class_name:
            raise ValueError(f"class_name not specified in config.json for {repo_dir}")
        
        # Add directory to Python path temporarily
        original_path = sys.path.copy()
        sys.path.insert(0, str(repo_dir))
        
        try:
            # Import the module
            import importlib.util
            spec = importlib.util.spec_from_file_location("qubot", qubot_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the class
            model_class = getattr(module, class_name)
            
            # Instantiate the model
            model = model_class()
            
            self.log(f"Successfully loaded {class_name} from {repo_dir.name}")
            return model
            
        finally:
            # Restore original Python path
            sys.path = original_path

    def test_container_simulation(self, problem_path: str, optimizer_path: str,
                                iterations: int = 1) -> Dict[str, Any]:
        """
        Test optimization in a simulated container environment.

        Args:
            problem_path: Path to problem repository
            optimizer_path: Path to optimizer repository
            iterations: Number of test iterations

        Returns:
            Dictionary containing test results
        """
        self.log("🐳 Testing container simulation...")

        results = {
            "test_type": "container_simulation",
            "iterations": iterations,
            "successful_runs": 0,
            "failed_runs": 0,
            "costs": [],
            "times": [],
            "errors": []
        }

        try:
            # Create isolated environment
            problem_dir, optimizer_dir = self.create_isolated_environment(problem_path, optimizer_path)

            # Install requirements in isolation
            if not self.install_requirements_in_isolation(problem_dir):
                raise RuntimeError("Failed to install problem requirements")
            if not self.install_requirements_in_isolation(optimizer_dir):
                raise RuntimeError("Failed to install optimizer requirements")

            # Load models from directories
            problem = self.load_model_from_directory(problem_dir)
            optimizer = self.load_model_from_directory(optimizer_dir)

            # Run optimization iterations
            for i in range(iterations):
                self.log(f"Container simulation iteration {i + 1}/{iterations}")

                try:
                    start_time = time.time()
                    result = optimizer.optimize(problem)
                    end_time = time.time()

                    iteration_time = end_time - start_time
                    results["times"].append(iteration_time)

                    # Extract cost from result
                    if isinstance(result, OptimizationResult):
                        cost = result.best_value
                    else:
                        # Handle different result formats
                        if hasattr(result, 'cost'):
                            cost = result.cost
                        elif hasattr(result, 'objective_value'):
                            cost = result.objective_value
                        else:
                            # Assume result is the solution itself
                            cost = problem.evaluate_solution(result)

                    results["costs"].append(cost)
                    results["successful_runs"] += 1

                    self.log(f"✅ Container simulation iteration {i + 1} completed: cost={cost}, time={iteration_time:.2f}s")

                except Exception as e:
                    results["failed_runs"] += 1
                    error_msg = f"Iteration {i + 1}: {str(e)}"
                    results["errors"].append(error_msg)
                    self.log(f"❌ Container simulation iteration {i + 1} failed: {e}", "error")

        except Exception as e:
            self.log(f"❌ Container simulation setup failed: {e}", "error")
            results["errors"].append(f"Setup error: {str(e)}")

        return results

    def test_playground_integration(self, problem_path: str, optimizer_path: str,
                                  iterations: int = 1) -> Dict[str, Any]:
        """
        Test optimization using playground integration.

        Args:
            problem_path: Path to problem repository
            optimizer_path: Path to optimizer repository
            iterations: Number of test iterations

        Returns:
            Dictionary containing test results
        """
        self.log("🎮 Testing playground integration...")

        results = {
            "test_type": "playground_integration",
            "iterations": iterations,
            "successful_runs": 0,
            "failed_runs": 0,
            "costs": [],
            "times": [],
            "errors": []
        }

        try:
            # Create isolated environment for playground testing
            problem_dir, optimizer_dir = self.create_isolated_environment(problem_path, optimizer_path)

            # Run optimization iterations using playground integration
            for i in range(iterations):
                self.log(f"Playground integration iteration {i + 1}/{iterations}")

                try:
                    start_time = time.time()

                    # Use playground integration function
                    result = execute_playground_optimization(
                        problem_dir=str(problem_dir),
                        optimizer_dir=str(optimizer_dir)
                    )

                    end_time = time.time()
                    iteration_time = end_time - start_time
                    results["times"].append(iteration_time)

                    # Extract cost from playground result
                    if isinstance(result, dict):
                        cost = result.get('best_value') or result.get('cost') or result.get('objective_value')
                        if cost is None:
                            raise ValueError("Could not extract cost from playground result")
                    else:
                        cost = float(result)

                    results["costs"].append(cost)
                    results["successful_runs"] += 1

                    self.log(f"✅ Playground integration iteration {i + 1} completed: cost={cost}, time={iteration_time:.2f}s")

                except Exception as e:
                    results["failed_runs"] += 1
                    error_msg = f"Iteration {i + 1}: {str(e)}"
                    results["errors"].append(error_msg)
                    self.log(f"❌ Playground integration iteration {i + 1} failed: {e}", "error")

        except Exception as e:
            self.log(f"❌ Playground integration setup failed: {e}", "error")
            results["errors"].append(f"Setup error: {str(e)}")

        return results

    def compare_results(self, container_results: Dict[str, Any],
                       playground_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare results from container simulation and playground integration.

        Args:
            container_results: Results from container simulation
            playground_results: Results from playground integration

        Returns:
            Dictionary containing comparison analysis
        """
        self.log("📊 Comparing container simulation vs playground integration...")

        comparison = {
            "container_successful": container_results["successful_runs"],
            "playground_successful": playground_results["successful_runs"],
            "container_failed": container_results["failed_runs"],
            "playground_failed": playground_results["failed_runs"],
            "consistency_check": "unknown",
            "cost_difference": None,
            "time_difference": None,
            "recommendations": []
        }

        # Check if both methods had successful runs
        if container_results["successful_runs"] > 0 and playground_results["successful_runs"] > 0:
            # Compare costs
            container_avg_cost = sum(container_results["costs"]) / len(container_results["costs"])
            playground_avg_cost = sum(playground_results["costs"]) / len(playground_results["costs"])

            cost_diff = abs(container_avg_cost - playground_avg_cost)
            relative_diff = cost_diff / max(abs(container_avg_cost), abs(playground_avg_cost), 1e-10)

            comparison["cost_difference"] = {
                "container_avg": container_avg_cost,
                "playground_avg": playground_avg_cost,
                "absolute_diff": cost_diff,
                "relative_diff": relative_diff
            }

            # Compare times
            container_avg_time = sum(container_results["times"]) / len(container_results["times"])
            playground_avg_time = sum(playground_results["times"]) / len(playground_results["times"])

            comparison["time_difference"] = {
                "container_avg": container_avg_time,
                "playground_avg": playground_avg_time,
                "difference": abs(container_avg_time - playground_avg_time)
            }

            # Determine consistency
            if relative_diff < 0.01:  # Less than 1% difference
                comparison["consistency_check"] = "excellent"
                comparison["recommendations"].append("✅ Results are highly consistent - safe to upload to Rastion")
            elif relative_diff < 0.05:  # Less than 5% difference
                comparison["consistency_check"] = "good"
                comparison["recommendations"].append("✅ Results are reasonably consistent - should work well in Rastion")
            elif relative_diff < 0.1:  # Less than 10% difference
                comparison["consistency_check"] = "fair"
                comparison["recommendations"].append("⚠️ Some variation detected - test thoroughly in Rastion playground")
            else:
                comparison["consistency_check"] = "poor"
                comparison["recommendations"].append("❌ Significant differences detected - investigate before uploading")

        elif container_results["successful_runs"] == 0 and playground_results["successful_runs"] == 0:
            comparison["consistency_check"] = "both_failed"
            comparison["recommendations"].append("❌ Both methods failed - fix issues before uploading")

        elif container_results["successful_runs"] == 0:
            comparison["consistency_check"] = "container_failed"
            comparison["recommendations"].append("❌ Container simulation failed - check dependencies and requirements")

        elif playground_results["successful_runs"] == 0:
            comparison["consistency_check"] = "playground_failed"
            comparison["recommendations"].append("❌ Playground integration failed - check qubots compatibility")

        return comparison

    def print_summary(self, container_results: Dict[str, Any],
                     playground_results: Dict[str, Any],
                     comparison: Dict[str, Any]):
        """Print a comprehensive test summary."""
        print("\n" + "="*80)
        print("🎯 PLAYGROUND EXECUTION TEST SUMMARY")
        print("="*80)

        # Container simulation results
        print(f"\n🐳 CONTAINER SIMULATION:")
        print(f"   ✅ Successful runs: {container_results['successful_runs']}")
        print(f"   ❌ Failed runs: {container_results['failed_runs']}")
        if container_results["costs"]:
            avg_cost = sum(container_results["costs"]) / len(container_results["costs"])
            avg_time = sum(container_results["times"]) / len(container_results["times"])
            print(f"   💰 Average cost: {avg_cost:.4f}")
            print(f"   ⏱️  Average time: {avg_time:.2f}s")

        # Playground integration results
        print(f"\n🎮 PLAYGROUND INTEGRATION:")
        print(f"   ✅ Successful runs: {playground_results['successful_runs']}")
        print(f"   ❌ Failed runs: {playground_results['failed_runs']}")
        if playground_results["costs"]:
            avg_cost = sum(playground_results["costs"]) / len(playground_results["costs"])
            avg_time = sum(playground_results["times"]) / len(playground_results["times"])
            print(f"   💰 Average cost: {avg_cost:.4f}")
            print(f"   ⏱️  Average time: {avg_time:.2f}s")

        # Comparison results
        print(f"\n📊 CONSISTENCY ANALYSIS:")
        print(f"   🎯 Consistency: {comparison['consistency_check'].upper()}")

        if comparison["cost_difference"]:
            cost_diff = comparison["cost_difference"]
            print(f"   💰 Cost difference: {cost_diff['relative_diff']:.2%}")

        if comparison["time_difference"]:
            time_diff = comparison["time_difference"]
            print(f"   ⏱️  Time difference: {time_diff['difference']:.2f}s")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in comparison["recommendations"]:
            print(f"   {rec}")

        # Error details if any
        all_errors = container_results["errors"] + playground_results["errors"]
        if all_errors:
            print(f"\n❌ ERRORS ENCOUNTERED:")
            for error in all_errors:
                print(f"   • {error}")

        print("="*80)


def main():
    """Main function to handle command line arguments and run playground tests."""
    parser = argparse.ArgumentParser(
        description="Test playground execution consistency for qubots repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Test TSP problem with HiGHS solver
  python test_playground_execution.py tsp highs_tsp_solver

  # Test MaxCut problem with OR-Tools optimizer (multiple iterations)
  python test_playground_execution.py maxcut_problem ortools_maxcut_optimizer --iterations 3

  # Test VRP with genetic algorithm (quiet mode)
  python test_playground_execution.py vehicle_routing_problem genetic_vrp_optimizer --quiet

  # Test with custom repository paths
  python test_playground_execution.py ./my_problem ./my_optimizer --iterations 5
        """
    )

    parser.add_argument("problem_repo", help="Problem repository name or path")
    parser.add_argument("optimizer_repo", help="Optimizer repository name or path")
    parser.add_argument("--iterations", type=int, default=1,
                       help="Number of test iterations to run (default: 1)")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimal output mode")
    parser.add_argument("--container-only", action="store_true",
                       help="Run only container simulation test")
    parser.add_argument("--playground-only", action="store_true",
                       help="Run only playground integration test")

    args = parser.parse_args()

    verbose = not args.quiet

    # Resolve repository paths
    def resolve_repo_path(repo_str):
        """Resolve repository string to actual path."""
        # Check if it's a relative or absolute path
        if os.path.exists(repo_str):
            return os.path.abspath(repo_str)

        # Check if it's in examples directory
        examples_path = Path(__file__).parent / repo_str
        if examples_path.exists():
            return str(examples_path)

        # If not found, assume it's a repository name in examples
        raise FileNotFoundError(f"Repository not found: {repo_str}")

    try:
        problem_path = resolve_repo_path(args.problem_repo)
        optimizer_path = resolve_repo_path(args.optimizer_repo)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Make sure the repository exists in the examples directory or provide a valid path")
        sys.exit(1)

    if verbose:
        print("🎯 Qubots Playground Execution Testing Tool")
        print("="*50)
        print(f"📁 Problem: {problem_path}")
        print(f"🔧 Optimizer: {optimizer_path}")
        print(f"🔄 Iterations: {args.iterations}")

    # Initialize tester
    tester = PlaygroundTester(verbose=verbose)

    try:
        container_results = None
        playground_results = None

        # Run container simulation test
        if not args.playground_only:
            container_results = tester.test_container_simulation(
                problem_path, optimizer_path, args.iterations
            )

        # Run playground integration test
        if not args.container_only:
            playground_results = tester.test_playground_integration(
                problem_path, optimizer_path, args.iterations
            )

        # Compare results if both tests were run
        if container_results and playground_results:
            comparison = tester.compare_results(container_results, playground_results)
            tester.print_summary(container_results, playground_results, comparison)

            # Exit with appropriate code based on consistency
            consistency = comparison["consistency_check"]
            if consistency in ["excellent", "good"]:
                sys.exit(0)  # Success
            elif consistency == "fair":
                sys.exit(1)  # Warning
            else:
                sys.exit(2)  # Error

        elif container_results:
            print(f"\n🐳 Container simulation completed: {container_results['successful_runs']} successful runs")
            sys.exit(0 if container_results['successful_runs'] > 0 else 2)

        elif playground_results:
            print(f"\n🎮 Playground integration completed: {playground_results['successful_runs']} successful runs")
            sys.exit(0 if playground_results['successful_runs'] > 0 else 2)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        if verbose:
            print(f"🔍 Error details: {traceback.format_exc()}")
        sys.exit(3)

    finally:
        # Clean up temporary directories
        tester.cleanup()


if __name__ == "__main__":
    main()
