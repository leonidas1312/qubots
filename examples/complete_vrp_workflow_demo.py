"""
Complete VRP Workflow Demonstration

This script demonstrates the complete qubots framework workflow using the existing
VRP (Vehicle Routing Problem) examples. It shows the end-to-end process from local
model creation to platform execution.

Workflow Steps:
1. Create/Load VRP Problem and Genetic Optimizer locally
2. Upload models to Rastion Platform
3. Load models from Rastion Platform
4. Execute optimization via Playground Service
5. Display results and performance metrics

Usage:
    python complete_vrp_workflow_demo.py

Requirements:
- Authenticated with Rastion platform
- qubots library installed
- VRP examples available in examples/ directory

Author: Qubots Community
Version: 1.0.0
"""

import sys
import os
import time
import json
from pathlib import Path

# Add qubots to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import qubots
    import qubots.rastion as rastion
    from qubots import execute_playground_optimization
    print("✅ Qubots framework imported successfully")
except ImportError as e:
    print(f"❌ Failed to import qubots: {e}")
    print("Please install qubots: pip install qubots")
    sys.exit(1)

# Import VRP models from examples
try:
    sys.path.insert(0, str(Path(__file__).parent / "vehicle_routing_problem"))
    from vehicle_routing_problem.qubot import VehicleRoutingProblem

    sys.path.insert(0, str(Path(__file__).parent / "genetic_vrp_optimizer"))
    from genetic_vrp_optimizer.qubot import GeneticVRPOptimizer

    print("✅ VRP examples imported successfully")
except ImportError as e:
    print(f"❌ Failed to import VRP examples: {e}")
    print("Please ensure VRP examples are available in examples/ directory")
    sys.exit(1)


class VRPWorkflowDemo:
    """Complete VRP workflow demonstration class."""

    def __init__(self):
        """Initialize the workflow demo."""
        import random
        self.demo_timestamp = int(time.time())
        self.random_suffix = random.randint(1000, 9999)
        self.problem_repo_name = f"demo_vrp_problem_{self.demo_timestamp}_{self.random_suffix}"
        self.optimizer_repo_name = f"demo_genetic_optimizer_{self.demo_timestamp}_{self.random_suffix}"

        # Track created repositories for cleanup
        self.created_repos = []

        print(f"🚀 VRP Workflow Demo Initialized")
        print(f"   Problem repo: {self.problem_repo_name}")
        print(f"   Optimizer repo: {self.optimizer_repo_name}")

    def step_1_create_local_models(self):
        """Step 1: Create/Load VRP Problem and Genetic Optimizer locally."""
        print("\n" + "="*60)
        print("📦 STEP 1: Create/Load Local Models")
        print("="*60)

        try:
            # Create VRP Problem instance
            print("🚛 Creating VRP Problem instance...")
            self.vrp_problem = VehicleRoutingProblem(
                n_customers=8,
                n_vehicles=3,
                depot_location=(50, 50),
                area_size=(100, 100),
                capacity_range=(80, 120),
                demand_range=(5, 25),
                penalty_unserved=1000.0,
                penalty_capacity=500.0
            )

            print(f"   ✅ VRP Problem created:")
            print(f"      Customers: {self.vrp_problem.n_customers}")
            print(f"      Vehicles: {self.vrp_problem.n_vehicles}")
            print(f"      Total demand: {sum(c.demand for c in self.vrp_problem.customers)}")
            print(f"      Total capacity: {sum(v.capacity for v in self.vrp_problem.vehicles)}")

            # Test problem functionality
            print("   🧪 Testing problem functionality...")
            test_solution = self.vrp_problem.get_random_solution()
            test_cost = self.vrp_problem.evaluate_solution(test_solution, verbose=True)
            print(f"      Random solution cost: {test_cost:.2f}")

            # Create Genetic VRP Optimizer instance
            print("\n🧬 Creating Genetic VRP Optimizer instance...")
            self.genetic_optimizer = GeneticVRPOptimizer(
                population_size=20,
                generations=15,
                crossover_rate=0.8,
                mutation_rate=0.15,
                elite_size=3,
                tournament_size=3,
                adaptive_parameters=True
            )

            print(f"   ✅ Genetic Optimizer created:")
            print(f"      Population size: {self.genetic_optimizer.population_size}")
            print(f"      Generations: {self.genetic_optimizer.generations}")
            print(f"      Crossover rate: {self.genetic_optimizer.crossover_rate}")
            print(f"      Mutation rate: {self.genetic_optimizer.mutation_rate}")

            # Test local optimization
            print("   🧪 Testing local optimization...")
            start_time = time.time()
            local_result = self.genetic_optimizer.optimize(self.vrp_problem)
            local_time = time.time() - start_time

            print(f"      ✅ Local optimization completed:")
            print(f"         Best cost: {local_result.best_value:.2f}")
            print(f"         Runtime: {local_time:.2f} seconds")
            print(f"         Iterations: {local_result.iterations}")

            return True

        except Exception as e:
            print(f"   ❌ Failed to create local models: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step_2_upload_to_platform(self):
        """Step 2: Upload models to Rastion Platform."""
        print("\n" + "="*60)
        print("📤 STEP 2: Upload Models to Rastion Platform")
        print("="*60)

        try:
            # Check authentication
            print("🔐 Checking authentication...")
            if not rastion.is_authenticated():
                print("   ❌ Not authenticated with Rastion platform")
                print("   Please authenticate first:")
                print("   import qubots.rastion as rastion")
                print("   rastion.authenticate('your_gitea_token')")
                return False

            print("   ✅ Authenticated with Rastion platform")

            # Upload VRP Problem using path-based upload
            print(f"\n🚛 Uploading VRP Problem as '{self.problem_repo_name}'...")
            problem_upload_start = time.time()

            # Use path-based upload to preserve the complete qubot.py file
            problem_path = str(Path(__file__).parent / "vehicle_routing_problem")
            problem_url = rastion.upload_qubots_model(
                path=problem_path,
                repository_name=self.problem_repo_name,
                description=f"Demo VRP Problem - {self.demo_timestamp}",
                requirements=["qubots", "numpy"],
                private=False
            )

            problem_upload_time = time.time() - problem_upload_start
            self.created_repos.append(self.problem_repo_name)

            print(f"   ✅ VRP Problem uploaded successfully:")
            print(f"      Repository: {self.problem_repo_name}")
            print(f"      URL: {problem_url}")
            print(f"      Upload time: {problem_upload_time:.2f} seconds")

            # Upload Genetic Optimizer using path-based upload
            print(f"\n🧬 Uploading Genetic Optimizer as '{self.optimizer_repo_name}'...")
            optimizer_upload_start = time.time()

            # Use path-based upload to preserve the complete qubot.py file
            optimizer_path = str(Path(__file__).parent / "genetic_vrp_optimizer")
            optimizer_url = rastion.upload_qubots_model(
                path=optimizer_path,
                repository_name=self.optimizer_repo_name,
                description=f"Demo Genetic VRP Optimizer - {self.demo_timestamp}",
                requirements=["qubots", "numpy"],
                private=False
            )

            optimizer_upload_time = time.time() - optimizer_upload_start
            self.created_repos.append(self.optimizer_repo_name)

            print(f"   ✅ Genetic Optimizer uploaded successfully:")
            print(f"      Repository: {self.optimizer_repo_name}")
            print(f"      URL: {optimizer_url}")
            print(f"      Upload time: {optimizer_upload_time:.2f} seconds")

            # Wait for platform processing
            print("\n⏳ Waiting for platform processing...")
            time.sleep(3)

            return True

        except Exception as e:
            print(f"   ❌ Failed to upload models: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step_3_load_from_platform(self):
        """Step 3: Load models from Rastion Platform."""
        print("\n" + "="*60)
        print("📥 STEP 3: Load Models from Rastion Platform")
        print("="*60)

        try:
            # Load VRP Problem
            print(f"🚛 Loading VRP Problem '{self.problem_repo_name}'...")
            load_start = time.time()

            self.loaded_problem = rastion.load_qubots_model(self.problem_repo_name)

            load_time = time.time() - load_start

            print(f"   ✅ VRP Problem loaded successfully:")
            print(f"      Type: {type(self.loaded_problem).__name__}")
            print(f"      Load time: {load_time:.2f} seconds")

            # Verify problem functionality
            print("   🧪 Verifying problem functionality...")
            if hasattr(self.loaded_problem, '_metadata') and self.loaded_problem._metadata:
                print(f"      Name: {self.loaded_problem._metadata.name}")
                print(f"      Description: {self.loaded_problem._metadata.description}")

            print(f"      Customers: {self.loaded_problem.n_customers}")
            print(f"      Vehicles: {self.loaded_problem.n_vehicles}")

            # Test loaded problem
            test_solution = self.loaded_problem.get_random_solution()
            test_cost = self.loaded_problem.evaluate_solution(test_solution)
            print(f"      Test evaluation: {test_cost:.2f}")

            # Load Genetic Optimizer
            print(f"\n🧬 Loading Genetic Optimizer '{self.optimizer_repo_name}'...")
            load_start = time.time()

            self.loaded_optimizer = rastion.load_qubots_model(self.optimizer_repo_name)

            load_time = time.time() - load_start

            print(f"   ✅ Genetic Optimizer loaded successfully:")
            print(f"      Type: {type(self.loaded_optimizer).__name__}")
            print(f"      Load time: {load_time:.2f} seconds")

            # Verify optimizer functionality
            print("   🧪 Verifying optimizer functionality...")
            if hasattr(self.loaded_optimizer, '_metadata') and self.loaded_optimizer._metadata:
                print(f"      Name: {self.loaded_optimizer._metadata.name}")
                print(f"      Description: {self.loaded_optimizer._metadata.description}")

            print(f"      Population size: {self.loaded_optimizer.population_size}")
            print(f"      Generations: {self.loaded_optimizer.generations}")
            print(f"      Adaptive parameters: {self.loaded_optimizer.adaptive_parameters}")

            return True

        except Exception as e:
            print(f"   ❌ Failed to load models: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step_4_test_direct_execution(self):
        """Step 4: Test direct execution of loaded models."""
        print("\n" + "="*60)
        print("🔧 STEP 4: Test Direct Execution")
        print("="*60)

        try:
            print("🚀 Running direct optimization with loaded models...")

            # Configure optimizer for demo (faster execution)
            self.loaded_optimizer.population_size = 15
            self.loaded_optimizer.generations = 10

            print(f"   Configuration:")
            print(f"      Population size: {self.loaded_optimizer.population_size}")
            print(f"      Generations: {self.loaded_optimizer.generations}")

            # Run direct optimization
            direct_start = time.time()
            direct_result = self.loaded_optimizer.optimize(self.loaded_problem)
            direct_time = time.time() - direct_start

            print(f"   ✅ Direct optimization completed:")
            print(f"      Best cost: {direct_result.best_value:.2f}")
            print(f"      Runtime: {direct_time:.2f} seconds")
            print(f"      Iterations: {direct_result.iterations}")

            # Analyze solution
            if hasattr(direct_result, 'best_solution') and direct_result.best_solution:
                solution_summary = self.loaded_problem.get_solution_summary(direct_result.best_solution)
                print(f"      Vehicles used: {solution_summary['vehicles_used']}/{solution_summary['total_vehicles']}")
                print(f"      Customers served: {solution_summary['served_customers']}/{solution_summary['total_customers']}")
                print(f"      Feasible solution: {solution_summary['feasible']}")

            return True

        except Exception as e:
            print(f"   ❌ Direct execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step_5_playground_execution(self):
        """Step 5: Execute optimization via Playground Service."""
        print("\n" + "="*60)
        print("🎮 STEP 5: Execute via Playground Service")
        print("="*60)

        try:
            print("🚀 Running optimization through Rastion Playground...")

            # Configure parameters for playground execution
            problem_params = {
                "penalty_unserved": 1000.0,
                "penalty_capacity": 500.0
            }

            optimizer_params = {
                "population_size": 20,
                "generations": 15,
                "crossover_rate": 0.8,
                "mutation_rate": 0.15,
                "elite_size": 3,
                "adaptive_parameters": True
            }

            print(f"   Problem parameters: {problem_params}")
            print(f"   Optimizer parameters: {optimizer_params}")

            # Execute through playground
            print("\n   🔄 Starting playground optimization...")
            playground_start = time.time()

            playground_result = execute_playground_optimization(
                problem_name=self.problem_repo_name,
                optimizer_name=self.optimizer_repo_name,
                problem_params=problem_params,
                optimizer_params=optimizer_params
            )

            playground_time = time.time() - playground_start

            # Analyze playground results
            if isinstance(playground_result, dict):
                if "error" in playground_result:
                    print(f"   ❌ Playground execution failed: {playground_result['error']}")
                    return False
                else:
                    print(f"   ✅ Playground optimization completed:")
                    print(f"      Execution time: {playground_time:.2f} seconds")

                    # Display results
                    if "best_value" in playground_result:
                        print(f"      Best cost: {playground_result['best_value']:.2f}")

                    if "runtime_seconds" in playground_result:
                        print(f"      Optimization runtime: {playground_result['runtime_seconds']:.2f} seconds")

                    if "iterations" in playground_result:
                        print(f"      Iterations: {playground_result['iterations']}")

                    # Show additional metrics if available
                    for key, value in playground_result.items():
                        if key not in ["best_value", "runtime_seconds", "iterations", "best_solution"]:
                            print(f"      {key}: {value}")

                    # Store result for comparison
                    self.playground_result = playground_result

            else:
                print(f"   ✅ Playground optimization completed:")
                print(f"      Result type: {type(playground_result)}")
                print(f"      Execution time: {playground_time:.2f} seconds")

                if hasattr(playground_result, 'best_value'):
                    print(f"      Best cost: {playground_result.best_value:.2f}")

                if hasattr(playground_result, 'runtime_seconds'):
                    print(f"      Runtime: {playground_result.runtime_seconds:.2f} seconds")

                self.playground_result = playground_result

            return True

        except Exception as e:
            print(f"   ❌ Playground execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step_6_results_analysis(self):
        """Step 6: Analyze and compare results."""
        print("\n" + "="*60)
        print("📊 STEP 6: Results Analysis and Comparison")
        print("="*60)

        try:
            print("📈 Analyzing optimization results...")

            # Compare local vs platform results if available
            if hasattr(self, 'playground_result'):
                print("\n   🎯 Playground Execution Results:")

                if isinstance(self.playground_result, dict):
                    for key, value in self.playground_result.items():
                        if key == "best_solution":
                            print(f"      {key}: [Solution with {len(value) if hasattr(value, '__len__') else 'N/A'} routes]")
                        else:
                            print(f"      {key}: {value}")
                else:
                    print(f"      Result object: {type(self.playground_result)}")
                    if hasattr(self.playground_result, '__dict__'):
                        for attr, value in self.playground_result.__dict__.items():
                            if not attr.startswith('_'):
                                print(f"      {attr}: {value}")

            # Performance summary
            print("\n   ⚡ Performance Summary:")
            print(f"      ✅ Local model creation: Success")
            print(f"      ✅ Platform upload: Success")
            print(f"      ✅ Platform loading: Success")
            print(f"      ✅ Direct execution: Success")
            print(f"      ✅ Playground execution: Success")

            # Model information
            print("\n   📋 Model Information:")
            print(f"      Problem: {self.problem_repo_name}")
            print(f"      Optimizer: {self.optimizer_repo_name}")
            print(f"      Framework: qubots v{qubots.__version__}")

            return True

        except Exception as e:
            print(f"   ❌ Results analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def cleanup_demo_repos(self):
        """Clean up demo repositories."""
        print("\n" + "="*60)
        print("🧹 CLEANUP: Demo Repositories")
        print("="*60)

        if self.created_repos:
            print("⚠️  Demo repositories created:")
            for repo in self.created_repos:
                print(f"   - {repo}")

            print("\n💡 To clean up these repositories:")
            print("   1. Go to your Rastion platform web interface")
            print("   2. Navigate to your repositories")
            print("   3. Delete the demo repositories listed above")
            print("   4. Or use the platform API if available")
        else:
            print("   No repositories to clean up")

    def run_complete_workflow(self):
        """Run the complete VRP workflow demonstration."""
        print("🎯 COMPLETE VRP WORKFLOW DEMONSTRATION")
        print("=" * 70)
        print("This demo shows the end-to-end qubots framework workflow:")
        print("1. Create/Load local VRP models")
        print("2. Upload models to Rastion Platform")
        print("3. Load models from Rastion Platform")
        print("4. Test direct execution")
        print("5. Execute via Playground Service")
        print("6. Analyze results")
        print("=" * 70)

        workflow_start = time.time()

        # Execute workflow steps
        steps = [
            ("Create Local Models", self.step_1_create_local_models),
            ("Upload to Platform", self.step_2_upload_to_platform),
            ("Load from Platform", self.step_3_load_from_platform),
            ("Test Direct Execution", self.step_4_test_direct_execution),
            ("Playground Execution", self.step_5_playground_execution),
            ("Results Analysis", self.step_6_results_analysis)
        ]

        completed_steps = 0

        for step_name, step_func in steps:
            try:
                if step_func():
                    completed_steps += 1
                    print(f"\n✅ {step_name} completed successfully")
                else:
                    print(f"\n❌ {step_name} failed")
                    break
            except KeyboardInterrupt:
                print(f"\n⏹️ Workflow interrupted by user")
                break
            except Exception as e:
                print(f"\n💥 {step_name} failed with exception: {e}")
                break

        workflow_time = time.time() - workflow_start

        # Final summary
        print("\n" + "="*70)
        print("🎯 WORKFLOW SUMMARY")
        print("="*70)
        print(f"📊 Steps completed: {completed_steps}/{len(steps)}")
        print(f"⏱️ Total workflow time: {workflow_time:.2f} seconds")

        if completed_steps == len(steps):
            print("🎉 Complete workflow executed successfully!")
            print("\n✨ Key Achievements:")
            print("   ✅ VRP models created and tested locally")
            print("   ✅ Models uploaded to Rastion platform")
            print("   ✅ Models loaded from platform successfully")
            print("   ✅ Direct execution validated")
            print("   ✅ Playground execution completed")
            print("   ✅ Real-time optimization results obtained")

            print("\n🚀 Next Steps:")
            print("   1. Explore different VRP configurations")
            print("   2. Try other optimization algorithms")
            print("   3. Integrate with your own applications")
            print("   4. Use the qubots testing framework for validation")
        else:
            print("⚠️ Workflow incomplete - check error messages above")

            print("\n🔧 Troubleshooting:")
            print("   1. Ensure you're authenticated: rastion.authenticate('token')")
            print("   2. Check network connectivity")
            print("   3. Verify VRP examples are available")
            print("   4. Check platform status")

        # Cleanup information
        self.cleanup_demo_repos()

        return completed_steps == len(steps)


def main():
    """Main function to run the VRP workflow demonstration."""
    print("🚀 Starting Complete VRP Workflow Demonstration")

    # Check authentication first
    try:
        if not rastion.is_authenticated():
            print("\n❌ Not authenticated with Rastion platform")
            print("\n🔐 Please authenticate first:")
            print("   import qubots.rastion as rastion")
            print("   rastion.authenticate('your_gitea_token')")
            print("\nThen run this demo again.")
            return False
    except Exception as e:
        print(f"\n❌ Authentication check failed: {e}")
        return False

    # Run the complete workflow
    demo = VRPWorkflowDemo()
    success = demo.run_complete_workflow()

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
