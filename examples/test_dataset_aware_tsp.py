#!/usr/bin/env python3
"""
Dataset-Aware TSP Testing Script

This script demonstrates the complete workflow for using dataset-aware TSP qubots:
1. Loading problems with dataset IDs from Rastion platform
2. Loading optimizers using AutoProblem and AutoOptimizer
3. Running optimization locally
4. Simulating workflow automation execution

This shows how developers can start locally and then use the same qubots
in workflow automation with identical results.
"""

import sys
import os
import time
import json
from typing import Optional, Dict, Any

# Add the examples directory to Python path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from qubots import AutoProblem, AutoOptimizer
    QUBOTS_AVAILABLE = True
except ImportError:
    print("Warning: qubots not available. Using local imports.")
    QUBOTS_AVAILABLE = False
    # Import local implementations
    from dataset_aware_tsp_problem.qubot import DatasetAwareTSPProblem
    from genetic_tsp_optimizer.qubot import GeneticTSPOptimizer


def test_with_dataset_id(dataset_id: str, auth_token: Optional[str] = None):
    """
    Test TSP optimization using a dataset ID from Rastion platform.
    
    Args:
        dataset_id: Rastion platform dataset ID
        auth_token: Optional authentication token
    """
    print(f"\n{'='*60}")
    print(f"Testing with Dataset ID: {dataset_id}")
    print(f"{'='*60}")
    
    try:
        if QUBOTS_AVAILABLE:
            # Load problem using AutoProblem (simulates workflow automation)
            print("Loading TSP problem from Rastion platform...")
            problem = AutoProblem.from_repo("Rastion/demo-tsp2", override_params={
                "dataset_id": dataset_id,
                "dataset_source": "platform",
                "auth_token": auth_token
            })
            
            # Load optimizer using AutoOptimizer
            print("Loading genetic TSP optimizer...")
            optimizer = AutoOptimizer.from_repo("Rastion/genetic-tsp-optimizer-demo", override_params={
                "population_size": 50,
                "generations": 100,
                "early_stopping": True,
                "stagnation_limit": 20
            })
        else:
            # Direct instantiation for local testing
            print("Creating TSP problem with dataset ID...")
            problem = DatasetAwareTSPProblem(
                dataset_id=dataset_id,
                dataset_source="platform",
                auth_token=auth_token
            )
            
            print("Creating genetic TSP optimizer...")
            optimizer = GeneticTSPOptimizer(
                population_size=50,
                generations=100,
                early_stopping=True,
                stagnation_limit=20
            )
        
        # Display problem information
        print("\nProblem Information:")
        problem_info = problem.get_problem_info()
        for key, value in problem_info.items():
            print(f"  {key}: {value}")
        
        # Test solution validation
        print("\nTesting solution validation...")
        random_solution = problem.random_solution()
        is_valid = problem.is_valid_solution(random_solution)
        print(f"  Random solution valid: {is_valid}")
        print(f"  Random solution: {random_solution[:10]}..." if len(random_solution) > 10 else f"  Random solution: {random_solution}")
        
        # Run optimization
        print("\nRunning optimization...")
        start_time = time.time()
        result = optimizer.optimize(problem)
        end_time = time.time()
        
        # Display results
        print(f"\nOptimization Results:")
        print(f"  Best tour distance: {result.best_value:.2f}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Runtime: {result.runtime_seconds:.3f} seconds")
        print(f"  Converged: {result.convergence_achieved}")
        print(f"  Solution valid: {problem.is_valid_solution(result.best_solution)}")
        
        if len(result.best_solution) <= 20:
            print(f"  Best tour: {result.best_solution}")
        else:
            print(f"  Best tour (first 10): {result.best_solution[:10]}...")
        
        # Additional metrics
        if result.additional_metrics:
            print(f"\nAdditional Information:")
            for key, value in result.additional_metrics.items():
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"Error testing with dataset ID {dataset_id}: {e}")
        return False


def test_with_local_file(instance_file: str):
    """
    Test TSP optimization using a local TSPLIB file.
    
    Args:
        instance_file: Path to local TSPLIB file
    """
    print(f"\n{'='*60}")
    print(f"Testing with Local File: {instance_file}")
    print(f"{'='*60}")
    
    try:
        if QUBOTS_AVAILABLE:
            # Load problem using AutoProblem
            print("Loading TSP problem from local file...")
            problem = AutoProblem.from_repo("Rastion/demo-tsp2", override_params={
                "instance_file": instance_file,
                "dataset_source": "local"
            })
            
            # Load optimizer using AutoOptimizer
            print("Loading genetic TSP optimizer...")
            optimizer = AutoOptimizer.from_repo("Rastion/genetic-tsp-optimizer-demo", override_params={
                "population_size": 30,
                "generations": 50,
                "adaptive_params": True
            })
        else:
            # Direct instantiation
            print("Creating TSP problem with local file...")
            problem = DatasetAwareTSPProblem(
                instance_file=instance_file,
                dataset_source="local"
            )
            
            print("Creating genetic TSP optimizer...")
            optimizer = GeneticTSPOptimizer(
                population_size=30,
                generations=50,
                adaptive_params=True
            )
        
        # Run optimization
        print("Running optimization...")
        result = optimizer.optimize(problem)
        
        # Display results
        print(f"\nResults:")
        print(f"  Best distance: {result.best_value:.2f}")
        print(f"  Runtime: {result.runtime_seconds:.3f}s")
        print(f"  Valid solution: {problem.is_valid_solution(result.best_solution)}")
        
        return True
        
    except Exception as e:
        print(f"Error testing with local file {instance_file}: {e}")
        return False


def test_generated_instance(n_cities: int = 20, seed: int = 42):
    """
    Test TSP optimization using a generated random instance.
    
    Args:
        n_cities: Number of cities to generate
        seed: Random seed for reproducibility
    """
    print(f"\n{'='*60}")
    print(f"Testing with Generated Instance: {n_cities} cities")
    print(f"{'='*60}")
    
    try:
        if QUBOTS_AVAILABLE:
            # Load problem using AutoProblem
            print("Loading TSP problem with generated instance...")
            problem = AutoProblem.from_repo("Rastion/demo-tsp2", override_params={
                "n_cities": n_cities,
                "seed": seed,
                "dataset_source": "none"
            })
            
            # Load optimizer using AutoOptimizer
            print("Loading genetic TSP optimizer...")
            optimizer = AutoOptimizer.from_repo("Rastion/genetic-tsp-optimizer-demo", override_params={
                "population_size": 40,
                "generations": 100,
                "mutation_rate": 0.15
            })
        else:
            # Direct instantiation
            print("Creating TSP problem with generated instance...")
            problem = DatasetAwareTSPProblem(
                n_cities=n_cities,
                seed=seed,
                dataset_source="none"
            )
            
            print("Creating genetic TSP optimizer...")
            optimizer = GeneticTSPOptimizer(
                population_size=40,
                generations=100,
                mutation_rate=0.15
            )
        
        # Run optimization
        print("Running optimization...")
        result = optimizer.optimize(problem)
        
        # Display results
        print(f"\nResults:")
        print(f"  Best distance: {result.best_value:.2f}")
        print(f"  Runtime: {result.runtime_seconds:.3f}s")
        print(f"  Generations: {result.iterations}")
        print(f"  Valid solution: {problem.is_valid_solution(result.best_solution)}")
        
        return True
        
    except Exception as e:
        print(f"Error testing generated instance: {e}")
        return False


def simulate_workflow_automation(dataset_id: str):
    """
    Simulate how the workflow automation would execute the same optimization.
    This demonstrates the seamless transition from local development to cloud execution.
    
    Args:
        dataset_id: Dataset ID to use in workflow automation
    """
    print(f"\n{'='*60}")
    print("SIMULATING WORKFLOW AUTOMATION EXECUTION")
    print(f"{'='*60}")
    
    print("In workflow automation, the following would happen:")
    print("1. User drags TSP Problem node to canvas")
    print("2. User sets dataset_id parameter to:", dataset_id)
    print("3. User drags Genetic TSP Optimizer node to canvas")
    print("4. User connects problem output to optimizer input")
    print("5. User clicks 'Execute' button")
    print("6. Platform loads qubots using AutoProblem/AutoOptimizer")
    print("7. Platform runs optimization in cloud environment")
    print("8. Results are displayed in workflow automation UI")
    
    # Simulate the actual execution
    print("\nSimulated execution:")
    return test_with_dataset_id(dataset_id)


def main():
    """Main testing function."""
    print("Dataset-Aware TSP Qubots Testing Suite")
    print("=" * 60)
    
    # Test scenarios
    test_results = []
    
    # Test 1: Generated instance (always works)
    print("\n🧪 Test 1: Generated Random Instance")
    result1 = test_generated_instance(n_cities=15, seed=123)
    test_results.append(("Generated Instance", result1))
    
    # Test 2: Local file (if available)
    print("\n🧪 Test 2: Local TSPLIB File")
    # Create a simple test file if it doesn't exist
    test_file = "test_instance.tsp"
    if not os.path.exists(test_file):
        create_test_tsp_file(test_file)
    
    result2 = test_with_local_file(test_file)
    test_results.append(("Local File", result2))
    
    # Test 3: Platform dataset (requires dataset ID)
    print("\n🧪 Test 3: Platform Dataset")
    print("Note: This test requires a valid dataset ID from Rastion platform")
    dataset_id = input("Enter dataset ID (or press Enter to skip): ").strip()
    
    if dataset_id:
        result3 = test_with_dataset_id(dataset_id)
        test_results.append(("Platform Dataset", result3))
        
        # Test 4: Workflow automation simulation
        print("\n🧪 Test 4: Workflow Automation Simulation")
        result4 = simulate_workflow_automation(dataset_id)
        test_results.append(("Workflow Automation", result4))
    else:
        print("Skipping platform dataset test (no dataset ID provided)")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for test_name, success in test_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total_tests = len(test_results)
    passed_tests = sum(1 for _, success in test_results if success)
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your qubots are ready for production.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")


def create_test_tsp_file(filename: str):
    """Create a simple test TSP file for local testing."""
    tsp_content = """NAME: test_instance
TYPE: TSP
COMMENT: Simple test instance for qubots
DIMENSION: 5
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 10.0 0.0
3 10.0 10.0
4 0.0 10.0
5 5.0 5.0
EOF
"""
    
    with open(filename, 'w') as f:
        f.write(tsp_content)
    
    print(f"Created test TSP file: {filename}")


if __name__ == "__main__":
    main()
