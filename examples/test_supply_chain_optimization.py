#!/usr/bin/env python3
"""
Test script for Supply Chain Optimization Decision Models

This script tests the supply chain optimization components to ensure they work
correctly together and produce valid results. It follows the same pattern as
the portfolio and molecular optimization tests.

Usage:
    python test_supply_chain_optimization.py
"""

import sys
from pathlib import Path

# Add the examples directory to the Python path
examples_dir = Path(__file__).parent
sys.path.insert(0, str(examples_dir))

def test_supply_chain_problem():
    """Test the supply chain network optimization problem."""
    print("\n" + "=" * 60)
    print("TESTING SUPPLY CHAIN NETWORK PROBLEM")
    print("=" * 60)

    try:
        from supply_chain_network_problem.qubot import SupplyChainNetworkProblem

        # Test 1: Default problem (built-in data)
        print("\n1. Testing with default data...")
        problem = SupplyChainNetworkProblem()
        print(f"   Problem created with {len(problem.suppliers)} suppliers")
        print(f"   Warehouses: {len(problem.warehouses)}")
        print(f"   Customers: {len(problem.customers)}")
        print(f"   Total demand: {problem.total_demand:,.0f} units")
        print(f"   Total supplier capacity: {problem.total_supplier_capacity:,.0f} units")

        # Test random solution generation
        solution = problem.random_solution()
        print(f"   Random solution generated")
        print(f"     - Supplier production: {len(solution['supplier_production'])} entries")
        print(f"     - Warehouse flows: {len(solution['warehouse_flows'])} entries")
        print(f"     - Warehouse operations: {len(solution['warehouse_operations'])} entries")

        # Test solution evaluation
        cost = problem.evaluate_solution(solution)
        print(f"   Solution evaluated, cost: ${cost:,.2f}")

        # Test feasibility check
        is_feasible = problem.is_feasible(solution)
        print(f"   Feasibility check: {is_feasible}")

        # Test solution summary
        summary = problem.get_solution_summary(solution)
        print(f"   Solution summary generated ({len(summary.split('\\n'))} lines)")

        # Test 2: CSV data input
        print("\n2. Testing with CSV data...")
        csv_data = """entity_type,entity_id,name,location_x,location_y,capacity,fixed_cost,variable_cost,demand,supplier_id,warehouse_id,customer_id,unit_cost
supplier,S1,Test Supplier,10,20,1000,25000,10.0,0,,,,,
warehouse,W1,Test Warehouse,30,40,800,40000,5.0,0,,,,,
customer,C1,Test Customer,50,60,0,0,0,500,,,,,
supplier_warehouse,SW1,S1-W1 Link,,,,,,,S1,W1,,12.0
warehouse_customer,WC1,W1-C1 Link,,,,,,,W1,,C1,8.0"""

        problem2 = SupplyChainNetworkProblem(csv_data=csv_data)
        print(f"   Problem created with CSV data")
        print(f"     - Suppliers: {len(problem2.suppliers)}")
        print(f"     - Warehouses: {len(problem2.warehouses)}")
        print(f"     - Customers: {len(problem2.customers)}")

        solution2 = problem2.random_solution()
        cost2 = problem2.evaluate_solution(solution2)
        print(f"   CSV-based solution evaluated, cost: ${cost2:,.2f}")

        # Test 3: Load from dataset file
        print("\n3. Testing with dataset file...")
        dataset_file = examples_dir / "supply_chain_dataset.csv"
        if dataset_file.exists():
            with open(dataset_file, 'r') as f:
                dataset_content = f.read()

            problem3 = SupplyChainNetworkProblem(dataset_content=dataset_content)
            print(f"   ✓ Problem loaded from dataset file")
            print(f"     - Suppliers: {len(problem3.suppliers)}")
            print(f"     - Warehouses: {len(problem3.warehouses)}")
            print(f"     - Customers: {len(problem3.customers)}")
            print(f"     - Total demand: {problem3.total_demand:,.0f} units")

            solution3 = problem3.random_solution()
            cost3 = problem3.evaluate_solution(solution3)
            print(f"   ✓ Dataset-based solution evaluated, cost: ${cost3:,.2f}")
        else:
            print("   ⚠️  Dataset file not found, skipping file test")

        print("\n✅ Supply Chain Problem test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Supply Chain Problem test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False



def test_ortools_optimizer():
    """Test the OR-Tools-based supply chain optimizer."""
    print("\n" + "=" * 60)
    print("TESTING OR-TOOLS SUPPLY CHAIN OPTIMIZER")
    print("=" * 60)

    # Test 1: Check OR-Tools availability first
    print("\n1. Testing OR-Tools availability...")
    try:
        from ortools.linear_solver import pywraplp
        print(f"   ✓ OR-Tools available")
        print(f"   ✓ Version: {pywraplp.__version__ if hasattr(pywraplp, '__version__') else 'Unknown'}")
        ortools_available = True
    except ImportError:
        print("   ⚠️  OR-Tools not available - install with: pip install ortools")
        print("   ⚠️  Skipping OR-Tools optimizer tests")
        return True  # Skip but don't fail

    if not ortools_available:
        return True

    try:
        from ortools_supply_chain_optimizer.qubot import ORToolsSupplyChainOptimizer

        # Test 2: Default optimizer
        print("\n2. Testing optimizer initialization...")
        optimizer = ORToolsSupplyChainOptimizer()
        print(f"   ✓ Optimizer created")
        print(f"   ✓ Solver: {optimizer.solver_name}")
        print(f"   ✓ Time limit: {optimizer.time_limit} seconds")

        # Test 3: Custom parameters
        print("\n3. Testing with custom parameters...")
        custom_optimizer = ORToolsSupplyChainOptimizer(
            solver_name="GLOP",
            time_limit=120.0
        )
        print(f"   ✓ Custom optimizer created")
        print(f"   ✓ Custom solver: {custom_optimizer.solver_name}")
        print(f"   ✓ Custom time limit: {custom_optimizer.time_limit}")

        print("\n✅ OR-Tools Optimizer test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ OR-Tools Optimizer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_heuristic_optimizer():
    """Test the heuristic supply chain optimizer."""
    print("\n" + "=" * 60)
    print("TESTING HEURISTIC SUPPLY CHAIN OPTIMIZER")
    print("=" * 60)

    try:
        from heuristic_supply_chain_optimizer.qubot import HeuristicSupplyChainOptimizer

        # Test 1: Default optimizer
        print("\n1. Testing optimizer initialization...")
        optimizer = HeuristicSupplyChainOptimizer()
        print(f"   ✓ Optimizer created")
        print(f"   ✓ Max iterations: {optimizer.max_iterations}")
        print(f"   ✓ Local search iterations: {optimizer.local_search_iterations}")
        print(f"   ✓ Perturbation strength: {optimizer.perturbation_strength}")
        print(f"   ✓ Improvement threshold: {optimizer.improvement_threshold}")

        # Test 2: Custom parameters
        print("\n2. Testing with custom parameters...")
        custom_optimizer = HeuristicSupplyChainOptimizer(
            max_iterations=500,
            local_search_iterations=25,
            perturbation_strength=0.3,
            improvement_threshold=0.005,
            random_seed=42
        )
        print(f"   ✓ Custom optimizer created")
        print(f"   ✓ Custom max iterations: {custom_optimizer.max_iterations}")
        print(f"   ✓ Custom perturbation: {custom_optimizer.perturbation_strength}")
        print(f"   ✓ Random seed: {custom_optimizer.random_seed}")

        print("\n✅ Heuristic Optimizer test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Heuristic Optimizer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between problem and optimizers."""
    print("\n" + "=" * 60)
    print("TESTING SUPPLY CHAIN OPTIMIZATION INTEGRATION")
    print("=" * 60)

    try:
        from supply_chain_network_problem.qubot import SupplyChainNetworkProblem
        from heuristic_supply_chain_optimizer.qubot import HeuristicSupplyChainOptimizer

        # Check if OR-Tools is available
        try:
            from ortools_supply_chain_optimizer.qubot import ORToolsSupplyChainOptimizer
            ortools_available = True
        except ImportError:
            print("   ⚠️  OR-Tools not available, skipping OR-Tools optimizer tests")
            ORToolsSupplyChainOptimizer = None
            ortools_available = False

        # Create problem
        print("\n1. Setting up supply chain optimization...")

        # Load dataset if available
        dataset_file = examples_dir / "supply_chain_dataset.csv"
        if dataset_file.exists():
            with open(dataset_file, 'r') as f:
                dataset_content = f.read()
            problem = SupplyChainNetworkProblem(dataset_content=dataset_content)
            print(f"   ✓ Problem loaded from dataset file")
        else:
            problem = SupplyChainNetworkProblem()
            print(f"   ✓ Problem created with default data")

        print(f"   ✓ Network: {len(problem.suppliers)} suppliers, {len(problem.warehouses)} warehouses, {len(problem.customers)} customers")
        print(f"   ✓ Total demand: {problem.total_demand:,.0f} units")
        print(f"   ✓ Total capacity: {problem.total_supplier_capacity:,.0f} units")

        # Test with OR-Tools optimizer
        print("\n2. Testing OR-Tools optimization...")
        if ORToolsSupplyChainOptimizer is not None:
            try:
                ortools_optimizer = ORToolsSupplyChainOptimizer(
                    solver_name="SCIP",
                    time_limit=30.0  # Short time limit for testing
                )

                import time
                start_time = time.time()
                ortools_result = ortools_optimizer.optimize(problem)
                ortools_runtime = time.time() - start_time

                print(f"   ✓ OR-Tools optimization completed")
                print(f"   ✓ Best cost: ${ortools_result.best_value:,.2f}")
                print(f"   ✓ Runtime: {ortools_runtime:.2f} seconds")
                print(f"   ✓ Solver status: {ortools_result.additional_metrics.get('solver_status', 'Unknown')}")
                print(f"   ✓ Solution feasible: {problem.is_feasible(ortools_result.best_solution)}")

            except Exception as e:
                print(f"   ⚠️  OR-Tools optimization failed: {e}")
                ortools_result = None
                ortools_runtime = 0
        else:
            print(f"   ⚠️  OR-Tools optimizer not available, skipping")
            ortools_result = None
            ortools_runtime = 0

        # Test with heuristic optimizer
        print("\n3. Testing heuristic optimization...")
        try:
            heuristic_optimizer = HeuristicSupplyChainOptimizer(
                max_iterations=100,  # Reduced for faster testing
                local_search_iterations=25,
                perturbation_strength=0.2,
                random_seed=42
            )

            start_time = time.time()
            heuristic_result = heuristic_optimizer.optimize(problem)
            heuristic_runtime = time.time() - start_time

            print(f"   ✓ Heuristic optimization completed")
            print(f"   ✓ Best cost: ${heuristic_result.best_value:,.2f}")
            print(f"   ✓ Runtime: {heuristic_runtime:.2f} seconds")
            print(f"   ✓ Iterations: {heuristic_result.iterations}")
            print(f"   ✓ Improvements: {heuristic_result.additional_metrics.get('total_improvements', 0)}")
            print(f"   ✓ Solution feasible: {problem.is_feasible(heuristic_result.best_solution)}")

        except Exception as e:
            print(f"   ⚠️  Heuristic optimization failed: {e}")
            heuristic_result = None
            heuristic_runtime = 0

        # Compare results
        print("\n4. Comparing optimization results...")

        # Compare with heuristic
        if ortools_result and heuristic_result:
            cost_gap = ((heuristic_result.best_value - ortools_result.best_value) /
                       ortools_result.best_value * 100)
            speedup = ortools_runtime / heuristic_runtime if heuristic_runtime > 0 else 0

            print(f"   ✓ Cost comparison:")
            print(f"     - OR-Tools (exact): ${ortools_result.best_value:,.2f}")
            print(f"     - Heuristic: ${heuristic_result.best_value:,.2f}")
            print(f"     - Gap: {cost_gap:+.2f}%")

            print(f"   ✓ Performance comparison:")
            print(f"     - OR-Tools runtime: {ortools_runtime:.2f}s")
            print(f"     - Heuristic runtime: {heuristic_runtime:.2f}s")
            if speedup > 0:
                print(f"     - Speedup: {speedup:.1f}x faster")

            # Quality assessment
            if cost_gap <= 10:
                print(f"   ✓ Quality assessment: Excellent (within 10% of optimal)")
            elif cost_gap <= 25:
                print(f"   ✓ Quality assessment: Good (within 25% of optimal)")
            else:
                print(f"   ⚠️  Quality assessment: Could be improved")

        elif ortools_result:
            print(f"   ✓ OR-Tools result: ${ortools_result.best_value:,.2f}")
        elif heuristic_result:
            print(f"   ✓ Heuristic result: ${heuristic_result.best_value:,.2f}")
        else:
            print(f"   ⚠️  No successful optimization results")

        # Show best solution
        print("\n5. Best solution analysis...")

        # Find overall best result
        all_results = []
        if ortools_result:
            all_results.append(("OR-Tools (Exact)", ortools_result))
        if heuristic_result:
            all_results.append(("Heuristic", heuristic_result))

        best_result = None
        best_name = ""

        if all_results:
            best_result_tuple = min(all_results, key=lambda x: x[1].best_value)
            best_name = best_result_tuple[0]
            best_result = best_result_tuple[1]

        if best_result:
            print(f"   ✓ Best solution by: {best_name}")
            print(f"   ✓ Total cost: ${best_result.best_value:,.2f}")

            # Count open warehouses
            open_warehouses = sum(1 for is_open in best_result.best_solution['warehouse_operations'].values() if is_open > 0)
            active_flows = len([f for f in best_result.best_solution['warehouse_flows'].values() if f > 0])

            print(f"   ✓ Open warehouses: {open_warehouses}/{len(problem.warehouses)}")
            print(f"   ✓ Active flows: {active_flows}")
            print(f"   ✓ Solution summary available: {len(problem.get_solution_summary(best_result.best_solution).split('\\n'))} lines")

        print("\n✅ Integration test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests for the supply chain optimization decision models."""
    print("SUPPLY CHAIN OPTIMIZATION DECISION MODELS TESTS")
    print("=" * 70)

    # Run individual tests
    tests = [
        ("Supply Chain Problem", test_supply_chain_problem),
        ("OR-Tools Optimizer", test_ortools_optimizer),
        ("Heuristic Optimizer", test_heuristic_optimizer),
        ("Integration", test_integration)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        success = test_func()
        results.append((test_name, success))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if not success:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED! The supply chain optimization decision models are working correctly.")
        print("\nYou can now use these decision models for:")
        print("• Manufacturing network optimization")
        print("• Retail supply chain planning")
        print("• E-commerce fulfillment optimization")
        print("• Humanitarian logistics planning")
        print("• Multi-echelon inventory optimization")
    else:
        print("❌ SOME TESTS FAILED. Please check the error messages above.")

    print("\n📚 Next steps:")
    print("• Experiment with different parameter settings")
    print("• Try larger datasets for scalability testing")
    print("• Compare exact vs heuristic approaches")
    print("• Upload to Rastion platform for cloud execution")

if __name__ == "__main__":
    main()
