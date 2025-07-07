#!/usr/bin/env python3
"""
Test script for Portfolio Optimization Problem and Optimizer

This script tests the portfolio optimization components to ensure they work
correctly together and produce valid results.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add the examples directory to the Python path
examples_dir = Path(__file__).parent
sys.path.insert(0, str(examples_dir))

def test_portfolio_problem():
    """Test the portfolio optimization problem."""
    print("=" * 60)
    print("TESTING PORTFOLIO OPTIMIZATION PROBLEM")
    print("=" * 60)
    
    try:
        # Import the problem class
        from portfolio_optimization_problem.qubot import PortfolioOptimizationProblem
        
        # Test 1: Default problem (built-in data)
        print("\n1. Testing with default data...")
        problem = PortfolioOptimizationProblem()
        print(f"   ✓ Problem created with {problem.n_stocks} stocks")
        print(f"   ✓ Target return: {problem.target_return:.1%}")
        print(f"   ✓ Risk-free rate: {problem.risk_free_rate:.1%}")
        
        # Test random solution generation
        solution = problem.random_solution()
        print(f"   ✓ Random solution generated: {len(solution['weights'])} weights")
        
        # Test solution evaluation
        risk = problem.evaluate_solution(solution)
        print(f"   ✓ Solution evaluated, risk: {risk:.6f}")
        
        # Test feasibility check
        is_feasible = problem.is_feasible(solution)
        print(f"   ✓ Feasibility check: {is_feasible}")
        
        # Test solution info
        info = problem.get_solution_info(solution)
        print(f"   ✓ Portfolio return: {info['portfolio_return']:.2%}")
        print(f"   ✓ Portfolio volatility: {info['portfolio_volatility']:.2%}")
        print(f"   ✓ Sharpe ratio: {info['sharpe_ratio']:.3f}")
        
        # Test 2: CSV data input
        print("\n2. Testing with CSV data...")
        csv_data = """symbol,expected_return,volatility
AAPL,0.12,0.25
GOOGL,0.15,0.30
MSFT,0.11,0.22"""
        
        problem2 = PortfolioOptimizationProblem(
            csv_data=csv_data,
            target_return=0.12
        )
        print(f"   ✓ Problem created with CSV data: {problem2.n_stocks} stocks")
        
        solution2 = problem2.random_solution()
        risk2 = problem2.evaluate_solution(solution2)
        print(f"   ✓ CSV-based solution evaluated, risk: {risk2:.6f}")
        
        # Test 3: Constraint violations
        print("\n3. Testing constraint violations...")
        bad_solution = [0.5, 0.3, 0.1]  # Doesn't sum to 1
        penalty = problem2.evaluate_solution(bad_solution)
        print(f"   ✓ Constraint violation penalty applied: {penalty:.2f}")
        
        print("\n✅ Portfolio Problem tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Portfolio Problem test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_portfolio_optimizer():
    """Test the portfolio optimization optimizer."""
    print("\n" + "=" * 60)
    print("TESTING PORTFOLIO OPTIMIZATION OPTIMIZER")
    print("=" * 60)
    
    try:
        # Import the optimizer class
        from portfolio_optimization_optimizer.qubot import PortfolioOptimizationOptimizer
        from portfolio_optimization_problem.qubot import PortfolioOptimizationProblem
        
        # Create a simple problem
        problem = PortfolioOptimizationProblem(target_return=0.10)
        
        # Test 1: SLSQP optimizer
        print("\n1. Testing SLSQP optimizer...")
        optimizer = PortfolioOptimizationOptimizer(
            algorithm="slsqp",
            max_iterations=500,
            tolerance=1e-6
        )
        
        result = optimizer.optimize(problem)
        print(f"   ✓ Optimization completed")
        print(f"   ✓ Algorithm used: {result.algorithm_used}")
        print(f"   ✓ Runtime: {result.runtime_seconds:.3f} seconds")
        print(f"   ✓ Iterations: {result.iterations}")
        print(f"   ✓ Portfolio return: {result.portfolio_return:.2%}")
        print(f"   ✓ Portfolio volatility: {result.portfolio_volatility:.2%}")
        print(f"   ✓ Sharpe ratio: {result.sharpe_ratio:.3f}")
        print(f"   ✓ Constraint violations: {len(result.constraint_violations)}")
        
        # Test 2: Automatic algorithm selection
        print("\n2. Testing automatic algorithm selection...")
        optimizer_auto = PortfolioOptimizationOptimizer(algorithm="auto")
        result_auto = optimizer_auto.optimize(problem)
        print(f"   ✓ Auto-selected algorithm: {result_auto.algorithm_used}")
        print(f"   ✓ Runtime: {result_auto.runtime_seconds:.3f} seconds")
        
        # Test 3: Differential Evolution
        print("\n3. Testing Differential Evolution...")
        optimizer_de = PortfolioOptimizationOptimizer(
            algorithm="differential_evolution",
            max_iterations=100  # Reduced for faster testing
        )
        result_de = optimizer_de.optimize(problem)
        print(f"   ✓ DE optimization completed")
        print(f"   ✓ Runtime: {result_de.runtime_seconds:.3f} seconds")
        
        print("\n✅ Portfolio Optimizer tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Portfolio Optimizer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between problem and optimizer with sample data."""
    print("\n" + "=" * 60)
    print("TESTING INTEGRATION WITH SAMPLE DATA")
    print("=" * 60)
    
    try:
        from portfolio_optimization_problem.qubot import PortfolioOptimizationProblem
        from portfolio_optimization_optimizer.qubot import PortfolioOptimizationOptimizer
        
        # Test with different sample datasets
        datasets = [
            ("tech_stocks.csv", 0.15, "Technology Portfolio"),
            ("conservative_portfolio.csv", 0.06, "Conservative Portfolio"),
            ("diversified_portfolio.csv", 0.08, "Diversified Portfolio")
        ]
        
        for dataset_file, target_return, description in datasets:
            print(f"\n{description}:")
            print("-" * 40)
            
            # Check if sample data file exists
            data_path = examples_dir / "portfolio_optimization_problem" / "sample_data" / dataset_file
            if not data_path.exists():
                print(f"   ⚠️  Sample data file not found: {dataset_file}")
                continue
            
            # Create problem with sample data
            problem = PortfolioOptimizationProblem(
                csv_file_path=str(data_path),
                target_return=target_return
            )
            
            # Create optimizer
            optimizer = PortfolioOptimizationOptimizer(
                algorithm="auto",
                max_iterations=500
            )
            
            # Optimize
            result = optimizer.optimize(problem)
            
            print(f"   ✓ Stocks: {problem.n_stocks}")
            print(f"   ✓ Target return: {target_return:.1%}")
            print(f"   ✓ Achieved return: {result.portfolio_return:.2%}")
            print(f"   ✓ Portfolio risk: {result.portfolio_volatility:.2%}")
            print(f"   ✓ Sharpe ratio: {result.sharpe_ratio:.3f}")
            print(f"   ✓ Algorithm: {result.algorithm_used}")
            print(f"   ✓ Runtime: {result.runtime_seconds:.3f}s")
            print(f"   ✓ Feasible: {len(result.constraint_violations) == 0}")
            
            # Show top 3 allocations
            allocations = result.stock_allocations
            top_allocations = sorted(allocations.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   ✓ Top allocations: {', '.join([f'{stock}: {weight:.1%}' for stock, weight in top_allocations])}")
        
        print("\n✅ Integration tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("PORTFOLIO OPTIMIZATION COMPONENT TESTING")
    print("=" * 60)
    
    # Run tests
    test_results = []
    
    test_results.append(test_portfolio_problem())
    test_results.append(test_portfolio_optimizer())
    test_results.append(test_integration())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Portfolio optimization components are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
