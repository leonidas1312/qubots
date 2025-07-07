#!/usr/bin/env python3
"""
Portfolio Optimization Demo with Visualization

This script demonstrates the portfolio optimization problem and optimizer
with automatic visualization plots for users new to optimization.
"""

import sys
from pathlib import Path

# Add the examples directory to the Python path
examples_dir = Path(__file__).parent
sys.path.insert(0, str(examples_dir))

def demo_portfolio_optimization():
    """Demonstrate portfolio optimization with different datasets and plots."""
    
    print("🎯 PORTFOLIO OPTIMIZATION DEMO WITH VISUALIZATION")
    print("=" * 60)
    print("This demo shows how to optimize investment portfolios using")
    print("the Qubots framework with automatic visualization plots.")
    print("=" * 60)
    
    try:
        from portfolio_optimization_problem.qubot import PortfolioOptimizationProblem
        from portfolio_optimization_optimizer.qubot import PortfolioOptimizationOptimizer
        
        # Demo 1: Tech Stocks Portfolio
        print("\n📱 DEMO 1: Technology Stocks Portfolio")
        print("-" * 40)
        print("Optimizing a high-growth tech portfolio...")
        
        # Load tech stocks data
        tech_data_path = examples_dir / "portfolio_optimization_problem" / "sample_data" / "tech_stocks.csv"
        if tech_data_path.exists():
            problem_tech = PortfolioOptimizationProblem(
                csv_file_path=str(tech_data_path),
                target_return=0.15  # 15% target return for growth portfolio
            )
            
            optimizer = PortfolioOptimizationOptimizer(
                algorithm="auto",
                max_iterations=500,
                create_plots=True  # Enable visualization
            )
            
            print(f"Stocks: {[stock.symbol for stock in problem_tech.stocks]}")
            print(f"Target Return: {problem_tech.target_return:.1%}")
            print("Optimizing... (plots will appear)")
            
            result_tech = optimizer.optimize(problem_tech)
            
            print(f"✅ Optimization completed!")
            print(f"   Algorithm: {result_tech.algorithm_used}")
            print(f"   Runtime: {result_tech.runtime_seconds:.2f} seconds")
            
            input("\nPress Enter to continue to the next demo...")
        
        # Demo 2: Conservative Portfolio
        print("\n🏦 DEMO 2: Conservative Portfolio")
        print("-" * 40)
        print("Optimizing a low-risk conservative portfolio...")
        
        conservative_data_path = examples_dir / "portfolio_optimization_problem" / "sample_data" / "conservative_portfolio.csv"
        if conservative_data_path.exists():
            problem_conservative = PortfolioOptimizationProblem(
                csv_file_path=str(conservative_data_path),
                target_return=0.06  # 6% target return for conservative portfolio
            )
            
            optimizer_conservative = PortfolioOptimizationOptimizer(
                algorithm="slsqp",  # Fast algorithm for demonstration
                max_iterations=300,
                create_plots=True
            )
            
            print(f"Stocks: {[stock.symbol for stock in problem_conservative.stocks]}")
            print(f"Target Return: {problem_conservative.target_return:.1%}")
            print("Optimizing... (plots will appear)")
            
            result_conservative = optimizer_conservative.optimize(problem_conservative)
            
            print(f"✅ Optimization completed!")
            print(f"   Algorithm: {result_conservative.algorithm_used}")
            print(f"   Runtime: {result_conservative.runtime_seconds:.2f} seconds")
            
            input("\nPress Enter to continue to comparison...")
        
        # Demo 3: Comparison
        print("\n📊 DEMO 3: Portfolio Comparison")
        print("-" * 40)
        print("Comparing the two optimized portfolios:")
        print()
        
        if 'result_tech' in locals() and 'result_conservative' in locals():
            print("TECH PORTFOLIO vs CONSERVATIVE PORTFOLIO")
            print("-" * 50)
            print(f"{'Metric':<20} {'Tech':<12} {'Conservative':<12}")
            print("-" * 50)
            print(f"{'Return':<20} {f'{result_tech.portfolio_return:.2%}':<12} {f'{result_conservative.portfolio_return:.2%}':<12}")
            print(f"{'Risk (Volatility)':<20} {f'{result_tech.portfolio_volatility:.2%}':<12} {f'{result_conservative.portfolio_volatility:.2%}':<12}")
            print(f"{'Sharpe Ratio':<20} {f'{result_tech.sharpe_ratio:.3f}':<12} {f'{result_conservative.sharpe_ratio:.3f}':<12}")
            print(f"{'Algorithm':<20} {result_tech.algorithm_used:<12} {result_conservative.algorithm_used:<12}")
            print("-" * 50)
            
            print("\n💡 Key Insights:")
            print("• Tech portfolio: Higher return but also higher risk")
            print("• Conservative portfolio: Lower risk but also lower return")
            print("• Sharpe ratio shows risk-adjusted performance")
            print("• Different algorithms may be optimal for different portfolio types")
        
        print("\n🎉 Demo completed! You've seen how portfolio optimization")
        print("   can help make informed investment decisions with visual insights.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the examples directory.")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

def demo_custom_data():
    """Demonstrate with custom CSV data."""
    print("\n🔧 BONUS: Custom Data Demo")
    print("-" * 30)
    print("You can also use your own CSV data:")
    
    custom_csv = """symbol,expected_return,volatility
AAPL,0.12,0.25
MSFT,0.11,0.22
GOOGL,0.15,0.30"""
    
    print("CSV Data:")
    print(custom_csv)
    print()
    
    try:
        from portfolio_optimization_problem.qubot import PortfolioOptimizationProblem
        from portfolio_optimization_optimizer.qubot import PortfolioOptimizationOptimizer
        
        problem_custom = PortfolioOptimizationProblem(
            csv_data=custom_csv,
            target_return=0.12
        )
        
        optimizer_custom = PortfolioOptimizationOptimizer(
            algorithm="slsqp",
            create_plots=True
        )
        
        print("Optimizing custom portfolio...")
        result_custom = optimizer_custom.optimize(problem_custom)
        
        print(f"✅ Custom portfolio optimized!")
        print(f"   Return: {result_custom.portfolio_return:.2%}")
        print(f"   Risk: {result_custom.portfolio_volatility:.2%}")
        
    except Exception as e:
        print(f"❌ Custom demo failed: {e}")

if __name__ == "__main__":
    print("Welcome to the Portfolio Optimization Demo!")
    print("This demo will show you how to optimize investment portfolios")
    print("with automatic visualization plots.\n")
    
    try:
        demo_portfolio_optimization()
        
        # Ask if user wants to see custom data demo
        response = input("\nWould you like to see the custom data demo? (y/n): ")
        if response.lower().startswith('y'):
            demo_custom_data()
            
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
    
    print("\nThank you for trying the Portfolio Optimization Demo!")
    print("For more information, check the README files in the examples directory.")
