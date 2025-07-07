"""
Portfolio Optimization Optimizer for Qubots Framework

This optimizer solves portfolio optimization problems using multiple optimization
algorithms including Sequential Least Squares Programming (SLSQP) and 
Differential Evolution for robust portfolio allocation.

Features:
- Multiple optimization algorithms (SLSQP, Differential Evolution, Trust-constr)
- Automatic algorithm selection based on problem characteristics
- Constraint handling for portfolio optimization
- Comprehensive result reporting with financial metrics
- Fallback mechanisms for robustness
"""

import numpy as np
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution, Bounds, LinearConstraint
import warnings
import matplotlib.pyplot as plt

from qubots import BaseOptimizer, OptimizationResult, OptimizerMetadata, OptimizerType, OptimizerFamily


@dataclass
class PortfolioOptimizationResult(OptimizationResult):
    """Extended result class for portfolio optimization with financial metrics."""
    portfolio_return: Optional[float] = None
    portfolio_volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    stock_allocations: Optional[Dict[str, float]] = None
    algorithm_used: Optional[str] = None
    constraint_violations: Optional[List[str]] = None


class PortfolioOptimizationOptimizer(BaseOptimizer):
    """
    Portfolio Optimization Optimizer using multiple algorithms.
    
    Supports:
    - SLSQP (Sequential Least Squares Programming) - primary method
    - Differential Evolution - for global optimization
    - Trust-constr - for constrained optimization
    - Automatic fallback between methods
    """
    
    def __init__(self,
                 algorithm: str = "auto",
                 max_iterations: int = 1000,
                 tolerance: float = 1e-8,
                 time_limit: float = 60.0,
                 random_seed: int = 42,
                 use_bounds: bool = True,
                 use_constraints: bool = True,
                 create_plots: bool = True,
                 **kwargs):
        """
        Initialize portfolio optimization optimizer.

        Args:
            algorithm: Optimization algorithm ("slsqp", "differential_evolution", "trust-constr", "auto")
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            time_limit: Maximum optimization time in seconds
            random_seed: Random seed for reproducibility
            use_bounds: Whether to use variable bounds
            use_constraints: Whether to use explicit constraints
            create_plots: Whether to create visualization plots after optimization
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)

        self.algorithm = algorithm
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.time_limit = time_limit
        self.random_seed = random_seed
        self.use_bounds = use_bounds
        self.use_constraints = use_constraints
        self.create_plots = create_plots
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Suppress scipy optimization warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)

    def _get_default_metadata(self) -> OptimizerMetadata:
        """Return default metadata for portfolio optimization optimizer."""
        return OptimizerMetadata(
            name="Portfolio Optimization Optimizer",
            description="Multi-algorithm portfolio optimizer using SLSQP, Differential Evolution, and Trust-Region methods",
            optimizer_type=OptimizerType.HYBRID,
            optimizer_family=OptimizerFamily.GRADIENT_BASED,
            author="Qubots Framework",
            version="1.0.0",
            is_deterministic=True,
            supports_constraints=True,
            supports_continuous=True,
            supports_discrete=False,
            time_complexity="O(n³)",
            space_complexity="O(n²)",
            convergence_guaranteed=False,
            parallel_capable=False
        )

    def _optimize_implementation(self, problem, initial_solution: Optional[Any] = None) -> OptimizationResult:
        """Core optimization implementation."""
        # Use the existing optimize method logic but return OptimizationResult
        portfolio_result = self._optimize_portfolio(problem)

        # Convert to base OptimizationResult
        return OptimizationResult(
            best_solution=portfolio_result.best_solution,
            best_value=portfolio_result.best_value,
            iterations=portfolio_result.iterations,
            runtime_seconds=portfolio_result.runtime_seconds,
            termination_reason=portfolio_result.termination_reason,
            optimization_history=[],
            parameter_values=self._parameters,
            additional_metrics={
                'portfolio_return': portfolio_result.portfolio_return,
                'portfolio_volatility': portfolio_result.portfolio_volatility,
                'sharpe_ratio': portfolio_result.sharpe_ratio,
                'algorithm_used': portfolio_result.algorithm_used
            }
        )

    def optimize(self, problem) -> PortfolioOptimizationResult:
        """Public interface that returns detailed portfolio result."""
        return self._optimize_portfolio(problem)

    def _optimize_portfolio(self, problem) -> PortfolioOptimizationResult:
        """
        Optimize the portfolio problem.

        Args:
            problem: Portfolio optimization problem instance

        Returns:
            PortfolioOptimizationResult with solution and financial metrics
        """
        start_time = time.time()
        
        # Get problem dimensions
        n_stocks = problem.n_stocks
        
        # Generate initial solution
        initial_solution = self._generate_initial_solution(problem)
        
        # Select optimization algorithm
        if self.algorithm == "auto":
            algorithm = self._select_algorithm(problem)
        else:
            algorithm = self.algorithm
        
        # Optimize using selected algorithm
        try:
            if algorithm == "slsqp":
                result = self._optimize_slsqp(problem, initial_solution)
            elif algorithm == "differential_evolution":
                result = self._optimize_differential_evolution(problem)
            elif algorithm == "trust-constr":
                result = self._optimize_trust_constr(problem, initial_solution)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
                
        except Exception as e:
            # Fallback to differential evolution if primary method fails
            if algorithm != "differential_evolution":
                try:
                    result = self._optimize_differential_evolution(problem)
                    algorithm = "differential_evolution (fallback)"
                except Exception:
                    # Last resort: return initial solution
                    result = self._create_fallback_result(problem, initial_solution)
                    algorithm = "fallback"
            else:
                result = self._create_fallback_result(problem, initial_solution)
                algorithm = "fallback"
        
        # Calculate execution time
        execution_time = time.time() - start_time

        # Get detailed solution information
        solution_info = problem.get_solution_info(result['x'])

        # Create visualization plots if requested
        if self.create_plots:
            self._create_portfolio_plots(problem, result['x'], solution_info, algorithm)
        
        # Create comprehensive result
        portfolio_result = PortfolioOptimizationResult(
            best_solution=result['x'].tolist(),
            best_value=result['fun'],
            iterations=result.get('nit', 0),
            runtime_seconds=execution_time,
            termination_reason=result.get('message', 'completed'),
            portfolio_return=solution_info['portfolio_return'],
            portfolio_volatility=solution_info['portfolio_volatility'],
            sharpe_ratio=solution_info['sharpe_ratio'],
            stock_allocations=solution_info['stock_allocations'],
            algorithm_used=algorithm,
            constraint_violations=self._check_constraints(problem, result['x']),
            additional_metrics={
                'problem_size': n_stocks,
                'target_return': problem.target_return,
                'risk_free_rate': problem.risk_free_rate,
                'is_feasible': solution_info['is_feasible'],
                'weights_sum': solution_info['weights_sum'],
                'optimization_success': result.get('success', False)
            }
        )
        
        return portfolio_result
    
    def _generate_initial_solution(self, problem) -> np.ndarray:
        """Generate a good initial solution for the portfolio problem."""
        n_stocks = problem.n_stocks
        
        # Start with equal weights
        weights = np.ones(n_stocks) / n_stocks
        
        # Adjust to meet return constraint if needed
        expected_returns = np.array([stock.expected_return for stock in problem.stocks])
        current_return = np.dot(weights, expected_returns)
        
        if current_return < problem.target_return:
            # Increase allocation to highest return stocks
            high_return_indices = np.argsort(expected_returns)[-min(3, n_stocks):]
            adjustment = (problem.target_return - current_return) / len(high_return_indices)
            
            for idx in high_return_indices:
                weights[idx] += adjustment
            
            # Renormalize and ensure non-negativity
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
        
        return weights
    
    def _select_algorithm(self, problem) -> str:
        """Select the best algorithm based on problem characteristics."""
        n_stocks = problem.n_stocks
        
        if n_stocks <= 10:
            return "slsqp"  # Fast and accurate for small problems
        elif n_stocks <= 50:
            return "trust-constr"  # Good for medium-sized problems
        else:
            return "differential_evolution"  # More robust for large problems
    
    def _optimize_slsqp(self, problem, initial_solution: np.ndarray) -> Dict[str, Any]:
        """Optimize using Sequential Least Squares Programming."""
        n_stocks = problem.n_stocks
        
        # Define bounds (0 <= weight <= 1)
        bounds = [(0, 1) for _ in range(n_stocks)] if self.use_bounds else None
        
        # Define constraints
        constraints = []
        if self.use_constraints:
            # Constraint: weights sum to 1
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x) - 1.0
            })
            
            # Constraint: expected return >= target
            expected_returns = np.array([stock.expected_return for stock in problem.stocks])
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: np.dot(x, expected_returns) - problem.target_return
            })
        
        # Optimize
        result = minimize(
            fun=lambda x: problem.evaluate_solution(x),
            x0=initial_solution,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.tolerance,
                'disp': False
            }
        )
        
        return result
    
    def _optimize_differential_evolution(self, problem) -> Dict[str, Any]:
        """Optimize using Differential Evolution."""
        n_stocks = problem.n_stocks
        
        # Define bounds
        bounds = [(0, 1) for _ in range(n_stocks)]
        
        # Define constraints for differential evolution
        constraints = []
        if self.use_constraints:
            # Linear constraint: weights sum to 1
            A_eq = np.ones((1, n_stocks))
            b_eq = np.array([1.0])
            constraints.append(LinearConstraint(A_eq, b_eq, b_eq))
            
            # Linear constraint: expected return >= target
            expected_returns = np.array([stock.expected_return for stock in problem.stocks])
            A_ineq = expected_returns.reshape(1, -1)
            b_lower = np.array([problem.target_return])
            b_upper = np.array([np.inf])
            constraints.append(LinearConstraint(A_ineq, b_lower, b_upper))
        
        # Optimize
        result = differential_evolution(
            func=lambda x: problem.evaluate_solution(x),
            bounds=bounds,
            constraints=constraints if constraints else None,
            maxiter=min(self.max_iterations, 1000),
            tol=self.tolerance,
            seed=self.random_seed,
            disp=False
        )
        
        return result
    
    def _optimize_trust_constr(self, problem, initial_solution: np.ndarray) -> Dict[str, Any]:
        """Optimize using Trust-Region Constrained algorithm."""
        n_stocks = problem.n_stocks
        
        # Define bounds
        bounds = Bounds(np.zeros(n_stocks), np.ones(n_stocks)) if self.use_bounds else None
        
        # Define constraints
        constraints = []
        if self.use_constraints:
            # Linear constraint: weights sum to 1
            A_eq = np.ones((1, n_stocks))
            constraints.append(LinearConstraint(A_eq, [1.0], [1.0]))
            
            # Linear constraint: expected return >= target
            expected_returns = np.array([stock.expected_return for stock in problem.stocks])
            A_ineq = expected_returns.reshape(1, -1)
            constraints.append(LinearConstraint(A_ineq, [problem.target_return], [np.inf]))
        
        # Optimize
        result = minimize(
            fun=lambda x: problem.evaluate_solution(x),
            x0=initial_solution,
            method='trust-constr',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.max_iterations,
                'gtol': self.tolerance,
                'disp': False
            }
        )
        
        return result
    
    def _create_fallback_result(self, problem, initial_solution: np.ndarray) -> Dict[str, Any]:
        """Create a fallback result when optimization fails."""
        return {
            'x': initial_solution,
            'fun': problem.evaluate_solution(initial_solution),
            'success': False,
            'message': 'Optimization failed, using initial solution',
            'nit': 0
        }
    
    def _check_constraints(self, problem, solution: np.ndarray) -> List[str]:
        """Check constraint violations in the solution."""
        violations = []
        
        # Check weight sum
        weight_sum = np.sum(solution)
        if abs(weight_sum - 1.0) > 1e-6:
            violations.append(f"Weights sum to {weight_sum:.6f}, should be 1.0")
        
        # Check non-negativity
        negative_weights = solution[solution < -1e-6]
        if len(negative_weights) > 0:
            violations.append(f"Found {len(negative_weights)} negative weights")
        
        # Check return constraint
        expected_returns = np.array([stock.expected_return for stock in problem.stocks])
        portfolio_return = np.dot(solution, expected_returns)
        if portfolio_return < problem.target_return - 1e-6:
            violations.append(f"Portfolio return {portfolio_return:.4f} below target {problem.target_return:.4f}")
        
        return violations

    def _create_portfolio_plots(self, problem, weights: np.ndarray, solution_info: Dict[str, Any], algorithm: str):
        """Create helpful visualization plots for portfolio optimization results."""
        try:
            # Set up the plotting style
            plt.style.use('default')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Portfolio Optimization Results ({algorithm.upper()})', fontsize=16, fontweight='bold')

            # Plot 1: Portfolio Allocation Pie Chart
            stock_symbols = [stock.symbol for stock in problem.stocks]
            allocations = solution_info['stock_allocations']

            # Filter out very small allocations for cleaner visualization
            min_allocation = 0.01  # 1%
            significant_allocations = {k: v for k, v in allocations.items() if v >= min_allocation}
            other_allocation = sum(v for v in allocations.values() if v < min_allocation)

            if other_allocation > 0:
                significant_allocations['Others'] = other_allocation

            colors = plt.cm.Set3(np.linspace(0, 1, len(significant_allocations)))
            wedges, texts, autotexts = ax1.pie(
                significant_allocations.values(),
                labels=significant_allocations.keys(),
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            ax1.set_title('Portfolio Allocation', fontweight='bold')

            # Plot 2: Risk vs Return Scatter Plot
            expected_returns = np.array([stock.expected_return for stock in problem.stocks])
            volatilities = np.array([stock.volatility for stock in problem.stocks])

            # Create scatter plot of individual stocks
            scatter = ax2.scatter(volatilities * 100, expected_returns * 100,
                                s=100, alpha=0.6, c='lightblue', edgecolors='navy')

            # Add portfolio point
            portfolio_vol = solution_info['portfolio_volatility'] * 100
            portfolio_ret = solution_info['portfolio_return'] * 100
            ax2.scatter(portfolio_vol, portfolio_ret, s=200, c='red', marker='*',
                       edgecolors='darkred', linewidth=2, label='Optimized Portfolio')

            # Add target return line
            target_ret = problem.target_return * 100
            ax2.axhline(y=target_ret, color='green', linestyle='--', alpha=0.7,
                       label=f'Target Return ({target_ret:.1f}%)')

            # Annotate stocks
            for i, stock in enumerate(problem.stocks):
                if weights[i] > 0.05:  # Only annotate stocks with >5% allocation
                    ax2.annotate(stock.symbol,
                               (volatilities[i] * 100, expected_returns[i] * 100),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)

            ax2.set_xlabel('Risk (Volatility %)', fontweight='bold')
            ax2.set_ylabel('Expected Return (%)', fontweight='bold')
            ax2.set_title('Risk vs Return Analysis', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Portfolio Metrics Bar Chart
            metrics = {
                'Portfolio Return': f"{solution_info['portfolio_return']:.1%}",
                'Portfolio Risk': f"{solution_info['portfolio_volatility']:.1%}",
                'Sharpe Ratio': f"{solution_info['sharpe_ratio']:.2f}",
                'Target Return': f"{problem.target_return:.1%}"
            }

            metric_values = [
                solution_info['portfolio_return'] * 100,
                solution_info['portfolio_volatility'] * 100,
                solution_info['sharpe_ratio'] * 10,  # Scale for visibility
                problem.target_return * 100
            ]

            colors_metrics = ['green', 'orange', 'blue', 'gray']
            bars = ax3.bar(range(len(metrics)), metric_values, color=colors_metrics, alpha=0.7)
            ax3.set_xticks(range(len(metrics)))
            ax3.set_xticklabels(metrics.keys(), rotation=45, ha='right')
            ax3.set_ylabel('Value (%)', fontweight='bold')
            ax3.set_title('Portfolio Performance Metrics', fontweight='bold')

            # Add value labels on bars
            for i, (bar, value, label) in enumerate(zip(bars, metric_values, metrics.values())):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        label, ha='center', va='bottom', fontweight='bold')

            # Plot 4: Stock Allocation Bar Chart
            # Sort stocks by allocation for better visualization
            sorted_allocations = sorted(allocations.items(), key=lambda x: x[1], reverse=True)
            stocks_sorted = [item[0] for item in sorted_allocations]
            weights_sorted = [item[1] * 100 for item in sorted_allocations]

            colors_stocks = plt.cm.viridis(np.linspace(0, 1, len(stocks_sorted)))
            bars_stocks = ax4.bar(range(len(stocks_sorted)), weights_sorted, color=colors_stocks)
            ax4.set_xticks(range(len(stocks_sorted)))
            ax4.set_xticklabels(stocks_sorted, rotation=45, ha='right')
            ax4.set_ylabel('Allocation (%)', fontweight='bold')
            ax4.set_title('Individual Stock Allocations', fontweight='bold')

            # Add percentage labels on bars
            for bar, weight in zip(bars_stocks, weights_sorted):
                if weight > 1:  # Only label significant allocations
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{weight:.1f}%', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            plt.show()

            # Print summary information
            print("\n" + "="*60)
            print("PORTFOLIO OPTIMIZATION SUMMARY")
            print("="*60)
            print(f"Algorithm Used: {algorithm.upper()}")
            print(f"Number of Stocks: {len(problem.stocks)}")
            print(f"Target Return: {problem.target_return:.1%}")
            print(f"Achieved Return: {solution_info['portfolio_return']:.2%}")
            print(f"Portfolio Risk: {solution_info['portfolio_volatility']:.2%}")
            print(f"Sharpe Ratio: {solution_info['sharpe_ratio']:.3f}")
            print(f"Feasible Solution: {'Yes' if solution_info['is_feasible'] else 'No'}")

            print(f"\nTop 5 Stock Allocations:")
            for i, (stock, weight) in enumerate(sorted_allocations[:5]):
                print(f"  {i+1}. {stock}: {weight:.1%}")

            if len(sorted_allocations) > 5:
                remaining_weight = sum(weight for _, weight in sorted_allocations[5:])
                print(f"  Others: {remaining_weight:.1%}")

            print("="*60)

        except Exception as e:
            print(f"Warning: Could not create plots - {e}")
            print("Note: Plotting requires matplotlib. Install with: pip install matplotlib")


# Export the optimizer class for qubots framework
__all__ = ['PortfolioOptimizationOptimizer', 'PortfolioOptimizationResult']
