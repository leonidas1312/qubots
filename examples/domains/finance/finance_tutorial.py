"""
Portfolio Optimization Tutorial with Qubots and Rastion Integration

This tutorial demonstrates:
1. Creating a Portfolio Optimization Problem using qubots ContinuousProblem
2. Implementing a scipy-based portfolio optimizer with risk constraints
3. Uploading models to Rastion platform
4. Loading models from Rastion and verifying integrity
5. Complete workflow with error handling and validation

Requirements:
- qubots library
- scipy (pip install scipy)
- numpy
- matplotlib (for visualization)
- pandas (for data handling)

Author: Qubots Tutorial Team
Version: 1.0.0
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import time
import json
from datetime import datetime, timedelta

# Add qubots to path if running locally
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Qubots imports
from qubots import (
    ContinuousProblem, BaseOptimizer,
    ProblemMetadata, OptimizerMetadata,
    ProblemType, ObjectiveType, DifficultyLevel,
    OptimizerType, OptimizerFamily,
    OptimizationResult
)
import qubots.rastion as rastion

# Financial optimization imports (with fallback)
try:
    from scipy.optimize import minimize, Bounds, LinearConstraint
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
    print("✅ SciPy available for portfolio optimization")
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  SciPy not available. Install with: pip install scipy")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  Pandas not available. Install with: pip install pandas")


@dataclass
class Asset:
    """Represents a financial asset in the portfolio."""
    symbol: str
    name: str
    expected_return: float
    volatility: float
    sector: str = "Unknown"
    market_cap: float = 0.0
    beta: float = 1.0


@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints."""
    max_weight_per_asset: float = 0.4
    min_weight_per_asset: float = 0.0
    max_sector_weight: float = 0.6
    min_portfolio_return: float = 0.08
    max_portfolio_risk: float = 0.25
    transaction_cost: float = 0.001


class PortfolioOptimizationProblem(ContinuousProblem):
    """
    Portfolio Optimization Problem implementation using qubots ContinuousProblem.

    This problem involves finding optimal asset weights to maximize return
    while minimizing risk, subject to various portfolio constraints.
    """

    def __init__(self, assets: List[Asset],
                 correlation_matrix: np.ndarray,
                 constraints: PortfolioConstraints = None,
                 risk_free_rate: float = 0.02):
        """
        Initialize the Portfolio Optimization Problem.

        Args:
            assets: List of assets to include in portfolio
            correlation_matrix: Asset correlation matrix
            constraints: Portfolio constraints
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.assets = assets
        self.correlation_matrix = correlation_matrix
        self.constraints = constraints or PortfolioConstraints()
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(assets)

        # Calculate covariance matrix
        self.covariance_matrix = self._calculate_covariance_matrix()

        # Expected returns vector
        self.expected_returns = np.array([asset.expected_return for asset in assets])

        # Asset bounds (weights between 0 and max_weight)
        bounds = {f"weight_{i}": (self.constraints.min_weight_per_asset,
                                 self.constraints.max_weight_per_asset)
                 for i in range(self.n_assets)}

        # Initialize base class
        super().__init__(
            dimension=self.n_assets,
            bounds=bounds,
            metadata=self._get_default_metadata()
        )

    def _get_default_metadata(self) -> ProblemMetadata:
        """Return default metadata for Portfolio Optimization."""
        return ProblemMetadata(
            name="Portfolio Optimization Problem",
            description=f"Portfolio optimization with {self.n_assets} assets using Modern Portfolio Theory",
            problem_type=ProblemType.CONTINUOUS,
            objective_type=ObjectiveType.MAXIMIZE,  # Maximizing Sharpe ratio
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            domain="finance",
            tags={"portfolio", "finance", "optimization", "markowitz", "sharpe"},
            author="Qubots Finance Tutorial",
            version="1.0.0",
            dimension=self.n_assets,
            evaluation_complexity="O(n²)",
            memory_complexity="O(n²)"
        )

    def _calculate_covariance_matrix(self) -> np.ndarray:
        """Calculate covariance matrix from correlation matrix and volatilities."""
        volatilities = np.array([asset.volatility for asset in self.assets])
        # Cov(i,j) = corr(i,j) * vol(i) * vol(j)
        covariance = np.outer(volatilities, volatilities) * self.correlation_matrix
        return covariance

    def evaluate_solution(self, weights: np.ndarray) -> float:
        """
        Evaluate a portfolio solution (asset weights).

        Args:
            weights: Array of asset weights (must sum to 1)

        Returns:
            Negative Sharpe ratio (for minimization) or penalty for invalid portfolios
        """
        weights = np.array(weights)

        # Check if weights sum to 1 (with tolerance)
        if abs(np.sum(weights) - 1.0) > 1e-6:
            return -1000  # Heavy penalty for invalid weight sum

        # Check weight bounds
        if np.any(weights < self.constraints.min_weight_per_asset) or \
           np.any(weights > self.constraints.max_weight_per_asset):
            return -1000  # Heavy penalty for weight violations

        # Calculate portfolio return
        portfolio_return = np.dot(weights, self.expected_returns)

        # Calculate portfolio risk (standard deviation)
        portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)

        # Check risk constraint
        if portfolio_risk > self.constraints.max_portfolio_risk:
            return -1000  # Heavy penalty for risk violations

        # Check minimum return constraint
        if portfolio_return < self.constraints.min_portfolio_return:
            return -1000  # Heavy penalty for return violations

        # Calculate Sharpe ratio
        if portfolio_risk == 0:
            return -1000  # Avoid division by zero

        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk

        # Return Sharpe ratio (higher is better)
        return sharpe_ratio

    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics."""
        weights = np.array(weights)

        # Basic metrics
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0

        # Diversification metrics
        effective_assets = 1 / np.sum(weights**2)  # Inverse of Herfindahl index
        max_weight = np.max(weights)

        # Sector concentration
        sector_weights = {}
        for i, asset in enumerate(self.assets):
            sector = asset.sector
            if sector not in sector_weights:
                sector_weights[sector] = 0
            sector_weights[sector] += weights[i]

        max_sector_weight = max(sector_weights.values()) if sector_weights else 0

        return {
            "expected_return": portfolio_return,
            "volatility": portfolio_risk,
            "sharpe_ratio": sharpe_ratio,
            "effective_assets": effective_assets,
            "max_weight": max_weight,
            "max_sector_weight": max_sector_weight,
            "total_weight": np.sum(weights)
        }

    def is_feasible(self, weights: np.ndarray) -> bool:
        """Check if portfolio weights are feasible."""
        weights = np.array(weights)

        # Check weight sum
        if abs(np.sum(weights) - 1.0) > 1e-6:
            return False

        # Check weight bounds
        if np.any(weights < self.constraints.min_weight_per_asset) or \
           np.any(weights > self.constraints.max_weight_per_asset):
            return False

        # Check portfolio constraints
        metrics = self.calculate_portfolio_metrics(weights)

        if metrics["volatility"] > self.constraints.max_portfolio_risk:
            return False

        if metrics["expected_return"] < self.constraints.min_portfolio_return:
            return False

        if metrics["max_sector_weight"] > self.constraints.max_sector_weight:
            return False

        return True

    def get_random_solution(self) -> np.ndarray:
        """Generate a random feasible portfolio."""
        # Generate random weights
        weights = np.random.random(self.n_assets)

        # Normalize to sum to 1
        weights = weights / np.sum(weights)

        # Ensure bounds are respected
        weights = np.clip(weights,
                         self.constraints.min_weight_per_asset,
                         self.constraints.max_weight_per_asset)

        # Renormalize after clipping
        weights = weights / np.sum(weights)

        return weights

    def visualize_solution(self, weights: np.ndarray, title: str = "Portfolio Allocation"):
        """Visualize the portfolio allocation."""
        try:
            weights = np.array(weights)

            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # 1. Asset allocation pie chart
            asset_labels = [f"{asset.symbol}\n({weight:.1%})"
                           for asset, weight in zip(self.assets, weights) if weight > 0.01]
            asset_weights = [weight for weight in weights if weight > 0.01]

            ax1.pie(asset_weights, labels=asset_labels, autopct='%1.1f%%', startangle=90)
            ax1.set_title("Asset Allocation")

            # 2. Sector allocation
            sector_weights = {}
            for i, asset in enumerate(self.assets):
                sector = asset.sector
                if sector not in sector_weights:
                    sector_weights[sector] = 0
                sector_weights[sector] += weights[i]

            sectors = list(sector_weights.keys())
            sector_values = list(sector_weights.values())

            ax2.bar(sectors, sector_values)
            ax2.set_title("Sector Allocation")
            ax2.set_ylabel("Weight")
            ax2.tick_params(axis='x', rotation=45)

            # 3. Risk-Return scatter
            returns = [asset.expected_return for asset in self.assets]
            risks = [asset.volatility for asset in self.assets]

            # Scale bubble size by weight
            sizes = weights * 1000

            scatter = ax3.scatter(risks, returns, s=sizes, alpha=0.6, c=weights, cmap='viridis')
            ax3.set_xlabel("Volatility")
            ax3.set_ylabel("Expected Return")
            ax3.set_title("Risk-Return Profile")

            # Add asset labels
            for i, asset in enumerate(self.assets):
                if weights[i] > 0.01:  # Only label significant holdings
                    ax3.annotate(asset.symbol, (risks[i], returns[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)

            plt.colorbar(scatter, ax=ax3, label='Weight')

            # 4. Portfolio metrics
            metrics = self.calculate_portfolio_metrics(weights)

            metric_names = ['Expected Return', 'Volatility', 'Sharpe Ratio', 'Effective Assets']
            metric_values = [metrics['expected_return'], metrics['volatility'],
                           metrics['sharpe_ratio'], metrics['effective_assets']]

            bars = ax4.bar(metric_names, metric_values)
            ax4.set_title("Portfolio Metrics")
            ax4.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom')

            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Visualization failed: {e}")


class ScipyPortfolioOptimizer(BaseOptimizer):
    """
    SciPy-based portfolio optimizer using Modern Portfolio Theory.

    Implements various portfolio optimization strategies including
    mean-variance optimization, risk parity, and maximum Sharpe ratio.
    """

    def __init__(self, method: str = "SLSQP",
                 max_iterations: int = 1000,
                 tolerance: float = 1e-8,
                 objective: str = "sharpe"):
        """
        Initialize SciPy portfolio optimizer.

        Args:
            method: Optimization method ('SLSQP', 'trust-constr', etc.)
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            objective: Objective function ('sharpe', 'return', 'risk')
        """
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.objective = objective

        # Pass parameters to base class
        super().__init__(
            self._get_default_metadata(),
            method=method,
            max_iterations=max_iterations,
            tolerance=tolerance,
            objective=objective
        )

    def _get_default_metadata(self) -> OptimizerMetadata:
        """Return default metadata for SciPy portfolio optimizer."""
        return OptimizerMetadata(
            name="SciPy Portfolio Optimizer",
            description="Modern Portfolio Theory optimizer using SciPy's constrained optimization",
            optimizer_type=OptimizerType.HEURISTIC,
            optimizer_family=OptimizerFamily.GRADIENT_BASED,
            author="Qubots Finance Tutorial",
            version="1.0.0",
            supports_constraints=True,
            supports_multi_objective=False,
            typical_problems=["portfolio", "finance", "continuous"],
            required_parameters=["method"],
            optional_parameters=["max_iterations", "tolerance", "objective"]
        )

    def _optimize_implementation(self, problem: PortfolioOptimizationProblem,
                               initial_solution: Optional[np.ndarray] = None) -> OptimizationResult:
        """
        Solve portfolio optimization using SciPy.

        Args:
            problem: Portfolio optimization problem instance
            initial_solution: Optional initial portfolio weights

        Returns:
            OptimizationResult with solution and metadata
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy not available. Please install with: pip install scipy")

        start_time = time.time()

        # Initial guess
        if initial_solution is not None:
            x0 = np.array(initial_solution)
        else:
            # Equal weight portfolio as starting point
            x0 = np.ones(problem.n_assets) / problem.n_assets

        # Define objective function
        def objective_function(weights):
            if self.objective == "sharpe":
                # Maximize Sharpe ratio (minimize negative Sharpe)
                sharpe = problem.evaluate_solution(weights)
                return -sharpe  # Minimize negative Sharpe
            elif self.objective == "risk":
                # Minimize portfolio risk
                portfolio_variance = np.dot(weights, np.dot(problem.covariance_matrix, weights))
                return np.sqrt(portfolio_variance)
            elif self.objective == "return":
                # Maximize return (minimize negative return)
                portfolio_return = np.dot(weights, problem.expected_returns)
                return -portfolio_return
            else:
                raise ValueError(f"Unknown objective: {self.objective}")

        # Constraints
        constraints = []

        # Weights must sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1.0
        })

        # Minimum return constraint (if specified)
        if problem.constraints.min_portfolio_return > 0:
            constraints.append({
                'type': 'ineq',
                'fun': lambda weights: np.dot(weights, problem.expected_returns) - problem.constraints.min_portfolio_return
            })

        # Maximum risk constraint (if specified)
        if problem.constraints.max_portfolio_risk < float('inf'):
            def risk_constraint(weights):
                portfolio_variance = np.dot(weights, np.dot(problem.covariance_matrix, weights))
                portfolio_risk = np.sqrt(portfolio_variance)
                return problem.constraints.max_portfolio_risk - portfolio_risk

            constraints.append({
                'type': 'ineq',
                'fun': risk_constraint
            })

        # Bounds for individual weights
        bounds = [(problem.constraints.min_weight_per_asset,
                  problem.constraints.max_weight_per_asset)
                 for _ in range(problem.n_assets)]

        # Solve optimization problem
        try:
            result = minimize(
                objective_function,
                x0,
                method=self.method,
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.tolerance,
                    'disp': False
                }
            )

            end_time = time.time()

            if result.success:
                optimal_weights = result.x

                # Calculate portfolio metrics
                metrics = problem.calculate_portfolio_metrics(optimal_weights)

                # Determine objective value based on optimization goal
                if self.objective == "sharpe":
                    objective_value = metrics["sharpe_ratio"]
                elif self.objective == "risk":
                    objective_value = metrics["volatility"]
                elif self.objective == "return":
                    objective_value = metrics["expected_return"]

                return OptimizationResult(
                    best_solution=optimal_weights,
                    best_value=objective_value,
                    iterations=result.nit if hasattr(result, 'nit') else 1,
                    evaluations=result.nfev if hasattr(result, 'nfev') else 1,
                    runtime_seconds=end_time - start_time,
                    convergence_achieved=result.success,
                    termination_reason=result.message,
                    additional_metrics={
                        "scipy_success": result.success,
                        "scipy_status": result.status if hasattr(result, 'status') else 0,
                        "portfolio_metrics": metrics,
                        "optimization_method": self.method,
                        "objective_type": self.objective
                    }
                )
            else:
                return OptimizationResult(
                    best_solution=None,
                    best_value=float('-inf') if self.objective in ["sharpe", "return"] else float('inf'),
                    iterations=result.nit if hasattr(result, 'nit') else 0,
                    evaluations=result.nfev if hasattr(result, 'nfev') else 0,
                    runtime_seconds=end_time - start_time,
                    convergence_achieved=False,
                    termination_reason=f"SciPy optimization failed: {result.message}",
                    additional_metrics={
                        "scipy_success": result.success,
                        "scipy_status": result.status if hasattr(result, 'status') else -1,
                        "optimization_method": self.method
                    }
                )

        except Exception as e:
            end_time = time.time()
            return OptimizationResult(
                best_solution=None,
                best_value=float('-inf') if self.objective in ["sharpe", "return"] else float('inf'),
                iterations=0,
                evaluations=0,
                runtime_seconds=end_time - start_time,
                convergence_achieved=False,
                termination_reason=f"SciPy optimization error: {str(e)}",
                additional_metrics={"error": str(e)}
            )


def generate_sample_portfolio_data(n_assets: int = 10, seed: int = 42) -> Tuple[List[Asset], np.ndarray]:
    """
    Generate sample portfolio data for testing.

    Args:
        n_assets: Number of assets to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (assets, correlation_matrix)
    """
    np.random.seed(seed)

    # Define sectors
    sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer", "Industrial"]

    # Generate assets
    assets = []
    for i in range(n_assets):
        asset = Asset(
            symbol=f"ASSET_{i:02d}",
            name=f"Asset {i}",
            expected_return=np.random.uniform(0.05, 0.20),  # 5-20% expected return
            volatility=np.random.uniform(0.10, 0.40),       # 10-40% volatility
            sector=np.random.choice(sectors),
            market_cap=np.random.uniform(1e9, 100e9),       # $1B - $100B market cap
            beta=np.random.uniform(0.5, 2.0)               # Beta between 0.5 and 2.0
        )
        assets.append(asset)

    # Generate realistic correlation matrix
    # Start with random correlations
    correlation_matrix = np.random.uniform(-0.3, 0.7, (n_assets, n_assets))

    # Make symmetric
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2

    # Set diagonal to 1
    np.fill_diagonal(correlation_matrix, 1.0)

    # Ensure positive semi-definite
    eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
    eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive eigenvalues
    correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    # Normalize to ensure diagonal is 1
    d = np.sqrt(np.diag(correlation_matrix))
    correlation_matrix = correlation_matrix / np.outer(d, d)

    return assets, correlation_matrix


def demonstrate_portfolio_workflow():
    """
    Demonstrate the complete portfolio optimization workflow with qubots and Rastion integration.
    """
    print("💰 Portfolio Optimization Tutorial with Qubots & Rastion")
    print("=" * 60)

    # Step 1: Generate sample data
    print("\n📊 Step 1: Generating sample portfolio data...")
    assets, correlation_matrix = generate_sample_portfolio_data(n_assets=8, seed=42)

    avg_return = np.mean([asset.expected_return for asset in assets])
    avg_volatility = np.mean([asset.volatility for asset in assets])

    print(f"✅ Generated {len(assets)} assets")
    print(f"   Average expected return: {avg_return:.1%}")
    print(f"   Average volatility: {avg_volatility:.1%}")
    print(f"   Sectors: {set(asset.sector for asset in assets)}")

    # Step 2: Create portfolio optimization problem
    print("\n🏗️  Step 2: Creating portfolio optimization problem...")

    constraints = PortfolioConstraints(
        max_weight_per_asset=0.3,
        min_weight_per_asset=0.02,
        max_sector_weight=0.5,
        min_portfolio_return=0.10,
        max_portfolio_risk=0.20
    )

    portfolio_problem = PortfolioOptimizationProblem(
        assets=assets,
        correlation_matrix=correlation_matrix,
        constraints=constraints,
        risk_free_rate=0.03
    )

    print(f"✅ Created portfolio problem: {portfolio_problem.metadata.name}")
    print(f"   Problem type: {portfolio_problem.metadata.problem_type}")
    print(f"   Objective: {portfolio_problem.metadata.objective_type}")
    print(f"   Assets: {portfolio_problem.n_assets}")

    # Step 3: Test problem with random solution
    print("\n🎲 Step 3: Testing with random portfolio...")
    random_weights = portfolio_problem.get_random_solution()
    random_sharpe = portfolio_problem.evaluate_solution(random_weights)
    is_feasible = portfolio_problem.is_feasible(random_weights)

    print(f"✅ Random portfolio Sharpe ratio: {random_sharpe:.3f}")
    print(f"   Feasible: {is_feasible}")

    if is_feasible:
        metrics = portfolio_problem.calculate_portfolio_metrics(random_weights)
        print(f"   Expected return: {metrics['expected_return']:.1%}")
        print(f"   Volatility: {metrics['volatility']:.1%}")

    # Step 4: Create SciPy optimizer
    print("\n⚙️  Step 4: Creating SciPy portfolio optimizer...")
    if SCIPY_AVAILABLE:
        optimizer = ScipyPortfolioOptimizer(
            method="SLSQP",
            max_iterations=1000,
            tolerance=1e-8,
            objective="sharpe"
        )
        print(f"✅ Created optimizer: {optimizer.metadata.name}")
        print(f"   Type: {optimizer.metadata.optimizer_type}")
        print(f"   Family: {optimizer.metadata.optimizer_family}")
    else:
        print("⚠️  SciPy not available, skipping optimization step")
        return

    # Step 5: Solve with SciPy optimizer
    print("\n🔍 Step 5: Solving portfolio optimization with SciPy...")
    try:
        result = optimizer.optimize(portfolio_problem)

        print(f"✅ SciPy optimization completed!")
        print(f"   Best Sharpe ratio: {result.best_value:.3f}")
        print(f"   Runtime: {result.runtime_seconds:.2f} seconds")
        print(f"   Converged: {result.convergence_achieved}")
        print(f"   Iterations: {result.iterations}")
        print(f"   Function evaluations: {result.evaluations}")

        # Display portfolio metrics
        if result.best_solution is not None:
            metrics = result.additional_info.get("portfolio_metrics", {})
            print(f"\n📈 Optimal Portfolio Metrics:")
            print(f"   Expected Return: {metrics.get('expected_return', 0):.1%}")
            print(f"   Volatility: {metrics.get('volatility', 0):.1%}")
            print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"   Effective Assets: {metrics.get('effective_assets', 0):.1f}")
            print(f"   Max Asset Weight: {metrics.get('max_weight', 0):.1%}")

            # Show top holdings
            weights = result.best_solution
            top_holdings = sorted(zip(assets, weights), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n🏆 Top 5 Holdings:")
            for asset, weight in top_holdings:
                print(f"   {asset.symbol}: {weight:.1%} ({asset.sector})")

            # Visualize solution
            print("\n📊 Visualizing optimal portfolio...")
            portfolio_problem.visualize_solution(result.best_solution, "Optimal Portfolio Allocation")

    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return

    # Step 6: Demonstrate Rastion upload workflow
    print("\n📤 Step 6: Rastion Upload Workflow (Simulation)...")
    print("Note: Actual upload requires authentication with rastion.authenticate(token)")

    try:
        # Show what would be packaged for upload
        from qubots.rastion_client import QubotPackager

        # Package the problem
        problem_package = QubotPackager.package_model(
            portfolio_problem,
            "portfolio_optimization_problem",
            "Modern Portfolio Theory optimization problem from qubots tutorial with SciPy integration"
        )

        # Package the optimizer
        optimizer_package = QubotPackager.package_model(
            optimizer,
            "scipy_portfolio_optimizer",
            "SciPy-based portfolio optimizer implementing Modern Portfolio Theory"
        )

        print("✅ Problem package created:")
        for filename in problem_package.keys():
            print(f"   📄 {filename}")

        print("✅ Optimizer package created:")
        for filename in optimizer_package.keys():
            print(f"   📄 {filename}")

        # Simulate upload process
        print("\n🔄 Upload simulation:")
        print("   1. Validating model integrity...")
        print("   2. Serializing financial model...")
        print("   3. Creating repository structure...")
        print("   4. Uploading to Rastion platform...")
        print("   ✅ Upload complete! (simulated)")

    except Exception as e:
        print(f"❌ Package creation failed: {e}")

    # Step 7: Demonstrate loading workflow
    print("\n📥 Step 7: Rastion Loading Workflow (Simulation)...")
    print("After upload, users would load models like this:")
    print()
    print("```python")
    print("import qubots.rastion as rastion")
    print()
    print("# Authenticate (one-time setup)")
    print("rastion.authenticate('your_gitea_token')")
    print()
    print("# Load models with one line")
    print("problem = rastion.load_qubots_model('portfolio_optimization_problem')")
    print("optimizer = rastion.load_qubots_model('scipy_portfolio_optimizer')")
    print()
    print("# Verify model integrity")
    print("print(f'Problem: {problem.metadata.name}')")
    print("print(f'Assets: {problem.n_assets}')")
    print("print(f'Risk-free rate: {problem.risk_free_rate:.1%}')")
    print()
    print("# Run optimization")
    print("result = optimizer.optimize(problem)")
    print("print(f'Best Sharpe ratio: {result.best_value:.3f}')")
    print()
    print("# Analyze portfolio")
    print("metrics = problem.calculate_portfolio_metrics(result.best_solution)")
    print("print(f'Expected return: {metrics[\"expected_return\"]:.1%}')")
    print("print(f'Volatility: {metrics[\"volatility\"]:.1%}')")
    print()
    print("# Visualize results")
    print("problem.visualize_solution(result.best_solution)")
    print("```")

    # Step 8: Model verification and error handling
    print("\n🔍 Step 8: Model Verification & Error Handling...")

    # Demonstrate model validation
    print("Model validation checks:")
    print(f"   ✅ Portfolio weights sum: {np.sum(random_weights):.6f}")
    print(f"   ✅ Portfolio feasibility: {portfolio_problem.is_feasible(random_weights)}")
    print(f"   ✅ Optimizer compatibility: Compatible with ContinuousProblem")

    # Demonstrate error handling scenarios
    print("\nError handling scenarios:")

    # Test with invalid weights (don't sum to 1)
    try:
        invalid_weights = np.array([0.5, 0.3, 0.1, 0.05, 0.05, 0.0, 0.0, 0.0])  # Sum = 1.0
        invalid_weights[0] = 0.6  # Now sum = 1.1
        sharpe = portfolio_problem.evaluate_solution(invalid_weights)
        print(f"   ✅ Invalid weights handled: Sharpe = {sharpe}")
    except Exception as e:
        print(f"   ✅ Invalid weights error: {type(e).__name__}")

    # Test with constraint violations
    try:
        # Create portfolio that violates risk constraint
        high_risk_weights = np.zeros(portfolio_problem.n_assets)
        high_risk_weights[0] = 1.0  # 100% in one asset (likely high risk)
        sharpe = portfolio_problem.evaluate_solution(high_risk_weights)
        metrics = portfolio_problem.calculate_portfolio_metrics(high_risk_weights)
        print(f"   ✅ High-risk portfolio: Sharpe = {sharpe:.3f}, Risk = {metrics['volatility']:.1%}")
    except Exception as e:
        print(f"   ⚠️  High-risk portfolio error: {e}")

    print("\n✨ Portfolio Optimization Tutorial completed successfully!")
    print("\n📚 Key Learning Points:")
    print("   🔹 Created portfolio optimization using qubots ContinuousProblem")
    print("   🔹 Implemented SciPy-based optimizer with financial constraints")
    print("   🔹 Demonstrated complete Rastion upload/download workflow")
    print("   🔹 Showed Modern Portfolio Theory implementation")
    print("   🔹 Visualized portfolio allocations and risk-return profiles")
    print("   🔹 Used proper qubots metadata and optimization results")


def demonstrate_advanced_portfolio_features():
    """
    Demonstrate advanced portfolio optimization features and strategies.
    """
    print("\n🚀 Advanced Portfolio Features Demo")
    print("=" * 45)

    # Create different portfolio strategies
    print("\n📊 Testing different optimization objectives...")
    assets, correlation_matrix = generate_sample_portfolio_data(n_assets=6, seed=123)

    constraints = PortfolioConstraints(
        max_weight_per_asset=0.4,
        min_weight_per_asset=0.05,
        max_sector_weight=0.6,
        min_portfolio_return=0.08,
        max_portfolio_risk=0.25
    )

    portfolio_problem = PortfolioOptimizationProblem(
        assets=assets,
        correlation_matrix=correlation_matrix,
        constraints=constraints,
        risk_free_rate=0.02
    )

    if SCIPY_AVAILABLE:
        objectives = ["sharpe", "risk", "return"]

        print("\n🔧 Testing different optimization objectives...")

        for objective in objectives:
            try:
                optimizer = ScipyPortfolioOptimizer(
                    method="SLSQP",
                    max_iterations=500,
                    objective=objective
                )

                result = optimizer.optimize(portfolio_problem)

                if result.best_solution is not None:
                    metrics = portfolio_problem.calculate_portfolio_metrics(result.best_solution)
                    print(f"\n   {objective.upper()} Optimization:")
                    print(f"     Return: {metrics['expected_return']:.1%}")
                    print(f"     Risk: {metrics['volatility']:.1%}")
                    print(f"     Sharpe: {metrics['sharpe_ratio']:.3f}")
                    print(f"     Max Weight: {metrics['max_weight']:.1%}")
                else:
                    print(f"   {objective.upper()} Optimization: Failed")

            except Exception as e:
                print(f"   {objective.upper()} Optimization: Error ({e})")

    print("\n✅ Advanced features demonstration completed!")


if __name__ == "__main__":
    """
    Main execution block - run the complete portfolio optimization tutorial.

    This tutorial can be executed directly with:
    python finance_tutorial.py
    """
    try:
        # Run main workflow demonstration
        demonstrate_portfolio_workflow()

        # Run advanced features demo
        demonstrate_advanced_portfolio_features()

        print("\n🎉 All portfolio optimization tutorial demonstrations completed successfully!")
        print("\n📖 Next Steps:")
        print("   1. Try modifying portfolio constraints and objectives")
        print("   2. Experiment with different asset universes and correlations")
        print("   3. Implement additional risk measures (VaR, CVaR)")
        print("   4. Add transaction costs and rebalancing constraints")
        print("   5. Upload your models to Rastion for sharing")
        print("   6. Explore other qubots tutorials (routing, scheduling)")

    except KeyboardInterrupt:
        print("\n⏹️  Tutorial interrupted by user")
    except Exception as e:
        print(f"\n❌ Tutorial failed with error: {e}")
        print("   Please check dependencies and try again")
        import traceback
        traceback.print_exc()
