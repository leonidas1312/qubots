"""
Portfolio Optimization Problem for Qubots Framework

This problem implements the Markowitz portfolio optimization model that reads
stock data from CSV files and optimizes portfolio allocation to minimize risk
while meeting return constraints.

The problem accepts CSV data with columns:
- symbol: Stock symbol/identifier
- expected_return: Expected annual return (as decimal, e.g., 0.12 for 12%)
- volatility: Annual volatility/standard deviation (as decimal)
- Additional columns for correlation data (optional)

Features:
- CSV data input for business-relevant stock information
- Markowitz mean-variance optimization model
- Configurable risk tolerance and return targets
- Support for correlation matrix or simplified uncorrelated model
- Comprehensive solution validation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import io

from qubots import BaseProblem, ProblemMetadata, ProblemType, ObjectiveType, DifficultyLevel


@dataclass
class StockData:
    """Data structure for individual stock information."""
    symbol: str
    expected_return: float
    volatility: float
    
    def __post_init__(self):
        """Validate stock data."""
        if self.expected_return < -1.0 or self.expected_return > 2.0:
            raise ValueError(f"Expected return {self.expected_return} seems unrealistic")
        if self.volatility < 0.0 or self.volatility > 2.0:
            raise ValueError(f"Volatility {self.volatility} must be non-negative and reasonable")


class PortfolioOptimizationProblem(BaseProblem):
    """
    Portfolio Optimization Problem using Markowitz mean-variance model.
    
    Minimizes portfolio risk (variance) subject to:
    - Portfolio weights sum to 1 (fully invested)
    - Expected portfolio return meets minimum target
    - All weights are non-negative (no short selling)
    """
    
    def __init__(self, 
                 csv_data: str = None,
                 csv_file_path: str = None,
                 target_return: float = 0.10,
                 risk_free_rate: float = 0.02,
                 correlation_matrix: Optional[np.ndarray] = None,
                 **kwargs):
        """
        Initialize portfolio optimization problem.
        
        Args:
            csv_data: CSV content as string
            csv_file_path: Path to CSV file (alternative to csv_data)
            target_return: Minimum required portfolio return (default: 10%)
            risk_free_rate: Risk-free rate for Sharpe ratio calculations (default: 2%)
            correlation_matrix: Optional correlation matrix between stocks
            **kwargs: Additional parameters
        """
        self.target_return = target_return
        self.risk_free_rate = risk_free_rate
        
        # Load stock data from CSV
        self.stocks = self._load_stock_data(csv_data, csv_file_path)
        self.n_stocks = len(self.stocks)
        
        # Build covariance matrix
        self.covariance_matrix = self._build_covariance_matrix(correlation_matrix)
        
        # Set up problem metadata
        metadata = self._get_default_metadata()
        super().__init__(metadata, **kwargs)
    
    def _get_default_metadata(self) -> ProblemMetadata:
        """Return default metadata for portfolio optimization problem."""
        return ProblemMetadata(
            name="Portfolio Optimization Problem",
            description=f"Markowitz portfolio optimization with {self.n_stocks} stocks, "
                       f"target return {self.target_return:.1%}",
            problem_type=ProblemType.CONTINUOUS,
            objective_type=ObjectiveType.MINIMIZE,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            domain="finance",
            tags={"portfolio", "markowitz", "finance", "optimization", "risk"},
            dimension=self.n_stocks,
            variable_bounds={f"weight_{i}": (0.0, 1.0) for i in range(self.n_stocks)},
            constraints_count=2,  # sum to 1, return constraint
            evaluation_complexity="O(n²)",
            memory_complexity="O(n²)"
        )
    
    def _load_stock_data(self, csv_data: str = None, csv_file_path: str = None) -> List[StockData]:
        """Load stock data from CSV source."""
        if csv_data:
            # Load from string data
            df = pd.read_csv(io.StringIO(csv_data))
        elif csv_file_path:
            # Load from file path
            df = pd.read_csv(csv_file_path)
        else:
            # Use default sample data
            df = self._create_default_data()
        
        # Validate required columns
        required_cols = ['symbol', 'expected_return', 'volatility']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert to StockData objects
        stocks = []
        for _, row in df.iterrows():
            stock = StockData(
                symbol=str(row['symbol']),
                expected_return=float(row['expected_return']),
                volatility=float(row['volatility'])
            )
            stocks.append(stock)
        
        if len(stocks) < 2:
            raise ValueError("Portfolio optimization requires at least 2 stocks")
        
        return stocks
    
    def _create_default_data(self) -> pd.DataFrame:
        """Create default sample stock data for demonstration."""
        return pd.DataFrame({
            'symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
            'expected_return': [0.12, 0.15, 0.11, 0.14, 0.20],
            'volatility': [0.25, 0.30, 0.22, 0.28, 0.45]
        })
    
    def _build_covariance_matrix(self, correlation_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """Build covariance matrix from volatilities and correlations."""
        volatilities = np.array([stock.volatility for stock in self.stocks])
        
        if correlation_matrix is not None:
            # Use provided correlation matrix
            if correlation_matrix.shape != (self.n_stocks, self.n_stocks):
                raise ValueError(f"Correlation matrix shape {correlation_matrix.shape} "
                               f"doesn't match number of stocks {self.n_stocks}")
            corr_matrix = correlation_matrix
        else:
            # Create simplified correlation matrix (moderate positive correlations)
            corr_matrix = np.eye(self.n_stocks)
            for i in range(self.n_stocks):
                for j in range(i + 1, self.n_stocks):
                    # Random correlation between 0.1 and 0.6
                    correlation = 0.1 + 0.5 * np.random.random()
                    corr_matrix[i, j] = correlation
                    corr_matrix[j, i] = correlation
        
        # Convert correlation to covariance: Cov(i,j) = σ_i * σ_j * ρ_ij
        covariance = np.outer(volatilities, volatilities) * corr_matrix
        return covariance
    
    def evaluate_solution(self, solution: Union[List[float], np.ndarray, Dict[str, Any]]) -> float:
        """
        Evaluate portfolio solution.
        
        Args:
            solution: Portfolio weights as list/array or dict with 'weights' key
            
        Returns:
            Portfolio risk (variance) with penalties for constraint violations
        """
        # Extract weights from solution
        if isinstance(solution, dict):
            weights = np.array(solution.get('weights', solution.get('portfolio', [])))
        else:
            weights = np.array(solution)
        
        if len(weights) != self.n_stocks:
            return 1e6  # Large penalty for wrong dimension
        
        # Check constraints and apply penalties
        penalty = 0.0
        
        # Constraint 1: Weights must sum to 1 (fully invested)
        weight_sum = np.sum(weights)
        if abs(weight_sum - 1.0) > 1e-6:
            penalty += 1000 * (weight_sum - 1.0) ** 2
        
        # Constraint 2: No negative weights (no short selling)
        negative_weights = weights[weights < 0]
        if len(negative_weights) > 0:
            penalty += 1000 * np.sum(negative_weights ** 2)
        
        # Constraint 3: Expected return must meet target
        expected_returns = np.array([stock.expected_return for stock in self.stocks])
        portfolio_return = np.dot(weights, expected_returns)
        if portfolio_return < self.target_return:
            penalty += 1000 * (self.target_return - portfolio_return) ** 2
        
        # Calculate portfolio variance (risk)
        portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
        
        return portfolio_variance + penalty
    
    def is_feasible(self, solution: Union[List[float], np.ndarray, Dict[str, Any]]) -> bool:
        """Check if solution satisfies all constraints."""
        if isinstance(solution, dict):
            weights = np.array(solution.get('weights', solution.get('portfolio', [])))
        else:
            weights = np.array(solution)
        
        if len(weights) != self.n_stocks:
            return False
        
        # Check weight sum constraint
        if abs(np.sum(weights) - 1.0) > 1e-6:
            return False
        
        # Check non-negativity
        if np.any(weights < -1e-6):
            return False
        
        # Check return constraint
        expected_returns = np.array([stock.expected_return for stock in self.stocks])
        portfolio_return = np.dot(weights, expected_returns)
        if portfolio_return < self.target_return - 1e-6:
            return False
        
        return True
    
    def random_solution(self) -> Dict[str, Any]:
        """Generate a random feasible solution."""
        # Generate random weights that sum to 1
        weights = np.random.random(self.n_stocks)
        weights = weights / np.sum(weights)
        
        # Adjust to meet return constraint if needed
        expected_returns = np.array([stock.expected_return for stock in self.stocks])
        current_return = np.dot(weights, expected_returns)
        
        if current_return < self.target_return:
            # Increase allocation to highest return stocks
            high_return_indices = np.argsort(expected_returns)[-2:]
            adjustment = (self.target_return - current_return) / len(high_return_indices)
            
            for idx in high_return_indices:
                weights[idx] += adjustment
            
            # Renormalize
            weights = weights / np.sum(weights)
        
        return {'weights': weights.tolist()}
    
    def get_solution_info(self, solution: Union[List[float], np.ndarray, Dict[str, Any]]) -> Dict[str, Any]:
        """Get detailed information about a solution."""
        if isinstance(solution, dict):
            weights = np.array(solution.get('weights', solution.get('portfolio', [])))
        else:
            weights = np.array(solution)
        
        expected_returns = np.array([stock.expected_return for stock in self.stocks])
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return {
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'portfolio_variance': portfolio_variance,
            'sharpe_ratio': sharpe_ratio,
            'weights_sum': np.sum(weights),
            'is_feasible': self.is_feasible(solution),
            'stock_allocations': {
                self.stocks[i].symbol: weights[i] 
                for i in range(len(weights))
            }
        }


# Export the problem class for qubots framework
__all__ = ['PortfolioOptimizationProblem']
