import numpy as np
from qubots.base_problem import BaseProblem

class PortfolioOptimizationProblem(BaseProblem):
    """
    Portfolio Optimization Problem:
    Minimize portfolio variance given asset returns and a covariance matrix.
    Constraint: The sum of asset weights must equal 1.
    """
    def __init__(self, expected_returns, covariance_matrix, target_return=None):
        self.expected_returns = np.array(expected_returns)
        self.covariance_matrix = np.array(covariance_matrix)
        self.n = len(expected_returns)
        self.target_return = target_return
    
    def evaluate_solution(self, weights) -> float:
        weights = np.array(weights)
        variance = weights.T @ self.covariance_matrix @ weights
        penalty = abs(np.sum(weights) - 1) * 1e6
        if self.target_return is not None:
            portfolio_return = np.dot(self.expected_returns, weights)
            if portfolio_return < self.target_return:
                penalty += (self.target_return - portfolio_return) * 1e6
        return variance + penalty
    
    def random_solution(self):
        weights = np.random.rand(self.n)
        return list(weights / np.sum(weights))
