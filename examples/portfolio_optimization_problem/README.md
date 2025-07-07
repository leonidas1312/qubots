# Portfolio Optimization Problem

A Markowitz mean-variance portfolio optimization problem implementation for the Qubots framework. This problem reads stock data from CSV files and optimizes portfolio allocation to minimize risk while meeting return constraints.

## 📊 Problem Description

The portfolio optimization problem implements the classic Markowitz model:

- **Objective**: Minimize portfolio risk (variance)
- **Constraints**: 
  - Portfolio weights sum to 1 (fully invested)
  - Expected portfolio return meets minimum target
  - All weights are non-negative (no short selling)

### Mathematical Formulation

```
Minimize: w^T Σ w  (portfolio variance)
Subject to:
  - Σ w_i = 1  (fully invested)
  - w^T μ ≥ r_target  (minimum return)
  - w_i ≥ 0  (no short selling)
```

Where:
- `w` = portfolio weights vector
- `Σ` = covariance matrix of stock returns
- `μ` = expected returns vector
- `r_target` = minimum required return

## 📁 CSV Input Format

The problem accepts CSV data with the following columns:

### Required Columns
- `symbol`: Stock symbol/identifier (string)
- `expected_return`: Expected annual return as decimal (e.g., 0.12 for 12%)
- `volatility`: Annual volatility/standard deviation as decimal (e.g., 0.25 for 25%)

### Optional Columns
- `sector`: Stock sector classification
- `market_cap`: Market capitalization
- `beta`: Beta coefficient

### Example CSV
```csv
symbol,expected_return,volatility
AAPL,0.12,0.25
GOOGL,0.15,0.30
MSFT,0.11,0.22
AMZN,0.14,0.28
TSLA,0.20,0.45
```

## 🚀 Usage

### Basic Usage

```python
from qubots import AutoProblem

# Load with default data (5 tech stocks)
problem = AutoProblem.from_repo("examples/portfolio_optimization_problem")

# Generate and evaluate a solution
solution = problem.random_solution()
risk = problem.evaluate_solution(solution)

print(f"Portfolio risk: {risk:.6f}")
print(f"Is feasible: {problem.is_feasible(solution)}")
```

### Custom CSV Data

```python
from qubots import AutoProblem

# CSV data as string
csv_data = """symbol,expected_return,volatility
AAPL,0.12,0.25
GOOGL,0.15,0.30
MSFT,0.11,0.22"""

problem = AutoProblem.from_repo("examples/portfolio_optimization_problem", 
                               override_params={
                                   "csv_data": csv_data,
                                   "target_return": 0.13
                               })
```

### From CSV File

```python
from qubots import AutoProblem

problem = AutoProblem.from_repo("examples/portfolio_optimization_problem",
                               override_params={
                                   "csv_file_path": "data/stocks.csv",
                                   "target_return": 0.10,
                                   "risk_free_rate": 0.02
                               })
```

### Advanced Configuration

```python
import numpy as np
from qubots import AutoProblem

# Custom correlation matrix
correlation_matrix = np.array([
    [1.0, 0.3, 0.2],
    [0.3, 1.0, 0.4], 
    [0.2, 0.4, 1.0]
])

problem = AutoProblem.from_repo("examples/portfolio_optimization_problem",
                               override_params={
                                   "target_return": 0.12,
                                   "risk_free_rate": 0.025,
                                   "correlation_matrix": correlation_matrix.tolist()
                               })
```

## 📈 Solution Analysis

Get detailed information about portfolio solutions:

```python
solution = problem.random_solution()
info = problem.get_solution_info(solution)

print(f"Portfolio Return: {info['portfolio_return']:.2%}")
print(f"Portfolio Volatility: {info['portfolio_volatility']:.2%}")
print(f"Sharpe Ratio: {info['sharpe_ratio']:.3f}")
print(f"Stock Allocations:")
for stock, weight in info['stock_allocations'].items():
    print(f"  {stock}: {weight:.1%}")
```

## ⚙️ Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_return` | number | 0.10 | Minimum required portfolio return (10%) |
| `risk_free_rate` | number | 0.02 | Risk-free rate for Sharpe ratio (2%) |
| `csv_data` | string | null | CSV content as string |
| `csv_file_path` | string | null | Path to CSV file |
| `correlation_matrix` | array | null | Optional correlation matrix |

## 🎯 Solution Format

Solutions should be provided as:

```python
{
    "weights": [0.2, 0.3, 0.25, 0.15, 0.1]  # Portfolio weights summing to 1.0
}
```

Or as a simple list/array:
```python
[0.2, 0.3, 0.25, 0.15, 0.1]
```

## 📊 Performance Characteristics

- **Evaluation Complexity**: O(n²) where n is number of stocks
- **Memory Complexity**: O(n²) for covariance matrix storage
- **Typical Runtime**: <50ms for portfolios with 5-20 stocks
- **Scalability**: Suitable for portfolios with 5-100 stocks

## 🔍 Validation

The problem includes comprehensive validation:

- **Format Validation**: Ensures correct solution dimensions
- **Constraint Checking**: Verifies all constraints are satisfied
- **Data Validation**: Validates CSV input format and ranges
- **Feasibility Testing**: Checks solution feasibility

## 💡 Tips for Optimizers

1. **Constraint Handling**: The problem uses penalty methods for constraint violations
2. **Initialization**: Use `random_solution()` for feasible starting points
3. **Gradient Information**: Portfolio variance is quadratic, enabling gradient-based methods
4. **Scaling**: Consider normalizing weights during optimization
5. **Convergence**: Monitor both objective value and constraint satisfaction

## 🏷️ Tags

`portfolio`, `markowitz`, `finance`, `optimization`, `risk`, `csv`, `continuous`, `quadratic`
