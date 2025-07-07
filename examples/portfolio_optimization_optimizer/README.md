# Portfolio Optimization Optimizer

A robust multi-algorithm optimizer for portfolio optimization problems in the Qubots framework. Uses advanced optimization methods including SLSQP, Differential Evolution, and Trust-Region algorithms with automatic algorithm selection.

## 🎯 Overview

This optimizer solves Markowitz portfolio optimization problems using multiple state-of-the-art algorithms:

- **SLSQP**: Sequential Least Squares Programming for fast, gradient-based optimization
- **Differential Evolution**: Global optimization for robust solutions
- **Trust-Region Constrained**: Advanced constrained optimization
- **Automatic Selection**: Intelligent algorithm choice based on problem characteristics

## 🔧 Algorithms

### SLSQP (Sequential Least Squares Programming)
- **Best for**: Small to medium portfolios (5-20 stocks)
- **Strengths**: Fast convergence, handles constraints well
- **Method**: Gradient-based local optimization
- **Complexity**: O(n³)

### Differential Evolution
- **Best for**: Any size portfolio, complex objective landscapes
- **Strengths**: Global optimization, robust to local minima
- **Method**: Population-based evolutionary algorithm
- **Complexity**: O(n × population × generations)

### Trust-Region Constrained
- **Best for**: Medium to large portfolios with many constraints
- **Strengths**: Advanced constraint handling, numerical stability
- **Method**: Trust-region with constraint linearization
- **Complexity**: O(n³)

### Automatic Selection
The optimizer automatically selects the best algorithm based on:
- Problem size (number of stocks)
- Constraint complexity
- Expected performance characteristics

## 🚀 Usage

### Basic Usage

```python
from qubots import AutoProblem, AutoOptimizer

# Load portfolio problem
problem = AutoProblem.from_repo("examples/portfolio_optimization_problem")

# Load optimizer with automatic algorithm selection
optimizer = AutoOptimizer.from_repo("examples/portfolio_optimization_optimizer")

# Solve the portfolio
result = optimizer.optimize(problem)

print(f"Portfolio Return: {result.portfolio_return:.2%}")
print(f"Portfolio Risk: {result.portfolio_volatility:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
print(f"Algorithm Used: {result.algorithm_used}")

# Visualization plots are automatically created and displayed
```

### Algorithm Selection

```python
from qubots import AutoOptimizer

# Force specific algorithm
optimizer = AutoOptimizer.from_repo("examples/portfolio_optimization_optimizer",
                                   override_params={
                                       "algorithm": "slsqp",
                                       "max_iterations": 1000,
                                       "tolerance": 1e-8
                                   })
```

### High-Precision Optimization

```python
from qubots import AutoOptimizer

# High-precision optimization
optimizer = AutoOptimizer.from_repo("examples/portfolio_optimization_optimizer",
                                   override_params={
                                       "algorithm": "trust-constr",
                                       "max_iterations": 2000,
                                       "tolerance": 1e-10,
                                       "time_limit": 120.0
                                   })
```

### Global Optimization

```python
from qubots import AutoOptimizer

# Global optimization for complex problems
optimizer = AutoOptimizer.from_repo("examples/portfolio_optimization_optimizer",
                                   override_params={
                                       "algorithm": "differential_evolution",
                                       "max_iterations": 3000,
                                       "random_seed": 123
                                   })
```

## ⚙️ Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm` | string | "auto" | Optimization algorithm to use |
| `max_iterations` | integer | 1000 | Maximum optimization iterations |
| `tolerance` | number | 1e-8 | Convergence tolerance |
| `time_limit` | number | 60.0 | Maximum time in seconds |
| `random_seed` | integer | 42 | Random seed for reproducibility |
| `use_bounds` | boolean | true | Enforce variable bounds (0 ≤ weight ≤ 1) |
| `use_constraints` | boolean | true | Use explicit constraints |
| `create_plots` | boolean | true | Create visualization plots after optimization |

### Algorithm Options
- `"auto"`: Automatic selection based on problem size
- `"slsqp"`: Sequential Least Squares Programming
- `"differential_evolution"`: Global evolutionary optimization
- `"trust-constr"`: Trust-region constrained optimization

## 📊 Visualization Features

The optimizer automatically creates helpful visualization plots for users who are new to portfolio optimization:

### Plot 1: Portfolio Allocation Pie Chart
- Shows how your investment is distributed across different stocks
- Combines small allocations (<1%) into "Others" for clarity
- Color-coded for easy identification

### Plot 2: Risk vs Return Scatter Plot
- Displays individual stocks as points (risk vs expected return)
- Shows your optimized portfolio as a red star
- Includes target return line for reference
- Helps understand the risk-return tradeoff

### Plot 3: Portfolio Performance Metrics
- Bar chart showing key metrics: return, risk, Sharpe ratio, target
- Easy-to-read percentage values
- Visual comparison of achieved vs target performance

### Plot 4: Individual Stock Allocations
- Bar chart of stock-by-stock allocation percentages
- Sorted by allocation size for easy reading
- Color-coded for visual appeal

### Disabling Plots
```python
# Turn off plots for automated workflows
optimizer = AutoOptimizer.from_repo("examples/portfolio_optimization_optimizer",
                                   override_params={"create_plots": False})
```

## 📊 Result Format

The optimizer returns a `PortfolioOptimizationResult` with:

```python
{
    "best_solution": [0.2, 0.3, 0.25, 0.15, 0.1],  # Optimal weights
    "best_value": 0.0234,                           # Portfolio variance
    "portfolio_return": 0.125,                      # Expected return
    "portfolio_volatility": 0.153,                  # Portfolio std dev
    "sharpe_ratio": 0.686,                          # Risk-adjusted return
    "stock_allocations": {                          # Stock-wise allocation
        "AAPL": 0.2,
        "GOOGL": 0.3,
        # ...
    },
    "algorithm_used": "slsqp",                      # Algorithm used
    "constraint_violations": [],                    # Any violations
    "runtime_seconds": 0.234,                      # Execution time
    "iterations": 45,                               # Iterations taken
    "termination_reason": "Optimization terminated successfully"
}
```

## 🎯 Performance Guidelines

### Small Portfolios (5-20 stocks)
- **Recommended**: `algorithm: "slsqp"`
- **Expected Runtime**: 0.1-1.0 seconds
- **Iterations**: 50-200

### Medium Portfolios (20-50 stocks)
- **Recommended**: `algorithm: "auto"` or `"trust-constr"`
- **Expected Runtime**: 1.0-10.0 seconds
- **Iterations**: 100-500

### Large Portfolios (50+ stocks)
- **Recommended**: `algorithm: "differential_evolution"`
- **Expected Runtime**: 10.0-60.0 seconds
- **Iterations**: 500-2000

## 🔍 Constraint Handling

The optimizer handles portfolio constraints through:

1. **Weight Sum Constraint**: Σ w_i = 1 (fully invested)
2. **Non-negativity**: w_i ≥ 0 (no short selling)
3. **Return Constraint**: w^T μ ≥ r_target (minimum return)
4. **Bounds**: 0 ≤ w_i ≤ 1 (reasonable allocation limits)

## 🛠️ Troubleshooting

### Optimization Fails
- Try `algorithm: "differential_evolution"` for global search
- Increase `max_iterations` and `time_limit`
- Check if target return is achievable

### Slow Convergence
- Use `algorithm: "slsqp"` for faster local optimization
- Reduce `tolerance` if high precision isn't needed
- Check problem scaling and conditioning

### Constraint Violations
- Ensure `use_constraints: true`
- Verify target return is realistic
- Check input data quality

## 💡 Tips for Best Results

1. **Algorithm Selection**: Use "auto" for most cases, "slsqp" for speed, "differential_evolution" for robustness
2. **Tolerance**: Use 1e-6 for fast results, 1e-8 for standard precision, 1e-10 for high precision
3. **Time Limits**: Set appropriate limits based on portfolio size and precision needs
4. **Random Seeds**: Use consistent seeds for reproducible results
5. **Constraint Handling**: Always use constraints for realistic portfolio optimization

## 🏷️ Tags

`portfolio`, `markowitz`, `finance`, `optimization`, `scipy`, `slsqp`, `differential_evolution`, `trust-region`, `constrained`
