# Pyomo MaxCut Optimizer

A flexible and versatile solver for the Maximum Cut Problem using the Pyomo modeling framework. This optimizer supports multiple solver backends and integrates seamlessly with the qubots framework and Rastion platform for interactive optimization.

## Overview

The Maximum Cut (MaxCut) problem is a fundamental graph optimization problem where the goal is to partition the vertices of a graph into two sets such that the total weight of edges crossing between the sets is maximized. This implementation uses Pyomo's algebraic modeling capabilities to formulate the problem and supports multiple solver backends for maximum flexibility.

### Mathematical Formulation

The optimizer formulates MaxCut as a quadratic binary optimization problem:

```
maximize: Σ w_ij * (x_i + x_j - 2*x_i*x_j) for all edges (i,j)
subject to: x_i ∈ {0,1} for all vertices i
```

Where:
- `x_i = 1` means vertex `i` is in set T, `x_i = 0` means vertex `i` is in set S
- `w_ij` is the weight of edge between vertices `i` and `j`
- The quadratic term `(x_i + x_j - 2*x_i*x_j)` equals 1 when vertices are in different sets (cut edge) and 0 when in the same set

## Features

- **Multi-Solver Support**: Works with Gurobi, CPLEX, CBC, GLPK, and other solvers
- **Automatic Fallback**: Automatically tries alternative solvers if primary solver is unavailable
- **Flexible Configuration**: Extensive parameter tuning with solver-specific options
- **Real-time Monitoring**: Live progress tracking and logging for playground integration
- **Warm Start Support**: Can initialize with existing solutions for faster convergence
- **Open Source Compatible**: Works with free solvers like CBC and GLPK

## Installation

### Prerequisites

Pyomo requires at least one optimization solver. Options include:

#### Commercial Solvers (High Performance)
- **Gurobi**: Academic licenses available, excellent performance
- **CPLEX**: IBM's solver, academic licenses available

#### Open Source Solvers (Free)
- **CBC**: Coin-OR Branch and Cut solver
- **GLPK**: GNU Linear Programming Kit

### Installation Steps

1. **Install Pyomo**:
   ```bash
   pip install pyomo
   ```

2. **Install a solver**:
   
   For CBC (recommended open-source option):
   ```bash
   # On Windows
   conda install -c conda-forge coincbc
   
   # On Linux/Mac
   sudo apt-get install coinor-cbc  # Ubuntu/Debian
   brew install cbc                 # macOS
   ```
   
   For Gurobi:
   ```bash
   pip install gurobipy
   # Requires Gurobi license
   ```

3. **Verify installation**:
   ```python
   import pyomo.environ as pyo
   solver = pyo.SolverFactory('cbc')
   print(f"CBC available: {solver.available()}")
   ```

## Usage

### Basic Usage

```python
from qubots import MaxCutProblem
from pyomo_maxcut_optimizer.qubot import PyomoMaxCutOptimizer

# Create problem
problem = MaxCutProblem(n_vertices=10, graph_type="random")

# Create optimizer with default solver
optimizer = PyomoMaxCutOptimizer(
    solver_name="cbc",  # Free open-source solver
    time_limit=300.0,
    mip_gap=0.01
)

# Solve
result = optimizer.optimize(problem)
print(f"Best cut weight: {result.best_value}")
print(f"Solution: {result.best_solution}")
```

### Multi-Solver Configuration

```python
# Try Gurobi first, fallback to CBC
optimizer = PyomoMaxCutOptimizer(
    solver_name="gurobi",
    time_limit=600.0,
    mip_gap=0.001,
    threads=4,
    solver_options={
        "Presolve": 2,      # Gurobi-specific option
        "Cuts": 2           # Aggressive cuts
    }
)

# If Gurobi is not available, it will automatically try CBC
result = optimizer.optimize(problem)
print(f"Used solver: {result.additional_metrics['solver']}")
```

### Advanced Configuration

```python
# Custom solver options for different backends
optimizer = PyomoMaxCutOptimizer(
    solver_name="cplex",
    time_limit=1800.0,     # 30 minutes
    mip_gap=0.0001,        # 0.01% gap
    threads=8,
    tee=True,              # Show solver output
    solver_options={
        "mip.strategy.heuristicfreq": 10,  # CPLEX heuristic frequency
        "mip.cuts.all": 2                  # Aggressive cuts
    }
)
```

## Parameters

| Parameter | Type | Default | Options/Range | Description |
|-----------|------|---------|---------------|-------------|
| `solver_name` | string | "gurobi" | ["gurobi", "cplex", "cbc", "glpk", "ipopt"] | Backend solver to use |
| `time_limit` | float | 300.0 | [1.0, 3600.0] | Maximum solving time in seconds |
| `mip_gap` | float | 0.01 | [0.0001, 0.5] | Relative optimality gap tolerance |
| `threads` | int | 0 | [0, 16] | Number of threads (0 = automatic) |
| `tee` | bool | false | - | Enable solver output streaming |
| `solver_options` | object | {} | - | Additional solver-specific options |

## Solver-Specific Options

### Gurobi Options
```python
solver_options = {
    "Presolve": 2,        # Aggressive presolve
    "Cuts": 2,            # Aggressive cuts
    "Heuristics": 0.1,    # 10% time on heuristics
    "Method": 2           # Barrier method
}
```

### CPLEX Options
```python
solver_options = {
    "mip.tolerances.mipgap": 0.001,
    "mip.strategy.heuristicfreq": 10,
    "mip.cuts.all": 2
}
```

### CBC Options
```python
solver_options = {
    "ratio": 0.01,        # MIP gap
    "cuts": "on",         # Enable cuts
    "heuristics": "on"    # Enable heuristics
}
```

## Performance Guidelines

### Solver Recommendations

- **For Speed**: CBC (free, good performance)
- **For Quality**: Gurobi or CPLEX (commercial, excellent performance)
- **For Compatibility**: GLPK (widely available, basic performance)

### Problem Size Recommendations

- **Small (≤20 vertices)**: Any solver, optimal solutions in seconds
- **Medium (20-50 vertices)**: CBC or commercial solvers, near-optimal in minutes
- **Large (50-100 vertices)**: Commercial solvers recommended
- **Very Large (>100 vertices)**: Consider heuristic methods

## Integration with Rastion Platform

This optimizer is fully compatible with the Rastion platform playground:

1. **Upload**: Use the qubots upload utilities to deploy to Rastion
2. **Configure**: Select solver and adjust parameters through the interface
3. **Execute**: Run optimizations with real-time progress monitoring
4. **Share**: Save and share optimization workflows

The optimizer automatically handles solver availability and provides fallback options for maximum compatibility across different environments.

## Examples

See the qubots examples directory for complete working examples including:
- Local testing scripts with different solvers
- Platform integration examples
- Performance comparisons between solvers
- Solver-specific parameter tuning guides

## Troubleshooting

### Common Issues

1. **Solver not found**: Install the solver or try a different one
2. **License issues**: Ensure commercial solver licenses are properly configured
3. **Performance issues**: Try different solvers or adjust solver-specific parameters

### Solver Installation Help

- **CBC**: `conda install -c conda-forge coincbc`
- **GLPK**: `conda install -c conda-forge glpk`
- **Gurobi**: Requires license from gurobi.com
- **CPLEX**: Requires license from IBM

## License

This optimizer uses the MIT license as part of the qubots framework. Individual solvers may have their own licensing requirements.
