# OR-Tools MaxCut Optimizer

A Google OR-Tools CP-SAT based optimizer for the Maximum Cut Problem using the qubots framework. This optimizer provides an exact solution approach using constraint programming techniques with linearization to handle the quadratic MaxCut objective.

## Overview

The Maximum Cut (MaxCut) problem is a classic NP-hard combinatorial optimization problem that involves partitioning the vertices of a graph into two sets such that the total weight of edges crossing between the sets is maximized.

This optimizer uses Google OR-Tools' CP-SAT solver, which is:
- **Open Source**: Apache License 2.0
- **Exact**: Guarantees optimal solutions (given sufficient time)
- **Efficient**: Uses advanced constraint programming techniques
- **Parallel**: Supports multi-threaded solving

## Mathematical Formulation

The optimizer formulates MaxCut as a constraint satisfaction problem:

### Variables
- `x_i ∈ {0,1}` for each vertex i (partition assignment)
- `y_ij ∈ {0,1}` for each edge (i,j) (cut indicator)

### Constraints (Linearization)
For each edge (i,j), the following constraints ensure `y_ij = 1` if and only if vertices i and j are in different partitions:

```
y_ij >= x_i - x_j
y_ij >= x_j - x_i  
y_ij <= x_i + x_j
y_ij <= 2 - x_i - x_j
```

### Objective
```
maximize: Σ w_ij * y_ij for all edges (i,j)
```

## Features

- **Linearization**: Converts the quadratic MaxCut objective into linear constraints
- **Symmetry Breaking**: Optional constraint to reduce search space
- **Parallel Solving**: Configurable number of search workers
- **Solution Hints**: Supports warm-starting with initial solutions
- **Detailed Logging**: Optional progress tracking and statistics

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_limit` | number | 300.0 | Maximum solving time in seconds |
| `num_search_workers` | integer | 0 | Number of parallel workers (0 = automatic) |
| `log_search_progress` | boolean | false | Enable detailed search progress logging |
| `use_symmetry` | boolean | true | Enable symmetry breaking techniques |
| `use_sat_preprocessing` | boolean | true | Enable SAT preprocessing |
| `enumerate_all_solutions` | boolean | false | Find all optimal solutions |

## Installation

```bash
pip install qubots
```

## Usage

### Basic Usage

```python
from qubots import AutoProblem, AutoOptimizer

# Load a MaxCut problem
problem = AutoProblem.from_repo("ileo/demo-maxcut")

# Load the OR-Tools optimizer
optimizer = AutoOptimizer.from_repo("ileo/demo-ortools-maxcut-optimizer")

# Solve the problem
result = optimizer.optimize(problem)

print(f"Best cut weight: {result.best_value}")
print(f"Solution: {result.best_solution}")
print(f"Runtime: {result.runtime_seconds:.3f} seconds")
```

### Advanced Configuration

```python
from qubots import AutoProblem, AutoOptimizer

# Load a MaxCut problem
problem = AutoProblem.from_repo("ileo/demo-maxcut")

# Configure optimizer parameters
optimizer = AutoOptimizer.from_repo("ileo/demo-ortools-maxcut-optimizer", override_params={
    "time_limit": 60.0,            # 1 minute
    "num_search_workers": 4,       # Use 4 parallel workers
    "log_search_progress": True,   # Enable detailed logging
    "use_symmetry": True,          # Use symmetry breaking
})

result = optimizer.optimize(problem)

print(f"Best cut weight: {result.best_value}")
print(f"Solution: {result.best_solution}")
print(f"Runtime: {result.runtime_seconds:.3f} seconds")
```

