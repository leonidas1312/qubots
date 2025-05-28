# OR-Tools MaxCut Optimizer

A Google OR-Tools CP-SAT based optimizer for the Maximum Cut Problem using the qubots framework. This optimizer provides an exact solution approach using constraint programming techniques with linearization to handle the quadratic MaxCut objective.

## Overview

The Maximum Cut (MaxCut) problem is a classic NP-hard combinatorial optimization problem that involves partitioning the vertices of a graph into two sets such that the total weight of edges crossing between the sets is maximized.

This optimizer uses Google OR-Tools' CP-SAT solver, which is:
- **Open Source**: Apache License 2.0 - no license required
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
- **Real-time Streaming**: Compatible with Rastion playground terminal viewer

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
pip install ortools>=9.8.0 numpy>=1.21.0 qubots>=0.1.0
```

## Usage

### Basic Usage

```python
from qubots import load_problem, load_optimizer

# Load a MaxCut problem
problem = load_problem("maxcut_problem")

# Load the OR-Tools optimizer
optimizer = load_optimizer("ortools_maxcut_optimizer")

# Solve the problem
result = optimizer.optimize(problem)

print(f"Best cut weight: {result.best_value}")
print(f"Solution: {result.best_solution}")
print(f"Runtime: {result.runtime_seconds:.3f} seconds")
```

### Advanced Configuration

```python
# Configure optimizer parameters
optimizer = load_optimizer("ortools_maxcut_optimizer", {
    "time_limit": 600.0,           # 10 minutes
    "num_search_workers": 4,       # Use 4 parallel workers
    "log_search_progress": True,   # Enable detailed logging
    "use_symmetry": True,          # Use symmetry breaking
})

# Solve with initial solution hint
initial_solution = [0, 1, 0, 1, 0]  # Example partition
result = optimizer.optimize(problem, initial_solution=initial_solution)
```

### Integration with Rastion Platform

This optimizer is fully compatible with the Rastion platform playground:

1. **Upload**: Use the qubots upload utilities to deploy to Rastion
2. **Configure**: Set parameters through the playground interface
3. **Execute**: Run optimizations with real-time terminal output
4. **Share**: Share working configurations with the community

## Performance Characteristics

- **Small instances** (≤20 vertices): Typically finds optimal solutions quickly
- **Medium instances** (20-50 vertices): Good performance with near-optimal solutions
- **Large instances** (>50 vertices): May require longer time limits or heuristic approaches

## Comparison with Other Solvers

| Solver | License | Type | Strengths |
|--------|---------|------|-----------|
| OR-Tools CP-SAT | Open Source | Exact | No license required, good for medium instances |
| Gurobi | Commercial | Exact | Fastest for large instances, requires license |
| CPLEX | Commercial | Exact | Enterprise-grade, requires license |
| Pyomo | Open Source | Modeling | Flexible modeling, depends on underlying solver |

## Algorithm Details

The OR-Tools CP-SAT solver uses:

1. **SAT-based solving**: Converts the problem to Boolean satisfiability
2. **Constraint propagation**: Efficiently reduces search space
3. **Conflict-driven learning**: Learns from failed search paths
4. **Parallel search**: Multiple workers explore different parts of the search tree
5. **Preprocessing**: Simplifies the model before solving

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure OR-Tools is installed: `pip install ortools`
2. **Slow Performance**: Try reducing `time_limit` or increasing `num_search_workers`
3. **No Solution Found**: Check if the problem is feasible or increase time limit

### Performance Tips

1. **Use symmetry breaking** for better performance on symmetric graphs
2. **Provide good initial solutions** to speed up convergence
3. **Adjust worker count** based on available CPU cores
4. **Enable preprocessing** for better model simplification

## Examples

See the qubots examples directory for complete working examples:
- Basic MaxCut solving
- Parameter tuning
- Performance comparison
- Integration with other optimizers

## Contributing

This optimizer is part of the qubots framework. Contributions are welcome:
- Bug reports and feature requests
- Performance improvements
- Additional constraint programming techniques
- Documentation improvements

## License

This optimizer uses Google OR-Tools which is licensed under Apache License 2.0.
No commercial license is required for any use case.

## References

- [Google OR-Tools Documentation](https://developers.google.com/optimization)
- [CP-SAT Primer](https://github.com/d-krupke/cpsat-primer)
- [MaxCut Problem on Wikipedia](https://en.wikipedia.org/wiki/Maximum_cut)
- [Qubots Framework](https://github.com/qubots/qubots)
