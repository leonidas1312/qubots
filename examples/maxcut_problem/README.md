# Maximum Cut Problem

A comprehensive implementation of the Maximum Cut (MaxCut) problem for the qubots optimization framework.

## Problem Description

The Maximum Cut problem is a fundamental combinatorial optimization problem where the goal is to partition the vertices of a weighted graph into two sets such that the total weight of edges crossing between the two sets is maximized.

### Mathematical Formulation

Given a graph G = (V, E) with vertex set V and edge set E, and edge weights w_ij, the MaxCut problem seeks to find a partition of V into two disjoint sets S and T such that:

```
maximize: Σ w_ij for all edges (i,j) where i ∈ S and j ∈ T
```

### Solution Representation

Solutions are represented as binary vectors where:
- `solution[i] = 0` means vertex i belongs to set S
- `solution[i] = 1` means vertex i belongs to set T

Example: `[0, 1, 0, 1]` means vertices 0,2 are in set S and vertices 1,3 are in set T.

## Features

- **Multiple Graph Types**: Support for random, complete, cycle, and grid graphs
- **Configurable Parameters**: Adjustable graph size, density, and edge weights
- **Real-time Evaluation**: Verbose evaluation with detailed cut analysis
- **Solution Analysis**: Comprehensive solution summaries with metrics
- **Neighbor Generation**: Local search support with vertex flipping
- **Rastion Integration**: Full compatibility with the Rastion platform playground

## Usage

### Basic Usage

```python
from qubots import MaxCutProblem

# Create a random graph with 10 vertices
problem = MaxCutProblem(n_vertices=10, graph_type="random", density=0.5)

# Generate and evaluate a random solution
solution = problem.get_random_solution()
cut_weight = problem.evaluate_solution(solution, verbose=True)

print(f"Cut weight: {cut_weight}")
print(f"Solution summary: {problem.get_solution_summary(solution)}")
```

### Custom Graph

```python
from qubots import MaxCutProblem
import numpy as np

# Create custom adjacency matrix
adj_matrix = np.array([
    [0, 2, 3, 0],
    [2, 0, 1, 4],
    [3, 1, 0, 2],
    [0, 4, 2, 0]
])

problem = MaxCutProblem(adjacency_matrix=adj_matrix)
```

## Graph Types

1. **Random**: Edges added with specified probability (density)
2. **Complete**: All vertices connected to all others
3. **Cycle**: Vertices connected in a ring structure
4. **Grid**: Approximate square grid topology

## Applications

- **Statistical Physics**: Ising model ground state problems
- **Circuit Design**: VLSI layout optimization
- **Machine Learning**: Feature selection and clustering
- **Network Analysis**: Community detection
- **Quantum Computing**: QAOA algorithm benchmarking

## Complexity

- **Problem Class**: NP-Hard
- **Approximation**: 0.878-approximation algorithm exists (Goemans-Williamson)
- **Exact Solutions**: Feasible for small graphs (< 20 vertices)
- **Heuristics**: Various metaheuristics perform well in practice

## Configuration Parameters

- `n_vertices`: Number of graph vertices (3-100)
- `graph_type`: Graph structure type
- `density`: Edge probability for random graphs (0.1-1.0)
- `weight_range`: Range for random edge weights

## Integration with Optimizers

This problem is designed to work with various optimization algorithms:
- CPLEX (exact solver)
- Genetic algorithms
- Simulated annealing
- Tabu search
- Quantum algorithms (QAOA)

## References

1. Goemans, M. X., & Williamson, D. P. (1995). Improved approximation algorithms for maximum cut and satisfiability problems.
2. Karp, R. M. (1972). Reducibility among combinatorial problems.
3. Barahona, F., et al. (1988). An application of combinatorial optimization to statistical physics and circuit layout design.
