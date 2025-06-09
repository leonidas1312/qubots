# HiGHS TSP Solver

A high-performance Integer Linear Programming (ILP) solver for the Traveling Salesman Problem (TSP) using the HiGHS optimization engine. This solver provides exact solutions with optimality guarantees for small to medium-sized TSP instances.

## 🔧 Algorithm Details

### Formulation
- **Type**: Integer Linear Programming (ILP)
- **Variables**: Binary variables x_ij (edge selection) + continuous variables u_i (MTZ ordering)
- **Objective**: Minimize total tour distance
- **Constraints**: 
  - Degree constraints (each city visited exactly once)
  - Miller-Tucker-Zemlin (MTZ) subtour elimination

### Key Features
- **Exact Algorithm**: Guarantees global optimum when solved to completion
- **Subtour Elimination**: Uses MTZ constraints for efficient subtour prevention
- **High Performance**: Leverages HiGHS state-of-the-art MIP solver
- **Parallel Processing**: Supports multi-threaded solving
- **Configurable**: Extensive parameter tuning options

## 📊 Performance Characteristics

| Problem Size | Expected Runtime | Memory Usage | Recommendation |
|--------------|------------------|--------------|----------------|
| ≤20 cities   | < 1 second      | Low          | Excellent      |
| 21-50 cities | 1-60 seconds    | Moderate     | Good           |
| 51-100 cities| 1-300 seconds   | High         | Feasible       |
| >100 cities  | >300 seconds    | Very High    | Use heuristics |

## 🚀 Usage

### Basic Usage

```python
from qubots import AutoProblem, AutoOptimizer

# Load TSP problem
problem = AutoProblem.from_repo("ileo/tsp-demo", override_params={
    "instance_file": "instances/att48.tsp"
})

# Load HiGHS solver
optimizer = AutoOptimizer.from_repo("ileo/highs-tsp-solver-demo")

# Solve the problem
result = optimizer.optimize(problem)

print(f"Optimal tour distance: {result.best_value}")
print(f"Tour: {result.best_solution}")
print(f"Runtime: {result.runtime_seconds:.3f} seconds")
print(f"Status: {result.termination_reason}")
```

### Advanced Configuration

```python
from qubots import AutoProblem, AutoOptimizer

# Load problem
problem = AutoProblem.from_repo("ileo/tsp-demo", override_params={
    "instance_file": "instances/berlin52.tsp"
})

# Configure solver with custom parameters
optimizer = AutoOptimizer.from_repo("ileo/highs-tsp-solver-demo", override_params={
    "time_limit": 120.0,      # 2 minutes
    "mip_gap": 0.001,         # 0.1% optimality gap
    "parallel": True,         # Enable parallel processing
    "log_level": 2            # Verbose logging
})

# Solve with progress tracking
def progress_callback(iteration, best_value, current_value):
    print(f"Iteration {iteration}: Best = {best_value}, Current = {current_value}")

result = optimizer.optimize(problem, progress_callback=progress_callback)

# Analyze results
print(f"\n=== Solution Analysis ===")
print(f"Optimal Value: {result.best_value}")
print(f"Solution Found: {result.is_feasible}")
print(f"Runtime: {result.runtime_seconds:.3f} seconds")
print(f"Termination: {result.termination_reason}")

if result.additional_metrics:
    print(f"Solver Status: {result.additional_metrics.get('solver_status', 'N/A')}")
    mip_gap = result.additional_metrics.get('mip_gap_achieved')
    if mip_gap is not None:
        print(f"Final MIP Gap: {mip_gap:.4f}")
```

## ⚙️ Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `time_limit` | float | 300.0 | >0 | Maximum solving time in seconds |
| `mip_gap` | float | 0.01 | 0-1 | MIP optimality gap tolerance (1% = 0.01) |
| `presolve` | bool | true | - | Enable presolving techniques |
| `parallel` | bool | true | - | Enable parallel processing |
| `log_level` | int | 1 | 0-2 | Logging verbosity (0=none, 1=basic, 2=verbose) |

## 🔬 Algorithm Information

### Mathematical Formulation

**Decision Variables:**
- x_ij ∈ {0,1}: Binary variable indicating if edge (i,j) is in the tour
- u_i ∈ ℝ: Continuous variable for MTZ ordering (i ≠ 0)

**Objective:**
```
minimize Σ_i Σ_j c_ij * x_ij
```

**Constraints:**
```
Σ_j x_ij = 1  ∀i (out-degree)
Σ_i x_ij = 1  ∀j (in-degree)
u_i - u_j + n*x_ij ≤ n-1  ∀i,j≠0, i≠j (MTZ subtour elimination)
1 ≤ u_i ≤ n  ∀i≠0
```

### Complexity Analysis
- **Variables**: O(n²) binary + O(n) continuous
- **Constraints**: O(n²) 
- **Time Complexity**: Exponential (worst case)
- **Space Complexity**: O(n²)

## 🎯 When to Use

### Recommended For:
- Small to medium TSP instances (≤100 cities)
- When exact optimal solutions are required
- Benchmarking and validation purposes
- Academic research and education

### Not Recommended For:
- Large instances (>200 cities)
- Real-time applications requiring fast solutions
- When good approximate solutions are sufficient

## 🔗 Integration

This solver integrates seamlessly with the qubots framework:

- **AutoOptimizer**: Automatic loading and configuration
- **Benchmarking**: Compatible with qubots benchmark suite
- **Problem Types**: Works with any TSP problem implementing `get_distance_matrix()`
- **Result Format**: Standard qubots `OptimizationResult` format

## 📚 References

1. Miller, C. E., Tucker, A. W., & Zemlin, R. A. (1960). Integer programming formulation of traveling salesman problems. Journal of the ACM, 7(4), 326-329.
2. HiGHS optimization solver: https://highs.dev/
3. Applegate, D., Bixby, R., Chvátal, V., & Cook, W. (2006). The traveling salesman problem: a computational study.
