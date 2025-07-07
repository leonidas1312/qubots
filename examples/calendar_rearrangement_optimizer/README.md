# OR-Tools Calendar Rearrangement Optimizer

A high-performance qubots optimizer that uses Google OR-Tools integer programming to find optimal calendar rearrangement solutions.

## 🎯 Optimizer Overview

This optimizer provides **guaranteed optimal solutions** for calendar rearrangement problems using advanced mathematical optimization techniques. Perfect for:

- **Optimal scheduling**: Find the mathematically best meeting arrangement
- **Complex constraints**: Handle multiple capacity and priority constraints
- **Large calendars**: Efficiently solve problems with many meetings
- **Professional planning**: Get optimal solutions for critical scheduling decisions

## 🚀 Key Features

- **Optimal Solutions**: Guaranteed to find the best possible arrangement
- **Multiple Solvers**: Support for SCIP, CBC, Gurobi, and CPLEX
- **Rich Visualization**: Comprehensive before/after analysis with plots
- **Fast Performance**: Efficient integer programming formulation
- **Constraint Handling**: Respects all capacity and assignment constraints

## 📊 Visualization Features

The optimizer automatically generates four comprehensive plots:

1. **Meeting Distribution**: Before/after meeting counts per day
2. **Daily Hours**: Meeting hours comparison with capacity limits
3. **Priority Analysis**: Distribution of moved meetings by priority
4. **Cost Breakdown**: Detailed rescheduling costs per meeting

## ⚙️ Solver Options

### Open Source Solvers

| Solver | Description | Best For |
|--------|-------------|----------|
| **SCIP** | Default, robust solver | General use, good performance |
| **CBC** | Coin-OR solver | Alternative to SCIP |

## 🔧 Configuration Options

## 📈 Solution Quality

The optimizer provides different solution qualities based on solving time:

- **OPTIMAL**: Mathematically proven best solution
- **FEASIBLE**: Good solution found within time limit
- **INFEASIBLE**: No valid solution exists (constraints too tight)

## 🎯 Mathematical Formulation

The optimizer solves this integer programming problem:

```
Minimize: Σ(base_cost + priority_cost) × assignment_variables

Subject to:
- Each meeting assigned to exactly one day
- Daily capacity constraints respected
- Binary assignment variables (0 or 1)
```

## 📊 Result Analysis

### Optimization Metrics

```python
result = optimizer.optimize(problem)

print(f"Status: {result.convergence_info['status']}")
print(f"Meetings moved: {result.additional_metrics['meetings_moved']}")
print(f"Solver time: {result.additional_metrics['solver_time']:.2f}s")
print(f"Nodes explored: {result.additional_metrics.get('nodes_explored', 'N/A')}")
```

### Visualization Access

```python
# Access generated plots
if 'calendar_analysis' in result.plots:
    fig = result.plots['calendar_analysis']
    fig.show()  # Display the comprehensive analysis
```

## 🔍 Advanced Features

### Warm Start (Future Enhancement)

```python
# Use initial solution to speed up solving
initial_solution = problem.random_solution()
result = optimizer.optimize(problem, initial_solution=initial_solution)
```

### Solution Validation

```python
# Verify solution optimality
if result.convergence_info['status'] == 'OPTIMAL':
    print("Solution is guaranteed optimal!")
    print(f"Optimal value: {result.objective_value}")
```

## 🔧 Troubleshooting

### Common Issues

**Infeasible Solutions**:
- Check if `max_hours_per_day` is sufficient
- Verify enough `available_days` are provided
- Ensure some meetings are marked as `flexible=True`

**Slow Solving**:
- Reduce `time_limit` for faster (possibly suboptimal) solutions
- Try different solvers (CBC vs SCIP)
- Consider problem simplification


## 📦 Dependencies

- **ortools**: Google's optimization tools
- **matplotlib**: Plotting and visualization
- **seaborn**: Enhanced statistical plots
- **pandas**: Data manipulation
- **numpy**: Numerical computations

Perfect for production scheduling systems requiring optimal solutions!
