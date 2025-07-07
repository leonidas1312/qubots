# Heuristic Supply Chain Optimizer

A fast heuristic optimizer for supply chain network problems using greedy construction and local search improvement. This optimizer provides high-quality solutions quickly with comprehensive visualization of the search process.

## 🚀 Key Features

- **Fast Execution**: Optimized for large-scale problems with quick convergence
- **Multi-Phase Algorithm**: Greedy construction + local search + perturbation
- **High-Quality Solutions**: Typically within 5-15% of optimal solutions
- **Adaptive Search**: Multiple neighborhood operators for diversification
- **Escape Mechanisms**: Perturbation to avoid local optima
- **Rich Visualizations**: 6 comprehensive plots showing search progress and results
- **Configurable Parameters**: Tunable for different problem characteristics

## 📦 Installation

```bash
pip install qubots matplotlib seaborn networkx
```

## 🎯 Usage

### Basic Usage

```python
from qubots import AutoProblem, AutoOptimizer

# Load supply chain problem
problem = AutoProblem.from_repo("examples/supply_chain_network_problem")

# Load heuristic optimizer
optimizer = AutoOptimizer.from_repo("examples/heuristic_supply_chain_optimizer")

# Solve the problem
result = optimizer.optimize(problem)

print(f"Best cost found: ${result.best_value:,.2f}")
print(f"Runtime: {result.runtime_seconds:.2f} seconds")
print(f"Iterations: {result.iterations}")
print(f"Improvements: {result.metadata['total_improvements']}")
```

### Custom Configuration

```python
from qubots import AutoOptimizer

# Configure for intensive search
optimizer = AutoOptimizer.from_repo("examples/heuristic_supply_chain_optimizer",
                                   override_params={
                                       "max_iterations": 2000,
                                       "local_search_iterations": 200,
                                       "perturbation_strength": 0.3,
                                       "improvement_threshold": 0.005,
                                       "random_seed": 42
                                   })

result = optimizer.optimize(problem)
```

### Quick Search Configuration

```python
# Configure for fast search
optimizer = AutoOptimizer.from_repo("examples/heuristic_supply_chain_optimizer",
                                   override_params={
                                       "max_iterations": 100,
                                       "local_search_iterations": 50,
                                       "perturbation_strength": 0.1
                                   })

result = optimizer.optimize(problem)
```

## 🔧 Algorithm Phases

### 1. Greedy Construction
- **Warehouse Selection**: Score warehouses by cost-effectiveness and location
- **Customer Assignment**: Assign customers to cheapest available warehouses
- **Supplier Allocation**: Assign suppliers to minimize transportation costs
- **Production Setting**: Set supplier production to meet demand

### 2. Local Search
Four neighborhood operators are used iteratively:
- **Warehouse Swap**: Exchange warehouse assignments between customers
- **Customer Reassignment**: Move customers to different warehouses
- **Supplier Adjustment**: Change supplier allocations to warehouses
- **Warehouse Toggle**: Open/close warehouses strategically

### 3. Perturbation
When stuck in local optima (20+ iterations without improvement):
- **Random Warehouse Toggle**: Randomly open/close warehouses
- **Random Reassignment**: Randomly reassign customers
- **Random Supplier Change**: Randomly change supplier allocations

## 📊 Visualization Features

The optimizer automatically generates 6 comprehensive visualizations:

### 1. Convergence Curve
- Shows cost improvement over iterations
- Displays total improvement percentage
- Tracks optimization progress

### 2. Network Visualization
- Interactive supply chain network graph
- Color-coded nodes by entity type
- Flow connections between entities

### 3. Cost Breakdown
- Pie chart of cost components
- Production, transportation, and fixed costs
- Percentage breakdown of total cost

### 4. Improvement History
- Bar chart of improvement events
- Shows magnitude of each improvement
- Tracks search effectiveness

### 5. Warehouse Utilization
- Horizontal bar chart of warehouse usage
- Open vs. closed warehouse status
- Capacity utilization percentages

### 6. Algorithm Performance
- Detailed performance metrics
- Timing and convergence statistics
- Parameter settings summary

## ⚙️ Parameters

### Core Parameters
- **max_iterations** (1000): Maximum main optimization iterations
- **local_search_iterations** (100): Local search iterations per main iteration
- **perturbation_strength** (0.2): Perturbation intensity (0.0-1.0)
- **improvement_threshold** (0.01): Minimum improvement to continue
- **random_seed** (None): Seed for reproducible results

### Parameter Tuning Guidelines

**For Large Problems:**
```python
{
    "max_iterations": 500,
    "local_search_iterations": 50,
    "perturbation_strength": 0.15
}
```

**For High Quality Solutions:**
```python
{
    "max_iterations": 2000,
    "local_search_iterations": 200,
    "perturbation_strength": 0.25,
    "improvement_threshold": 0.005
}
```

**For Fast Results:**
```python
{
    "max_iterations": 100,
    "local_search_iterations": 25,
    "perturbation_strength": 0.1
}
```

## 📈 Performance Characteristics

- **Speed**: Very fast, typically seconds to minutes
- **Quality**: Usually 5-15% from optimal for medium-scale problems
- **Scalability**: Excellent for large problems (1000+ variables)
- **Memory**: Low memory usage, scales linearly
- **Convergence**: Fast initial improvement, gradual refinement

## 🧪 Example Results

```python
# Example optimization result
result = optimizer.optimize(problem)

print(f"""
Heuristic Optimization Results:
- Best Cost: ${result.best_value:,.2f}
- Runtime: {result.runtime_seconds:.2f} seconds
- Total Iterations: {result.iterations}
- Improvements Found: {result.metadata['total_improvements']}
- Improvement Rate: {result.metadata['total_improvements']/result.iterations*100:.1f}%
- Convergence Rate: {result.metadata['convergence_rate']:.1%}
- Avg Iteration Time: {result.metadata['average_iteration_time']:.4f}s
""")

# Check solution quality
if problem.is_feasible(result.best_solution):
    print("✓ Solution is feasible")
    print(problem.get_solution_summary(result.best_solution))
```

## 🔗 Integration & Comparison

Compare with exact optimizer:

```python
from qubots import AutoProblem, AutoOptimizer

problem = AutoProblem.from_repo("examples/supply_chain_network_problem")

# Heuristic solution
heuristic_optimizer = AutoOptimizer.from_repo("examples/heuristic_supply_chain_optimizer")
heuristic_result = heuristic_optimizer.optimize(problem)

# Exact solution (for comparison)
exact_optimizer = AutoOptimizer.from_repo("examples/pulp_supply_chain_optimizer")
exact_result = exact_optimizer.optimize(problem)

# Compare results
gap = ((heuristic_result.best_value - exact_result.best_value) / 
       exact_result.best_value * 100)
speedup = exact_result.runtime_seconds / heuristic_result.runtime_seconds

print(f"Heuristic vs Exact:")
print(f"- Quality gap: {gap:.2f}%")
print(f"- Speed improvement: {speedup:.1f}x faster")
print(f"- Heuristic time: {heuristic_result.runtime_seconds:.2f}s")
print(f"- Exact time: {exact_result.runtime_seconds:.2f}s")
```

## 🛠️ Troubleshooting

### Slow Convergence
```python
# Increase perturbation and reduce local search
optimizer = AutoOptimizer.from_repo("examples/heuristic_supply_chain_optimizer",
                                   override_params={
                                       "perturbation_strength": 0.4,
                                       "local_search_iterations": 50
                                   })
```

### Poor Quality Solutions
```python
# Increase search intensity
optimizer = AutoOptimizer.from_repo("examples/heuristic_supply_chain_optimizer",
                                   override_params={
                                       "max_iterations": 3000,
                                       "local_search_iterations": 300,
                                       "improvement_threshold": 0.001
                                   })
```

### Reproducible Results
```python
# Set random seed
optimizer = AutoOptimizer.from_repo("examples/heuristic_supply_chain_optimizer",
                                   override_params={"random_seed": 42})
```

## 📚 Applications

- **Manufacturing Networks**: Fast production and distribution optimization
- **Retail Supply Chains**: Quick inventory positioning decisions
- **E-commerce Fulfillment**: Rapid warehouse allocation optimization
- **Emergency Logistics**: Fast disaster relief distribution planning
- **Food Distribution**: Quick perishable product routing
- **Large-Scale Networks**: Optimization of complex multi-echelon systems
