# OR-Tools Supply Chain Optimizer

A high-performance linear programming optimizer for supply chain network optimization using Google OR-Tools. This qubot provides optimal solutions for complex supply chain problems with comprehensive visualization and analysis.

## 🚀 Key Features

- **Optimal Solutions**: Guaranteed optimal solutions for linear and mixed-integer problems
- **Multiple Solvers**: Support for SCIP, GLOP, CBC, Gurobi, and CPLEX solvers
- **High Performance**: Efficient solving with advanced optimization techniques
- **Rich Visualizations**: Network diagrams, cost breakdowns, and geographic layouts
- **Scalable**: Handles medium to large-scale supply chain networks
- **Open Source**: No licensing restrictions unlike PuLP

## 📦 Installation

```bash
pip install qubots ortools matplotlib seaborn networkx
```

## 🎯 Usage

### Basic Usage

```python
from qubots import AutoProblem, AutoOptimizer

# Load supply chain problem
problem = AutoProblem.from_repo("examples/supply_chain_network_problem")

# Load OR-Tools optimizer
optimizer = AutoOptimizer.from_repo("examples/ortools_supply_chain_optimizer")

# Solve the problem
result = optimizer.optimize(problem)

print(f"Optimal cost: ${result.best_value:,.2f}")
print(f"Solver status: {result.metadata['solver_status']}")
print(f"Runtime: {result.runtime_seconds:.2f} seconds")
```

### Custom Solver Configuration

```python
from qubots import AutoOptimizer

# Use SCIP solver with extended time limit
optimizer = AutoOptimizer.from_repo(
    "examples/ortools_supply_chain_optimizer",
    solver_name="SCIP",
    time_limit=600.0
)

result = optimizer.optimize(problem)
```

### Available Solvers

- **SCIP**: Default mixed-integer programming solver (open source)
- **GLOP**: Google's linear programming solver (fast for LP problems)
- **CBC**: COIN-OR Branch and Cut solver (open source)
- **GUROBI**: Commercial high-performance solver (requires license)
- **CPLEX**: IBM's commercial solver (requires license)

## 🔧 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `solver_name` | string | "SCIP" | Solver to use for optimization |
| `time_limit` | float | 300.0 | Maximum solving time in seconds |

## 📊 Optimization Model

### Decision Variables
- **Production**: Continuous variables for supplier production amounts
- **Warehouse Operations**: Binary variables for warehouse open/close decisions
- **Flows**: Continuous variables for product flows between entities

### Objective Function
```
Minimize: Production Costs + Transportation Costs + Fixed Warehouse Costs
```

### Constraints
1. **Demand Satisfaction**: All customer demands must be met
2. **Supplier Capacity**: Production cannot exceed supplier capacity
3. **Warehouse Capacity**: Flows cannot exceed warehouse capacity (if open)
4. **Flow Balance**: Inflow equals outflow at each warehouse
5. **Binary Operations**: Warehouses are either fully open or closed

## 📈 Performance Characteristics

- **Time Complexity**: Polynomial for LP, Exponential for MILP
- **Space Complexity**: O(variables + constraints)
- **Scalability**: Excellent for medium to large problems
- **Deterministic**: Always produces the same result for the same input
- **Parallel**: Supports multi-threaded solving

## 🎨 Visualizations

The optimizer generates comprehensive visualizations:

1. **Network Flow Diagram**: Interactive network showing suppliers, warehouses, and customers
2. **Cost Breakdown**: Pie chart of production, transportation, and fixed costs
3. **Capacity Utilization**: Bar chart showing resource utilization
4. **Geographic Layout**: Spatial view of supply chain entities
5. **Flow Distribution**: Histogram of flow amounts
6. **Solution Summary**: Key metrics and statistics

## 🔗 Integration

Works seamlessly with supply chain problems:

```python
from qubots import AutoProblem, AutoOptimizer

# Compare with heuristic optimizer
problem = AutoProblem.from_repo("examples/supply_chain_network_problem")
ortools_optimizer = AutoOptimizer.from_repo("examples/ortools_supply_chain_optimizer")
heuristic_optimizer = AutoOptimizer.from_repo("examples/heuristic_supply_chain_optimizer")

# Get optimal solution
optimal_result = ortools_optimizer.optimize(problem)
print(f"Optimal cost: ${optimal_result.best_value:,.2f}")

# Compare with heuristic
heuristic_result = heuristic_optimizer.optimize(problem)
print(f"Heuristic cost: ${heuristic_result.best_value:,.2f}")

# Calculate optimality gap
gap = ((heuristic_result.best_value - optimal_result.best_value) / 
       optimal_result.best_value * 100)
print(f"Optimality gap: {gap:.2f}%")
```

## 🆚 Advantages over PuLP

- **No License Restrictions**: Completely open source
- **Better Performance**: More efficient solving algorithms
- **Advanced Features**: Better presolving and cutting planes
- **Modern Architecture**: Actively maintained by Google
- **Broader Solver Support**: More solver options available

## 📚 Applications

Perfect for optimizing:
- Manufacturing supply chains
- Retail distribution networks
- E-commerce fulfillment centers
- Humanitarian logistics
- Multi-echelon inventory systems
- Transportation networks

## 🔍 Example Output

```
OPTIMIZATION SUMMARY

Solver: SCIP
Status: OPTIMAL

COSTS:
Total Cost: $234,567.89

PRODUCTION:
Total Production: 2,450 units
Total Demand: 2,450 units
Demand Coverage: 100.0%

FACILITIES:
Open Warehouses: 2/3
Utilization Rate: 66.7%

NETWORK:
Suppliers: 3
Customers: 5
Active Flows: 8
```

## 🚀 Getting Started

1. Install OR-Tools: `pip install ortools`
2. Load your supply chain problem
3. Create the optimizer with desired parameters
4. Run optimization and analyze results
5. Use visualizations to understand the solution

The OR-Tools optimizer provides a powerful, license-free alternative to PuLP with superior performance and features for supply chain optimization.
