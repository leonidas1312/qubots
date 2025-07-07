# Supply Chain Network Optimization Problem

A comprehensive supply chain network optimization problem that models multi-echelon supply chains with suppliers, warehouses, and customers. This qubot minimizes total cost while satisfying demand and capacity constraints.

## 🚀 Key Features

- **Multi-Echelon Network**: Models suppliers → warehouses → customers flow
- **Cost Optimization**: Minimizes production, transportation, and fixed facility costs
- **Capacity Constraints**: Respects supplier production and warehouse storage limits
- **Demand Satisfaction**: Ensures all customer demands are met
- **Facility Decisions**: Optimizes which warehouses to open/close
- **Flexible Input**: Accepts CSV data or uses default realistic dataset

## 📦 Installation

```bash
pip install qubots
```

## 🎯 Usage

### Basic Usage

```python
from qubots import AutoProblem

# Load with default supply chain data
problem = AutoProblem.from_repo("examples/supply_chain_network_problem")

# Generate and evaluate a solution
solution = problem.random_solution()
cost = problem.evaluate_solution(solution)

print(f"Total supply chain cost: ${cost:,.2f}")
print(f"Is feasible: {problem.is_feasible(solution)}")
print(problem.get_solution_summary(solution))
```

### Custom CSV Data

```python
from qubots import AutoProblem

# Custom supply chain data
csv_data = """entity_type,entity_id,name,location_x,location_y,capacity,fixed_cost,variable_cost,demand,supplier_id,warehouse_id,customer_id,unit_cost
supplier,S1,Acme Manufacturing,10,20,1000,25000,10.0,0,,,,,
warehouse,W1,Central Hub,30,40,800,40000,5.0,0,,,,,
customer,C1,Big Retailer,50,60,0,0,0,500,,,,,
supplier_warehouse,SW1,S1-W1 Link,,,,,,,S1,W1,,12.0
warehouse_customer,WC1,W1-C1 Link,,,,,,,W1,,C1,8.0"""

problem = AutoProblem.from_repo("examples/supply_chain_network_problem", 
                               override_params={"csv_data": csv_data})
```

### With Dataset from Platform

```python
from qubots import AutoProblem, load_dataset_from_platform

# Load dataset from Rastion platform
dataset_content = load_dataset_from_platform(
    token="your_token",
    dataset_id="supply_chain_dataset_id"
)

problem = AutoProblem.from_repo("examples/supply_chain_network_problem",
                               override_params={"dataset_content": dataset_content})
```

## 📊 Problem Structure

### Entities

- **Suppliers**: Production facilities with capacity and variable costs
- **Warehouses**: Distribution centers with capacity and fixed opening costs  
- **Customers**: Demand points requiring product delivery

### Decision Variables

- **Supplier Production**: How much each supplier produces
- **Warehouse Operations**: Which warehouses to open (binary decision)
- **Flow Amounts**: Product flow between suppliers→warehouses→customers

### Constraints

1. **Supplier Capacity**: Production ≤ supplier capacity
2. **Warehouse Capacity**: Total flow ≤ warehouse capacity (if open)
3. **Demand Satisfaction**: Each customer receives required demand
4. **Flow Balance**: Inflow = outflow at each warehouse
5. **Binary Operations**: Warehouses are either open (1) or closed (0)

### Objective

Minimize total cost = Production costs + Transportation costs + Fixed warehouse costs

## 🔧 Solution Format

Solutions are dictionaries with three components:

```python
solution = {
    'supplier_production': {
        'S1': 1500.0,  # Supplier S1 produces 1500 units
        'S2': 800.0    # Supplier S2 produces 800 units
    },
    'warehouse_flows': {
        ('S1', 'W1'): 1000.0,  # 1000 units from S1 to W1
        ('S2', 'W1'): 500.0,   # 500 units from S2 to W1
        ('W1', 'C1'): 850.0,   # 850 units from W1 to C1
        ('W1', 'C2'): 650.0    # 650 units from W1 to C2
    },
    'warehouse_operations': {
        'W1': 1,  # Warehouse W1 is open
        'W2': 0   # Warehouse W2 is closed
    }
}
```

## 📈 Example Output

```
=== Supply Chain Network Solution ===

Supplier Production:
  Global Materials Inc: 2300.0 units
  Regional Supply Co: 1650.0 units

Open Warehouses:
  Central Distribution
  North Hub

Customer Deliveries:
  Metro Retail Chain: 850.0 units from Central Distribution
  Downtown Stores: 620.0 units from North Hub
  Suburban Markets: 480.0 units from Central Distribution
```

## 🧪 Testing

```python
# Test problem functionality
problem = AutoProblem.from_repo("examples/supply_chain_network_problem")

# Generate multiple solutions
for i in range(5):
    solution = problem.random_solution()
    cost = problem.evaluate_solution(solution)
    feasible = problem.is_feasible(solution)
    print(f"Solution {i+1}: Cost=${cost:,.2f}, Feasible={feasible}")
```

## 🔗 Integration

This problem works seamlessly with supply chain optimizers:

```python
from qubots import AutoProblem, AutoOptimizer

# Load problem and optimizer
problem = AutoProblem.from_repo("examples/supply_chain_network_problem")
optimizer = AutoOptimizer.from_repo("examples/pulp_supply_chain_optimizer")

# Solve the problem
result = optimizer.optimize(problem)
print(f"Optimal cost: ${result.best_value:,.2f}")
```

## 📚 Applications

- **Manufacturing Networks**: Optimize production and distribution
- **Retail Supply Chains**: Minimize costs while meeting store demands
- **E-commerce Fulfillment**: Optimize warehouse locations and flows
- **Humanitarian Logistics**: Efficient disaster relief distribution
- **Food Distribution**: Fresh product supply chain optimization
