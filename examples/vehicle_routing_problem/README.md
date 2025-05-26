# Vehicle Routing Problem (VRP) - Qubots Implementation

A comprehensive Vehicle Routing Problem implementation using the qubots framework, designed for the Rastion platform playground.

## 🚛 Problem Description

The Vehicle Routing Problem (VRP) is a classic optimization challenge in logistics and operations research. Given a fleet of vehicles and a set of customers with known demands, the goal is to find optimal routes that:

- **Minimize total travel distance/cost**
- **Serve all customers exactly once**
- **Respect vehicle capacity constraints**
- **Start and end routes at depot locations**

## 🎯 Features

- **Flexible Problem Configuration**: Customizable customers, vehicles, and constraints
- **Real-time Evaluation**: Fast solution evaluation with detailed feedback
- **Constraint Handling**: Capacity constraints with penalty-based violations
- **Solution Validation**: Comprehensive format and feasibility checking
- **Detailed Analytics**: Rich solution summaries and performance metrics
- **Playground Compatible**: Optimized for Rastion platform integration

## 📊 Problem Instance

### Default Configuration
- **Customers**: 10 randomly distributed locations
- **Vehicles**: 3 vehicles with different capacities (40-60 units)
- **Depot**: Central location at coordinates (0, 0)
- **Constraints**: Vehicle capacity limits and customer coverage

### Customer Properties
- **Location**: (x, y) coordinates
- **Demand**: Required delivery quantity (5-25 units)
- **Service Time**: Time required at location (10-30 minutes)
- **Time Windows**: Optional delivery time constraints

### Vehicle Properties
- **Capacity**: Maximum load capacity
- **Cost per km**: Operating cost coefficient
- **Start/End Depot**: Route origin and destination

## 🔧 Usage

### Basic Usage

```python
from qubot import VehicleRoutingProblem

# Create problem instance
vrp = VehicleRoutingProblem()

# Generate random solution
solution = vrp.get_random_solution()
# Example: [[1, 3, 5], [2, 4], [6, 7, 8, 9, 10]]

# Evaluate solution
cost = vrp.evaluate_solution(solution)
print(f"Total cost: {cost:.2f}")

# Get detailed analysis
summary = vrp.get_solution_summary(solution)
print(f"Customers served: {summary['served_customers']}")
print(f"Feasible solution: {summary['feasible']}")
```

### Custom Configuration

```python
from qubot import Customer, Vehicle, VehicleRoutingProblem

# Define custom customers
customers = [
    Customer(id=1, x=2.0, y=3.0, demand=15),
    Customer(id=2, x=-1.0, y=4.0, demand=20),
    Customer(id=3, x=3.0, y=-2.0, demand=10)
]

# Define custom vehicles
vehicles = [
    Vehicle(id=1, capacity=50, cost_per_km=1.0),
    Vehicle(id=2, capacity=40, cost_per_km=0.8)
]

# Create custom problem
vrp = VehicleRoutingProblem(
    customers=customers,
    vehicles=vehicles,
    depot_location=(0.0, 0.0),
    penalty_unserved=1500.0,
    penalty_capacity=750.0
)
```

## 🎮 Rastion Playground Integration

### Loading in Playground

```python
import qubots.rastion as rastion

# Load the VRP problem
problem = rastion.load_qubots_model("vehicle_routing_problem")

# Access problem properties
print(f"Customers: {problem.n_customers}")
print(f"Vehicles: {problem.n_vehicles}")
print(f"Total demand: {sum(c.demand for c in problem.customers)}")
```

### Real-Time Execution with Streaming

The VRP problem supports real-time output streaming in the playground:

```python
# Enable verbose mode for real-time evaluation details
solution = problem.get_random_solution()
cost = problem.evaluate_solution(solution, verbose=True)

# Output will show:
# 🔍 Evaluating solution with 3 routes...
#    Route 1: 5 customers, cost: 45.23, demand: 85/100 (85.0%)
#    Route 2: 4 customers, cost: 38.67, demand: 70/100 (70.0%)
#    ⚠️ Capacity violation: +25.00 penalty
# ✅ Total cost: 108.90
```

### Parameter Configuration

The playground supports real-time parameter adjustment:

- **max_route_duration**: Maximum time per route (60-1440 minutes)
- **penalty_unserved**: Cost penalty for unserved customers (0-10000)
- **penalty_capacity**: Cost penalty per capacity violation (0-5000)

### Solution Format

Solutions are represented as lists of routes:
```python
solution = [
    [1, 3, 5],      # Vehicle 1 serves customers 1, 3, 5
    [2, 4],         # Vehicle 2 serves customers 2, 4
    [6, 7, 8, 9, 10] # Vehicle 3 serves customers 6, 7, 8, 9, 10
]
```

## 📈 Performance Metrics

The problem provides comprehensive evaluation metrics:

- **Total Cost**: Combined distance and penalty costs
- **Service Coverage**: Percentage of customers served
- **Capacity Utilization**: Vehicle load efficiency
- **Route Feasibility**: Constraint satisfaction status
- **Distance Statistics**: Total travel distance per vehicle

## 🔍 Solution Analysis

```python
# Get detailed solution breakdown
summary = vrp.get_solution_summary(solution)

print(f"Total cost: {summary['total_cost']:.2f}")
print(f"Served: {summary['served_customers']}/{summary['total_customers']}")
print(f"Vehicles used: {summary['vehicles_used']}/{summary['total_vehicles']}")

# Analyze individual routes
for route_info in summary['route_details']:
    print(f"Vehicle {route_info['vehicle_id']}:")
    print(f"  Route: {route_info['route']}")
    print(f"  Demand: {route_info['total_demand']}/{route_info['vehicle_capacity']}")
    print(f"  Utilization: {route_info['capacity_utilization']:.1%}")
    print(f"  Cost: {route_info['route_cost']:.2f}")
    print(f"  Feasible: {route_info['feasible']}")
```

## 🚀 Next Steps

1. **Load an optimizer** from the Rastion platform
2. **Run optimization** to find better solutions
3. **Compare algorithms** using different optimization approaches
4. **Visualize results** using the built-in dashboard features
5. **Share solutions** with the community

## 📚 Related Problems

- **Traveling Salesman Problem (TSP)**: Single vehicle routing
- **Capacitated VRP**: Focus on capacity constraints
- **VRP with Time Windows**: Additional temporal constraints
- **Multi-Depot VRP**: Multiple starting locations

---

**Compatible with**: Qubots Framework, Rastion Platform, OR-Tools optimizers
**Difficulty**: Intermediate
**Domain**: Routing & Logistics
