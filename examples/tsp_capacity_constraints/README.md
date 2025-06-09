# Traveling Salesman Problem with Capacity Constraints (TSP-CC)

A sophisticated variant of the Traveling Salesman Problem that adds vehicle capacity constraints. This implementation extends the classic TSP by requiring that the vehicle's capacity not be exceeded when serving customers.

## Problem Description

The TSP-CC seeks to find the shortest tour that visits all cities while respecting vehicle capacity constraints. The problem features:

- **Vehicle Capacity**: Maximum load the vehicle can carry
- **Customer Demands**: Each city has a demand/pickup requirement
- **Depot Operations**: Vehicle starts and can return to depot (city 0)
- **Multi-trip Capability**: Option to allow multiple trips when capacity is exceeded

**Objective**: Minimize total travel distance (including depot returns if multi-trip)

**Constraints**:
- Each city must be visited exactly once
- Vehicle capacity must not be exceeded
- Vehicle starts and ends at depot
- Depot has zero demand

## Features

- **Capacity Constraints**: Configurable vehicle capacity and customer demands
- **Multi-trip Mode**: Automatic trip splitting when capacity is reached
- **Single-trip Mode**: Penalty-based soft constraints for capacity violations
- **TSPLIB Compatibility**: Uses the same distance data as standard TSP instances
- **Flexible Evaluation**: Different evaluation methods for multi-trip vs single-trip modes
- **Detailed Analysis**: Comprehensive capacity feasibility and trip analysis

## Installation

```bash
pip install qubots
```

## Usage

### Basic Usage (Multi-trip Mode)

```python
from qubots import AutoProblem

# Load TSP-CC problem with default settings
problem = AutoProblem.from_repo("examples/tsp_capacity_constraints")

# Generate and evaluate a solution
solution = problem.random_solution()
cost = problem.evaluate_solution(solution)

print(f"Total cost: {cost}")
print(f"Tour: {solution}")

# Check capacity feasibility
feasible, details = problem.is_capacity_feasible(solution)
print(f"Capacity feasible: {feasible}")
print(f"Number of trips: {details['num_trips']}")
```

### Custom Capacity and Demands

```python
from qubots import AutoProblem

# Define custom demands for a 5-city problem
demands = [0.0, 25.0, 30.0, 20.0, 35.0]  # Depot has 0 demand

problem = AutoProblem.from_repo("examples/tsp_capacity_constraints", override_params={
    "instance_file": "../tsp/instances/ulysses16.tsp",
    "vehicle_capacity": 80.0,
    "demands": demands[:16],  # Adjust for 16 cities
    "allow_multi_trip": True
})

# Get instance information
info = problem.get_instance_info()
print(f"Total demand: {info['total_demand']}")
print(f"Vehicle capacity: {info['vehicle_capacity']}")
print(f"Minimum trips needed: {info['min_trips_needed']:.1f}")
```

### Single-trip Mode with Penalties

```python
from qubots import AutoProblem

# Use single-trip mode with capacity penalties
problem = AutoProblem.from_repo("examples/tsp_capacity_constraints", override_params={
    "vehicle_capacity": 50.0,
    "allow_multi_trip": False,
    "capacity_penalty": 500.0
})

solution = problem.random_solution()
info = problem.get_solution_info(solution)

print(f"Total cost: {info['total_cost']}")
print(f"Pure distance: {info['total_distance']}")
print(f"Capacity penalty: {info['capacity_penalty']}")
print(f"Capacity feasible: {info['capacity_feasible']}")
```

### Solution Analysis

```python
# Generate a solution
solution = problem.random_solution()

# Get detailed solution information
info = problem.get_solution_info(solution)

print(f"Feasible format: {info['feasible']}")
print(f"Capacity feasible: {info['capacity_feasible']}")
print(f"Total cost: {info['total_cost']}")

if info['allow_multi_trip']:
    print(f"Number of trips: {info['num_trips']}")
    print(f"Trip loads: {info['trip_loads']}")
    
    # Show individual trips
    for i, trip in enumerate(info['trips']):
        load = info['trip_loads'][i]
        print(f"Trip {i+1}: {trip} (load: {load:.1f})")
else:
    print(f"Total load: {info['total_load']}")
    print(f"Capacity violation: {info['violation']}")
```

### Capacity-aware Local Search

```python
# Perform local search considering capacity constraints
current_solution = problem.random_solution()
current_cost = problem.evaluate_solution(current_solution)

print(f"Starting cost: {current_cost}")

for iteration in range(100):
    neighbor = problem.get_neighbor_solution(current_solution)
    neighbor_cost = problem.evaluate_solution(neighbor)
    
    if neighbor_cost < current_cost:
        current_solution = neighbor
        current_cost = neighbor_cost
        
        # Analyze capacity utilization
        feasible, details = problem.is_capacity_feasible(current_solution)
        if problem.allow_multi_trip:
            trips_info = f"trips={details['num_trips']}"
        else:
            trips_info = f"load={details['total_load']:.1f}/{problem.vehicle_capacity}"
        
        print(f"Iteration {iteration}: cost={current_cost:.1f}, {trips_info}")

print(f"Final cost: {current_cost}")
```

## Configuration Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instance_file` | string | "../tsp/instances/att48.tsp" | TSPLIB instance file |
| `vehicle_capacity` | number | 100.0 | Maximum vehicle capacity |
| `allow_multi_trip` | boolean | true | Allow multiple trips to depot |

### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `demands` | array | auto-generated | Demand for each city |
| `capacity_penalty` | number | 1000.0 | Penalty for capacity violations (single-trip mode) |

## Operating Modes

### Multi-trip Mode (`allow_multi_trip=true`)

- **Behavior**: Vehicle automatically returns to depot when capacity would be exceeded
- **Evaluation**: Sum of distances for all trips including depot returns
- **Feasibility**: Hard capacity constraints - always feasible
- **Use case**: Realistic vehicle routing scenarios

### Single-trip Mode (`allow_multi_trip=false`)

- **Behavior**: Single continuous tour with capacity violations penalized
- **Evaluation**: Distance + penalty for capacity violations
- **Feasibility**: Soft capacity constraints with configurable penalties
- **Use case**: Theoretical analysis or when depot returns are not allowed

## Demand Generation

When demands are not explicitly provided, the system generates reasonable defaults:

1. **Depot demand**: Always set to 0.0
2. **Average demand**: Calculated as `capacity / (cities * 0.6)` for flexibility
3. **Variation**: Random demands around average with ±50% variation
4. **Constraint satisfaction**: Ensures interesting but solvable instances

## Solution Format

Solutions are represented as permutations of city indices, same as standard TSP:

```python
# Example for 5 cities
solution = [0, 3, 1, 4, 2]  # Visit order: 0 → 3 → 1 → 4 → 2 → 0
```

## Evaluation Methods

### Multi-trip Evaluation
```
total_cost = sum(trip_distances for all trips)
where each trip includes return to depot
```

### Single-trip Evaluation
```
total_cost = tour_distance + capacity_violations * penalty
where capacity_violations = max(0, total_load - capacity)
```

## Available Instances

The TSP-CC implementation can use any TSPLIB instance from the base TSP collection:

- **Small**: gr17, ulysses16, att48, berlin52
- **Medium**: kroA100, ch150, d198  
- **Large**: a280, pcb442, d657

## Performance Considerations

- **Evaluation Complexity**: O(n) where n is the number of cities
- **Memory Complexity**: O(n²) for distance matrix + O(n) for demand data
- **Trip Splitting**: Additional O(n) overhead for multi-trip mode

## API Reference

### Main Methods

- `evaluate_solution(solution)`: Calculate total cost
- `is_capacity_feasible(solution)`: Check capacity constraints
- `split_into_trips(tour)`: Split tour into capacity-feasible trips
- `get_solution_info(solution)`: Get comprehensive solution analysis
- `get_demands()`: Get demand values for all cities
- `get_vehicle_capacity()`: Get vehicle capacity

### Capacity-Specific Methods

- `calculate_route_load(route)`: Calculate total load for a route segment
- `is_feasible_format(solution)`: Check solution format validity

## Applications

TSP-CC is relevant for many real-world scenarios:

- **Delivery Services**: Package delivery with vehicle capacity limits
- **Waste Collection**: Garbage trucks with limited capacity
- **Supply Chain**: Distribution with load constraints
- **Service Industries**: Mobile services with equipment/supply limits

## Contributing

This implementation is part of the Qubots community. Contributions for additional constraint types, heuristics, or performance improvements are welcome.

## References

- Laporte, G. "The Vehicle Routing Problem: An overview of exact and approximate algorithms." European Journal of Operational Research, 1992.
- Toth, P. and Vigo, D. "The Vehicle Routing Problem." SIAM Monographs on Discrete Mathematics and Applications, 2002.
