# Traveling Salesman Problem with Time Windows (TSPTW)

A sophisticated variant of the Traveling Salesman Problem that adds time window constraints for each city. This implementation extends the classic TSP by requiring that each city be visited within its specified time window.

## Problem Description

The TSPTW seeks to find the shortest tour that visits all cities exactly once while respecting time window constraints. Each city has:

- **Time Window**: An earliest and latest time when the city can be visited
- **Service Time**: Time required to serve the customer at the city
- **Travel Time**: Time to travel between cities (based on distance and speed)

**Objective**: Minimize total cost (travel distance + time window violation penalties)

**Constraints**:
- Each city must be visited exactly once
- The tour must return to the starting city
- Cities should be visited within their time windows (soft constraint with penalty)
- Travel time affects arrival times at subsequent cities

## Features

- **Time Window Constraints**: Each city has configurable earliest and latest visit times
- **Travel Speed Control**: Configurable speed factor for travel time calculation
- **Service Times**: Configurable service time required at each city
- **Penalty System**: Soft constraints with configurable penalties for time window violations
- **TSPLIB Compatibility**: Uses the same distance data as standard TSP instances
- **Detailed Analysis**: Comprehensive time feasibility and violation reporting

## Installation

```bash
pip install qubots
```

## Usage

### Basic Usage

```python
from qubots import AutoProblem

# Load TSPTW problem with default settings
problem = AutoProblem.from_repo("examples/tsp_time_windows")

# Generate and evaluate a solution
solution = problem.random_solution()
cost = problem.evaluate_solution(solution)

print(f"Total cost: {cost}")
print(f"Tour: {solution}")
```

### Custom Time Windows

```python
from qubots import AutoProblem

# Define custom time windows for a 5-city problem
time_windows = [
    (0, 100),    # City 0: available 0-100
    (20, 80),    # City 1: available 20-80
    (40, 120),   # City 2: available 40-120
    (10, 60),    # City 3: available 10-60
    (50, 150)    # City 4: available 50-150
]

problem = AutoProblem.from_repo("examples/tsp_time_windows", override_params={
    "instance_file": "../tsp/instances/ulysses16.tsp",
    "time_windows": time_windows[:16],  # Adjust for 16 cities
    "travel_speed": 2.0,
    "time_penalty": 500.0
})
```

### Solution Analysis

```python
# Generate a solution
solution = problem.random_solution()

# Get detailed solution information
info = problem.get_solution_info(solution)

print(f"Feasible format: {info['feasible']}")
print(f"Time feasible: {info['time_feasible']}")
print(f"Total cost: {info['total_cost']}")
print(f"Pure distance: {info['total_distance']}")
print(f"Time penalty: {info['time_penalty']}")
print(f"Time violations: {info['time_violations']}")
print(f"Total tour time: {info['total_tour_time']}")

# Check specific violations
for violation in info['violations_detail']:
    print(f"City {violation['city']}: arrived at {violation['arrival_time']:.1f}, "
          f"latest allowed {violation['latest_time']:.1f}")
```

### Time Feasibility Check

```python
# Check if a solution satisfies time constraints
solution = problem.random_solution()
is_feasible, details = problem.is_time_feasible(solution)

print(f"Time feasible: {is_feasible}")
print(f"Total tour time: {details['total_time']:.1f}")

if not is_feasible:
    print(f"Violations: {len(details['violations'])}")
    for violation in details['violations']:
        print(f"  City {violation['city']}: late by {violation['violation']:.1f}")
```

### Local Search with Time Awareness

```python
# Perform local search considering time constraints
current_solution = problem.random_solution()
current_cost = problem.evaluate_solution(current_solution)

print(f"Starting cost: {current_cost}")

for iteration in range(100):
    neighbor = problem.get_neighbor_solution(current_solution)
    neighbor_cost = problem.evaluate_solution(neighbor)
    
    if neighbor_cost < current_cost:
        current_solution = neighbor
        current_cost = neighbor_cost
        
        # Check time feasibility of improvement
        time_feasible, _ = problem.is_time_feasible(current_solution)
        print(f"Iteration {iteration}: cost={current_cost:.1f}, "
              f"time_feasible={time_feasible}")

print(f"Final cost: {current_cost}")
```

## Configuration Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instance_file` | string | "../tsp/instances/att48.tsp" | TSPLIB instance file |
| `travel_speed` | number | 1.0 | Speed factor for travel time calculation |
| `time_penalty` | number | 1000.0 | Penalty for time window violations |

### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_windows` | array | auto-generated | List of (earliest, latest) times for each city |
| `service_times` | array | [10.0, ...] | Service time required at each city |

## Time Window Generation

When time windows are not explicitly provided, the system generates reasonable defaults:

1. **Estimates total tour time** based on average distances
2. **Creates staggered windows** to provide interesting constraints
3. **Ensures depot availability** throughout the planning horizon
4. **Allows flexibility** with window sizes proportional to tour time

## Solution Format

Solutions are represented as permutations of city indices, same as standard TSP:

```python
# Example for 5 cities
solution = [0, 3, 1, 4, 2]  # Visit order: 0 → 3 → 1 → 4 → 2 → 0
```

## Evaluation Method

The total cost combines distance and time penalties:

```
total_cost = sum(distances) + sum(time_violations * penalty)
```

Where time violations are calculated as:
```
violation = max(0, arrival_time - latest_allowed_time)
```

## Available Instances

The TSPTW implementation can use any TSPLIB instance from the base TSP collection:

- **Small**: gr17, ulysses16, att48, berlin52
- **Medium**: kroA100, ch150, d198
- **Large**: a280, pcb442, d657

## Performance Considerations

- **Evaluation Complexity**: O(n) where n is the number of cities
- **Memory Complexity**: O(n²) for distance matrix + O(n) for time data
- **Time Calculation**: Additional O(n) overhead for time window checking

## API Reference

### Main Methods

- `evaluate_solution(solution)`: Calculate total cost (distance + penalties)
- `is_time_feasible(solution)`: Check time window constraints
- `get_solution_info(solution)`: Get comprehensive solution analysis
- `get_time_windows()`: Get time windows for all cities
- `get_service_times()`: Get service times for all cities

### Time-Specific Methods

- `calculate_travel_time(from_city, to_city)`: Get travel time between cities
- `is_feasible_format(solution)`: Check solution format validity

## Applications

TSPTW is relevant for many real-world scenarios:

- **Delivery Services**: Package delivery with customer availability windows
- **Service Scheduling**: Maintenance visits with appointment times
- **Public Transportation**: Bus/train scheduling with timetables
- **Healthcare**: Home healthcare visits with patient preferences

## Contributing

This implementation is part of the Qubots community. Contributions for additional constraint types, heuristics, or performance improvements are welcome.

## References

- Solomon, M.M. "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints." Operations Research, 1987.
- Savelsbergh, M.W.P. "Local Search in Routing Problems with Time Windows." Annals of Operations Research, 1985.
