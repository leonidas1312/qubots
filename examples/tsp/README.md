# Traveling Salesman Problem (TSP)

A comprehensive implementation of the Traveling Salesman Problem for the Qubots framework, supporting the full TSPLIB format with multiple distance calculation methods.

## Problem Description

The Traveling Salesman Problem (TSP) is a classic combinatorial optimization problem where the goal is to find the shortest possible route that visits each city exactly once and returns to the starting city.

**Objective**: Minimize the total travel distance of the tour.

**Constraints**: 
- Each city must be visited exactly once
- The tour must return to the starting city
- All cities must be included in the tour

## Features

- **TSPLIB Format Support**: Full compatibility with TSPLIB instances
- **Multiple Distance Types**: Support for EUC_2D, ATT, GEO, MAN_2D, MAX_2D, CEIL_2D, and more
- **Flexible Input**: Handles both explicit distance matrices and coordinate-based instances
- **2-opt Neighborhood**: Built-in local search neighborhood generation
- **Comprehensive Validation**: Robust solution format checking
- **Detailed Analysis**: Rich solution information and statistics

## Installation

```bash
pip install qubots
```

## Usage

### Basic Usage

```python
from qubots import AutoProblem

# Load TSP problem with default instance (att48)
problem = AutoProblem.from_repo("examples/tsp")

# Generate and evaluate a random solution
solution = problem.random_solution()
distance = problem.evaluate_solution(solution)

print(f"Tour distance: {distance}")
print(f"Tour: {solution}")
```

### Custom Instance

```python
from qubots import AutoProblem

# Load with a specific TSPLIB instance
problem = AutoProblem.from_repo("examples/tsp", override_params={
    "instance_file": "instances/berlin52.tsp"
})

# Get problem information
info = problem.get_instance_info()
print(f"Instance: {info['name']}")
print(f"Cities: {info['dimension']}")
print(f"Distance type: {info['edge_weight_type']}")
```

### Solution Analysis

```python
# Generate a solution
solution = problem.random_solution()

# Get detailed solution information
solution_info = problem.get_solution_info(solution)
print(f"Feasible: {solution_info['feasible']}")
print(f"Total distance: {solution_info['total_distance']}")
print(f"Average edge distance: {solution_info['avg_edge_distance']:.2f}")
print(f"Min/Max edge: {solution_info['min_edge_distance']}/{solution_info['max_edge_distance']}")
```

### Local Search

```python
# Start with a random solution
current_solution = problem.random_solution()
current_distance = problem.evaluate_solution(current_solution)

# Perform local search with 2-opt moves
for _ in range(100):
    neighbor = problem.get_neighbor_solution(current_solution)
    neighbor_distance = problem.evaluate_solution(neighbor)
    
    if neighbor_distance < current_distance:
        current_solution = neighbor
        current_distance = neighbor_distance
        print(f"Improved to: {current_distance}")
```

## Available Instances

The implementation includes a comprehensive collection of TSPLIB instances:

### Small Instances (< 50 cities)
- `gr17.tsp` - 17 cities
- `ulysses16.tsp` - 16 cities  
- `att48.tsp` - 48 cities (default)
- `berlin52.tsp` - 52 cities

### Medium Instances (50-200 cities)
- `kroA100.tsp` - 100 cities
- `ch150.tsp` - 150 cities
- `d198.tsp` - 198 cities

### Large Instances (200+ cities)
- `a280.tsp` - 280 cities
- `pcb442.tsp` - 442 cities
- `d657.tsp` - 657 cities

## Distance Types Supported

- **EUC_2D**: Euclidean distance in 2D
- **ATT**: Pseudo-Euclidean distance (ATT)
- **GEO**: Geographical distance
- **MAN_2D**: Manhattan distance in 2D
- **MAX_2D**: Maximum distance in 2D
- **CEIL_2D**: Ceiling of Euclidean distance
- **EXPLICIT**: Explicit distance matrix

## Solution Format

Solutions are represented as permutations of city indices:

```python
# Example for 5 cities (0, 1, 2, 3, 4)
solution = [0, 3, 1, 4, 2]  # Visit cities in this order
# Tour: 0 → 3 → 1 → 4 → 2 → 0 (return to start)
```

## API Reference

### Main Methods

- `evaluate_solution(solution)`: Calculate tour distance
- `random_solution()`: Generate random valid tour
- `is_feasible(solution)`: Check if solution is valid
- `get_neighbor_solution(solution)`: Generate 2-opt neighbor
- `get_solution_info(solution)`: Get detailed solution analysis
- `get_instance_info()`: Get problem instance information

### Properties

- `nb_cities`: Number of cities
- `dist_matrix`: Distance matrix
- `coords`: City coordinates (if available)
- `instance_name`: Name of the instance

## Performance Notes

- **Evaluation Complexity**: O(n) where n is the number of cities
- **Memory Complexity**: O(n²) for distance matrix storage
- **Neighborhood Generation**: O(1) for single 2-opt move

## Contributing

This implementation is part of the Qubots community. Contributions for additional distance types, neighborhood operators, or performance improvements are welcome.

## References

- Reinelt, G. "TSPLIB--A Traveling Salesman Problem Library." ORSA Journal on Computing, Vol. 3, No. 4, pp. 376-384. Fall 1991.
- [TSPLIB Website](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)
