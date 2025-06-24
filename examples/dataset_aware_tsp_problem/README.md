# Dataset-Aware TSP Problem

A highly efficient Traveling Salesman Problem (TSP) implementation that accepts pre-loaded dataset content or loads from the Rastion platform. Designed for optimal performance in the modular qubots architecture.

## 🚀 Key Features

- **Efficient Dataset Handling**: Accepts pre-loaded content to avoid redundant API calls
- **Fallback Platform Loading**: Automatically loads from Rastion when needed
- **TSPLIB Format Support**: Full compatibility with TSPLIB instances
- **Multiple Distance Types**: EUC_2D, MAN_2D, MAX_2D, ATT, CEIL_2D
- **Flexible Input**: Coordinates or explicit distance matrices
- **Solution Validation**: Robust tour validation and neighbor generation

## 📦 Installation

```bash
pip install qubots
```

## 🎯 Efficient Usage Pattern (Recommended)

The most efficient way to use this qubot is to load the dataset once and pass the content directly:

```python
from qubots import AutoProblem, load_dataset_from_platform

# Step 1: Load dataset once
dataset_content = load_dataset_from_platform(
    token="your_rastion_token",
    dataset_id="your_dataset_id"
)

# Step 2: Create problem with pre-loaded content (efficient!)
problem = AutoProblem.from_repo(
    "examples/dataset_aware_tsp_problem",
    override_params={
        "dataset_content": dataset_content  # No redundant API calls!
    }
)

# Step 3: Use the problem
solution = problem.random_solution()
distance = problem.evaluate_solution(solution)
print(f"Tour distance: {distance}")
```

## 🔄 Fallback Usage Pattern

If you prefer, the qubot can also load datasets internally (less efficient):

```python
from qubots import AutoProblem

# Problem loads dataset internally (triggers API call)
problem = AutoProblem.from_repo(
    "examples/dataset_aware_tsp_problem",
    override_params={
        "dataset_id": "your_dataset_id",
        "auth_token": "your_rastion_token"
    }
)
```

## 📊 Performance Comparison

| Pattern | API Calls | Efficiency | Use Case |
|---------|-----------|------------|----------|
| **Pre-loaded content** | 1 (external) | ⭐⭐⭐⭐⭐ | Production, multiple problems |
| **Platform loading** | 1 (internal) | ⭐⭐⭐ | Quick testing, single use |

## 🧪 Complete Example

```python
from qubots import AutoProblem, AutoOptimizer, load_dataset_from_platform

# Load dataset efficiently
dataset_content = load_dataset_from_platform(
    token="your_token",
    dataset_id="tsp_berlin52"
)

# Create problem with pre-loaded content
problem = AutoProblem.from_repo(
    "examples/dataset_aware_tsp_problem",
    override_params={"dataset_content": dataset_content}
)

# Load an optimizer
optimizer = AutoOptimizer.from_repo("examples/genetic_tsp_optimizer")

# Solve the problem
result = optimizer.optimize(problem)

print(f"Best tour distance: {result.best_value}")
print(f"Runtime: {result.runtime_seconds:.2f}s")
```

## 🔧 Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `dataset_content` | string | No* | Pre-loaded dataset content (preferred) |
| `dataset_id` | string | No* | Rastion platform dataset ID (fallback) |
| `auth_token` | string | No* | Rastion authentication token (fallback) |

*Either `dataset_content` OR (`dataset_id` + `auth_token`) must be provided.

## 📋 Supported Dataset Formats

### TSPLIB Format
```
NAME: berlin52
TYPE: TSP
DIMENSION: 52
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 565.0 575.0
2 25.0 185.0
...
EOF
```

### Distance Matrix Format
```
NAME: example
TYPE: TSP
DIMENSION: 4
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX
EDGE_WEIGHT_SECTION
0 10 15 20
10 0 35 25
15 35 0 30
20 25 30 0
EOF
```

## 🎮 Problem Methods

### Core Methods
```python
# Generate random solution
solution = problem.random_solution()

# Evaluate solution
result = problem.evaluate_solution(solution)

# Check solution validity
is_valid = problem.is_valid_solution(solution)

# Generate neighbor solution (2-opt)
neighbor = problem.get_neighbor_solution(solution)

# Get problem information
info = problem.get_problem_info()
```

### Problem Information
```python
info = problem.get_problem_info()
# Returns:
# {
#     "n_cities": 52,
#     "edge_weight_type": "EUC_2D",
#     "edge_weight_format": "FUNCTION",
#     "has_coordinates": True,
#     "has_distance_matrix": True,
#     "dataset_id": "berlin52",
#     "dataset_loaded": True
# }
```

## 🏃‍♂️ Running the Example

```bash
# Run the efficient usage example
python examples/dataset_aware_tsp_problem/example_efficient_usage.py
```

## 🔍 Troubleshooting

### Common Issues

1. **"Dataset input required" Error**
   - Provide either `dataset_content` or (`dataset_id` + `auth_token`)

2. **"Failed to load dataset" Error**
   - Check your auth token and dataset ID
   - Verify network connectivity
   - Ensure dataset exists and is accessible

3. **"Invalid tour" Error**
   - Solution must be a valid permutation of city indices
   - All cities must be visited exactly once

### Debug Tips

```python
# Check if dataset loaded correctly
info = problem.get_problem_info()
print(f"Dataset loaded: {info['dataset_loaded']}")
print(f"Number of cities: {info['n_cities']}")

# Validate solution format
solution = [0, 1, 2, 3]  # Example for 4-city problem
print(f"Valid solution: {problem.is_valid_solution(solution)}")
```

## 🎯 Best Practices

1. **Use Pre-loaded Content**: Always prefer `dataset_content` parameter for efficiency
2. **Cache Dataset Content**: Store loaded content for reuse across multiple problems
3. **Validate Early**: Check dataset loading before creating multiple problem instances
4. **Handle Errors Gracefully**: Wrap dataset loading in try-catch blocks

## 🔗 Integration with Workflow Automation

This qubot is fully compatible with Rastion's workflow automation system:

```python
# Workflow automation will automatically pass dataset_content
# when connecting dataset nodes to problem nodes
```

## 📈 Performance Characteristics

- **Evaluation Complexity**: O(n) where n is number of cities
- **Memory Complexity**: O(n²) for distance matrix storage
- **Typical Sizes**: 
  - Small: ≤50 cities
  - Medium: 51-200 cities  
  - Large: 201-1000 cities

## 🤝 Contributing

This qubot follows the standard qubots framework patterns. To contribute:

1. Maintain backward compatibility
2. Add comprehensive tests
3. Update documentation
4. Follow the efficient dataset loading pattern
