# Qubots Examples

This directory contains a comprehensive collection of optimization problems and solvers that demonstrate the capabilities of the qubots framework. These examples showcase different optimization domains, solver types, and integration patterns with the Rastion platform.

## 📋 Available Examples

### Problems

#### 🔗 MaxCut Problem (`maxcut_problem/`)
- **Description**: Graph partitioning optimization to maximize edge cuts between two sets
- **Features**: Multiple graph types (random, complete, cycle, grid), configurable density and weights
- **Use Cases**: Network analysis, clustering, VLSI design
- **Compatible Solvers**: OR-Tools MaxCut Optimizer

#### 🚚 Vehicle Routing Problem (`vehicle_routing_problem/`)
- **Description**: Multi-vehicle routing optimization with capacity constraints
- **Features**: Configurable fleet size, customer demands, depot locations
- **Use Cases**: Logistics, delivery optimization, supply chain management
- **Compatible Solvers**: Genetic VRP Optimizer

#### 🗺️ Traveling Salesman Problem (`tsp/`)
- **Description**: Classic TSP with TSPLIB format support
- **Features**: Multiple distance types (EUC_2D, ATT, GEO), instance loading
- **Use Cases**: Route optimization, scheduling, manufacturing
- **Compatible Solvers**: HiGHS TSP Solver

#### ⏰ TSP with Time Windows (`tsp_time_windows/`)
- **Description**: TSP variant with delivery time constraints
- **Features**: Time window validation, penalty-based evaluation
- **Use Cases**: Delivery scheduling, service routing
- **Compatible Solvers**: HiGHS TSP Solver (with modifications)

#### 📦 TSP with Capacity Constraints (`tsp_capacity_constraints/`)
- **Description**: TSP with vehicle capacity limitations
- **Features**: Multi-trip support, capacity feasibility checking
- **Use Cases**: Pickup and delivery, cargo transportation
- **Compatible Solvers**: HiGHS TSP Solver (with modifications)

### Optimizers

#### 🔧 OR-Tools MaxCut Optimizer (`ortools_maxcut_optimizer/`)
- **Description**: Integer programming solver using Google OR-Tools
- **Features**: SAT preprocessing, symmetry breaking, parallel search
- **Solver Type**: Exact/Heuristic hybrid
- **Compatible Problems**: MaxCut Problem

#### 🧬 Genetic VRP Optimizer (`genetic_vrp_optimizer/`)
- **Description**: Evolutionary algorithm for vehicle routing optimization
- **Features**: Adaptive parameters, tournament selection, diversity control
- **Solver Type**: Metaheuristic
- **Compatible Problems**: Vehicle Routing Problem

#### ⚡ HiGHS TSP Solver (`highs_tsp_solver/`)
- **Description**: Linear programming solver using HiGHS for TSP
- **Features**: Miller-Tucker-Zemlin formulation, subtour elimination
- **Solver Type**: Exact (Integer Linear Programming)
- **Compatible Problems**: TSP, TSP with Time Windows, TSP with Capacity Constraints

## 🚀 Running Examples Locally

### Prerequisites

1. **Install qubots framework**:
   ```bash
   pip install qubots
   ```

2. **Install example-specific dependencies** (if needed):
   ```bash
   # For OR-Tools examples
   pip install ortools

   # For HiGHS examples  
   pip install highspy

   # For genetic algorithm examples
   pip install numpy scipy
   ```

### Basic Usage

#### Method 1: Using AutoProblem and AutoOptimizer

```python
from qubots import AutoProblem, AutoOptimizer

# Load problem and optimizer from local examples
problem = AutoProblem.from_repo("examples/maxcut_problem")
optimizer = AutoOptimizer.from_repo("examples/ortools_maxcut_optimizer")

# Run optimization
result = optimizer.optimize(problem)
print(f"Best solution value: {result.best_value}")
print(f"Runtime: {result.runtime_seconds:.2f} seconds")
```

#### Method 2: Using the Testing Script

```bash
# Test compatibility and run optimization
python examples/load_and_test_optimization.py examples/maxcut_problem examples/ortools_maxcut_optimizer

# Run multiple iterations for statistical analysis
python examples/load_and_test_optimization.py examples/tsp examples/highs_tsp_solver --iterations 5

# Quiet mode for automated testing
python examples/load_and_test_optimization.py examples/vehicle_routing_problem examples/genetic_vrp_optimizer --quiet
```

### Advanced Configuration

#### Custom Parameters

```python
from qubots import AutoProblem, AutoOptimizer

# Load problem with custom parameters
problem = AutoProblem.from_repo("examples/maxcut_problem", override_params={
    "n_vertices": 20,
    "graph_type": "random",
    "density": 0.3
})

# Load optimizer with custom parameters
optimizer = AutoOptimizer.from_repo("examples/ortools_maxcut_optimizer", override_params={
    "time_limit": 60.0,
    "num_search_workers": 4,
    "log_search_progress": True
})

# Run optimization
result = optimizer.optimize(problem)
```

#### Progress Monitoring

```python
def progress_callback(iteration, best_value, current_value):
    print(f"Iteration {iteration}: Best={best_value}, Current={current_value}")

def log_callback(level, message, source):
    print(f"[{level.upper()}] {source}: {message}")

result = optimizer.optimize(
    problem,
    progress_callback=progress_callback,
    log_callback=log_callback
)
```

## 📤 How Examples Were Uploaded to Rastion

The examples in this directory have been uploaded to the Rastion platform using the `upload_repo_to_rastion.py` script. This allows them to be accessed remotely and tested in the Rastion playground environment.

### Upload Process

1. **Prepare Repository**: Each example follows the standard qubots repository structure:
   ```
   example_name/
   ├── qubot.py          # Main implementation
   ├── config.json       # Configuration and metadata
   ├── requirements.txt  # Dependencies (optional)
   └── README.md        # Documentation (optional)
   ```

2. **Upload to Platform**:
   ```bash
   # Upload a problem
   python examples/upload_repo_to_rastion.py ./examples/maxcut_problem \
       --name "maxcut_problem" \
       --description "Graph partitioning optimization problem" \
       --token YOUR_RASTION_TOKEN

   # Upload an optimizer
   python examples/upload_repo_to_rastion.py ./examples/ortools_maxcut_optimizer \
       --name "ortools_maxcut_optimizer" \
       --description "OR-Tools based MaxCut solver" \
       --token YOUR_RASTION_TOKEN
   ```

3. **Verification**: Test uploaded examples using the platform:
   ```python
   from qubots import AutoProblem, AutoOptimizer

   # Load from Rastion platform (uploaded examples)
   problem = AutoProblem.from_repo("username/maxcut_problem")
   optimizer = AutoOptimizer.from_repo("username/ortools_maxcut_optimizer")
   ```

### Repository Configuration

Each example includes a `config.json` file with proper metadata:

```json
{
    "type": "problem",  // or "optimizer"
    "entry_point": "qubot",
    "class_name": "MaxCutProblem",
    "metadata": {
        "name": "MaxCut Problem",
        "description": "Graph partitioning optimization problem",
        "author": "Qubots Community",
        "version": "1.0.0"
    },
    "parameters": {
        "n_vertices": {
            "type": "integer",
            "default": 10,
            "min": 3,
            "max": 100,
            "description": "Number of vertices in the graph"
        }
    }
}
```

## 🧪 Testing and Validation

### Automated Testing

Run the comprehensive test suite to validate all examples:

```bash
# Test all problem-optimizer combinations
python examples/run_playground_tests.py

# Test specific combinations
python examples/playground_consistency_test.py
```

### Manual Validation

```python
from qubots import AutoProblem, AutoOptimizer

# Test each example individually
examples = [
    ("examples/maxcut_problem", "examples/ortools_maxcut_optimizer"),
    ("examples/tsp", "examples/highs_tsp_solver"),
    ("examples/vehicle_routing_problem", "examples/genetic_vrp_optimizer")
]

for problem_repo, optimizer_repo in examples:
    try:
        problem = AutoProblem.from_repo(problem_repo)
        optimizer = AutoOptimizer.from_repo(optimizer_repo)
        result = optimizer.optimize(problem)
        print(f"✅ {problem_repo} + {optimizer_repo}: {result.best_value}")
    except Exception as e:
        print(f"❌ {problem_repo} + {optimizer_repo}: {e}")
```

## 🔗 Integration with Rastion Playground

All examples are designed to work seamlessly with the Rastion playground environment:

1. **Interactive Testing**: Load and test examples directly in the web interface
2. **Parameter Experimentation**: Modify parameters and see real-time results
3. **Performance Comparison**: Compare different solvers on the same problem
4. **Collaborative Development**: Share and modify examples with the community

### Playground Usage

1. Visit [Rastion Playground](https://rastion.com/playground)
2. Select a problem from the examples (e.g., "maxcut_problem")
3. Select a compatible optimizer (e.g., "ortools_maxcut_optimizer")
4. Adjust parameters as needed
5. Run optimization and analyze results

## 📚 Learning Path

### For Beginners
1. Start with **MaxCut Problem** + **OR-Tools MaxCut Optimizer**
2. Experiment with different graph sizes and types
3. Try **TSP** + **HiGHS TSP Solver** for exact optimization
4. Explore **VRP** + **Genetic VRP Optimizer** for metaheuristics

### For Advanced Users
1. Modify existing examples to create new variants
2. Implement custom solvers for existing problems
3. Create new problem types following the established patterns
4. Contribute improvements back to the community

## 🤝 Contributing New Examples

To add new examples to this collection:

1. **Follow the Repository Structure**: Use the standard qubots format
2. **Include Comprehensive Documentation**: README with usage examples
3. **Add Configuration**: Proper `config.json` with parameter definitions
4. **Test Thoroughly**: Ensure compatibility and performance
5. **Upload to Platform**: Make available for community use

For detailed contribution guidelines, see the main [CONTRIBUTING.md](../CONTRIBUTING.md) file.
