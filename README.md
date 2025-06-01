# Qubots: Enterprise Optimization Framework

[![PyPI version](https://img.shields.io/pypi/v/qubots.svg)](https://pypi.org/project/qubots/)
[![Build Status](https://github.com/leonidas1312/qubots/actions/workflows/publish.yml/badge.svg)](https://github.com/leonidas1312/qubots/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/leonidas1312/qubots.svg)](https://github.com/leonidas1312/qubots/issues)
[![GitHub forks](https://img.shields.io/github/forks/leonidas1312/qubots.svg)](https://github.com/leonidas1312/qubots/network)

**Qubots** is a  Python optimization framework that enables developers to build, share, and deploy optimization solutions. With its innovative `AutoProblem` and `AutoOptimizer` components, qubots transforms complex optimization challenges into modular, reusable components that integrate seamlessly with the [Rastion platform](https://rastion.com) for collaborative development and deployment.

## 🎯 Framework Overview

Qubots is designed for environments where optimization problems need to be:
- **Modular and Reusable**: Problems and optimizers are independent components that can be mixed and matched
- **Collaborative**: Teams can share and build upon each other's optimization components
- **Production-Ready**: Robust error handling, logging, and monitoring capabilities

The framework centers around two core concepts:
- **AutoProblem**: Automatically loads and instantiates optimization problems from repositories
- **AutoOptimizer**: Automatically loads and instantiates optimization algorithms from repositories

## 🚀 Key Features

- **🔧 Auto-Loading Architecture**: Seamlessly load problems and optimizers from repositories with `AutoProblem` and `AutoOptimizer`
- **🌐 Repository Integration**: Direct integration with Git repositories for version-controlled optimization components
- **🎯 Enterprise-Ready**: Built for production environments with comprehensive error handling and logging
- **⚡ High Performance**: Integration with industry-standard optimization libraries (OR-Tools, CasADi, CPLEX, Gurobi)
- **📊 Advanced Benchmarking**: Built-in performance testing, comparison tools, and dashboard generation
- **🔍 Model Discovery**: Search and discover optimization models from the community and enterprise repositories
- **📈 Real-time Monitoring**: Progress tracking, logging callbacks, and optimization result visualization
- **🛠️ Utility Tools**: Comprehensive scripts for repository management and optimization testing

## 📦 Installation

### Basic Installation

Install qubots from PyPI:

```bash
pip install qubots
```

### Domain-Specific Dependencies

For domain-specific optimizations, install optional dependencies:

```bash
# For routing and scheduling 
pip install qubots[routing]

# For continuous optimization 
pip install qubots[continuous]

# For finance optimization 
pip install qubots[finance]

# For energy optimization 
pip install qubots[energy]

# For all features
pip install qubots[all]
```

### Development Installation

For development and testing:

```bash
# Clone the repository
git clone https://github.com/Rastion/qubots.git
cd qubots

# Install in development mode
pip install -e .[dev]
```

## 🏗️ Architecture

The qubots framework is built around a modular architecture with the following core components:

### Core Components

1. **AutoProblem**: Dynamically loads optimization problems from repositories
   - Handles Git repository cloning and caching
   - Validates problem structure and metadata
   - Instantiates problem classes with configurable parameters

2. **AutoOptimizer**: Dynamically loads optimization algorithms from repositories
   - Supports parameter overrides and customization
   - Validates optimizer compatibility with problems
   - Provides comprehensive optimization result tracking

3. **BaseProblem**: Abstract base class for all optimization problems
   - Standardized evaluation interface
   - Metadata management and validation
   - Solution format validation and neighbor generation

4. **BaseOptimizer**: Abstract base class for all optimization algorithms
   - Comprehensive optimization result tracking
   - Progress monitoring and logging callbacks
   - Parameter validation and management

### Repository Structure

Each qubots repository follows a standardized structure:

```
repository/
├── qubot.py          # Main implementation file
├── config.json       # Configuration and metadata
├── requirements.txt  # Python dependencies (optional)
└── README.md        # Documentation (optional)
```

## 🚀 Quick Start

### Loading and Running Optimizations

Here's how to load problems and optimizers from repositories and run optimizations:

```python
from qubots import AutoProblem, AutoOptimizer

# Load a problem from a repository
problem = AutoProblem.from_repo("ileo/demo-maxcut")

# Load an optimizer from a repository
optimizer = AutoOptimizer.from_repo("ileo/demo-ortools-maxcut-optimizer")

# Run optimization
result = optimizer.optimize(problem)

print(f"Best Solution: {result.best_solution}")
print(f"Best Value: {result.best_value}")
print(f"Runtime: {result.runtime_seconds:.3f} seconds")
print(f"Iterations: {result.iterations}")
```

### Advanced Configuration

You can override parameters when loading problems and optimizers:

```python
from qubots import AutoProblem, AutoOptimizer

# Load problem with custom parameters
problem = AutoProblem.from_repo("ileo/demo-maxcut", override_params={
    "n_vertices": 20,
    "graph_type": "random",
    "density": 0.3
})

# Load optimizer with custom parameters
optimizer = AutoOptimizer.from_repo("ileo/demo-ortools-maxcut-optimizer", override_params={
    "time_limit": 60.0,
    "num_search_workers": 4,
    "log_search_progress": True
})

# Run optimization with callbacks
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

## 🌐 Rastion Platform Integration

Qubots integrates with the Rastion platform for collaborative optimization development:

### Repository-Based Loading

The framework loads optimization components directly from Git repositories:

```python
from qubots import AutoProblem, AutoOptimizer

# Load from public repositories
problem = AutoProblem.from_repo("username/problem-repo")
optimizer = AutoOptimizer.from_repo("username/optimizer-repo")

# Load from specific branches or revisions
problem = AutoProblem.from_repo("username/problem-repo", revision="v1.2.0")
optimizer = AutoOptimizer.from_repo("username/optimizer-repo", revision="development")
```

### Caching and Performance

The framework automatically caches repositories for improved performance:

```python
# Repositories are cached in ~/.cache/rastion_hub by default
# Subsequent loads are much faster

# Custom cache directory
problem = AutoProblem.from_repo(
    "username/problem-repo",
    cache_dir="./my_cache"
)
```

## 🛠️ Utility Scripts

Qubots includes two powerful utility scripts for repository management and testing:

### upload_repo_to_rastion.py

This script uploads qubots repositories to the Rastion platform for sharing and collaboration.

#### Purpose
- Upload optimization problems and algorithms to the Rastion platform
- Validate repository structure and configuration
- Handle authentication and error reporting
- Support for both public and private repositories

#### Usage

```bash
# Basic upload with auto-detected metadata
python examples/upload_repo_to_rastion.py ./my_optimizer --token YOUR_RASTION_TOKEN

# Upload with custom name and description
python examples/upload_repo_to_rastion.py ./my_problem \
    --name "custom_vrp_solver" \
    --description "Advanced VRP solver with time windows"
    --token YOUR_RASTION_TOKEN

# Upload as private repository with overwrite
python examples/upload_repo_to_rastion.py ./my_optimizer \
    --private --overwrite --token YOUR_RASTION_TOKEN

# Upload with custom requirements
python examples/upload_repo_to_rastion.py ./my_problem \
    --requirements "qubots,numpy>=1.20.0,ortools>=9.0.0"
    --token YOUR_RASTION_TOKEN
```

#### Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `repo_path` | Path to the repository directory | Yes |
| `--name` | Repository name (auto-detected if not provided) | No |
| `--description` | Repository description (auto-detected if not provided) | No |
| `--private` | Make repository private | No |
| `--overwrite` | Overwrite existing repository | No |
| `--requirements` | Comma-separated Python requirements | No |
| `--token` | Rastion authentication token | Yes* |

*Token can also be set via `RASTION_TOKEN` environment variable

#### Repository Structure Requirements

Your repository must contain:
- `qubot.py`: Main implementation file
- `config.json`: Configuration with required fields:
  ```json
  {
    "type": "problem" or "optimizer",
    "entry_point": "qubot",
    "class_name": "YourClassName",
    "metadata": {
      "name": "Your Model Name",
      "description": "Model description"
    }
  }
  ```

#### Examples

```bash
# Upload a VRP problem
python examples/upload_repo_to_rastion.py ./vrp_problem \
    --name "vrp_timewindows" \
    --description "Vehicle Routing Problem with time windows and capacity constraints"
    --token YOUR_RASTION_TOKEN

# Upload an optimizer with specific requirements
python examples/upload_repo_to_rastion.py ./genetic_optimizer \
    --requirements "qubots,numpy>=1.20.0,scipy>=1.7.0" \
    --private
    --token YOUR_RASTION_TOKEN
```

### load_and_test_optimization.py

This script loads problems and optimizers from repositories and runs comprehensive testing to validate compatibility and performance.

#### Purpose
- Load and test optimization models from repositories
- Validate problem-optimizer compatibility
- Run multiple optimization iterations for statistical analysis
- Provide detailed performance metrics and error reporting

#### Usage

```bash
# Basic testing with single iteration
python examples/load_and_test_optimization.py user/my_problem user/my_optimizer

# Run multiple iterations for statistical analysis
python examples/load_and_test_optimization.py user/my_problem user/my_optimizer --iterations 5

# Quiet mode with minimal output
python examples/load_and_test_optimization.py user/my_problem user/my_optimizer --quiet

# With authentication for private repositories
python examples/load_and_test_optimization.py user/private_problem user/private_optimizer --token YOUR_TOKEN
```

#### Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `problem_repo` | Problem repository name (format: [username/]repo_name) | Yes |
| `optimizer_repo` | Optimizer repository name (format: [username/]repo_name) | Yes |
| `--iterations` | Number of optimization iterations to run (default: 1) | No |
| `--quiet` | Minimal output mode | No |
| `--token` | Rastion authentication token | No* |

*Token can also be set via `RASTION_TOKEN` environment variable

#### What It Tests

1. **Repository Loading**: Validates that both repositories can be loaded successfully
2. **Compatibility**: Checks that the problem and optimizer have compatible interfaces
3. **Functionality**: Tests basic problem evaluation and optimizer execution
4. **Performance**: Measures optimization time and solution quality
5. **Reliability**: Runs multiple iterations to assess consistency

#### Output

The script provides comprehensive output including:
- Loading status and validation results
- Compatibility check results
- Per-iteration optimization results
- Statistical summary (best/worst/average performance)
- Success rate and error analysis
- Overall assessment and recommendations

#### Examples

```bash
# Test a MaxCut problem with OR-Tools optimizer
python examples/load_and_test_optimization.py ileo/demo-maxcut ileo/demo-ortools-maxcut-optimizer

# Test VRP problem with genetic algorithm (5 iterations)
python examples/load_and_test_optimization.py ileo/demo-vrp-problem ileo/demo-genetic-vrp-optimizer --iterations 5

# Test private repositories with authentication
python examples/load_and_test_optimization.py company/private_problem company/private_optimizer \
    --token YOUR_RASTION_TOKEN --iterations 3
```

#### Exit Codes

- `0`: All optimization runs completed successfully
- `1`: Some optimization runs failed (partial success)
- `2`: All optimization runs failed
- `3`: Script error (loading, compatibility, etc.)

## 📚 Usage Examples

### Vehicle Routing Problem (VRP)

```python
from qubots import AutoProblem, AutoOptimizer

# Load VRP problem with custom parameters
problem = AutoProblem.from_repo("ileo/demo-vrp-problem", override_params={
    "n_customers": 25,
    "n_vehicles": 3,
    "depot_location": (0, 0),
    "vehicle_capacity": 100
})

# Load genetic algorithm optimizer
optimizer = AutoOptimizer.from_repo("ileo/demo-genetic-vrp-optimizer", override_params={
    "population_size": 100,
    "generations": 500,
    "mutation_rate": 0.1
})

# Run optimization
result = optimizer.optimize(problem)
print(f"Total distance: {result.best_value}")
print(f"Number of routes: {len(result.best_solution)}")
```

### Maximum Cut Problem

```python
from qubots import AutoProblem, AutoOptimizer

# Load MaxCut problem
problem = AutoProblem.from_repo("ileo/demo-maxcut", override_params={
    "n_vertices": 15,
    "graph_type": "random",
    "density": 0.4
})

# Load OR-Tools optimizer
optimizer = AutoOptimizer.from_repo("ileo/demo-ortools-maxcut-optimizer", override_params={
    "time_limit": 30.0,
    "use_symmetry": True
})

# Run optimization with monitoring
def progress_callback(iteration, best_value, current_value):
    print(f"Iteration {iteration}: Best cut weight = {best_value}")

result = optimizer.optimize(problem, progress_callback=progress_callback)
print(f"Maximum cut weight: {result.best_value}")
print(f"Cut partition: {result.best_solution}")
```

## 📊 API Reference

### Core Classes

#### AutoProblem

The `AutoProblem` class provides automatic loading and instantiation of optimization problems from repositories.

```python
class AutoProblem:
    @classmethod
    def from_repo(
        cls,
        repo_id: str,
        revision: str = "main",
        cache_dir: str = "~/.cache/rastion_hub",
        override_params: Optional[dict] = None,
        validate_metadata: bool = True
    ) -> BaseProblem
```

**Parameters:**
- `repo_id`: Repository identifier in format "username/repository-name"
- `revision`: Git branch or tag to load (default: "main")
- `cache_dir`: Local cache directory for repositories
- `override_params`: Dictionary of parameters to override default values
- `validate_metadata`: Whether to validate problem metadata

#### AutoOptimizer

The `AutoOptimizer` class provides automatic loading and instantiation of optimization algorithms from repositories.

```python
class AutoOptimizer:
    @classmethod
    def from_repo(
        cls,
        repo_id: str,
        revision: str = "main",
        cache_dir: str = "~/.cache/rastion_hub",
        override_params: Optional[dict] = None,
        validate_metadata: bool = True,
        register_in_registry: bool = True
    ) -> BaseOptimizer
```

**Parameters:**
- `repo_id`: Repository identifier in format "username/repository-name"
- `revision`: Git branch or tag to load (default: "main")
- `cache_dir`: Local cache directory for repositories
- `override_params`: Dictionary of parameters to override default values
- `validate_metadata`: Whether to validate optimizer metadata
- `register_in_registry`: Whether to register in global registry

#### BaseProblem

Abstract base class for all optimization problems.

**Key Methods:**
- `evaluate_solution(solution)`: Evaluate a candidate solution
- `get_random_solution()`: Generate a random valid solution
- `validate_solution_format(solution)`: Validate solution format
- `get_neighbor_solution(solution, step_size)`: Generate neighboring solution

#### BaseOptimizer

Abstract base class for all optimization algorithms.

**Key Methods:**
- `optimize(problem, initial_solution=None, progress_callback=None, log_callback=None)`: Run optimization
- `_optimize_implementation(problem, initial_solution)`: Core optimization logic (must be implemented)
- `stop_optimization()`: Stop running optimization
- `get_parameters()`: Get current optimizer parameters

#### OptimizationResult

Data class containing optimization results.

**Attributes:**
- `best_solution`: Best solution found
- `best_value`: Best objective value
- `iterations`: Number of iterations performed
- `runtime_seconds`: Total optimization time
- `convergence_history`: History of objective values
- `metadata`: Additional result metadata

### Dashboard Integration

#### QubotsAutoDashboard

Provides automatic dashboard generation for optimization results.

```python
@staticmethod
def auto_optimize_with_dashboard(
    problem,
    optimizer,
    problem_name: str = "Unknown Problem",
    optimizer_name: str = "Unknown Optimizer",
    log_callback=None,
    progress_callback=None
) -> DashboardResult
```

### Benchmarking

#### BenchmarkSuite

Comprehensive benchmarking and comparison tools.

```python
class BenchmarkSuite:
    def add_optimizer(self, name: str, optimizer: BaseOptimizer)
    def run_benchmarks(self, problem: BaseProblem, num_runs: int = 10)
    def generate_report(self, results: BenchmarkResult, output_file: str)
```

## 🧪 Testing and Validation

### Automated Testing

Use the provided utility script for comprehensive testing:

```bash
# Test problem-optimizer compatibility
python examples/load_and_test_optimization.py ileo/demo-maxcut ileo/demo-ortools-maxcut-optimizer --iterations 5

# Test with custom parameters
python examples/load_and_test_optimization.py user/my_problem user/my_optimizer --iterations 10 --quiet
```

### Manual Testing

```python
from qubots import AutoProblem, AutoOptimizer

# Load components
problem = AutoProblem.from_repo("user/test-problem")
optimizer = AutoOptimizer.from_repo("user/test-optimizer")

# Validate compatibility
try:
    # Test basic functionality
    solution = problem.get_random_solution()
    cost = problem.evaluate_solution(solution)

    # Run optimization
    result = optimizer.optimize(problem)
    print(f"Test successful: {result.best_value}")

except Exception as e:
    print(f"Test failed: {e}")
```

### Benchmarking

Compare multiple optimizers on the same problem:

```python
from qubots import AutoProblem, AutoOptimizer, BenchmarkSuite

# Load problem
problem = AutoProblem.from_repo("ileo/demo-maxcut")

# Create benchmark suite
suite = BenchmarkSuite()

# Add optimizers to compare
suite.add_optimizer("OR-Tools", AutoOptimizer.from_repo("ileo/demo-ortools-maxcut-optimizer"))
suite.add_optimizer("Genetic Algorithm", AutoOptimizer.from_repo("user/genetic-maxcut-optimizer"))

# Run benchmarks
results = suite.run_benchmarks(problem, num_runs=10)

# Generate report
suite.generate_report(results, "benchmark_results.html")
```

## 🤝 Contributing

We welcome contributions to the qubots framework! Here's how you can contribute:

### Types of Contributions

1. **New Optimization Problems**: Create and share new problem types
2. **New Optimization Algorithms**: Implement and share new optimizers
3. **Bug Fixes**: Report and fix issues in the framework
4. **Documentation**: Improve documentation and examples
5. **Performance Improvements**: Optimize existing components

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Rastion/qubots.git
cd qubots

# Install in development mode with all dependencies
pip install -e .

```

### Creating a New Optimizer

1. **Create Repository Structure**:
```
my_optimizer/
├── qubot.py          # Main implementation
├── config.json       # Configuration
├── requirements.txt  # Dependencies
└── README.md        # Documentation
```

2. **Implement Optimizer** (`qubot.py`):
```python
from qubots import BaseOptimizer, OptimizationResult, OptimizerMetadata
import time

class MyOptimizer(BaseOptimizer):
    def _get_default_metadata(self):
        return OptimizerMetadata(
            name="My Custom Optimizer",
            description="A novel optimization algorithm",
            author="Your Name",
            version="1.0.0"
        )

    def _optimize_implementation(self, problem, initial_solution=None):
        start_time = time.time()
        best_solution = problem.get_random_solution()
        best_value = problem.evaluate_solution(best_solution)

        # Your optimization logic here
        for iteration in range(100):
            solution = problem.get_random_solution()
            value = problem.evaluate_solution(solution)

            if value < best_value:
                best_solution = solution
                best_value = value

        return OptimizationResult(
            best_solution=best_solution,
            best_value=best_value,
            iterations=100,
            runtime_seconds=time.time() - start_time
        )
```

3. **Configure** (`config.json`):
```json
{
    "type": "optimizer",
    "entry_point": "qubot",
    "class_name": "MyOptimizer",
    "metadata": {
        "name": "My Custom Optimizer",
        "description": "A novel optimization algorithm"
    },
    "default_params": {}
}
```

4. **Upload to Platform**:
```bash
python examples/upload_repo_to_rastion.py ./my_optimizer --name "my_custom_optimizer" --token YOUR_RASTION_TOKEN
```

## 📄 License

This project is licensed under the [Apache License 2.0](./LICENSE).

## 🔗 Links

- **Homepage**: https://rastion.com
- **Documentation**: https://docs.rastion.com
- **Repository**: https://github.com/Rastion/qubots
- **PyPI**: https://pypi.org/project/qubots/
- **Issues**: https://github.com/Rastion/qubots/issues

---

**Qubots** empowers organizations to build scalable, collaborative optimization solutions. With its architecture and seamless repository integration, teams can rapidly develop, share, and deploy optimization components that solve real-world challenges across industries.
