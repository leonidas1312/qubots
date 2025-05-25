# Qubots: A Collaborative Optimization Framework

[![PyPI version](https://img.shields.io/pypi/v/qubots.svg)](https://pypi.org/project/qubots/)
[![Build Status](https://github.com/leonidas1312/qubots/actions/workflows/publish.yml/badge.svg)](https://github.com/leonidas1312/qubots/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/leonidas1312/qubots.svg)](https://github.com/leonidas1312/qubots/issues)
[![GitHub forks](https://img.shields.io/github/forks/leonidas1312/qubots.svg)](https://github.com/leonidas1312/qubots/network)

**Qubots** is a powerful Python framework that transforms optimization problems and algorithms into shareable, modular components called "qubots". With seamless integration to the [Rastion platform](https://rastion.com), qubots enables collaborative optimization development, sharing, and deployment across domains including routing, scheduling, logistics, finance, energy, and more.

## 🚀 Key Features

- **🔧 Modular Design**: Create reusable optimization components that work together seamlessly
- **🌐 Cloud Integration**: Upload, share, and load optimization models with the Rastion platform
- **🎯 Domain-Specific**: Pre-built optimizers for routing, scheduling, logistics, finance, energy, and fantasy sports
- **⚡ High Performance**: Integration with OR-Tools, CasADi, and other optimization libraries
- **📊 Benchmarking**: Built-in performance testing and comparison tools
- **🔍 Discovery**: Search and discover optimization models from the community
- **📚 Educational**: Comprehensive tutorials and examples for learning optimization

## 📦 Installation

Install qubots from PyPI:

```bash
pip install qubots
```

For domain-specific optimizations, install optional dependencies:

```bash
# For routing and scheduling (OR-Tools)
pip install qubots[routing]

# For continuous optimization (CasADi)
pip install qubots[continuous]

# For all features
pip install qubots[all]
```

## 🚀 Quick Start

### Basic Usage

Here's a simple example showing how to create and solve an optimization problem:

```python
from qubots import BaseProblem, BaseOptimizer
import qubots.rastion as rastion

# Load a problem from the Rastion platform
problem = rastion.load_qubots_model("traveling_salesman_problem")

# Load an optimizer
optimizer = rastion.load_qubots_model("ortools_tsp_solver")

# Run optimization
result = optimizer.optimize(problem)
print(f"Best Solution: {result.best_solution}")
print(f"Best Cost: {result.best_value}")
```

### Creating Custom Optimizers

```python
from qubots import BaseOptimizer, OptimizationResult

class MyOptimizer(BaseOptimizer):
    def _optimize_implementation(self, problem, initial_solution=None):
        # Your optimization logic here
        solution = problem.get_random_solution()
        cost = problem.evaluate_solution(solution)
        
        return OptimizationResult(
            best_solution=solution,
            best_value=cost,
            iterations=1,
            runtime_seconds=0.1
        )

# Use your optimizer
optimizer = MyOptimizer()
result = optimizer.optimize(problem)
```

## 🌐 Rastion Platform Integration

Qubots seamlessly integrates with the Rastion platform for model sharing and collaboration:

### Authentication

```python
import qubots.rastion as rastion

# Authenticate with your Rastion token
rastion.authenticate("your_rastion_token_here")
```

### Loading Models

```python
# Load any available model with one line
problem = rastion.load_qubots_model("traveling_salesman_problem")
optimizer1 = rastion.load_qubots_model("genetic_algorithm_tsp")

# Load with specific username
optimizer2 = rastion.load_qubots_model("custom_optimizer", username="researcher123")
```

### Uploading Models

```python
# Share your optimization models with the community
my_optimizer = MyOptimizer()
url = rastion.upload_model(
    model=my_optimizer,
    name="my_awesome_optimizer", 
    description="A novel optimization algorithm for routing problems",
    requirements=["numpy", "scipy", "qubots"]
)
```

### Model Discovery

```python
# Search for specific algorithms
genetic_algorithms = rastion.search_models("genetic algorithm")

# Discover routing optimization models
routing_models = rastion.discover_models("routing")

# List all available models
all_models = rastion.discover_models()
```

## 📚 Domain Examples

Qubots includes comprehensive examples across multiple optimization domains:

### 🚛 Routing and Logistics
- **Vehicle Routing Problem (VRP)**: Multi-vehicle delivery optimization
- **Traveling Salesman Problem (TSP)**: Classic route optimization
- **Supply Chain Optimization**: Warehouse and distribution planning

### ⏰ Scheduling
- **Job Shop Scheduling**: Manufacturing and production planning
- **Resource Allocation**: Optimal resource assignment
- **Project Scheduling**: Timeline and dependency management

### 💰 Finance
- **Portfolio Optimization**: Risk-return optimization
- **Asset Allocation**: Investment strategy optimization
- **Risk Management**: Financial risk minimization

### ⚡ Energy
- **Power Grid Optimization**: Energy distribution planning
- **Renewable Energy**: Solar and wind optimization
- **Energy Storage**: Battery and storage optimization

### 🏈 Fantasy Sports
- **Fantasy Football**: Lineup optimization with salary constraints
- **Daily Fantasy Sports**: Multi-contest optimization
- **Player Selection**: Statistical analysis and optimization

## 🛠️ Creating Custom Optimizers

### Step-by-Step Tutorial

1. **Inherit from Base Classes**:
```python
from qubots import BaseOptimizer, OptimizerMetadata

class MyOptimizer(BaseOptimizer):
    def __init__(self, **params):
        metadata = OptimizerMetadata(
            name="My Custom Optimizer",
            description="Custom optimization algorithm",
            author="Your Name",
            version="1.0.0"
        )
        super().__init__(metadata, **params)
```

2. **Implement Optimization Logic**:
```python
def _optimize_implementation(self, problem, initial_solution=None):
    # Your optimization algorithm here
    best_solution = None
    best_value = float('inf')
    
    for iteration in range(self.max_iterations):
        # Generate or improve solution
        solution = self.generate_solution(problem)
        value = problem.evaluate_solution(solution)
        
        if value < best_value:
            best_solution = solution
            best_value = value
    
    return OptimizationResult(
        best_solution=best_solution,
        best_value=best_value,
        iterations=iteration + 1,
        runtime_seconds=time.time() - start_time
    )
```

3. **Upload to Rastion**:
```python
url = rastion.upload_model(
    model=MyOptimizer(),
    name="my_optimizer",
    description="My custom optimization algorithm"
)
```

## 📊 Benchmarking and Testing

Qubots includes comprehensive benchmarking tools:

```python
from qubots import BenchmarkSuite

# Create benchmark suite
suite = BenchmarkSuite()

# Add optimizers to compare
suite.add_optimizer("Random Search", RandomSearchOptimizer())
suite.add_optimizer("Genetic Algorithm", GeneticOptimizer())
suite.add_optimizer("My Optimizer", MyOptimizer())

# Run benchmarks
results = suite.run_benchmarks(problem, num_runs=10)

# Generate report
suite.generate_report(results, "benchmark_results.html")
```

## 📖 Documentation

- **[Getting Started Guide](docs/guides/getting_started.md)**: Complete beginner tutorial
- **[Domain Tutorials](docs/tutorials/)**: Step-by-step domain-specific examples
- **[Rastion Integration](docs/guides/rastion_integration.md)**: Platform integration guide
- **[API Reference](docs/api/)**: Complete API documentation
- **[Fantasy Football Tutorial](docs/tutorials/fantasy_football.md)**: 3-file structure examples

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Rastion/qubots.git
cd qubots

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/

# Run benchmarks
python -m pytest tests/benchmarks/
```

## 📄 License

This project is licensed under the [Apache License 2.0](./LICENSE).

## 🔗 Links

- **Homepage**: https://rastion.com
- **Documentation**: https://rastion.com/docs
- **Repository**: https://github.com/Rastion/qubots
- **PyPI**: https://pypi.org/project/qubots/
- **Issues**: https://github.com/Rastion/qubots/issues

---

By leveraging the flexible design of qubots and the collaborative power of Rastion, you can rapidly prototype, share, and improve optimization solutions—be it for classical problems, quantum algorithms, or hybrid systems.
