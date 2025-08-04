# Qubots: Visual Optimization Workflow Platform

[![PyPI version](https://img.shields.io/pypi/v/qubots.svg)](https://pypi.org/project/qubots/)
[![Build Status](https://github.com/leonidas1312/qubots/actions/workflows/publish.yml/badge.svg)](https://github.com/leonidas1312/qubots/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

**Qubots** is a comprehensive optimization platform that combines visual workflow design with powerful code generation and AI agent integration. Create, share, and execute optimization workflows through an intuitive drag-and-drop interface or programmatically via Python and CLI tools.

## 🎯 Key Features

- **🎨 Visual Workflow Designer** - Drag-and-drop interface for building optimization workflows
- **🧩 Modular Components** - Reusable problems and optimizers with automatic discovery
- **🤖 AI Agent Integration** - MCP-compatible tools for seamless AI agent interaction
- **⚡ Code Generation** - Automatic Python script generation from visual workflows
- **🐳 Local Development** - Complete Docker-based development environment
- **📊 Testing Framework** - Comprehensive validation and quality assurance
- **🔧 CLI Tools** - Command-line interface for workflow and component management

## 🚀 Quick Start

### Option 1: Visual Interface (Recommended)

Start the complete visual workflow designer:

```bash
# Clone and setup
git clone https://github.com/leonidas1312/qubots.git
cd qubots

# Start full web interface (includes Gitea + API + Web UI)
./start_web_interface.sh

# Access the visual designer
open http://localhost:3001
```

**Features:**
- 🎨 Drag-and-drop workflow creation
- 🔧 Real-time parameter configuration
- 📊 Component library browser
- ⚡ Instant code generation
- 📤 Multiple export formats

### Option 2: AI Agent Integration

Use qubots directly with AI agents via NPX (no installation required):

```bash
# Create a new workflow
npx -y @qubots/mcp-tools workflow-create --name "Portfolio Optimization" --interactive

# Search for components
npx -y @qubots/mcp-tools component-search --type optimizer --domain finance

# Execute a workflow
npx -y @qubots/mcp-tools workflow-execute portfolio.json

# Generate Python code
npx -y @qubots/mcp-tools code-generate portfolio.json --format python
```

### Option 3: Python API

Use qubots programmatically in Python:

```bash
# Install qubots
pip install -e .

# Set up local environment
python setup_local.py
```

## 💡 Usage Examples

### Visual Workflow Creation

1. **Start the web interface**: `./start_web_interface.sh`
2. **Open browser**: Navigate to `http://localhost:3001`
3. **Drag components**: Add problems and optimizers from the sidebar
4. **Connect nodes**: Link components by dragging between connection points
5. **Configure parameters**: Click nodes to adjust settings in the parameter panel
6. **Generate code**: Export as Python script, JSON, or MCP format

### Python API Usage

```python
from qubots import AutoProblem, AutoOptimizer

# Load components from local repositories
problem = AutoProblem.from_repo("examples/portfolio_optimization_problem")
optimizer = AutoOptimizer.from_repo("examples/genetic_optimizer")

# Configure parameters
problem.set_parameters(target_return=0.12, num_assets=10)
optimizer.set_parameters(population_size=100, max_generations=500)

# Run optimization
result = optimizer.optimize(problem)

# Display results
print(f"Best Value: {result.best_value}")
print(f"Runtime: {result.runtime_seconds:.3f} seconds")
print(f"Solution: {result.best_solution}")
```

### CLI Workflow Management

```bash
# Validate a workflow
qubots workflow validate --file my_workflow.json

# Generate Python code from workflow
qubots workflow generate --file my_workflow.json --format python

# Export complete package
qubots workflow export --file my_workflow.json --package

# Create new component
qubots component create --name "My Problem" --type problem
```

### AI Agent Integration

```bash
# Check system status
npx -y @qubots/mcp-tools status

# Interactive workflow creation
npx -y @qubots/mcp-tools workflow-create --interactive

# Component management
npx -y @qubots/mcp-tools component-install portfolio-optimizer
```

## 🧩 Creating Custom Components

### Using the Component Generator

The easiest way to create new components is using the built-in generator:

```bash
# Create a new problem component
qubots component create \
    --name "Vehicle Routing Problem" \
    --type problem \
    --description "VRP with time windows and capacity constraints" \
    --parameters '{"num_vehicles": 5, "capacity": 100, "time_limit": 480}'

# Create a new optimizer component
qubots component create \
    --name "Simulated Annealing" \
    --type optimizer \
    --description "SA algorithm with adaptive cooling" \
    --parameters '{"initial_temp": 1000, "cooling_rate": 0.95, "min_temp": 0.01}'
```

This generates a complete component structure:
```
my_component/
├── qubot.py           # Main implementation
├── config.json        # Component configuration
├── requirements.txt   # Dependencies
├── test_qubot.py      # Unit tests
└── README.md          # Documentation
```

### Component Repository Structure

Each component repository should contain:

- **`qubot.py`**: Main implementation file with problem/optimizer class
- **`config.json`**: Component metadata and parameter definitions
- **`requirements.txt`**: Python dependencies
- **`README.md`**: Documentation and usage examples
- **`test_qubot.py`**: Unit tests (optional but recommended)

### Manual Implementation

For advanced customization, implement the base classes directly:

```python
from qubots import BaseProblem, ProblemMetadata, ProblemType, ObjectiveType

class CustomProblem(BaseProblem):
    def __init__(self, size=100):
        self.size = size
        super().__init__()

    def _get_default_metadata(self):
        return ProblemMetadata(
            name="Custom Problem",
            description="A custom optimization problem",
            problem_type=ProblemType.DISCRETE,
            objective_type=ObjectiveType.MINIMIZE,
            domain="custom",
            tags={"custom", "example"}
        )

    def evaluate_solution(self, solution):
        # Your evaluation logic
        return sum(solution)

    def random_solution(self):
        import random
        return [random.randint(0, 1) for _ in range(self.size)]
```

## 📊 Development Tools

### Testing and Validation

```bash
# Validate workflow structure
qubots workflow validate --file my_workflow.json --detailed

# Run component tests
python -m pytest examples/test_*.py

# Check system status
qubots status
```

### Local Development Environment

```bash
# Start complete development environment
./start_local.sh

# Access services:
# - Gitea: http://localhost:3000
# - API: http://localhost:8000
# - Web UI: http://localhost:3001
```

## Documentation

For comprehensive documentation, examples, and API reference, visit:

**📚 [https://docs.rastion.com](https://docs.rastion.com)**

The documentation includes:
- Detailed tutorials and examples
- Complete API reference
- Advanced configuration options
- Best practices and patterns
- Community contributions guide

## License

This project is licensed under the [Apache License 2.0](./LICENSE).

## 🔗 Links

- **Repository**: [github.com/leonidas1312/qubots](https://github.com/leonidas1312/qubots)
- **Issues**: [github.com/leonidas1312/qubots/issues](https://github.com/leonidas1312/qubots/issues)
- **Releases**: [github.com/leonidas1312/qubots/releases](https://github.com/leonidas1312/qubots/releases)
- **PyPI**: [pypi.org/project/qubots](https://pypi.org/project/qubots/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- 🐛 Reporting bugs and requesting features
- 🔧 Setting up the development environment
- 📝 Code style and testing requirements
- 🚀 Submitting pull requests

## 🙏 Acknowledgments

Built with modern web technologies and optimization frameworks:
- **React Flow** for visual workflow editing
- **FastAPI** for high-performance API backend
- **Docker** for containerized development
- **MCP Protocol** for AI agent integration
