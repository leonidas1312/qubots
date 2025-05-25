# Qubots-Rastion Platform Integration

This document describes the seamless integration between the Qubots optimization library and the Rastion platform, enabling users to upload, share, and load optimization models with minimal code.

## Overview

The integration provides a cloud-based model repository experience similar to Hugging Face or PyTorch Hub, but specifically designed for optimization models. Users can:

1. **Upload** their qubots-based optimization models to the Rastion platform
2. **Load** models from the platform with a single line of code
3. **Discover** and search for available optimization models
4. **Share** models with the community seamlessly

## Key Features

### 🚀 One-Line Model Loading
```python
import qubots.rastion as rastion
model = rastion.load_qubots_model("traveling_salesman_problem")
```

### 📤 Easy Model Upload
```python
url = rastion.upload_model(my_optimizer, "my_algorithm", "Description here")
```

### 🔍 Model Discovery
```python
models = rastion.discover_models("genetic algorithm")
```

### 🔐 Secure Authentication
```python
rastion.authenticate("your_gitea_token")
```

## Quick Start Guide

### 1. Authentication (One-time setup)

First, authenticate with the Rastion platform using your Gitea token:

```python
import qubots.rastion as rastion

# Get your token from https://hub.rastion.com (Profile Settings > Applications)
rastion.authenticate("your_gitea_token_here")
```

### 2. Loading Models

Load any available model with one line:

```python
# Load a problem
problem = rastion.load_qubots_model("traveling_salesman_problem")

# Load an optimizer  
optimizer = rastion.load_qubots_model("genetic_algorithm_tsp")

# Load with specific username
model = rastion.load_qubots_model("custom_optimizer", username="researcher123")
```

### 3. Using Loaded Models

Once loaded, models work exactly like local qubots:

```python
# Load models
problem = rastion.load_qubots_model("tsp_problem")
optimizer = rastion.load_qubots_model("ga_optimizer")

# Run optimization
result = optimizer.optimize(problem)
print(f"Best solution: {result['best_solution']}")
print(f"Best cost: {result['best_fitness']}")
```

### 4. Uploading Your Models

Share your optimization models with the community:

```python
# Create your model (inheriting from BaseProblem or BaseOptimizer)
class MyOptimizer(BaseOptimizer):
    # ... your implementation ...

# Upload to platform
my_optimizer = MyOptimizer()
url = rastion.upload_model(
    model=my_optimizer,
    name="my_awesome_optimizer", 
    description="A novel optimization algorithm for routing problems",
    requirements=["numpy", "scipy", "qubots"]
)

print(f"Model uploaded: {url}")
```

### 5. Discovering Models

Find models that match your needs:

```python
# Search for specific algorithms
genetic_algorithms = rastion.search_models("genetic algorithm")

# Discover routing optimization models
routing_models = rastion.discover_models("routing")

# List all available models
all_models = rastion.discover_models()

# List your uploaded models
my_models = rastion.list_my_models()
```

## Advanced Usage

### Custom Requirements

Specify custom Python requirements for your models:

```python
rastion.upload_model(
    model=my_model,
    name="advanced_optimizer",
    description="Uses OR-Tools and CasADi",
    requirements=["qubots", "ortools", "casadi", "numpy>=1.20.0"]
)
```

### Private Models

Upload private models for your organization:

```python
rastion.upload_model(
    model=my_model,
    name="proprietary_algorithm",
    description="Internal optimization algorithm",
    private=True  # Only you can access this model
)
```

### Version Management

The platform automatically handles versioning through Git:

```python
# Load specific version
model = rastion.load_qubots_model("my_model", revision="v1.2.0")

# Load latest development version
model = rastion.load_qubots_model("my_model", revision="develop")
```

## Integration Architecture

### Components

1. **RastionClient**: Core API wrapper for Gitea backend
2. **QubotPackager**: Automatic packaging of models for upload
3. **Auto-loading**: Enhanced loading with search capabilities
4. **CLI Integration**: Command-line tools for model management
5. **Registry Integration**: Local caching and metadata management

### File Structure

When you upload a model, the following files are automatically created:

```
my_model_repo/
├── qubot.py          # Your model's source code
├── config.json       # Model configuration and metadata
├── requirements.txt  # Python dependencies
└── README.md         # Auto-generated documentation
```

### Model Metadata

Models include rich metadata for discovery and compatibility:

```json
{
  "type": "optimizer",
  "entry_point": "qubot",
  "class_name": "MyOptimizer",
  "default_params": {},
  "metadata": {
    "name": "My Optimizer",
    "description": "A custom optimization algorithm",
    "author": "Your Name",
    "version": "1.0.0",
    "tags": ["genetic", "routing", "metaheuristic"]
  }
}
```

## CLI Integration

Enhanced CLI commands for power users:

```bash
# Interactive upload wizard
python -m qubots.cli_integration upload

# Quick upload
python -m qubots.cli_integration quick-upload model.py MyOptimizer my_optimizer "Description"

# List available models
python -m qubots.cli_integration list

# Search models
python -m qubots.cli_integration search "genetic algorithm"

# Validate model before upload
python -m qubots.cli_integration validate model.py MyOptimizer
```

## Best Practices

### Model Design

1. **Inherit from Base Classes**: Always inherit from `BaseProblem` or `BaseOptimizer`
2. **Include Metadata**: Provide rich metadata for better discoverability
3. **Document Parameters**: Use clear parameter names and defaults
4. **Handle Dependencies**: Specify all required packages

### Naming Conventions

1. **Repository Names**: Use descriptive, lowercase names with underscores
2. **Class Names**: Use PascalCase for class names
3. **Tags**: Use relevant, searchable tags

### Security

1. **Private Models**: Use private repositories for proprietary algorithms
2. **Token Security**: Keep your Gitea token secure and never commit it
3. **Code Review**: Review uploaded code for security issues

## Error Handling

The integration includes comprehensive error handling:

```python
try:
    model = rastion.load_qubots_model("nonexistent_model")
except ValueError as e:
    print(f"Model not found: {e}")

try:
    rastion.upload_model(invalid_model, "test", "test")
except ValueError as e:
    print(f"Upload failed: {e}")
```

## Examples

See `examples/rastion_integration_demo.py` for a comprehensive demonstration of all features.

## Support

- **Documentation**: Full API documentation available in docstrings
- **Examples**: Multiple examples in the `examples/` directory  
- **Community**: Join the Rastion community at https://rastion.com
- **Issues**: Report issues on the GitHub repository

## Future Enhancements

- **Model Versioning**: Enhanced version management and compatibility checking
- **Collaborative Features**: Team workspaces and shared model collections
- **Performance Metrics**: Benchmarking and performance tracking
- **Model Recommendations**: AI-powered model suggestions
- **Integration APIs**: REST APIs for external tool integration
