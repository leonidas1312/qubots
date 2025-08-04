# Contributing to Qubots

Thank you for your interest in contributing to Qubots! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/qubots.git
   cd qubots
   ```

2. **Install Python dependencies**
   ```bash
   pip install -e .
   ```

3. **Set up local development environment**
   ```bash
   # Set up local Gitea and services
   python setup_local.py

   # Start the complete development stack
   ./start_web_interface.sh
   ```

4. **Verify setup**
   ```bash
   # Test CLI
   qubots status

   # Access web interface at http://localhost:3001
   # Access Gitea at http://localhost:3000
   # Access API at http://localhost:8000
   ```

### Prerequisites

- **Python 3.8+** for core development
- **Node.js 16+** for web interface development
- **Docker** for local development environment
- **Git** for version control

## 🛠️ Development Workflow

### Code Style

We maintain code quality across multiple languages:

**Python:**
- Follow PEP 8 with 88-character line limit
- Use type hints for better code documentation
- Write clear docstrings for all public methods

**JavaScript/React:**
- Use Prettier with default settings
- Follow React best practices and hooks patterns
- Implement responsive design principles

**General:**
- Write clear, self-documenting code
- Add comments for complex logic
- Use meaningful variable and function names

### Testing

**Python Tests:**
```bash
# Run core tests
python -m pytest examples/test_*.py

# Test CLI functionality
qubots workflow validate --file examples/portfolio_workflow.json
```

**Web Interface Tests:**
```bash
cd web_interface
npm test
```

**Integration Tests:**
```bash
# Test complete workflow
python test_integration.py

# Test MCP tools
cd mcp_tools && npm test
```

### Adding New Features

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement your feature**
   - Add tests for new functionality
   - Update documentation as needed
   - Follow existing code patterns

3. **Test your changes**
   ```bash
   pytest
   black qubots tests
   isort qubots tests
   flake8 qubots tests
   ```

4. **Submit a pull request**

## 📝 Types of Contributions

### 1. Bug Reports

When reporting bugs, please include:
- Python version and OS
- Qubots version
- Minimal code example that reproduces the issue
- Expected vs actual behavior
- Full error traceback

### 2. Feature Requests

For new features:
- Describe the use case and motivation
- Provide examples of how the feature would be used
- Consider backward compatibility

### 3. Code Contributions

#### New Problem Types
- Extend `BaseProblem` or specialized problem classes
- Include comprehensive tests
- Add documentation and examples
- Consider domain-specific optimizations

#### New Optimizer Types
- Extend `BaseOptimizer` or specialized optimizer classes
- Implement proper metadata and parameter handling
- Include benchmarking examples
- Document algorithm-specific parameters

#### Integration with Optimization Libraries
- Follow existing patterns (OR-Tools, CasADi, etc.)
- Add optional dependencies to pyproject.toml
- Include installation and usage examples
- Test with different library versions

#### Web Interface Components
- Use React functional components with hooks
- Follow the existing component structure in `web_interface/src/`
- Implement responsive design with Tailwind CSS
- Add proper error boundaries and loading states
- Test UI components with user interactions

#### MCP Tools and AI Integration
- Follow MCP protocol specifications for tool definitions
- Add JSON schemas for input validation in `mcp_tools/schemas/`
- Ensure NPX compatibility for easy AI agent usage
- Test cross-platform functionality (Windows, macOS, Linux)
- Document tool usage with clear examples

### 4. Documentation

- Fix typos and improve clarity
- Add examples and tutorials
- Update API documentation
- Improve README and guides

## 🏗️ Architecture Guidelines

### Base Classes

- All problems must inherit from `BaseProblem`
- All optimizers must inherit from `BaseOptimizer`
- Use proper metadata for discoverability
- Implement abstract methods completely

### Specialized Classes

- Use specialized base classes when appropriate
- Follow domain-specific conventions
- Maintain consistency with existing patterns

### Registry Integration

- Register new qubots in the global registry
- Use appropriate metadata and tags
- Enable search and discovery features

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── test_base_problem.py
├── test_base_optimizer.py
├── test_specialized_problems.py
├── test_specialized_optimizers.py
├── test_benchmarking.py
├── test_registry.py
└── integration/
    ├── test_ortools_integration.py
    ├── test_casadi_integration.py
    └── ...
```

### Test Requirements

- Unit tests for all public methods
- Integration tests for external libraries
- Performance tests for optimization algorithms
- Mock external dependencies when appropriate

### Test Data

- Use reproducible random seeds
- Include edge cases and boundary conditions
- Test both valid and invalid inputs

## 📋 Pull Request Process

1. **Ensure all tests pass**
2. **Update documentation** if needed
3. **Add changelog entry** for significant changes
4. **Request review** from maintainers
5. **Address feedback** promptly

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
- [ ] No breaking changes (or properly documented)

## 🤝 Community Guidelines

- Be respectful and inclusive
- Help others learn and contribute
- Share knowledge and best practices
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)

## 📞 Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community discussions
- **Email**: For sensitive issues, contact the maintainers directly

## 🏷️ Release Process

Releases follow semantic versioning:
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

## 📄 License

By contributing to Qubots, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to Qubots! Together, we're building the future of collaborative optimization. 🚀
