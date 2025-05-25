# Qubots Test Suite

Comprehensive testing framework for the qubots optimization library, ensuring reliability, performance, and compatibility across all components.

## 📁 Test Structure

### 🧪 [Unit Tests](unit/)
Test individual components in isolation:
- **Core Classes**: BaseProblem, BaseOptimizer, metadata classes
- **Specialized Classes**: Domain-specific base classes
- **Utilities**: Helper functions and utilities
- **Rastion Integration**: Platform integration components
- **Registry**: Model registry and discovery

### 🔗 [Integration Tests](integration/)
Test component interactions and workflows:
- **Problem-Optimizer Integration**: Compatibility testing
- **Rastion Platform**: End-to-end platform integration
- **Example Validation**: Ensure all examples work correctly
- **Cross-Platform**: Testing across different environments

### 📊 [Benchmarks](benchmarks/)
Performance testing and comparison:
- **Algorithm Performance**: Speed and quality benchmarks
- **Scalability Tests**: Performance vs. problem size
- **Memory Usage**: Memory efficiency testing
- **Platform Performance**: Rastion platform performance

## 🚀 Running Tests

### Quick Test Run

```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/benchmarks/

# Run with coverage
pytest tests/ --cov=qubots --cov-report=html
```

### Detailed Test Commands

```bash
# Run tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_base_optimizer.py

# Run specific test function
pytest tests/unit/test_base_optimizer.py::test_optimizer_initialization

# Run tests matching pattern
pytest tests/ -k "test_genetic"

# Run tests with markers
pytest tests/ -m "slow"  # Run only slow tests
pytest tests/ -m "not slow"  # Skip slow tests
```

### Test Configuration

```bash
# Run with different Python versions (using tox)
tox

# Run with specific Python version
tox -e py39
tox -e py310
tox -e py311

# Run linting and formatting checks
tox -e lint
tox -e format
```

## 🧪 Test Categories

### Unit Tests

#### Core Components
- **BaseProblem**: Problem interface and validation
- **BaseOptimizer**: Optimizer interface and execution
- **Metadata Classes**: Problem and optimizer metadata
- **Result Classes**: Optimization result handling

#### Specialized Components
- **Continuous Problems**: Continuous optimization base classes
- **Discrete Problems**: Discrete optimization base classes
- **Combinatorial Problems**: Combinatorial optimization base classes
- **Specialized Optimizers**: Domain-specific optimizer base classes

#### Utilities
- **Benchmarking**: Performance testing utilities
- **Registry**: Model registry and discovery
- **Validation**: Input validation and error handling

### Integration Tests

#### Workflow Testing
- **Problem Creation**: End-to-end problem creation
- **Optimization Execution**: Complete optimization workflows
- **Result Analysis**: Result processing and validation

#### Platform Integration
- **Authentication**: Rastion platform authentication
- **Model Upload/Download**: Model sharing workflows
- **Search and Discovery**: Model discovery functionality

#### Example Validation
- **Domain Examples**: All domain-specific examples
- **Tutorial Examples**: Tutorial code validation
- **Documentation Examples**: Code in documentation

### Benchmark Tests

#### Performance Benchmarks
- **Algorithm Speed**: Execution time measurements
- **Solution Quality**: Optimization quality assessment
- **Convergence Rate**: Convergence speed analysis

#### Scalability Tests
- **Problem Size**: Performance vs. problem complexity
- **Population Size**: Genetic algorithm scaling
- **Memory Usage**: Memory efficiency testing

#### Comparison Benchmarks
- **Algorithm Comparison**: Compare different algorithms
- **Platform Performance**: Local vs. Rastion execution
- **Version Comparison**: Performance across versions

## 📋 Test Requirements

### Dependencies

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock pytest-benchmark

# Install all dependencies for comprehensive testing
pip install qubots[all,test]
```

### Environment Setup

```bash
# Set up test environment variables
export QUBOTS_TEST_MODE=true
export RASTION_TEST_TOKEN=test_token

# Create test configuration
cp tests/config/test_config.example.yml tests/config/test_config.yml
```

### Test Data

Test data is organized in `tests/data/`:
- **Sample Problems**: Small test problems for each domain
- **Expected Results**: Known optimal solutions for validation
- **Performance Baselines**: Historical performance data

## 🎯 Test Guidelines

### Writing Unit Tests

```python
import pytest
from qubots import BaseProblem, ProblemMetadata

class TestBaseProblem:
    """Test suite for BaseProblem class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metadata = ProblemMetadata(
            name="Test Problem",
            description="Test problem for unit testing"
        )
    
    def test_problem_initialization(self):
        """Test problem initialization."""
        problem = TestProblem(self.metadata)
        assert problem.metadata.name == "Test Problem"
        assert hasattr(problem, 'evaluate_solution')
    
    def test_solution_evaluation(self):
        """Test solution evaluation."""
        problem = TestProblem(self.metadata)
        solution = [1, 2, 3]
        result = problem.evaluate_solution(solution)
        assert isinstance(result, (int, float))
    
    @pytest.mark.parametrize("solution,expected", [
        ([1, 2, 3], 6),
        ([0, 0, 0], 0),
        ([-1, -2, -3], -6)
    ])
    def test_evaluation_cases(self, solution, expected):
        """Test specific evaluation cases."""
        problem = TestProblem(self.metadata)
        result = problem.evaluate_solution(solution)
        assert result == expected
```

### Writing Integration Tests

```python
import pytest
from qubots import AutoProblem, AutoOptimizer
import qubots.rastion as rastion

class TestRastionIntegration:
    """Test Rastion platform integration."""
    
    @pytest.fixture(autouse=True)
    def setup_rastion(self):
        """Set up Rastion test environment."""
        rastion.authenticate("test_token")
        yield
        rastion.clear_cache()
    
    def test_model_upload_download(self):
        """Test model upload and download workflow."""
        # Create test model
        optimizer = TestOptimizer()
        
        # Upload model
        url = rastion.upload_model(optimizer, "test_model")
        assert url is not None
        
        # Download model
        downloaded = rastion.load_qubots_model("test_model")
        assert downloaded.metadata.name == optimizer.metadata.name
    
    @pytest.mark.slow
    def test_optimization_workflow(self):
        """Test complete optimization workflow."""
        problem = rastion.load_qubots_model("test_problem")
        optimizer = rastion.load_qubots_model("test_optimizer")
        
        result = optimizer.optimize(problem)
        assert result.best_solution is not None
        assert result.best_value is not None
```

### Writing Benchmark Tests

```python
import pytest
from qubots import BenchmarkSuite

class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.benchmark
    def test_genetic_algorithm_performance(self, benchmark):
        """Benchmark genetic algorithm performance."""
        problem = create_test_problem(size=100)
        optimizer = GeneticAlgorithm(population_size=50, max_generations=100)
        
        result = benchmark(optimizer.optimize, problem)
        
        # Performance assertions
        assert result.runtime_seconds < 10.0  # Should complete in 10 seconds
        assert result.best_value > 0  # Should find positive solution
    
    @pytest.mark.parametrize("problem_size", [10, 50, 100, 500])
    def test_scalability(self, problem_size):
        """Test algorithm scalability."""
        problem = create_test_problem(size=problem_size)
        optimizer = TestOptimizer()
        
        start_time = time.time()
        result = optimizer.optimize(problem)
        runtime = time.time() - start_time
        
        # Scalability assertions
        expected_max_time = problem_size * 0.1  # Linear scaling expectation
        assert runtime < expected_max_time
```

## 🔧 Test Configuration

### pytest.ini

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    benchmark: marks tests as benchmarks
    rastion: marks tests requiring Rastion platform
addopts = 
    --strict-markers
    --disable-warnings
    --tb=short
```

### Coverage Configuration

```ini
[coverage:run]
source = qubots
omit = 
    */tests/*
    */venv/*
    setup.py

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

## 📊 Test Metrics

### Coverage Targets
- **Unit Tests**: > 90% code coverage
- **Integration Tests**: > 80% workflow coverage
- **Documentation**: 100% example validation

### Performance Targets
- **Unit Tests**: < 1 second per test
- **Integration Tests**: < 30 seconds per test
- **Benchmarks**: Baseline performance tracking

### Quality Metrics
- **Code Quality**: Pylint score > 8.0
- **Type Coverage**: mypy validation
- **Documentation**: 100% docstring coverage

## 🚨 Continuous Integration

### GitHub Actions

```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .[test]
    
    - name: Run tests
      run: |
        pytest tests/ --cov=qubots --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Pre-commit Hooks

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
  
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest tests/unit/
        language: system
        pass_filenames: false
```

## 🛠️ Development Workflow

### Test-Driven Development

1. **Write Test First**: Create failing test for new feature
2. **Implement Feature**: Write minimal code to pass test
3. **Refactor**: Improve code while keeping tests passing
4. **Add Integration Tests**: Test feature in context
5. **Update Documentation**: Document new feature

### Running Tests During Development

```bash
# Watch mode for continuous testing
pytest-watch tests/

# Run tests on file change
ptw tests/ --runner "pytest tests/ -x"

# Quick smoke test
pytest tests/unit/ -x --ff
```

## 📞 Getting Help

### Test Issues
- **Failing Tests**: Check test logs and error messages
- **Performance Issues**: Run benchmarks to identify bottlenecks
- **Platform Issues**: Verify Rastion authentication and connectivity

### Contributing Tests
- **New Features**: Add corresponding unit and integration tests
- **Bug Fixes**: Add regression tests
- **Performance**: Add benchmark tests for new algorithms

---

A comprehensive test suite ensures qubots remains reliable, performant, and user-friendly across all use cases! 🧪✅
