# Getting Started with Qubots

Welcome to qubots! This tutorial will guide you through your first optimization project using the qubots framework. By the end of this tutorial, you'll understand the core concepts and be able to create, run, and share optimization models.

## 🎯 Learning Objectives

After completing this tutorial, you will be able to:
- Install and set up qubots
- Understand the core qubots architecture
- Load optimization models from the Rastion platform
- Create your first custom optimization problem
- Implement a basic optimization algorithm
- Run optimizations and analyze results
- Upload your models to the Rastion platform

## 📋 Prerequisites

- Basic Python knowledge (functions, classes, imports)
- Understanding of basic optimization concepts (optional but helpful)
- Python 3.9 or higher installed

## 🚀 Installation and Setup

### Step 1: Install Qubots

```bash
# Install the base qubots package
pip install qubots

# Or install with optional dependencies for specific domains
pip install qubots[all]  # All features
pip install qubots[routing]  # OR-Tools for routing
pip install qubots[continuous]  # CasADi for continuous optimization
```

### Step 2: Verify Installation

```python
import qubots
print(f"Qubots version: {qubots.__version__}")

# Test basic imports
from qubots import BaseProblem, BaseOptimizer
print("✅ Qubots installed successfully!")
```

### Step 3: Set Up Rastion Platform (Optional)

```python
import qubots.rastion as rastion

# Get your token from https://rastion.com (Profile Settings > Applications)
# For this tutorial, we'll use demo mode
rastion.authenticate("demo_mode")  # Use your actual token in real projects
print("✅ Rastion authentication configured!")
```

## 🧠 Understanding Qubots Architecture

Qubots is built around two core concepts:

### 1. Problems (`BaseProblem`)
Define what you want to optimize:
- **Objective Function**: How to evaluate solutions
- **Solution Space**: What constitutes a valid solution
- **Constraints**: Rules that solutions must follow

### 2. Optimizers (`BaseOptimizer`)
Define how to find good solutions:
- **Search Strategy**: How to explore the solution space
- **Termination Criteria**: When to stop searching
- **Solution Improvement**: How to refine solutions

## 🏃‍♂️ Your First Optimization: Simple Function Minimization

Let's start with a classic optimization problem: finding the minimum of a mathematical function.

### Step 1: Create a Simple Problem

```python
import numpy as np
from qubots import BaseProblem, ProblemMetadata, ProblemType, ObjectiveType

class SimpleFunction(BaseProblem):
    """
    Minimize f(x) = x^2 + 2*x + 1 where x is between -10 and 10
    The minimum is at x = -1 with f(-1) = 0
    """
    
    def __init__(self):
        # Define the problem metadata
        metadata = ProblemMetadata(
            name="Simple Quadratic Function",
            description="Minimize f(x) = x^2 + 2*x + 1",
            problem_type=ProblemType.CONTINUOUS,
            objective_type=ObjectiveType.MINIMIZE,
            author="Getting Started Tutorial",
            version="1.0.0"
        )
        
        # Initialize with bounds
        self.lower_bound = -10.0
        self.upper_bound = 10.0
        
        super().__init__(metadata)
    
    def evaluate_solution(self, solution):
        """Evaluate f(x) = x^2 + 2*x + 1"""
        x = solution
        return x**2 + 2*x + 1
    
    def get_random_solution(self):
        """Generate a random solution between bounds"""
        return np.random.uniform(self.lower_bound, self.upper_bound)
    
    def is_feasible(self, solution):
        """Check if solution is within bounds"""
        return self.lower_bound <= solution <= self.upper_bound

# Test the problem
problem = SimpleFunction()
random_solution = problem.get_random_solution()
cost = problem.evaluate_solution(random_solution)
print(f"Random solution: x = {random_solution:.3f}, f(x) = {cost:.3f}")
```

### Step 2: Create a Simple Optimizer

```python
import time
from qubots import BaseOptimizer, OptimizerMetadata, OptimizationResult

class RandomSearchOptimizer(BaseOptimizer):
    """
    Simple random search optimizer that tries random solutions
    and keeps track of the best one found.
    """
    
    def __init__(self, num_trials=1000):
        metadata = OptimizerMetadata(
            name="Random Search Optimizer",
            description="Simple random search algorithm",
            author="Getting Started Tutorial",
            version="1.0.0"
        )
        
        self.num_trials = num_trials
        super().__init__(metadata, num_trials=num_trials)
    
    def _optimize_implementation(self, problem, initial_solution=None):
        """
        Run random search optimization.
        
        Args:
            problem: The problem to optimize
            initial_solution: Optional starting solution
            
        Returns:
            OptimizationResult with best solution found
        """
        start_time = time.time()
        
        # Initialize with random solution or provided initial solution
        if initial_solution is not None and problem.is_feasible(initial_solution):
            best_solution = initial_solution
        else:
            best_solution = problem.get_random_solution()
        
        best_value = problem.evaluate_solution(best_solution)
        
        # Random search loop
        for iteration in range(self.num_trials):
            # Generate random solution
            candidate = problem.get_random_solution()
            
            # Evaluate candidate
            candidate_value = problem.evaluate_solution(candidate)
            
            # Update best if better (assuming minimization)
            if candidate_value < best_value:
                best_solution = candidate
                best_value = candidate_value
        
        end_time = time.time()
        
        return OptimizationResult(
            best_solution=best_solution,
            best_value=best_value,
            iterations=self.num_trials,
            evaluations=self.num_trials,
            runtime_seconds=end_time - start_time,
            convergence_achieved=True,
            termination_reason="Maximum iterations reached"
        )

# Test the optimizer
optimizer = RandomSearchOptimizer(num_trials=5000)
print(f"Created optimizer: {optimizer.metadata.name}")
```

### Step 3: Run the Optimization

```python
# Run the optimization
print("\n🚀 Running optimization...")
result = optimizer.optimize(problem)

# Display results
print(f"\n📊 Optimization Results:")
print(f"Best solution: x = {result.best_solution:.6f}")
print(f"Best value: f(x) = {result.best_value:.6f}")
print(f"Runtime: {result.runtime_seconds:.3f} seconds")
print(f"Iterations: {result.iterations}")
print(f"Evaluations: {result.evaluations}")

# The theoretical minimum is at x = -1, f(-1) = 0
theoretical_min = -1.0
theoretical_value = 0.0
error = abs(result.best_solution - theoretical_min)
print(f"\n🎯 Accuracy:")
print(f"Theoretical minimum: x = {theoretical_min}, f(x) = {theoretical_value}")
print(f"Error in solution: {error:.6f}")
```

## 🌐 Loading Models from Rastion Platform

Now let's try loading a pre-built optimization model from the Rastion platform:

```python
import qubots.rastion as rastion

try:
    # Load a traveling salesman problem from Rastion
    print("\n🌐 Loading problem from Rastion platform...")
    tsp_problem = rastion.load_qubots_model("traveling_salesman_problem")
    print(f"✅ Loaded: {tsp_problem.metadata.name}")
    print(f"Description: {tsp_problem.metadata.description}")
    
    # Load an optimizer for TSP
    print("\n🔧 Loading optimizer from Rastion platform...")
    tsp_optimizer = rastion.load_qubots_model("genetic_algorithm_tsp")
    print(f"✅ Loaded: {tsp_optimizer.metadata.name}")
    
    # Run optimization
    print("\n🚀 Running TSP optimization...")
    tsp_result = tsp_optimizer.optimize(tsp_problem)
    print(f"TSP Best cost: {tsp_result.best_value:.2f}")
    print(f"TSP Runtime: {tsp_result.runtime_seconds:.3f} seconds")
    
except Exception as e:
    print(f"⚠️ Could not load from Rastion: {e}")
    print("This is normal if you're not connected to the internet or using demo mode")
```

## 📤 Uploading Your Models to Rastion

Share your optimization models with the community:

```python
try:
    # Upload your problem
    print("\n📤 Uploading models to Rastion...")
    
    problem_url = rastion.upload_model(
        model=problem,
        name="simple_quadratic_function",
        description="Tutorial example: minimize f(x) = x^2 + 2*x + 1",
        requirements=["numpy", "qubots"]
    )
    print(f"✅ Problem uploaded: {problem_url}")
    
    # Upload your optimizer
    optimizer_url = rastion.upload_model(
        model=optimizer,
        name="tutorial_random_search",
        description="Tutorial example: simple random search optimizer",
        requirements=["numpy", "qubots"]
    )
    print(f"✅ Optimizer uploaded: {optimizer_url}")
    
except Exception as e:
    print(f"⚠️ Could not upload to Rastion: {e}")
    print("This is normal if you're using demo mode or don't have upload permissions")
```

## 🔍 Discovering Models

Explore what's available on the platform:

```python
try:
    # Search for optimization models
    print("\n🔍 Discovering models on Rastion...")
    
    # Search for genetic algorithms
    genetic_models = rastion.search_models("genetic algorithm")
    print(f"Found {len(genetic_models)} genetic algorithm models")
    
    # Search for routing problems
    routing_models = rastion.search_models("routing")
    print(f"Found {len(routing_models)} routing models")
    
    # List some available models
    all_models = rastion.discover_models()
    print(f"Total available models: {len(all_models)}")
    
    if all_models:
        print("\nSample models:")
        for model in all_models[:3]:  # Show first 3
            print(f"  - {model['name']}: {model['description']}")
    
except Exception as e:
    print(f"⚠️ Could not discover models: {e}")
```

## 🎓 What You've Learned

Congratulations! You've completed your first qubots tutorial. You now know how to:

✅ **Install and set up qubots**
✅ **Create optimization problems** by inheriting from `BaseProblem`
✅ **Implement optimization algorithms** by inheriting from `BaseOptimizer`
✅ **Run optimizations** and analyze results
✅ **Load models** from the Rastion platform
✅ **Upload and share** your models with the community
✅ **Discover** optimization models created by others

## 🚀 Next Steps

Now that you understand the basics, explore these advanced topics:

### 1. Domain-Specific Tutorials
- **[Routing Optimization](routing_optimization.md)**: Vehicle routing and TSP
- **[Scheduling](scheduling_optimization.md)**: Job scheduling and resource allocation
- **[Finance](finance_optimization.md)**: Portfolio and risk optimization
- **[Fantasy Football](fantasy_football.md)**: Sports optimization with constraints

### 2. Advanced Topics
- **[Custom Optimizers](custom_optimizers.md)**: Advanced algorithm implementation
- **[Benchmarking](../guides/benchmarking.md)**: Performance testing and comparison
- **[Rastion Integration](../guides/rastion_integration.md)**: Complete platform guide

### 3. Real-World Examples
- Check out the [examples directory](../../examples/) for complete implementations
- Explore domain-specific examples in [examples/domains/](../../examples/domains/)

## 💡 Tips for Success

1. **Start Simple**: Begin with basic problems and gradually add complexity
2. **Test Frequently**: Validate your implementations with known solutions
3. **Use the Community**: Share your work and learn from others on Rastion
4. **Document Well**: Good documentation helps others understand and use your models
5. **Benchmark**: Compare your algorithms against existing solutions

Happy optimizing! 🎯
