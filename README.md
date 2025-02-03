# rastion-hub

A collaborative platform for building, testing, and sharing optimization algorithms ("solvers") and optimization problems. This repository contains:

- A set of base classes for problems (`BaseProblem`) and optimizers (`BaseOptimizer`).
- A library of common solvers (e.g., GeneticAlgorithm, GradientDescent, HillClimb, SimulatedAnnealing) in [`rastion_core/algorithms`](./rastion_core/algorithms).
- A library of example problem classes in [`rastion_core/problems`](./rastion_core/problems).
- An automated loader for solver repos (`AutoOptimizer`) and problem repos (`AutoProblem`) in [`rastion_hub`](./rastion_hub), enabling you to fetch Python classes from GitHub repos.
- A CLI tool (`rastion_cli`) that can create, clone, push, and run solver/problem repositories.

## Table of Contents
1. [Installation](#installation)
2. [Usage Overview](#usage-overview)
3. [Core Concepts](#core-concepts)
   - [Creating Custom Problems](#creating-custom-problems)
   - [Creating Custom Solvers](#creating-custom-solvers)
   - [AutoProblem and AutoOptimizer](#autoproblem-and-autooptimizer)
4. [Examples](#examples)
   - [Example 1: Custom Problem with a Predefined Optimizer](#example-1-custom-problem-with-a-predefined-optimizer)
   - [Example 2: Custom Optimizer with a Predefined Problem](#example-2-custom-optimizer-with-a-predefined-problem)
5. [License](#license)
6. [Download the Codebase](#download-the-codebase)

## Installation
To install the `rastion-hub` package along with the command-line interface (CLI):

```bash
pip install .  # Installs from this directory
```

You can also install via `pip install git+https://github.com/leonidas1312/rastion-hub.git` if you have the repository available online.

**Note:** You must have `git` installed for the CLI commands that work with remote GitHub repositories.

## Usage Overview

### Command-Line Interface

Once installed, you will have a `rastion` CLI command available. The top-level commands are:

- `rastion create_repo <repo_name>`: Create a new GitHub repository under an organization (default: `Rastion`).
- `rastion clone_repo <repo_name>`: Clone a GitHub repository locally.
- `rastion push_solver <repo_name> --file my_solver.py --config solver_config.json`: Push a local solver Python file and `solver_config.json` to a GitHub repo.
- `rastion push_problem <repo_name> --file my_problem.py --config problem_config.json`: Push a local problem Python file and `problem_config.json` to a GitHub repo.
- `rastion run_solver --solver-rev main --problem-repo <problem_repo>`: Clone or pull both solver and problem repos and run the solver.

These commands help you develop and publish new solvers and problems.

### Code Layout

- **`rastion_core`**: Core library containing base classes and built-in solvers/problems.
- **`rastion_hub/auto_optimizer.py`**: The `AutoOptimizer` class that can clone a solver repo and load its Python class.
- **`rastion_hub/auto_problem.py`**: The `AutoProblem` class that can do the same for problem repos.
- **`rastion_cli/cli.py`**: The CLI entry point.

## Core Concepts

### Creating Custom Problems

All custom problems in `rastion-hub` must derive from `BaseProblem`, implementing:

- `evaluate_solution(solution) -> float`: Returns the objective value for the given solution (lower is better by convention).
- `random_solution() -> solution`: Generates a feasible (random) solution.
- Optionally: `is_feasible(solution) -> bool` if your problem has constraints.

The following is an example stub:

```python
from rastion_core.base_problem import BaseProblem
import random

class MyCustomProblem(BaseProblem):
    def __init__(self, some_param=1):
        self.some_param = some_param

    def evaluate_solution(self, solution):
        # your logic, return float
        pass

    def random_solution(self):
        # return a random solution
        pass

    def is_feasible(self, solution):
        # optional constraint check
        return True
```

### Creating Custom Solvers

Similarly, all custom solvers must derive from `BaseOptimizer`, implementing:

- `optimize(problem, **kwargs) -> (best_solution, best_value)`.

```python
from rastion_core.base_optimizer import BaseOptimizer

class MyCustomOptimizer(BaseOptimizer):
    def __init__(self, param=123):
        self.param = param

    def optimize(self, problem, **kwargs):
        # your optimization logic
        # example:
        best_sol = problem.random_solution()
        best_val = problem.evaluate_solution(best_sol)
        return best_sol, best_val
```

### AutoProblem and AutoOptimizer

`AutoProblem` and `AutoOptimizer` are classes designed to fetch your custom problem or solver from a remote GitHub repository.
They rely on a `problem_config.json` or `solver_config.json` file describing the entry point:

- **`solver_config.json`**:

```json
{
  "entry_point": "my_module:MySolverClass",
  "default_params": {
    "some_param": 123
  }
}
```

- **`problem_config.json`**:

```json
{
  "entry_point": "my_module:MyProblemClass",
  "default_params": {
    "some_param": 100
  }
}
```

With these config files in your repo, you can do:

```python
from rastion_hub.auto_optimizer import AutoOptimizer
from rastion_hub.auto_problem import AutoProblem

# solver = AutoOptimizer.from_repo("YourOrg/YourSolverRepo", revision="main")
# problem = AutoProblem.from_repo("YourOrg/YourProblemRepo", revision="main")

best_sol, best_cost = solver.optimize(problem)
```

## Examples
Below are two real examples showing how to:
1. Create a custom problem that extends `BaseProblem`.
2. Create a custom solver that extends `BaseOptimizer`.
3. Load them via `AutoProblem` and `AutoOptimizer`.

### Example 1: Custom Problem with a Predefined Optimizer

**Step 1: Create a custom problem**

```python
# my_custom_problem.py

from rastion_core.base_problem import BaseProblem
import random

class MyBinaryProblem(BaseProblem):
    def __init__(self, size=10):
        self.size = size

    def evaluate_solution(self, solution):
        # Suppose our objective is the number of 1's in the binary vector.
        # We'll actually MINIMIZE negative of that, so that more 1's => better.
        return -sum(solution)

    def random_solution(self):
        # Return a random binary list of length self.size
        return [random.randint(0, 1) for _ in range(self.size)]

    def is_feasible(self, solution):
        # Here, all binary vectors are feasible.
        return True
```

**Step 2: Provide `problem_config.json`**

```json
{
  "entry_point": "my_custom_problem:MyBinaryProblem",
  "default_params": {
    "size": 10
  }
}
```

**Step 3: Create a repo and push them** (using CLI or manually).
Then, in some other script:

```python
from rastion_hub.auto_problem import AutoProblem
from rastion_core.algorithms.genetic_algorithm import GeneticAlgorithm

problem = AutoProblem.from_repo("YourOrg/MyBinaryProblemRepo")
# or problem = MyBinaryProblem(size=10) if local

solver = GeneticAlgorithm(population_size=20, max_generations=50)
best_sol, best_cost = solver.optimize(problem)
print("Best solution:", best_sol)
print("Best cost:", best_cost)
```

### Example 2: Custom Optimizer with a Predefined Problem

**Step 1: Create a custom solver**

```python
# my_custom_solver.py

from rastion_core.base_optimizer import BaseOptimizer
import random

class MyRandomSearchSolver(BaseOptimizer):
    def __init__(self, max_iterations=1000):
        self.max_iterations = max_iterations

    def optimize(self, problem, **kwargs):
        best_sol = None
        best_cost = float("inf")
        for _ in range(self.max_iterations):
            candidate = problem.random_solution()
            cost = problem.evaluate_solution(candidate)
            if cost < best_cost:
                best_sol, best_cost = candidate, cost
        return best_sol, best_cost
```

**Step 2: Provide `solver_config.json`**

```json
{
  "entry_point": "my_custom_solver:MyRandomSearchSolver",
  "default_params": {
    "max_iterations": 500
  }
}
```

**Step 3: Upload to a GitHub repo**

Then in a separate script:

```python
from rastion_core.problems.knapsack import KnapsackProblem
from rastion_hub.auto_optimizer import AutoOptimizer

# or use AutoProblem if your problem is also external.
items = [(10, 5), (6, 3), (5, 2)]
capacity = 10
problem = KnapsackProblem(items, capacity)

solver = AutoOptimizer.from_repo("YourOrg/MyRandomSolverRepo")

best_sol, best_cost = solver.optimize(problem)
print("Best knapsack solution:", best_sol)
print("Best cost:", best_cost)
```

## License

This project is licensed under the [MIT License](./LICENSE).
