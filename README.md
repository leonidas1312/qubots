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
   - [Example 3: Using the CLI to Create and Run a Custom Solver & Problem](#example-3-using-the-cli-to-create-and-run-a-custom-solver--problem)
5. [Potential Problems & Solvers](#potential-problems--solvers)
6. [License](#license)
7. [Download the Codebase](#download-the-codebase)

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

- \`\`: Core library containing base classes and built-in solvers/problems.
- \`\`: The `AutoOptimizer` class that can clone a solver repo and load its Python class.
- \`\`: The `AutoProblem` class that can do the same for problem repos.
- \`\`: The CLI entry point.

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

`AutoProblem` and `AutoOptimizer` are classes designed to fetch your custom problem or solver from a remote GitHub repository. They rely on a `problem_config.json` or `solver_config.json` file describing the entry point:

- \`\`:

```json
{
  "entry_point": "my_module:MySolverClass",
  "default_params": {
    "some_param": 123
  }
}
```

- \`\`:

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

\*\*Step 2: Provide \*\*\`\`

```json
{
  "entry_point": "my_custom_problem:MyBinaryProblem",
  "default_params": {
    "size": 10
  }
}
```

**Step 3: Create a repo and push them** (using CLI or manually). Then, in some other script:

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

\*\*Step 2: Provide \*\*\`\`

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

### Example 3: Using the CLI to Create and Run a Custom Solver & Problem

This section shows how to use the `rastion` CLI to create two repositories on GitHub—one for a custom solver and one for a custom problem—and then run them together.

> **Prerequisites**:
>
> 1. You must have your GitHub token set as an environment variable `GITHUB_TOKEN`.
> 2. You must have `git` installed and accessible.

**Step 0: Installation**

Make sure you have installed `rastion-hub` in your environment (e.g., `pip install .`).

```bash
pip install .
```

**Step 1: Create a new solver repository**

```bash
rastion create_repo my-solver-repo --org MyOrg --private False
```

This creates a new public repo under `MyOrg` called `my-solver-repo`. You can verify it on GitHub.

**Step 2: Push a custom solver**

1. Create a Python file `my_solver.py`:

   ```python
   from rastion_core.base_optimizer import BaseOptimizer
   import random

   class MyExampleSolver(BaseOptimizer):
       def __init__(self, iterations=100):
           self.iterations = iterations

       def optimize(self, problem, **kwargs):
           best_sol = None
           best_cost = float("inf")
           for _ in range(self.iterations):
               candidate = problem.random_solution()
               cost = problem.evaluate_solution(candidate)
               if cost < best_cost:
                   best_sol, best_cost = candidate, cost
           return best_sol, best_cost
   ```

2. Create the `solver_config.json`:

   ```json
   {
     "entry_point": "my_solver:MyExampleSolver",
     "default_params": {
       "iterations": 50
     }
   }
   ```

3. Push them to GitHub:

   ```bash
   rastion push_solver my-solver-repo --file my_solver.py --config solver_config.json --org MyOrg
   ```

**Step 3: Create a new problem repository**

```bash
rastion create_repo my-problem-repo --org MyOrg --private False
```

**Step 4: Push a custom problem**

1. Create a Python file `my_problem.py`:

   ```python
   from rastion_core.base_problem import BaseProblem
   import random

   class MyExampleProblem(BaseProblem):
       def __init__(self, size=5):
           self.size = size

       def evaluate_solution(self, solution):
           # We'll treat the sum of the solution's elements as a cost.
           # Minimizing sum => smaller sum is better.
           return sum(solution)

       def random_solution(self):
           # Generate random integers in [0, 10]
           return [random.randint(0, 10) for _ in range(self.size)]
   ```

2. Create the `problem_config.json`:

   ```json
   {
     "entry_point": "my_problem:MyExampleProblem",
     "default_params": {
       "size": 5
     }
   }
   ```

3. Push them to GitHub:

   ```bash
   rastion push_problem my-problem-repo --file my_problem.py --config problem_config.json --org MyOrg
   ```

**Step 5: Run the solver on the problem**

To run everything together, call:

```bash
rastion run_solver MyOrg/my-solver-repo \
    --solver-rev main \
    --problem-repo MyOrg/my-problem-repo \
    --problem-rev main
```

This will:

1. Clone or pull the solver repo `my-solver-repo`.
2. Load the `MyExampleSolver` with `iterations=50` (from `solver_config.json`).
3. Clone or pull the problem repo `my-problem-repo`.
4. Load the `MyExampleProblem` with `size=5` (from `problem_config.json`).
5. Call `solver.optimize(problem)`.
6. Print out the best solution found and its cost.

You should see output similar to:

```
Loading solver from: MyOrg/my-solver-repo@main
Loading problem from: MyOrg/my-problem-repo@main
Optimization completed: best_sol=[2,0,1,0,3], best_cost=6
```

Feel free to tweak the parameters in your JSON config, or pass `--override_params` if you want to override them on the command line in the future.

## Potential Problems & Solvers

Below is a list of various classical and advanced problems and solvers that could be integrated into Rastion:

### Potential Problems

- [x] Traveling Salesman Problem (TSP)
- [x] Knapsack Problem
- [x] Quadratic Optimization Problem
- [x] QUBO Problem
- [ ] Vehicle Routing Problem (VRP)
   - Optimize routes for a fleet of vehicles delivering to various locations under capacity constraints.
- [ ] Job Scheduling Problem (JSP)
   - Schedule jobs on machines while minimizing the total completion time or meeting deadlines.
- [ ] Facility Location Problem (FLP)
   - Decide where to place facilities (warehouses, factories) to minimize distribution costs.
- [ ] Portfolio Optimization Problem
   - Allocate asset weights to minimize risk (or maximize the Sharpe ratio) given returns and covariance data.
- [ ] Network Flow Optimization
   - Optimize the flow through a network (e.g., transportation or communication networks) while minimizing cost or maximizing throughput.
- [ ] Graph Coloring Problem
   - Color the nodes of a graph such that adjacent nodes do not share the same color while minimizing the number of colors used.
- [ ] Bin Packing Problem
   - Pack objects of varying sizes into a finite number of bins in a way that minimizes the number of bins.
- [ ] Resource Allocation Problem
   - Allocate limited resources among competing activities to optimize a given objective.
- [ ] Inventory Management Problem
   - Optimize order quantities and reorder points in supply chains to balance holding costs with shortage risks.
- [ ] Hyperparameter Optimization
   - Optimize hyperparameters for machine learning models using black‐box optimization methods.
- [ ] Neural Architecture Search (NAS)
   - Automatically design neural network architectures for specific tasks.
- [ ] Optimal Control Problems
   - Solve dynamic optimization problems (e.g., in robotics or finance) using methods like dynamic programming.
- [ ] Energy Management / Smart Grid Optimization
   - Optimize energy generation/distribution in smart grids to balance supply, demand, and cost.
- [ ] Sensor Placement Optimization
   - Determine the optimal sensor locations in a network to maximize coverage or information gain.
- [ ] Supply Chain Optimization
   - Optimize decisions along a supply chain such as transportation, inventory, and production scheduling.
- [ ] Data Center Resource Optimization
   - Allocate computing resources (CPU, memory, bandwidth) in data centers to minimize energy consumption and maximize throughput.


### Potential Solvers (Optimizers)

-

## License

This project is licensed under the [MIT License](./LICENSE).
