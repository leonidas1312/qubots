# Rastion Hub: A Collaborative Platform for Optimization

## Introduction
Welcome to **Rastion Hub**, a collaborative platform for hosting and sharing both **problems** and **solvers** within the field of optimization. **Rastion Hub** focuses specifically on optimization tasks and aims to streamline how individuals and organizations share, maintain, and experiment with optimization algorithms (solvers) and problem definitions.

## Key Objectives
- Provide an open, modular, and easily extensible structure for optimization.
- Support best practices for code sharing: using GitHub for version control, following a consistent structure for solver and problem definitions.
- You can push a `solver_config.json` and a `.py` solver file to share your optimizer. Similarly, for a problem, you push a `problem_config.json` and a `.py` file describing the problem.

## Website
You can find all the repos hosted with Rastion at our website : https://repo-bloom-portal.lovable.app/

## Installation
```bash
pip install rastion
```

## Example 1: Run PSO for portfolio optimization problem
```bash
from rastion_hub.auto_problem import AutoProblem
from rastion_hub.auto_optimizer import AutoOptimizer

problem = AutoProblem.from_repo(f"Rastion/portfolio-optimization", revision="main")
optimizer = AutoOptimizer.from_repo(f"Rastion/particle-swarm",
                                    revision="main",
                                    override_params={"swarm_size":60,"max_iters":500})

best_solution, best_cost = optimizer.optimize(problem)
print("Portfolio Optimization with PSO")
print("Best Solution:", best_solution)
print("Best Cost:", best_cost)
```

## Example 2: Use variational quantum algorithms as warm starters for custom classical optimization algorithms 
```bash
from rastion_hub.auto_problem import AutoProblem
from rastion_hub.auto_optimizer import AutoOptimizer
from rastion_hub.quantum_classical_pipeline import create_quantum_classical_pipeline

# 1. Load the problem instance (assumed to be a QUBO problem with a get_qubo() method).
problem = AutoProblem.from_repo(f"Rastion/max-cut", revision="main")

# 2. Load the quantum optimizer for the VQA pipeline.
quantum_optimizer = AutoOptimizer.from_repo(
   f"Rastion/vqa-qubit-eff",
   revision="main",
   override_params={
      "num_layers": 6,        
      "max_iters": 100,
      "nbitstrings": 5,
   }
)
# 3. Load the classical optimizer for the VQA pipeline. Here we use a custom RL local search.
classical_optimizer = AutoOptimizer.from_repo(
      f"Rastion/rl-optimizer",
      revision="main",
      override_params={
            "time_limit": 1  # seconds
      }
)
      
# Compose the quantum-classical pipeline.
pipeline = create_quantum_classical_pipeline(
   quantum_routine=quantum_optimizer,
   classical_optimizer=classical_optimizer
)

# Run the VQA pipeline and time its execution.
print("Running VQA pipeline ...")
vqa_solution, vqa_cost = pipeline.optimize(problem)
print(f"VQA Pipeline Solution: {vqa_solution}")
print(f"VQA Pipeline Cost: {vqa_cost}")
```

## Example 3: Run multiple solvers independently for a problem
```bash
from rastion_hub.auto_problem import AutoProblem
from rastion_hub.auto_optimizer import AutoOptimizer
from rastion_hub.optimizer_runner import run_optimizers_independently
# Load a small maxcut optimization problem.
problem = AutoProblem.from_repo(f"Rastion/max-cut", revision="main")

# Load several optimizers with optional parameter overrides.
optimizer1 = AutoOptimizer.from_repo(
   f"Rastion/particle-swarm",
   revision="main",
   override_params={"swarm_size": 50, "max_iters": 100}
)
optimizer2 = AutoOptimizer.from_repo(
   f"Rastion/tabu-search",
   revision="main",
   override_params={"max_iters": 100, "tabu_tenure": 10, "verbose": True}
)
optimizer3 = AutoOptimizer.from_repo(
   f"Rastion/exhaustive-search",
   revision="main",
)

optimizers = [optimizer1, optimizer2, optimizer3]

results = run_optimizers_independently(problem, optimizers)

# Find the best result (assuming lower cost is better).
best_optimizer, best_solution, best_cost = min(results, key=lambda x: x[2])

print("=== Independent Runs Results ===")
for name, sol, cost in results:
   print(f"Optimizer {name}: Cost = {cost}, Solution = {sol}")
print(f"\nBest optimizer: {best_optimizer} with cost = {best_cost}, solution = {best_solution}\n")
```

## Example 4: Run multiple solvers chained together for a problem
```bash
from rastion_hub.auto_problem import AutoProblem
from rastion_hub.auto_optimizer import AutoOptimizer
from rastion_hub.optimizer_runner import run_optimizers_in_chain
# Load a small maxcut optimization problem.
problem = AutoProblem.from_repo(f"Rastion/max-cut", revision="main", override_params={"num_nodes": 8})

# Load a chain of optimizers.
# For example, start with a global search particle swarm, then refine using
# tabu search and finally the custom rl optimizer.
optimizer1 = AutoOptimizer.from_repo(
   f"Rastion/particle-swarm",
   revision="main",
   override_params={"swarm_size": 50, "max_iters": 100}
)
optimizer2 = AutoOptimizer.from_repo(
   f"Rastion/tabu-search",
   revision="main",
   override_params={"max_iters": 100, "tabu_tenure": 10, "verbose": True}
)
optimizer3 = AutoOptimizer.from_repo(
   f"Rastion/rl-optimizer",
   revision="main",
   override_params={"time_limit": 1  # seconds
   
   }
)

optimizers_chain = [optimizer1, optimizer2, optimizer3]

final_solution, final_cost = run_optimizers_in_chain(problem, optimizers_chain)

print("=== Chained Refinement Results ===")
print(f"Final refined solution: {final_solution} with cost: {final_cost}\n")

exhaustive_optimizer = AutoOptimizer.from_repo(
   f"Rastion/exhaustive-search",
   revision="main",
)

best_solution, best_cost = exhaustive_optimizer.optimize(problem)
print("=== Exhaustive Results ===")
print(f"Best solution: {best_solution} with cost: {best_cost}\n")

```

## Repository Overview
Below is a brief overview of the main files and directories included in this codebase:

### 1. Examples
- **examples/**: Demonstration scripts and subdirectories that showcase how to interact with the Rastion Hub. Notable subfolders:
  - **optimizers/**: Example solver implementations. Each subfolder typically contains a `.py` solver file and a `solver_config.json`.
  - **problems/**: Example optimization problems. Each folder typically has a `.py` problem file and a `problem_config.json`.
  - **usage/**: Scripts that demonstrate how to run solvers against problems, how to chain them, how to do quantum-classical pipelines, etc.

### 2. Rastion CLI
- **rastion_cli/cli.py**: A command-line interface to interact with the Rastion Hub. Provides commands like:
  - `create_repo`: Create a new repository on GitHub under a specified organization.
  - `update_repo`: Update an existing repository with local changes.
  - `push_solver`: Push a new solver to a GitHub repo.
  - `push_problem`: Push a new problem definition to a GitHub repo.
  - `run_solver`: Clone a solver from a GitHub repo, optionally load a problem, and run the optimization.

### 3. Core Library (`rastion_hub/`)
- **auto_optimizer.py** / **auto_problem.py**: Classes for automatically retrieving solver or problem definitions from a GitHub repository, installing them in a local cache, and dynamically importing them.
- **base_optimizer.py** / **base_problem.py**: Abstract base classes that define minimal interfaces for any solver (`optimize(problem)`) or problem (`evaluate_solution(solution)` / `random_solution()`).
- **optimizer_runner.py**: Helper functions for running multiple solvers either independently or in a chain.
- **quantum_classical_pipeline.py**: A pipeline for combining a quantum routine and a classical optimizer.


## How the Platform Works
1. **Solvers**:
   - Each solver is defined by a Python file implementing a class derived from `BaseOptimizer`.
   - A `solver_config.json` must be present, specifying:
     - `entry_point`: The module and class name (e.g., `my_solver:MySolver`).
     - `default_params`: Default hyperparameters for the solver.
   - Once pushed to a repository on GitHub (e.g., `Rastion/my-solver-repo`), it can be retrieved with:
     ```bash
     from rastion_hub.auto_optimizer import AutoOptimizer
     solver = AutoOptimizer.from_repo("Rastion/my-solver-repo", revision="main")
     ```
   - Then call `solver.optimize(problem)`.

2. **Problems**:
   - Each problem extends `BaseProblem`, implementing `evaluate_solution(solution)` and optionally `random_solution()`. QUBO-based problems also provide `get_qubo()`.
   - A `problem_config.json` indicates the `entry_point` and default parameters.
   - Retrieve with:
     ```python
     from rastion_hub.auto_problem import AutoProblem
     problem = AutoProblem.from_repo("Rastion/my-problem-repo", revision="main")
     ```

3. **Rastion CLI**:
   - You can create, clone, push, or update repos for both problems and solvers. It’s designed to simplify the process:
     ```bash
     # Create a new repo on GitHub
     rastion create_repo my-cool-solver

     # Push local solver code and config
     rastion push_solver my-cool-solver --file my_solver.py --config solver_config.json
     ```

4. **Chain or Combine**:
   - Multiple solvers can be combined to refine solutions. Or a quantum solver can be combined with a classical refinement step, as in the quantum-classical pipeline.

## Key Classes and Modules
1. **`BaseProblem`**:
   - An abstract class that requires at least `evaluate_solution(solution) -> float`.
   - Optionally override `random_solution()`.

2. **`BaseOptimizer`**:
   - Must implement `optimize(problem, **kwargs) -> (best_solution, best_value)`.

3. **`AutoOptimizer`** and **`AutoProblem`**:
   - Provide the static method `from_repo(...)` to automatically clone/pull from GitHub and load the respective config.

4. **`optimizer_runner`**:
   - `run_optimizers_independently(problem, [opt1, opt2, ...])` -> Compares each separately.
   - `run_optimizers_in_chain(problem, [opt1, opt2, ...])` -> Sequential refinement.

5. **`quantum_classical_pipeline`**:
   - `create_quantum_classical_pipeline(quantum_routine, classical_optimizer)` -> Returns a composite pipeline optimizer.
   - Runs quantum routine first, then classical refinement.

6. **`vqa_interface`**:
   - `VQACycleInterface` is a more specialized approach for VQA loops.
   - Ties together a circuit ansatz, cost function, and classical optimization of quantum parameters.

## Detailed Usage and Examples
Below are some showcased scripts from the `examples/usage` directory, along with a fully fledged example of how you might interact with Rastion from scratch.

### 1. `run_portfolio_optimization.py`
Shows how to load a portfolio optimization problem and solve it with a Particle Swarm Optimizer. Example usage:
```bash
cd examples/usage
python run_portfolio_optimization.py
```
You will see output describing the best portfolio weights and the corresponding cost.

### 2. `run_portfolio_with_chains.py`
Demonstrates running multiple optimizers independently vs. chaining them sequentially. This is ideal for prototyping a hybrid strategy (e.g., a global search approach followed by a local refinement). Example:
```bash
cd examples/usage
python run_portfolio_with_chains.py
```

### 3. `run_vqa_pipeline.py` 
Illustrates a variational quantum algorithm pipeline, combining a quantum ansatz with a classical optimizer (like Torch Adam). Example:
```bash
cd examples/usage
python run_vqa_pipeline.py
```

### 4. `benchmark_vqa_pipeline.py`
Compares a quantum-classical pipeline approach to an exhaustive search on small QUBO problems, providing insights into performance, accuracy, and runtime. Example:
```bash
cd examples/usage
python benchmark_vqa_pipeline.py
```

## Complete Rastion CLI Walkthrough
Below is an example session demonstrating how to use the **Rastion** CLI to manage both solver and problem repos, then run them:

```bash
# 1. Create a new solver repository in the 'Rastion' org.
$ export GITHUB_TOKEN="<YOUR_PERSONAL_ACCESS_TOKEN>"
$ rastion create_repo my-solver --github-token $GITHUB_TOKEN
```

```bash
# 2. Push your solver code and config to GitHub.
$ cd /path/to/local/my-solver
$ ls
my_solver.py solver_config.json
$ rastion push_solver my-solver \
    --file my_solver.py \
    --config solver_config.json \
    --github-token $GITHUB_TOKEN
```

```bash
# 3. Create a new problem repository in the 'Rastion' org.
$ rastion create_repo my-problem --github-token $GITHUB_TOKEN
```

```bash
# 4. Push your problem code and config.
$ cd /path/to/local/my-problem
$ ls
my_problem.py problem_config.json
$ rastion push_problem my-problem \
    --file my_problem.py \
    --config problem_config.json \
    --github-token $GITHUB_TOKEN
```

```bash
# 5. Optionally, update existing repos if local changes are made.
$ rastion update_repo my-solver \
    --local-dir /path/to/local/my-solver \
    --branch main \
    --github-token $GITHUB_TOKEN
```

```bash
# 6. (Optional) Clone a repo to inspect or develop further.
$ rastion clone_repo my-solver --branch main
```

```bash
# 7. Use 'run_solver' to test the solver with a problem from the hub.
$ rastion run_solver Rastion/my-solver-repo \
    --solver-rev main \
    --problem-repo Rastion/my-problem-repo \
    --problem-rev main
```

```bash
# 8. Delete a repo if no longer needed.
$ rastion delete_repo my-solver --github-token $GITHUB_TOKEN
```

This sequence covers the primary Rastion CLI commands—creating, pushing, updating, cloning, running, and optionally deleting.

## Contributing
We encourage you to add new solvers and problems! Follow these steps:
1. **Create a new solver**: Implement a class extending `BaseOptimizer`, supply a `solver_config.json`, and push to a GitHub repo.
2. **Create a new problem**: Implement a class extending `BaseProblem`, supply a `problem_config.json`, and push similarly.
3. **Reference** your code from other scripts using the `AutoOptimizer` or `AutoProblem` classes.

Feel free to open issues, propose enhancements, or submit pull requests.

## License
This project is licensed under the [Apache 2.0 License](LICENSE). See the `LICENSE` file for more details.
