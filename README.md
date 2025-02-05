# Rastion Hub: A Collaborative Platform for Optimization

## Introduction
Welcome to **Rastion Hub**, a collaborative platform for hosting and sharing both **problems** and **solvers** within the field of optimization. Drawing inspiration from the Hugging Face model hub, **Rastion Hub** focuses specifically on optimization tasks and aims to streamline how individuals and organizations share, maintain, and experiment with optimization algorithms (solvers) and problem definitions.

This repository illustrates what an example codebase for **Rastion Hub** looks like. It includes:
1. **Problems**: Various well-known (and toy) optimization problems, each structured in a standardized format.
2. **Solvers**: A collection of classical and quantum-inspired optimizers.
3. **Utilities and Scripts**: Tools to demonstrate how to upload/download solvers and problems from the Rastion Hub, along with example usage scenarios.

## Key Objectives
- Provide an open, modular, and easily extensible structure for optimization.
- Support best practices for code sharing: using GitHub for version control, following a consistent structure for solver and problem definitions.
- Encourage a format similar to Hugging Face's approach: you can push a `solver_config.json` and a `.py` solver file to share your optimizer. Similarly, for a problem, you push a `problem_config.json` and a `.py` file describing the problem.

## Repository Overview
Below is a brief overview of the main files and directories included in this codebase:

### 1. Licensing
- **LICENSE**: Uses the Apache License 2.0, a permissive open-source license.

### 2. Build Configuration
- **pyproject.toml**: Build-system config, specifying dependencies (e.g., `setuptools`, `wheel`) and listing project metadata (`name`, `version`, `description`, etc.). Also includes an `entry_point` for `rastion` CLI.
- **requirements.txt**: Simple list of Python dependencies.

### 3. Examples
- **examples/**: Demonstration scripts and subdirectories that showcase how to interact with the Rastion Hub. Notable subfolders:
  - **optimizers/**: Example solver implementations (Ant Colony, Bayesian Optimization, Differential Evolution, Evolution Strategies, Particle Swarm, RL-based local search, Tabu Search, etc.). Each subfolder typically contains a `.py` solver file and a `solver_config.json`.
  - **problems/**: Example optimization problems (Bin Packing, Data Center, Energy Management, Facility Location, Gate Assignment, Graph Coloring, Knapsack, Max-Cut, etc.). Each folder typically has a `.py` problem file and a `problem_config.json`.
  - **usage/**: Scripts that demonstrate how to run solvers against problems, how to chain them, how to do quantum-classical pipelines, etc.

### 4. Rastion CLI
- **rastion_cli/cli.py**: A command-line interface to interact with the Rastion Hub. Provides commands like:
  - `create_repo`: Create a new repository on GitHub under a specified organization.
  - `update_repo`: Update an existing repository with local changes.
  - `push_solver`: Push a new solver to a GitHub repo.
  - `push_problem`: Push a new problem definition to a GitHub repo.
  - `run_solver`: Clone a solver from a GitHub repo, optionally load a problem, and run the optimization.

### 5. Core Library (`rastion_hub/`)
- **auto_optimizer.py** / **auto_problem.py**: Classes for automatically retrieving solver or problem definitions from a GitHub repository, installing them in a local cache, and dynamically importing them.
- **base_optimizer.py** / **base_problem.py**: Abstract base classes that define minimal interfaces for any solver (`optimize(problem)`) or problem (`evaluate_solution(solution)` / `random_solution()`).
- **optimizer_runner.py**: Helper functions for running multiple solvers either independently or in a chain.
- **quantum_classical_pipeline.py**: A pipeline for combining a quantum routine and a classical optimizer.
- **qubit_eff.py**: An example quantum helper that sets up a hardware-efficient ansatz for QUBO problems (Under developement).
- **vqa_interface.py**: A wrapper class (`VQACycleInterface`) for building VQA workflows. It sets up a `QuantumParameterProblem` so a classical optimizer can tune quantum circuit parameters (Under developement).

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
     rastion create_repo my-cool-solver --org Rastion --private

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

6. **`vqa_interface`** (Under developement):
   - `VQACycleInterface` is a more specialized approach for VQA loops.
   - Ties together a circuit ansatz, cost function, and classical optimization of quantum parameters.

## Usage Examples

A few example scripts are provided in `examples/usage`:

1. **`run_vehicle_routing.py`**:
   - Demonstrates loading a Vehicle Routing Problem from the hub and solving it with an Ant Colony Optimizer.

2. **`run_portfolio_optimization.py`**:
   - Loads a portfolio optimization problem and solves it with a Particle Swarm Optimizer.

3. **`benchmark_vqa_pipeline.py`**:
   - Compares a quantum-classical pipeline approach with an exhaustive search for small QUBO problems.

4. **`run_portfolio_with_chains.py`**:
   - Shows how to run multiple optimizers independently vs. in a chain.

## Getting Started
1. **Installation**:
   - Clone this repo locally.
   - Install the dependencies:
     ```bash
     pip install -r requirements.txt
     ```
2. **Using the CLI**:
   - Install the package (so that `rastion` CLI is recognized) by running:
     ```bash
     pip install .
     ```
   - Now you can use commands like:
     ```bash
     rastion create_repo my-solver
     rastion push_solver my-solver --file my_solver.py --config solver_config.json
     ```

3. **Running Examples**:
   - Explore `examples/`. For instance:
     ```bash
     cd examples/usage
     python run_vehicle_routing.py
     ```

## Contributing
We encourage you to add new solvers and problems! Follow these steps:
1. **Create a new solver**: Implement a class extending `BaseOptimizer`, supply a `solver_config.json`, and push to a GitHub repo.
2. **Create a new problem**: Implement a class extending `BaseProblem`, supply a `problem_config.json`, and push similarly.
3. **Reference** your code from other scripts using the `AutoOptimizer` or `AutoProblem` classes.

Feel free to open issues, propose enhancements, or submit pull requests.

## License
This project is licensed under the [Apache 2.0 License](LICENSE). See the `LICENSE` file for more details.

