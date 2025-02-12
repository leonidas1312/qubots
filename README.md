# Rastion Hub: A Collaborative Platform for Optimization

## Introduction
Welcome to **Rastion Hub**, a collaborative platform for hosting and sharing both **problems** and **optimizers** within the field of optimization. **Rastion Hub** focuses specifically on optimization tasks and aims to streamline how individuals and organizations share, maintain, and experiment with optimization algorithms (optimizers) and problem definitions.

## Key Objectives
- Provide an open, modular, and easily extensible structure for optimization.
- Support best practices for code sharing: using GitHub for version control, following a consistent structure for optimizer and problem definitions.
- You can push a `optimizer_config.json` and a `.py` optimizer file to share your optimizer. Similarly, for a problem, you push a `problem_config.json` and a `.py` file describing the problem.

## Website
You can find all the Rastion repos (optimizers & problems) as well as the docs and guides at our website : https://repo-bloom-portal.lovable.app/

## Installation
```bash
pip install rastion
```

## Repository Overview
Below is a brief overview of the main files and directories included in this codebase:

### 1. Examples
- **examples/**: Demonstration scripts and subdirectories that showcase how to interact with the Rastion Hub. Notable subfolders:
  - **implemented_tests/**: Testing functions that passed and return meaniningful results.
  - **optimizers/**: Example optimizers. Each folder typically has a `.py` file and a `optimizer_config.json`.
  - **problems/**: Example optimization problems. Each folder typically has a `.py` file and a `problem_config.json`.

### 2. Rastion CLI
- **rastion_cli/cli.py**: A command-line interface to interact with the Rastion Hub. Provides commands like:
  - `create_repo`: Create a new repository on GitHub on Rastion organization.
  - `update_repo`: Update an existing repository with local changes.
  - `push_optimizer`: Push a new optimizer to a GitHub repo (use after create_repo).
  - `push_problem`: Push a new problem definition to a GitHub repo (use after create_repo).
  - `run_optimizer`: Clone a optimizer from a GitHub repo, optionally load a problem, and run the optimization.
  - `delete_repo`: Delete a GitHub repo.

### 3. Core Library (`rastion_hub/`)
- **auto_optimizer.py** / **auto_problem.py**: Classes for automatically retrieving optimizer or problem definitions from a GitHub repository, installing them in a local cache, and dynamically importing them.
- **base_optimizer.py** / **base_problem.py**: Abstract base classes that define minimal interfaces for any optimizer or problem.
- **optimizer_runner.py**: Helper functions for running multiple optimizers either independently or in a chain.
- **quantum_classical_pipeline.py**: A pipeline for combining a quantum routine and a classical optimizer.


## How the Platform Works
1. **optimizers**:
   - Each optimizer is defined by a Python file implementing a class derived from `BaseOptimizer`.
   - A `optimizer_config.json` must be present, specifying:
     - `entry_point`: The module and class name (e.g., `my_optimizer:Myoptimizer`).
     - `default_params`: Default hyperparameters for the optimizer.
   - Once pushed to a repository on GitHub (e.g., `Rastion/my-optimizer-repo`), it can be retrieved with:
     ```bash
     from rastion_hub.auto_optimizer import AutoOptimizer
     optimizer = AutoOptimizer.from_repo("Rastion/my-optimizer-repo", revision="main")
     ```
   - Then call `optimizer.optimize(problem)`.

2. **Problems**:
   - Each problem extends `BaseProblem`, implementing `evaluate_solution(solution)` and optionally `random_solution()`. QUBO-based problems also provide `get_qubo()`.
   - A `problem_config.json` indicates the `entry_point` and default parameters.
   - Retrieve with:
     ```python
     from rastion_hub.auto_problem import AutoProblem
     problem = AutoProblem.from_repo("Rastion/my-problem-repo", revision="main")
     ```

3. **Rastion CLI**:
   - You can create, clone, push, or update repos for both problems and optimizers. It’s designed to simplify the process:
     ```bash
     # Create a new repo on GitHub
     rastion create_repo my-cool-optimizer

     # Push local optimizer code and config
     rastion push_optimizer my-cool-optimizer --file my_optimizer.py --config optimizer_config.json
     ```

4. **Chain or Combine**:
   - Multiple optimizers can be combined to refine solutions. Or a quantum optimizer can be combined with a classical refinement step, as in the quantum-classical pipeline.

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


## Contributing
We encourage you to add new optimizers and problems! Follow these steps:
1. **Create a new optimizer**: Implement a class extending `BaseOptimizer`, supply a `optimizer_config.json`, and push to a GitHub repo.
2. **Create a new problem**: Implement a class extending `BaseProblem`, supply a `problem_config.json`, and push similarly.
3. **Reference** your code from other scripts using the `AutoOptimizer` or `AutoProblem` classes.

Feel free to open issues, propose enhancements, or submit pull requests.

## License
This project is licensed under the [Apache 2.0 License](LICENSE). See the `LICENSE` file for more details.
