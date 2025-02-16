# Qubots: A Collaborative Optimization Framework

[![PyPI version](https://img.shields.io/pypi/v/qubots.svg)](https://pypi.org/project/qubots/)
[![Build Status](https://github.com/leonidas1312/qubots/actions/workflows/publish.yml/badge.svg)](https://github.com/leonidas1312/qubots/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/leonidas1312/qubots.svg)](https://github.com/leonidas1312/qubots/issues)
[![GitHub forks](https://img.shields.io/github/forks/leonidas1312/qubots.svg)](https://github.com/leonidas1312/qubots/network)

Qubots is a Python library that turns optimization problems and optimization algorithms (optimizers) into shareable, modular “qubots”. Whether you’re developing a new optimizer or formulating a complex problem, qubots makes it easy to package, share, and run your work. Through our central hub, Rastion, you can browse, download, and contribute to a growing ecosystem of GitHub repositories dedicated to cutting-edge optimization – spanning classical, quantum, and hybrid methods.

## Rastion

Rastion serves as the central repository hub for all qubots-related projects. On the Rastion page you’ll find detailed documentation, guides on usage, contribution instructions, and real-world use cases that demonstrate how to combine and chain qubots for advanced optimization workflows. Visit the demo page here: https://rastion.netlify.app/

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Technical Overview](#technical-overview)
  - [Base Classes](#base-classes)
  - [Dynamic Qubot Loading: AutoProblem & AutoOptimizer](#dynamic-qubot-loading-autoproblem--autooptimizer)
  - [Chaining and Pipelines](#chaining-and-pipelines)
- [Use Cases & Examples](#use-cases--examples)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Modular Design**: Formulate your optimization problems and solvers as independent “qubots” that follow a simple interface.
- **Dynamic Loading**: Use the built-in `AutoProblem` and `AutoOptimizer` classes to dynamically load and execute GitHub repositories containing your qubots.
- **Hybrid Optimization**: Seamlessly combine quantum and classical optimizers using our quantum-classical pipeline.
- **CLI Integration**: Manage repositories (create, update, delete, push) via command-line tools for smooth collaboration.
- **Extensive Examples**: Learn from a rich collection of examples covering classical optimization (e.g., Particle Swarm, Tabu Search), quantum approaches (e.g., QAOA, VQE), and continuous problem solvers.

## Installation

Qubots is available on PyPI. To install, simply run:

```bash
pip install qubots
```

For full documentation and guides, please visit the Rastion Hub demo page https://rastion.netlify.app/docs .

## Getting Started

Here’s a brief example showing how to load a problem and a solver from the Rastion hub, then run the optimization:

```python
from qubots.auto_problem import AutoProblem
from qubots.auto_optimizer import AutoOptimizer

# Load the portfolio optimization problem from the Rastion GitHub repository.
# By default this toy portfolio optimization problem is solved if the sum of the 3 elements to be equal to 1.
problem = AutoProblem.from_repo("Rastion/portfolio-optimization")

# Load the Particle Swarm Optimizer with overridden parameters.
optimizer = AutoOptimizer.from_repo(
    "Rastion/particle-swarm",
    override_params={"swarm_size": 60, "max_iters": 500}
)

# Run the optimization and print results.
best_solution, best_cost = optimizer.optimize(problem)
print("Portfolio Optimization with PSO")
print("Best Solution:", best_solution)
print("Best Cost:", best_cost)
```

This simple workflow demonstrates how qubots allow you to plug and play various optimization modules with minimal boilerplate.

## Technical Overview

### Base Classes

At the core of qubots are two abstract base classes:

- **BaseProblem**
  - Defines the interface for any optimization problem. Every problem qubot must implement:
    - `evaluate_solution(solution) -> float`
      - Computes the objective value (or cost) for a given candidate solution.
    - `random_solution() (optional)`
      - Generates a random feasible solution for the problem.

- **BaseOptimizer**
  - Provides the interface for optimization algorithms. Every optimizer qubot must implement:
    - `optimize(problem, initial_solution=None, **kwargs) -> (solution, cost)`
      - Runs the optimization on a given problem, optionally starting from an initial solution.

These interfaces ensure that every qubot—whether problem or solver—can be seamlessly interchanged and composed.

### Dynamic Qubot Loading: AutoProblem & AutoOptimizer

To encourage modularity and collaboration, qubots can be dynamically loaded from GitHub repositories. This is accomplished using:

- **AutoProblem**
  - Clones (or pulls) a repository from GitHub.
  - Installs required packages (via `requirements.txt`).
  - Reads a `problem_config.json` file that specifies an `entry_point` (formatted as `module:ClassName`) and default parameters.
  - Dynamically imports and instantiates the problem qubot.

- **AutoOptimizer**
  - Follows a similar process using a `solver_config.json` file.
  - Installs required packages (via `requirements.txt`).
  - Merges default parameters with any user-supplied `override_params`.
  - Dynamically loads the optimizer class and returns an instance ready for use.

This design allows developers to share their work as self-contained GitHub repos that anyone can load, test, and incorporate into larger workflows. **Remote execution of python code files, including installing packages via requirements.txt, is not a good practice**. For this reason it is suggested to use Rastion & Qubots in a secure environment using `python -m venv` or `conda create --name my_rastion_env python=3.9`. Developing a sandbox environment & shareable object for qubots should definitely be in the future plans.

### Chaining and Pipelines

Qubots also supports more advanced patterns:

- **Independent Runs**: Use helper functions (in `optimizer_runner.py`) to run several optimizers independently on the same problem, then compare their results.

- **Chained Refinement**: Sequentially refine a solution by passing the output of one optimizer as the initial solution for the next.

- **Quantum-Classical Pipelines**: The `QuantumClassicalPipeline` class (and its helper function `create_quantum_classical_pipeline`) enables you to combine a quantum routine with a classical optimizer. For example, a quantum solver might generate a candidate solution which is then refined by a classical optimizer.

## Examples

The `examples/` directory contains some examples and test cases, including:

- **Implemented tests**: Tests about the qubots library that have been tested to working locally.
- **Testing to do**: Tests that fail locally.

## Contributing

We welcome contributions to expand the qubots ecosystem! Here’s how you can get involved:

- **Report Issues**: If you encounter bugs or have feature suggestions, please open an issue on the GitHub repository.
- **Submit Pull Requests**: Follow the coding guidelines and ensure that your changes include tests and documentation updates as needed.
- **Share Your Qubots**: Have an innovative optimizer or an interesting problem formulation? Create a GitHub repo under the Rastion organization and share it via the Rastion hub.
- **Improve Documentation**: Enhance guides, tutorials, and examples to help others get started with qubots.

For more detailed contribution guidelines, please refer to our `CONTRIBUTING.md`.

## License

This project is licensed under the [Apache License 2.0](./LICENSE).

By leveraging the flexible design of qubots and the collaborative power of Rastion, you can rapidly prototype, share, and improve optimization solutions—be it for classical problems, quantum algorithms, or hybrid systems.


For more information, contact gleonidas303@gmail.com.

