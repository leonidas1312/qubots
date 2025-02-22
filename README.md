# Qubots: A Collaborative Optimization Framework

[![PyPI version](https://img.shields.io/pypi/v/qubots.svg)](https://pypi.org/project/qubots/)
[![Build Status](https://github.com/leonidas1312/qubots/actions/workflows/publish.yml/badge.svg)](https://github.com/leonidas1312/qubots/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/leonidas1312/qubots.svg)](https://github.com/leonidas1312/qubots/issues)
[![GitHub forks](https://img.shields.io/github/forks/leonidas1312/qubots.svg)](https://github.com/leonidas1312/qubots/network)

Qubots is a Python library that turns optimization problems and optimization algorithms (optimizers) into shareable, modular “qubots”. Whether you’re developing a new optimizer or formulating a complex problem, qubots makes it easy to package, share, and run your work. Through our central hub, Rastion, you can browse, download, and contribute to a growing ecosystem of GitHub repositories dedicated to cutting-edge optimization – spanning classical, quantum, and hybrid methods.

## Rastion

Rastion serves as the central repository hub for all qubots-related projects. On the Rastion page you’ll find detailed documentation, guides on usage, contribution instructions, and real-world use cases that demonstrate how to combine and chain qubots for advanced optimization workflows. Visit the demo page here: https://rastion.com

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Technical Overview](#technical-overview)
  - [Base Classes](#base-classes)
  - [Dynamic Qubot Loading: AutoProblem & AutoOptimizer](#dynamic-qubot-loading-autoproblem--autooptimizer)
  - [Chaining and Pipelines](#chaining-and-pipelines)
- [Roadmap](#roadmap)
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

For full documentation and guides, please visit the Rastion Hub demo page https://rastion.com/docs .

## Getting Started

Here’s a brief example showing how to load a problem and a solver from the Rastion hub, then run the optimization:

```python
from qubots.auto_problem import AutoProblem
from qubots.auto_optimizer import AutoOptimizer

# Load qubot problem
problem = AutoProblem.from_repo("Rastion/traveling_salesman_problem")

# Load first qubot optimizer based on ortools and constraint programming logic for TSP
optimizer1 = AutoOptimizer.from_repo(
    "Rastion/ortools_tsp_solver"
)

# Run optimization
best_solution, best_cost = optimizer1.optimize(problem)
print(f"Best cost: {best_cost}")
print(f"Best solution: {best_solution}")

# Load second qubot optimizer based on a heuristic simulated annealing algorithm
# that solves the QUBO formulation of the TSP qubot problem and decodes the solution back to 
# original format
optimizer2 = AutoOptimizer.from_repo(
    "Rastion/sa_tsp_qubo_optimizer",
    #override_params={"initial_temp": 100000, "cooling_rate": 0.9}
)

# Run optimization
best_solution, best_cost = optimizer2.optimize(problem)
print(f"Best cost: {best_cost}")
print(f"Best solution: {best_solution}")

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

## Roadmap

We're building qubots into an open source optimization community! Here's our trajectory:

- ✅ **Core Framework (v0.1.2)**  
  Launched dynamic loading of qubots, hybrid pipelines, CLI tools and guides

- 🚧 **Metadata & Compatibility (Current Focus)**  
  Adding problem/optimizer tags and validation tools for:  
  - *Qubot problems* (Combinatorial, Linear & Integer programming, Math functions)  
  - *Qubot optimizers* (TSP solvers, scheduling problems)

- ⏳ **Hardware-Compatible Ecosystem**  
  - Quantum Backends (Qiskit, Braket, D-Wave integration)  
  - GPU Accelerated Optimizers (PyTorch/TF integration)  
  - Edge Device Deployment (ONNX-optimized qubots)

- ⏳ **Domain-Specific Qubots**  
  **First Wave Targets:**  
  - 🧪 *Chemical Engineering*  
    - Molecular docking problems  
    - Quantum chemistry optimizers  
  - ✈️ *Aerospace Design*  
    - CFD parameter optimization  
    - Lightweight structure solvers  
  - 🔐 *Cryptography*  
    - Lattice-based optimization  
    - Post-quantum crypto challenges  

- ⏳ **Rastion Hub Expansion**  
  - Domain-specific leaderboards  
  - Hardware compatibility filters  
  - User-curated collections  
  - Live optimization visualizations




## Examples

The `examples/` directory contains some examples and test cases, including:

- **Implemented tests**: Tests about the qubots library that have been tested to working locally.
- **Testing to do**: Tests that fail locally.

## Contributing

We welcome contributions to expand the qubots ecosystem! 

For more detailed contribution guidelines, please refer to our www.rastion.com/docs.

## License

This project is licensed under the [Apache License 2.0](./LICENSE).

By leveraging the flexible design of qubots and the collaborative power of Rastion, you can rapidly prototype, share, and improve optimization solutions—be it for classical problems, quantum algorithms, or hybrid systems.


For more information, contact gleonidas303@gmail.com.

