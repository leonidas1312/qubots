# Qubots: A Collaborative Optimization Framework

[![PyPI version](https://img.shields.io/pypi/v/qubots.svg)](https://pypi.org/project/qubots/)
[![Build Status](https://github.com/leonidas1312/qubots/actions/workflows/publish.yml/badge.svg)](https://github.com/leonidas1312/qubots/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/leonidas1312/qubots.svg)](https://github.com/leonidas1312/qubots/issues)
[![GitHub forks](https://img.shields.io/github/forks/leonidas1312/qubots.svg)](https://github.com/leonidas1312/qubots/network)

Qubots is a Python library that turns optimization problems and optimization algorithms (optimizers) into shareable, modular “qubots”. The github organization, called Rastion (https://rastion.com), currently serves as a central repository system. 

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Technical Overview](#technical-overview)
  - [Base Classes](#base-classes)
  - [Formulations](#formulations)
  - [Dynamic Qubot Loading: AutoProblem & AutoOptimizer](#dynamic-qubot-loading-autoproblem--autooptimizer)
- [Community & Qubot Cards](#community--qubot-cards)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

Qubots is available on PyPI. To install, simply run:

```bash
pip install qubots
```

## Getting Started

Here’s a brief example showing how to load a problem and a optimizer from the Rastion hub, then run the optimization:

```python
from qubots.auto_problem import AutoProblem
from qubots.auto_optimizer import AutoOptimizer

# TSP from the Rastion GitHub repository.
problem = AutoProblem.from_repo("Rastion/traveling_salesman_problem")

# Optimizer that uses ortools to solve the TSP
ortools_optimizer = AutoOptimizer.from_repo("Rastion/ortools_tsp_solver")

# Run the optimization and print results.
best_solution, best_cost = optimizer.optimize(problem)
print("Best Solution:", best_solution)
print("Best Cost:", best_cost)
```

## Technical Overview

### Base Classes

At the core of qubots are two abstract base classes:

- **BaseProblem**
  - Defines the interface for any optimization problem. Every problem qubot must implement:
    - `evaluate_solution(solution) -> float`
      - Computes the objective value (or cost) for a given candidate solution.
    - `random_solution() (optional)`
      - Generates a random feasible solution for the problem. Having this function serves as a starting point for optimizers when initial solution is not given. 

- **BaseOptimizer**
  - Provides the interface for optimization algorithms. Every optimizer qubot must implement:
    - `optimize(problem, initial_solution=None, **kwargs) -> (solution, cost)`
      - Runs the optimization on a given problem, optionally starting from an initial solution.

These interfaces ensure that every qubot—whether problem or optimizer—can be seamlessly interchanged and composed. Splitting problem and optimizer information like this helps us build shareable specialized optimizers for **BaseProblem**. That is, we achieve a many-to-one connection of optimizers to problems.

### Formulations

Formulations are what makes problems versatile for optimizers. Different problems can have many formulations or mathematical representations. Since formulations are variants of problem, thus problems themselves, it makes sense to define them as **BaseProblem**.

A formulation should contain the logic of transforming **BaseProblem** to the formulation we want **and** the logic of decoding a solution from the formulation problem space back to **BaseProblem**. This ensures that for all the optimizers that uses different formulations of the problem, we have a way to compare them in the original **BaseProblem**. `\qubots\formulations` holds some formulations we tried on the TSP.

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

This design allows developers to share their work as self-contained GitHub repos that anyone can load, test, and incorporate into larger workflows. **Remote execution of python code files, including installing packages via requirements.txt, is not a good practice**. For this reason it is suggested to use Rastion & Qubots in a secure environment using `python -m venv` or `conda create --name my_rastion_env python=3.9`.

## Community & Qubot Cards 

Each qubot problem and optimizer is served with an additional configuration file we call "qubot card". This file helps the **AutoProblem** and **AutoOptimizer** correctly load qubots by telling them which file contains the base class of the qubot. The qubot card is also designed to keep metadata for helping the **community** better share their work and better organize it on the website where we show all the repositories. 

## Examples

Please visit https://rastion.com/docs
## Contributing

Currently under developement, please sent me an email at jonhkarystos@gmail.com .

## License

This project is licensed under the [Apache License 2.0](./LICENSE).

By leveraging the flexible design of qubots and the collaborative power of Rastion, you can rapidly prototype, share, and improve optimization solutions—be it for classical problems, quantum algorithms, or hybrid systems.

