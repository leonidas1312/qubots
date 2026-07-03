# Qubots

> Reproducible benchmarks and pluggable components for optimization.
> Every problem and every solver is a small repo with a manifest — compose them, run them, leaderboard them.

[![CI](https://github.com/leonidas1312/qubots/actions/workflows/ci.yml/badge.svg)](https://github.com/leonidas1312/qubots/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/qubots.svg)](https://pypi.org/project/qubots/)
[![Python](https://img.shields.io/pypi/pyversions/qubots.svg)](https://pypi.org/project/qubots/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p>
  <img alt="HiGHS" src="https://img.shields.io/badge/HiGHS-LP%2FMILP-005F73?style=for-the-badge">
  <img alt="OR-Tools" src="https://img.shields.io/badge/OR--Tools-CP--SAT-4285F4?style=for-the-badge&logo=google&logoColor=white">
  <img alt="NetworkX" src="https://img.shields.io/badge/NetworkX-Graphs-2E7D32?style=for-the-badge&logo=python&logoColor=white">
  <img alt="SciPy" src="https://img.shields.io/badge/SciPy-Assignment-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white">
  <img alt="PuLP" src="https://img.shields.io/badge/PuLP-MILP-7A3E9D?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img alt="CVXPY" src="https://img.shields.io/badge/CVXPY-Convex%20LP-1F77B4?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Pyomo" src="https://img.shields.io/badge/Pyomo-Modeling-4B8BBE?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Optuna" src="https://img.shields.io/badge/Optuna-Search-00A98F?style=for-the-badge">
  <img alt="JAX" src="https://img.shields.io/badge/JAX-Accelerated-FF6F00?style=for-the-badge&logo=jax&logoColor=white">
  <img alt="D-Wave" src="https://img.shields.io/badge/D--Wave%20Ocean-QUBO-1B365D?style=for-the-badge">
  <img alt="Qiskit" src="https://img.shields.io/badge/Qiskit-QAOA%2FVQE-6929C4?style=for-the-badge&logo=qiskit&logoColor=white">
</p>

## Install

```bash
pip install qubots[highs]
# or
pip install qubots[cpsat]
# or, for the integration examples
pip install qubots[integrations]
```

Or just the core for now and add backends later:

```bash
pip install qubots
qubots --help
```

## Quickstart

```python
from qubots import AutoProblem, AutoOptimizer

problem = AutoProblem.from_repo("examples/one_max_problem")
optimizer = AutoOptimizer.from_repo("examples/random_search_optimizer")

result = optimizer.optimize(problem)
print(result.best_value, result.best_solution)
```

## Why qubots

- **Pluggable components**: every problem and every solver is a small repo with a `qubots.yaml` manifest — drop one in, run it, share it.
- **Autodetect + import**: detect common optimization data files and turn them into runnable problem repos with stable metadata.
- **Real solvers**: HiGHS (LP/MILP) and OR-Tools CP-SAT (combinatorial / scheduling) ship as first-class qubot components.
- **MIPLIB-ready**: read industry-standard `.mps` files with sparse MILP support; `fetch_miplib("flugpl")` downloads + caches benchmark instances.
- **Framework agnostic**: components can wrap normal Python libraries such as HiGHS, OR-Tools, NetworkX, SciPy, PuLP, Pandas, CVXPY, Pyomo, Optuna, JAX, D-Wave Ocean, Qiskit, or custom research code behind the same qubots contract.
- **Git-native hub primitive**: `AutoProblem.from_repo("github:owner/repo@sha:subdir")` — pin to a SHA, share with anyone.
- **Cross-solver leaderboards**: `qubots benchmark` runs any optimizer set against any dataset and emits a markdown table.

## Autodetect and import data

Qubots v3 can detect common optimization file/data patterns and create reusable
problem repos:

```bash
qubots detect examples/pilots/campaign_budget.csv
qubots import examples/pilots/campaign_budget.csv \
  --family knapsack \
  --out imported_campaign_budget
qubots validate imported_campaign_budget
qubots publish-check imported_campaign_budget
```

Supported deterministic detectors:

| Data shape | Imported family |
|---|---|
| MPS / LP file | MILP |
| TSPLIB `.tsp` file | TSP |
| edge list text/CSV | MaxCut |
| numeric CSV/JSON matrix | assignment |
| item table with value, weight, capacity | knapsack |

For in-process use:

```python
from qubots import AutoProblem

problem = AutoProblem.from_data("examples/pilots/campaign_budget.csv", family="knapsack")
print(problem.random_solution(), problem.as_milp().n_vars)
```

See `docs/pilots/README.md` for small pilot use cases that can be imported and
benchmarked locally.

## Framework Integrations

Qubots does not require problems or optimizers to use a specific modeling
framework. A component repo declares dependencies in `qubots.yaml`, imports
whatever Python library it needs, and exposes the standard qubots problem or
optimizer contract.

The `integrations/` folder contains self-contained examples:

| Framework | Component | What it shows |
|---|---|---|
| HiGHS | `examples/highs_optimizer` | exact sparse/dense MILP solving |
| OR-Tools CP-SAT | `examples/cpsat_optimizer` | integer-only MILP / CP-SAT solving |
| NetworkX | `integrations/networkx_maxcut_optimizer` | graph construction inside an optimizer |
| SciPy | `integrations/scipy_assignment_optimizer` | exact assignment via `linear_sum_assignment` |
| PuLP | `integrations/pulp_milp_optimizer` | generic MILP adapter through PuLP/CBC |
| Pandas | `integrations/pandas_knapsack_problem` | data-frame-backed problem component |
| CVXPY | `integrations/cvxpy_lp_optimizer` | continuous LP solve through CVXPY |
| Pyomo | `integrations/pyomo_milp_optimizer` | algebraic MILP modeling with `appsi_highs` |
| Optuna | `integrations/optuna_binary_optimizer` | black-box binary search via TPE |
| JAX | `integrations/jax_maxcut_optimizer` | JIT-compiled graph objective evaluation |
| D-Wave Ocean | `integrations/dwave_neal_maxcut_optimizer` | QUBO/BQM sampling with dimod + neal |
| Qiskit | `integrations/qiskit_qaoa_maxcut_optimizer` | small local QAOA-style MaxCut |
| Qiskit | `integrations/qiskit_vqe_maxcut_optimizer` | small local VQE-style MaxCut |

The NetworkX optimizer is a concrete example:

```yaml
type: optimizer
name: networkx_maxcut
entrypoint: qubot.py:NetworkXMaxCutOptimizer
requirements:
  - "networkx>=3.0"
capabilities:
  - blackbox
  - graph
```

The QOBLIB Karate pilot benchmarks that optimizer next to the generic
black-box optimizers:

```bash
qubots benchmark \
  --dataset docs/pilots/artifacts/qoblib-karate-maxcut/dataset.yaml \
  --optimizers examples/random_search_optimizer \
  --optimizers examples/hill_climb_optimizer \
  --optimizers examples/simulated_annealing_optimizer \
  --optimizers integrations/networkx_maxcut_optimizer
```

The Pandas and PuLP examples compose the same way:

```bash
qubots benchmark \
  --problem integrations/pandas_knapsack_problem \
  --dataset integrations/datasets/pandas_knapsack.yaml \
  --optimizers integrations/pulp_milp_optimizer
```

The newer integration examples can be run directly too:

```bash
qubots benchmark \
  --dataset integrations/datasets/continuous_blending.yaml \
  --optimizers integrations/cvxpy_lp_optimizer \
  --optimizers integrations/pyomo_milp_optimizer
```

The Qiskit demos run locally with statevector simulation on a tiny MaxCut graph:

```bash
qubots benchmark \
  --dataset integrations/datasets/tiny_maxcut.yaml \
  --optimizers integrations/qiskit_qaoa_maxcut_optimizer \
  --optimizers integrations/qiskit_vqe_maxcut_optimizer
```

## Author a new component (60 seconds)

```bash
qubots new problem   --name shift_scheduler         # flavor: dual (default)
qubots new problem   --name flow_lp     --flavor milp
qubots new problem   --name one_max     --flavor blackbox
qubots new optimizer --name my_solver                # flavor: blackbox (default)
qubots new optimizer --name my_milp_solver --flavor milp
```

Each command writes a self-contained component repo (`qubots.yaml`, `qubot.py`,
`README.md`) that passes validation immediately:

```bash
qubots validate shift_scheduler
# [OK] /path/to/shift_scheduler
```

Flavors:

| Kind | Flavor | What you implement | Use when |
|---|---|---|---|
| problem | `blackbox` | `evaluate()` + `random_solution()` | Metaheuristics-only path |
| problem | `milp` | `as_milp() -> MILPModel` | Structured LP/MILP/CP-SAT |
| problem | `dual` (default) | both interfaces | Same instance under any optimizer |
| optimizer | `blackbox` (default) | `optimize(problem)` calling `evaluate` | New metaheuristic |
| optimizer | `milp` | `optimize(problem)` consuming `as_milp()` | New structured solver |

## Fine-tune (MVP)

```bash
qubots finetune \
  --problem examples/one_max_problem \
  --optimizer examples/random_search_optimizer \
  --dataset examples/one_max_dataset/dataset.yaml \
  --budget 20 \
  --out trained/random-search-run
```

If your dataset uses header format with `problem`, you can omit `--problem`:

```bash
qubots finetune \
  --optimizer examples/random_search_optimizer \
  --dataset examples/one_max_dataset_header/dataset.yaml \
  --budget 20
```

Load a tuned optimizer from artifact:

```python
from qubots import AutoOptimizer

opt = AutoOptimizer.from_trained("trained/random-search-run/trained.json")
```

Pipeline API:

```python
from qubots import pipeline

run = pipeline(
    problem="examples/one_max_problem",
    optimizer="examples/random_search_optimizer",
    trained="trained/random-search-run/trained.json",
)
result = run(
    problem_params={"n_bits": 64},
    optimizer_params={"iterations": 500},
)
print(result.best_value)
```

## MIPLIB benchmarks (real OR instances)

Read any industry-standard MPS / LP file as a qubots problem, including
[MIPLIB](https://miplib.zib.de) benchmark instances:

```python
from qubots import AutoOptimizer
from qubots.contrib.miplib import fetch_miplib
from qubots.contrib.mps import MPSProblem

# Downloads + caches under ~/.cache/qubots/miplib (or $QUBOTS_MIPLIB_CACHE).
mps_path = fetch_miplib("flugpl")

problem = MPSProblem(mps_path=mps_path)
solver = AutoOptimizer.from_repo("examples/highs_optimizer")
solver.set_parameters(time_limit_seconds=30)
result = solver.optimize(problem)
print(result.status, result.metadata["objective"])
# ok 1201500.0   <- known MIPLIB optimum
```

Or via the benchmark CLI on a bundled tiny MPS dataset:

```bash
qubots benchmark \
  --dataset examples/datasets/mps_tiny.yaml \
  --optimizers examples/highs_optimizer \
  --optimizers examples/cpsat_optimizer
```

```
| optimizer | type | mean_best_value | mean_runtime_seconds | success_rate |
|---|---:|---:|---:|---:|
| highs | repo | -25.000000 | 0.001955 | 100.00% |
| cpsat | repo | -25.000000 | 0.340345 | 100.00% |
```

To benchmark on real MIPLIB instances, generate a dataset YAML from a list
of names:

```python
import yaml
from qubots.contrib.miplib import fetch_miplib

names = ["flugpl", "gen-ip002", "mas74"]
spec = {
    "problem": "examples/mps_problem",
    "instances": [{"mps_path": str(fetch_miplib(n))} for n in names],
}
with open("miplib_easy.yaml", "w") as f:
    yaml.safe_dump(spec, f)
```

**Note:** `read_mps()` preserves the original dense `MILPModel` behavior for
small examples. Use imported MPS problems or `read_mps_sparse()` for sparse
rows on larger instances.

## Benchmark

```bash
qubots benchmark \
  --dataset examples/one_max_dataset/dataset.yaml \
  --problem examples/one_max_problem \
  --optimizers examples/random_search_optimizer \
  --repeats 1 \
  --out reports/benchmark.json
```

Example output:

```markdown
| optimizer | type | mean_best_value | mean_runtime_seconds | success_rate |
|---|---:|---:|---:|---:|
| /path/to/examples/random_search_optimizer | repo | -20.000000 | 0.001200 | 100.00% |
```

## Export Trained

```bash
qubots export-trained \
  --trained trained/random-search-run/trained.json \
  --out exported/random-search-optimizer \
  --name random-search-trained
```

## Structured Solving (HiGHS, CP-SAT)

Problems can optionally expose structure via `as_milp() -> MILPModel`. Structured
solvers consume that structure directly and solve to optimality, while
metaheuristic optimizers continue to work on the same problem object via
`evaluate()`.

Two solvers ship in `examples/`:

| Solver | Backend | Best for | Extra |
|---|---|---|---|
| `highs_optimizer` | [HiGHS](https://highs.dev) | LP, MILP (mixed integer + continuous) | `pip install qubots[highs]` |
| `cpsat_optimizer` | OR-Tools CP-SAT | Integer-only combinatorial / scheduling / packing | `pip install qubots[cpsat]` |

Native solver wheels can conflict when HiGHS and OR-Tools are imported in the
same Python process on some platforms. If you need both, use separate virtual
environments or separate CLI runs until the upstream wheel conflict is resolved.

```python
from qubots import AutoOptimizer, AutoProblem

problem = AutoProblem.from_repo("examples/knapsack_milp_problem")
problem.set_parameters(n_items=20, capacity_ratio=0.4, seed=7)

solver = AutoOptimizer.from_repo("examples/highs_optimizer")  # or cpsat_optimizer
result = solver.optimize(problem)

print(result.status, result.best_value, result.best_solution)
```

Defining your own MILP problem:

```python
from qubots import BaseProblem, MILPModel

class AssignmentProblem(BaseProblem):
    def as_milp(self) -> MILPModel:
        return MILPModel(
            sense="min",
            c=[...],            # objective coefficients
            integrality=[...],  # True for integer/binary
            lb=[...], ub=[...], # variable bounds
            A_ub=[...], b_ub=[...],  # A_ub @ x <= b_ub
            A_eq=[...], b_eq=[...],  # A_eq @ x == b_eq
        )
```

## More Examples

- Problems:
  - `examples/knapsack_problem`
  - `examples/maxcut_problem`
  - `integrations/pandas_knapsack_problem`
  - `integrations/continuous_blending_problem`
- Optimizers:
  - `examples/hill_climb_optimizer`
  - `examples/simulated_annealing_optimizer`
  - `integrations/networkx_maxcut_optimizer`
  - `integrations/scipy_assignment_optimizer`
  - `integrations/pulp_milp_optimizer`
  - `integrations/cvxpy_lp_optimizer`
  - `integrations/pyomo_milp_optimizer`
  - `integrations/optuna_binary_optimizer`
  - `integrations/jax_maxcut_optimizer`
  - `integrations/dwave_neal_maxcut_optimizer`
  - `integrations/qiskit_qaoa_maxcut_optimizer`
  - `integrations/qiskit_vqe_maxcut_optimizer`
- Header-format datasets:
  - `examples/datasets/knapsack_small.yaml`
  - `examples/datasets/maxcut_small.yaml`
  - `integrations/datasets/pandas_knapsack.yaml`
  - `integrations/datasets/continuous_blending.yaml`
  - `integrations/datasets/tiny_maxcut.yaml`

## Remote Repos (GitHub)

Qubots can load component repos from GitHub specs:

- `github:<owner>/<repo>@<ref>`
- `github:<owner>/<repo>@<ref>:<subdir>`

**Loading a remote spec executes arbitrary third-party Python on your machine.**
Treat it the same way you'd treat `pip install` or `curl | bash`. Two opt-in
mechanisms exist:

- `QUBOTS_TRUST_REMOTE_CODE=1` env var, or `--trust-remote-code` CLI flag (preferred).
- `QUBOTS_ALLOW_REMOTE=1` / `--allow-remote` (legacy alias, same effect).

Each remote load emits a `RuntimeWarning` showing the resolved
`owner/repo:subdir@ref`. Pin to a full commit SHA (40 hex chars) when you can —
branch and tag refs can be silently moved by the upstream.

```bash
qubots validate github:alice/my-solver@<sha>:component --trust-remote-code
```

```bash
qubots benchmark \
  --dataset examples/datasets/knapsack_small.yaml \
  --optimizers github:alice/my-solver@<sha>:component \
  --trust-remote-code
```

## Manifest schema

Every `qubots.yaml` is parsed against a versioned schema. The current version
is **1**; manifests written for any other version are refused with a clear
"upgrade qubots" error. New repos should declare the version explicitly:

```yaml
qubots_schema_version: 1
type: problem
name: my_problem
entrypoint: qubot.py:MyProblem
```

Manifests without `qubots_schema_version` are treated as v1 for backward
compatibility.

Path-traversal protection: entrypoint module paths are restricted to
relative paths inside the repo. `../escape.py:Foo`,
`/etc/passwd.py:Foo`, and similar are rejected at load time.
