# Qubots

> Reproducible benchmarks and pluggable components for optimization.
> Every problem and every solver is a small repo with a manifest — compose them, run them, leaderboard them.

[![CI](https://github.com/leonidas1312/qubots/actions/workflows/ci.yml/badge.svg)](https://github.com/leonidas1312/qubots/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/qubots.svg)](https://pypi.org/project/qubots/)
[![Python](https://img.shields.io/pypi/pyversions/qubots.svg)](https://pypi.org/project/qubots/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Install

```bash
pip install qubots[highs,cpsat,miplib]
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
- **Real solvers**: HiGHS (LP/MILP) and OR-Tools CP-SAT (combinatorial / scheduling) ship as first-class qubot components.
- **MIPLIB-ready**: read industry-standard `.mps` files; `fetch_miplib("flugpl")` downloads + caches benchmark instances.
- **Git-native hub primitive**: `AutoProblem.from_repo("github:owner/repo@sha:subdir")` — pin to a SHA, share with anyone.
- **Cross-solver leaderboards**: `qubots benchmark` runs any optimizer set against any dataset and emits a markdown table.

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

**Note:** the MPS reader currently produces a dense `MILPModel`, so very
large MIPLIB instances (millions of nonzeros) are not yet supported. Sparse
representation is on the roadmap.

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
- Optimizers:
  - `examples/hill_climb_optimizer`
  - `examples/simulated_annealing_optimizer`
- Header-format datasets:
  - `examples/datasets/knapsack_small.yaml`
  - `examples/datasets/maxcut_small.yaml`

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
