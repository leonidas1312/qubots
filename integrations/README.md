# Qubots Integration Examples

These are self-contained qubot components that wrap common Python optimization
and data frameworks. They are intentionally kept outside `examples/` so they
can grow into a catalog of integration recipes.

## Integration Matrix

| Framework | Component | Role | Families |
|---|---|---|---|
| HiGHS | `../examples/highs_optimizer` | exact LP/MILP solver | MILP, assignment, knapsack |
| OR-Tools CP-SAT | `../examples/cpsat_optimizer` | exact integer solver | integer-only MILP |
| NetworkX | `networkx_maxcut_optimizer` | graph construction + MaxCut heuristic | graph, MaxCut |
| SciPy | `scipy_assignment_optimizer` | exact linear assignment solver | assignment |
| PuLP | `pulp_milp_optimizer` | generic MILP modeling/solver adapter | dense/sparse MILP |
| Pandas | `pandas_knapsack_problem` | data-frame-backed problem component | knapsack |
| CVXPY | `cvxpy_lp_optimizer` | convex continuous LP adapter | continuous LP |
| Pyomo | `pyomo_milp_optimizer` | algebraic MILP modeling adapter | dense/sparse MILP |
| Optuna | `optuna_binary_optimizer` | black-box binary TPE search | binary blackbox |
| JAX | `jax_maxcut_optimizer` | JIT-compiled graph objective evaluation | graph, MaxCut |
| D-Wave Ocean | `dwave_neal_maxcut_optimizer` | BQM/QUBO sampling with dimod + neal | graph, MaxCut |
| Qiskit | `qiskit_qaoa_maxcut_optimizer` | small local QAOA-style statevector demo | graph, MaxCut |
| Qiskit | `qiskit_vqe_maxcut_optimizer` | small local VQE-style statevector demo | graph, MaxCut |

## Smoke Commands

```bash
qubots validate integrations/networkx_maxcut_optimizer
qubots validate integrations/scipy_assignment_optimizer
qubots validate integrations/pulp_milp_optimizer
qubots validate integrations/pandas_knapsack_problem
qubots validate integrations/cvxpy_lp_optimizer
qubots validate integrations/continuous_blending_problem
qubots validate integrations/pyomo_milp_optimizer
qubots validate integrations/optuna_binary_optimizer
qubots validate integrations/jax_maxcut_optimizer
qubots validate integrations/dwave_neal_maxcut_optimizer
qubots validate integrations/qiskit_qaoa_maxcut_optimizer
qubots validate integrations/qiskit_vqe_maxcut_optimizer
```

Run the integration tests:

```bash
python -m pytest -q tests/test_integrations.py
```

Benchmark the Pandas-backed problem with the PuLP adapter:

```bash
qubots benchmark \
  --problem integrations/pandas_knapsack_problem \
  --dataset integrations/datasets/pandas_knapsack.yaml \
  --optimizers integrations/pulp_milp_optimizer
```

Benchmark CVXPY and Pyomo on a continuous LP:

```bash
qubots benchmark \
  --dataset integrations/datasets/continuous_blending.yaml \
  --optimizers integrations/cvxpy_lp_optimizer \
  --optimizers integrations/pyomo_milp_optimizer
```

Benchmark the small Qiskit QAOA/VQE demos:

```bash
qubots benchmark \
  --dataset integrations/datasets/tiny_maxcut.yaml \
  --optimizers integrations/qiskit_qaoa_maxcut_optimizer \
  --optimizers integrations/qiskit_vqe_maxcut_optimizer
```

Stored evidence for that composition lives in
`integrations/artifacts/pandas-pulp-knapsack/`.

Stored evidence for the newer CVXPY, Pyomo, Optuna, JAX, D-Wave neal, and
Qiskit examples lives in `integrations/artifacts/modern-frameworks/`.

The point is not that qubots replaces these frameworks. Qubots gives them a
shared packaging, validation, benchmark, and sharing contract.
