# Modern Framework Integration Batch

This artifact records smoke benchmarks for the newer integration examples.

## Continuous LP

Problem: `integrations/continuous_blending_problem`

Optimizers:

- `integrations/cvxpy_lp_optimizer`
- `integrations/pyomo_milp_optimizer`

Result:

| optimizer | objective | runtime seconds |
|---|---:|---:|
| cvxpy_lp | 13.000000 | 0.019081 |
| pyomo_milp | 13.000000 | 0.007219 |

Both solve the blending LP at the known optimum `x=2`, `y=3`, objective `13`.

## Graph / QUBO

Problem: `docs/pilots/artifacts/qoblib-karate-maxcut/dataset.yaml`

Optimizers:

- `integrations/jax_maxcut_optimizer`
- `integrations/dwave_neal_maxcut_optimizer`

Result:

| optimizer | cut weight | runtime seconds |
|---|---:|---:|
| jax_maxcut | 54.000000 | 0.685049 |
| dwave_neal_maxcut | 61.000000 | 0.044110 |

JAX uses JIT-compiled cut evaluation. D-Wave neal builds a `dimod` binary
quadratic model and samples it locally, without quantum hardware access.

## Black-Box Search

Problem: `integrations/datasets/one_max_8.yaml`

Optimizer: `integrations/optuna_binary_optimizer`

Result:

| optimizer | best value | runtime seconds |
|---|---:|---:|
| optuna_binary | -7.000000 | 1.194892 |

The OneMax objective is negative count of selected bits, so lower is better.

## Qiskit QAOA / VQE

Problem: `integrations/datasets/tiny_maxcut.yaml`

Optimizers:

- `integrations/qiskit_qaoa_maxcut_optimizer`
- `integrations/qiskit_vqe_maxcut_optimizer`

Result:

| optimizer | cut weight | runtime seconds |
|---|---:|---:|
| qiskit_qaoa_maxcut | 7.000000 | 0.087681 |
| qiskit_vqe_maxcut | 7.000000 | 0.078747 |

Both demos use Qiskit `Statevector` locally on a 4-qubit MaxCut fixture. They
do not require Aer, cloud execution, or quantum hardware. The components reject
graphs above their configured `max_qubits` limit.
