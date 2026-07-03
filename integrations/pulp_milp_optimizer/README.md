# PuLP MILP Optimizer

Uses `pulp` with CBC to solve any qubots problem exposing `as_milp()`.

```bash
qubots benchmark \
  --problem integrations/pandas_knapsack_problem \
  --dataset integrations/datasets/pandas_knapsack.yaml \
  --optimizers integrations/pulp_milp_optimizer
```
