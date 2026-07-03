# Pandas + PuLP Knapsack Integration

This artifact demonstrates framework composition:

- `integrations/pandas_knapsack_problem` loads campaign-budget data with
  Pandas and exposes both black-box and MILP interfaces.
- `integrations/pulp_milp_optimizer` consumes `as_milp()` and solves the model
  through PuLP/CBC.

Run:

```bash
qubots benchmark \
  --problem integrations/pandas_knapsack_problem \
  --dataset integrations/datasets/pandas_knapsack.yaml \
  --optimizers integrations/pulp_milp_optimizer \
  --repeats 1 \
  --seed 7 \
  --out integrations/artifacts/pandas-pulp-knapsack/benchmark.json
```

Result:

- Status: `ok`
- Objective: 122
- Best value: -122
- Runtime: 0.012538 seconds

The same Pandas-backed problem can also run with `examples/highs_optimizer`,
`examples/cpsat_optimizer`, or black-box optimizers because it keeps the normal
qubots component contract.
