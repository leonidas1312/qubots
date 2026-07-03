# Campaign Budget Pilot

## Decision

Choose marketing campaigns under a fixed budget capacity.

Source data: `examples/pilots/campaign_budget.csv`

The table has five campaigns with value, weight, and capacity columns. Weight
represents budget consumption. Capacity is 35.

## Baseline

A common spreadsheet baseline is greedy selection by value-per-weight ratio:

1. `email_offer` value 25, weight 5
2. `retargeting` value 36, weight 9
3. `search_brand` value 42, weight 12

This uses 26 budget units and produces value 103. The remaining candidates do
not fit after that greedy sequence.

## Qubots Workflow

```bash
qubots detect examples/pilots/campaign_budget.csv \
  --json \
  --out docs/pilots/artifacts/campaign-budget/detection.json

qubots import examples/pilots/campaign_budget.csv \
  --family knapsack \
  --out docs/pilots/artifacts/campaign-budget/imported_problem \
  --name campaign_budget_knapsack \
  --force

qubots publish-check \
  docs/pilots/artifacts/campaign-budget/imported_problem \
  --json \
  --out docs/pilots/artifacts/campaign-budget/publish-check.json

qubots benchmark \
  --dataset docs/pilots/artifacts/campaign-budget/dataset.yaml \
  --optimizers examples/random_search_optimizer \
  --optimizers examples/highs_optimizer \
  --repeats 3 \
  --seed 7 \
  --out docs/pilots/artifacts/campaign-budget/benchmark.json
```

## Result

Qubots detects the file as `knapsack` with detector `item_table` at confidence
0.91. The imported repo passes `publish-check` with no issues.

Both random search and HiGHS found the same feasible selection:

- `search_brand`
- `search_nonbrand`
- `email_offer`

Total value: 122

Total weight: 35

This improves the simple greedy baseline by 19 value points on the same budget.

## Limitations

- This pilot uses a small public fixture, not a private production marketing
  dataset.
- Value estimates are assumed to be already supplied by the user; qubots does
  not infer business value from raw campaign history.
- CP-SAT was not included in this artifact because the current Python 3.14
  environment has a native shared-library conflict between the installed
  `highspy` and `ortools` wheels when both are imported in one process.
