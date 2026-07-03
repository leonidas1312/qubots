# QOBLIB Karate Graph MaxCut Pilot

## Decision

Partition a real benchmark graph to maximize cut weight, using a qubots-imported
edge-list problem and several compatible optimizers.

Source data: QOBLIB maximum independent set `karate.lp.xz`. The pilot input
`examples/pilots/qoblib_karate_edges.gph` is a derived edge list from the LP
constraints. QOBLIB declares data under CC BY 4.0.

## Baseline

A common graph-optimization baseline is random partition search. The benchmark
keeps that baseline and compares it with local search, simulated annealing, and
a `networkx`-backed optimizer.

## Qubots Workflow

```bash
qubots detect examples/pilots/qoblib_karate_edges.gph \
  --family maxcut \
  --json \
  --out docs/pilots/artifacts/qoblib-karate-maxcut/detection.json

qubots import examples/pilots/qoblib_karate_edges.gph \
  --family maxcut \
  --out docs/pilots/artifacts/qoblib-karate-maxcut/imported_problem \
  --name qoblib_karate_maxcut \
  --force

qubots publish-check \
  docs/pilots/artifacts/qoblib-karate-maxcut/imported_problem \
  --json \
  --out docs/pilots/artifacts/qoblib-karate-maxcut/publish-check.json

qubots benchmark \
  --dataset docs/pilots/artifacts/qoblib-karate-maxcut/dataset.yaml \
  --optimizers examples/random_search_optimizer \
  --optimizers examples/hill_climb_optimizer \
  --optimizers examples/simulated_annealing_optimizer \
  --optimizers integrations/networkx_maxcut_optimizer \
  --repeats 3 \
  --seed 7 \
  --out docs/pilots/artifacts/qoblib-karate-maxcut/benchmark.json
```

## Result

Qubots detects the file as `maxcut` with detector `edge_list` at confidence
0.86. The imported repo passes `publish-check` with no issues or warnings.

The benchmark minimizes negative cut weight, so more negative is better.

| optimizer | mean cut weight | mean runtime seconds |
|---|---:|---:|
| random_search | 49.666667 | 0.004024 |
| hill_climb_optimizer | 58.666667 | 0.008297 |
| simulated_annealing_optimizer | 59.666667 | 0.003137 |
| networkx_maxcut | 54.000000 | 0.001279 |

The `networkx_maxcut` optimizer is intentionally not special-cased by qubots:
its manifest declares `requirements: ["networkx>=3.0"]`, its code imports
`networkx`, and the benchmark runs it next to built-in black-box optimizers
through the same `AutoOptimizer.from_repo` path.

## Limitations

- This is a MaxCut demo derived from a QOBLIB independent-set graph. It is not a
  claim about solving the original independent-set model.
- The graph is small enough for local demonstrations; larger QOBLIB instances
  should be added as separate datasets with longer benchmark budgets.
