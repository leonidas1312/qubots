# NetworkX MaxCut Optimizer

Uses `networkx` to build a graph from a qubots edge-list MaxCut problem, then
runs deterministic one-flip local improvement.

```bash
qubots benchmark \
  --dataset docs/pilots/artifacts/qoblib-karate-maxcut/dataset.yaml \
  --optimizers integrations/networkx_maxcut_optimizer
```
