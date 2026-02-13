# Qubots Core

Qubots is a minimal local-first optimization framework. It loads problems and optimizers from local component repositories and runs them together.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
qubots --help
```

```python
from qubots import AutoProblem, AutoOptimizer

problem = AutoProblem.from_repo("examples/one_max_problem")
optimizer = AutoOptimizer.from_repo("examples/random_search_optimizer")

result = optimizer.optimize(problem)
print(result.best_value, result.best_solution)
```

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

Enable remote loading with either:

- environment variable: `QUBOTS_ALLOW_REMOTE=1`
- CLI flag: `--allow-remote` on `run`, `finetune`, `benchmark`, `validate`

Examples:

```bash
qubots validate github:alice/my-solver@<sha>:component --allow-remote
```

```bash
qubots benchmark \
  --dataset examples/datasets/knapsack_small.yaml \
  --optimizers github:alice/my-solver@<sha>:component \
  --allow-remote
```
