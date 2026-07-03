# Qubots Pilots

These pilots are small, inspectable use cases for showing what qubots v3 does:
detect common optimization data patterns, import them as reusable problem repos,
and benchmark compatible optimizers with stable artifacts.

## Pilot 1: Field Technician Assignment

Decision: assign technicians to jobs using a travel/fit cost matrix.

Data: `examples/pilots/technician_job_costs.csv`

Why it matters: many teams still make this decision in a spreadsheet. Qubots
turns the matrix into a reusable assignment problem and records a benchmark
artifact that can be compared across solvers.

```bash
qubots detect examples/pilots/technician_job_costs.csv
qubots import examples/pilots/technician_job_costs.csv \
  --family assignment \
  --out /tmp/qubots-pilot-technicians \
  --force
qubots publish-check /tmp/qubots-pilot-technicians
```

## Pilot 2: Marketing Budget Packing

Decision: choose campaigns under a fixed budget.

Data: `examples/pilots/campaign_budget.csv`

Why it matters: the input is a common business table with value, weight/cost,
and capacity. Qubots imports it as a knapsack problem that can run through
blackbox optimizers or structured MILP solvers.

```bash
qubots detect examples/pilots/campaign_budget.csv
qubots import examples/pilots/campaign_budget.csv \
  --family knapsack \
  --out /tmp/qubots-pilot-campaigns \
  --force
qubots publish-check /tmp/qubots-pilot-campaigns
```

## Pilot 3: Delivery Route Sanity Check

Decision: compare candidate routes across delivery stops.

Data: `examples/pilots/local_delivery.tsp`

Why it matters: TSPLIB-style files are a standard exchange format for routing
demos. Qubots imports the file as a runnable TSP problem without an ad hoc
adapter.

```bash
qubots detect examples/pilots/local_delivery.tsp
qubots import examples/pilots/local_delivery.tsp \
  --family tsp \
  --out /tmp/qubots-pilot-route \
  --force
qubots publish-check /tmp/qubots-pilot-route
```

## Pilot 4: Public MILP Benchmark Repro

Decision: solve a public MIPLIB MPS instance through an imported qubots problem
repo.

Data: MIPLIB `flugpl`, fetched by `qubots.contrib.miplib.fetch_miplib("flugpl")`
and stored in `docs/pilots/artifacts/miplib-flugpl/imported_problem/data/`.

Why it matters: this shows the structured path for real mathematical
programming data. Qubots detects MPS, imports a sparse MILP problem spec, and
routes it to the HiGHS optimizer.

Artifacts: `docs/pilots/artifacts/miplib-flugpl/`

```bash
qubots publish-check docs/pilots/artifacts/miplib-flugpl/imported_problem
qubots benchmark \
  --dataset docs/pilots/artifacts/miplib-flugpl/dataset.yaml \
  --optimizers examples/highs_optimizer
```

## Pilot 5: QOBLIB Karate Graph MaxCut

Decision: partition a benchmark graph to maximize cut weight.

Data: `examples/pilots/qoblib_karate_edges.gph`, derived from the QOBLIB
maximum independent set `karate.lp.xz` LP constraints.

Why it matters: this shows real graph data imported by the edge-list detector
and benchmarked with interchangeable black-box optimizers, including a
third-party `networkx` optimizer.

Artifacts: `docs/pilots/artifacts/qoblib-karate-maxcut/`

```bash
qubots publish-check docs/pilots/artifacts/qoblib-karate-maxcut/imported_problem
qubots benchmark \
  --dataset docs/pilots/artifacts/qoblib-karate-maxcut/dataset.yaml \
  --optimizers examples/random_search_optimizer \
  --optimizers examples/hill_climb_optimizer \
  --optimizers examples/simulated_annealing_optimizer \
  --optimizers integrations/networkx_maxcut_optimizer
```

## Framework Agnostic Components

`integrations/` demonstrates the framework boundary: qubots loads each
component from `qubots.yaml`, the component imports and uses its own framework
internally, and the benchmark still treats it as a normal problem or optimizer.

Current examples include HiGHS, OR-Tools CP-SAT, NetworkX, SciPy, PuLP, Pandas,
CVXPY, Pyomo, Optuna, JAX, D-Wave neal/dimod, and Qiskit QAOA/VQE demos. The
same pattern applies to CVXPY extensions, custom research code, or another
Python framework as long as the component exposes the qubots problem/optimizer
contract and declares its runtime requirements.

## Pilot Evidence Standard

For each pilot, keep these artifacts:

- source data file
- imported qubot repo or `problem_spec.yaml`
- `qubots publish-check --json` output
- benchmark JSON/markdown
- one-page note listing the baseline, objective, constraints, and limitations

Avoid claiming qubots infers arbitrary business logic. The credible v3 claim is
that it deterministically recognizes common optimization data patterns and
makes them runnable, shareable, and benchmarkable.
