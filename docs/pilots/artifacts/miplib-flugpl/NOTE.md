# MIPLIB flugpl MILP Pilot

## Decision

Reproduce a public MILP benchmark instance through a reusable qubots problem
repo and a real structured solver.

Source data: MIPLIB `flugpl`, fetched by `qubots.contrib.miplib.fetch_miplib`
and imported from the cached MPS file.

## Baseline

The expected manual workflow is: download an MPS file, point a solver CLI at it,
and record the objective elsewhere. This pilot shows the same instance as a
portable qubots problem spec with detection, validation, source hash, and
benchmark artifacts.

## Qubots Workflow

```bash
python - <<'PY'
from qubots.contrib.miplib import fetch_miplib
print(fetch_miplib("flugpl"))
PY

qubots detect /home/ileo/.cache/qubots/miplib/flugpl.mps \
  --family milp \
  --json \
  --out docs/pilots/artifacts/miplib-flugpl/detection.json

qubots import /home/ileo/.cache/qubots/miplib/flugpl.mps \
  --family milp \
  --out docs/pilots/artifacts/miplib-flugpl/imported_problem \
  --name miplib_flugpl_milp \
  --force

qubots publish-check \
  docs/pilots/artifacts/miplib-flugpl/imported_problem \
  --json \
  --out docs/pilots/artifacts/miplib-flugpl/publish-check.json

qubots benchmark \
  --dataset docs/pilots/artifacts/miplib-flugpl/dataset.yaml \
  --optimizers examples/highs_optimizer \
  --repeats 1 \
  --seed 7 \
  --out docs/pilots/artifacts/miplib-flugpl/benchmark.json
```

## Result

Qubots detects the source file as `milp` with detector `mps_lp` at confidence
0.96. The imported repo passes `publish-check` with no issues or warnings after
source metadata is declared.

HiGHS solved the imported sparse MPS problem with status `ok`.

- Objective: 1201500.0
- Feasibility: true
- Runtime: 0.164513 seconds
- Data hash: `sha256:a1f0cb79a95639450dd984473a00efca24365457153d7fcb859c6581ddab4c2c`

## Limitations

- This pilot demonstrates import and benchmark reproducibility, not a new
  solver result.
- The source terms are delegated to MIPLIB; downstream publishers should review
  the current MIPLIB terms before redistributing benchmark files.
