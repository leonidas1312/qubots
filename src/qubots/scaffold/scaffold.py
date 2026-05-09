"""Generate fresh qubots component repositories.

Each scaffold produces a directory with:

- ``qubots.yaml`` (declared at ``qubots_schema_version: 1``)
- ``qubot.py`` (a runnable subclass of ``BaseProblem`` / ``BaseOptimizer``)
- ``README.md`` (component-author-facing notes)

The output passes ``validate_repo`` immediately so authors can run
``qubots validate <out>`` and ``qubots benchmark`` against it before
filling in real logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


PROBLEM_FLAVORS: tuple[str, ...] = ("blackbox", "milp", "dual")
OPTIMIZER_FLAVORS: tuple[str, ...] = ("blackbox", "milp")


@dataclass(frozen=True)
class ScaffoldResult:
    path: Path
    files: list[Path]


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_VALID_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _slugify(name: str) -> str:
    s = _SLUG_RE.sub("_", name.strip().lower()).strip("_")
    if not s:
        raise ValueError(f"Cannot derive a slug from name: {name!r}")
    if s[0].isdigit():
        s = f"q_{s}"
    return s


def _class_name(slug: str) -> str:
    parts = [p for p in slug.split("_") if p]
    return "".join(p[:1].upper() + p[1:] for p in parts)


def _ensure_empty_dir(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(
                f"Target path already exists: {path}. Pass force=True to overwrite."
            )
        if not path.is_dir():
            raise NotADirectoryError(f"Target path exists and is not a directory: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# ---------- problem scaffolds ------------------------------------------------


_PROBLEM_BLACKBOX_PY = '''"""{name}: blackbox problem.

Implements ``evaluate(solution) -> float`` (lower is better) and
``random_solution()`` so any blackbox optimizer (random search, hill
climb, simulated annealing) can attack instances.
"""

from __future__ import annotations

import random

from qubots.core.problem import BaseProblem


class {class_name}(BaseProblem):
    def __init__(self) -> None:
        super().__init__()
        self.n_bits = 20
        self.seed = 0

    def evaluate(self, solution: list[int]) -> float:
        # Lower is better. Replace with the real objective + any penalty.
        return -float(sum(solution))

    def random_solution(self) -> list[int]:
        rng = random.Random(int(self.seed))
        return [rng.randint(0, 1) for _ in range(int(self.n_bits))]
'''


_PROBLEM_MILP_PY = '''"""{name}: structured MILP problem.

Implements ``as_milp() -> MILPModel`` so structured solvers (HiGHS,
CP-SAT) can solve to optimality. ``BaseProblem`` provides default
``evaluate`` / ``random_solution`` derived from the MILP, so blackbox
optimizers also work without extra code.
"""

from __future__ import annotations

from qubots.core.milp import MILPModel
from qubots.core.problem import BaseProblem


class {class_name}(BaseProblem):
    def __init__(self) -> None:
        super().__init__()
        # Replace with real instance parameters.
        self.size = 3

    def as_milp(self) -> MILPModel:
        n = int(self.size)
        # Toy template: maximize sum(x_i) subject to sum(x_i) <= n - 1,
        # binary x_i. Replace with the real formulation.
        return MILPModel(
            sense="max",
            c=[1.0] * n,
            integrality=[True] * n,
            lb=[0.0] * n,
            ub=[1.0] * n,
            A_ub=[[1.0] * n],
            b_ub=[float(max(0, n - 1))],
            constraint_names=["budget"],
        )
'''


_PROBLEM_DUAL_PY = '''"""{name}: dual blackbox + structured problem.

Exposes both ``as_milp()`` (structured solvers) and a tight
problem-specific ``evaluate()`` / ``random_solution()`` (blackbox
optimizers). Run the same instance under any optimizer in the hub.
"""

from __future__ import annotations

import random

from qubots.core.milp import MILPModel
from qubots.core.problem import BaseProblem


class {class_name}(BaseProblem):
    def __init__(self) -> None:
        super().__init__()
        self.n_items = 10
        self.seed = 0
        self._values: list[int] = []
        self._weights: list[int] = []
        self._capacity = 0
        self._instance_key: tuple[int, int] | None = None

    def _ensure_instance(self) -> None:
        key = (int(self.n_items), int(self.seed))
        if self._instance_key == key:
            return
        rng = random.Random(self.seed)
        n = int(self.n_items)
        self._values = [rng.randint(1, 30) for _ in range(n)]
        self._weights = [rng.randint(1, 20) for _ in range(n)]
        self._capacity = max(1, sum(self._weights) // 2)
        self._instance_key = key

    def as_milp(self) -> MILPModel:
        self._ensure_instance()
        n = len(self._values)
        return MILPModel(
            sense="max",
            c=[float(v) for v in self._values],
            integrality=[True] * n,
            lb=[0.0] * n,
            ub=[1.0] * n,
            A_ub=[[float(w) for w in self._weights]],
            b_ub=[float(self._capacity)],
            constraint_names=["capacity"],
        )

    def evaluate(self, solution: list[int]) -> float:
        self._ensure_instance()
        n = len(self._values)
        x = list(solution[:n]) + [0] * max(0, n - len(solution))
        weight = sum(w for w, b in zip(self._weights, x) if b)
        value = sum(v for v, b in zip(self._values, x) if b)
        overweight = max(0, weight - self._capacity)
        return -float(value) + 100.0 * overweight

    def random_solution(self) -> list[int]:
        self._ensure_instance()
        rng = random.Random(int(self.seed) + 1)
        return [rng.randint(0, 1) for _ in range(len(self._values))]
'''


_PROBLEM_README = """# {name}

A qubots problem component (`type: problem`) generated by `qubots new`.

## Contract

This problem implements: **{flavor}**.

{flavor_notes}

## Run it

```bash
qubots validate {name}
```

```python
from qubots import AutoProblem, AutoOptimizer

problem = AutoProblem.from_repo("{name}")
optimizer = AutoOptimizer.from_repo("path/to/some_optimizer")
result = optimizer.optimize(problem)
print(result.best_value, result.best_solution)
```

## Editing checklist

- [ ] Replace the placeholder objective / constraints in `qubot.py`.
- [ ] Update `qubots.yaml` parameters with the real instance knobs and defaults.
- [ ] Add a small dataset YAML if you want others to benchmark against your problem.
"""


_PROBLEM_FLAVOR_NOTES = {
    "blackbox": (
        "It defines `evaluate(solution) -> float` (lower is better) and "
        "`random_solution()`. Any blackbox optimizer in the hub can solve it."
    ),
    "milp": (
        "It defines `as_milp() -> MILPModel`. Structured solvers (HiGHS, "
        "CP-SAT) consume the model directly; `BaseProblem` provides default "
        "blackbox shims so metaheuristics still work."
    ),
    "dual": (
        "It defines both `as_milp()` and a tight problem-specific "
        "`evaluate()` / `random_solution()` so the same instance runs "
        "fairly under any optimizer in the hub."
    ),
}


_PROBLEM_FLAVOR_PARAMETERS = {
    "blackbox": {"n_bits": 20, "seed": 0},
    "milp": {"size": 3},
    "dual": {"n_items": 10, "seed": 0},
}


_PROBLEM_FLAVOR_REQUIREMENTS: dict[str, list[str]] = {
    "blackbox": [],
    "milp": [],
    "dual": [],
}


# ---------- optimizer scaffolds ----------------------------------------------


_OPTIMIZER_BLACKBOX_PY = '''"""{name}: blackbox optimizer.

Calls ``problem.random_solution()`` + ``problem.evaluate()`` directly.
Works on any problem that implements the blackbox interface.
"""

from __future__ import annotations

import random
import time
from typing import Any

from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


class {class_name}(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.iterations = 100
        self.seed = 0

    def optimize(self, problem: Any) -> Result:
        start = time.perf_counter()
        rng = random.Random(int(self.seed))

        best_solution = problem.random_solution()
        best_value = float(problem.evaluate(best_solution))
        trace: list[float] = [best_value]

        for _ in range(int(self.iterations)):
            candidate = problem.random_solution()
            value = float(problem.evaluate(candidate))
            if value < best_value:
                best_value = value
                best_solution = candidate
            trace.append(best_value)

        # ``rng`` is reserved for stochastic neighborhoods; the template
        # is just a random restart loop.
        _ = rng
        return Result(
            best_value=float(best_value),
            best_solution=best_solution,
            runtime_seconds=time.perf_counter() - start,
            status="ok",
            trace=trace,
            metadata={{"iterations": int(self.iterations)}},
        )
'''


_OPTIMIZER_MILP_PY = '''"""{name}: structured MILP optimizer.

Consumes ``problem.as_milp() -> MILPModel`` and returns a Result with the
optimal (or best feasible) assignment. Replace the placeholder body with
a real solver call (HiGHS, CP-SAT, SCIP, Gurobi, ...).
"""

from __future__ import annotations

import time
from typing import Any

from qubots.core.milp import MILPModel, SupportsMILP
from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


class {class_name}(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.time_limit_seconds: float | None = None

    def optimize(self, problem: Any) -> Result:
        if isinstance(problem, MILPModel):
            milp = problem
        elif isinstance(problem, SupportsMILP):
            milp = problem.as_milp()
        else:
            raise TypeError(
                "{class_name} requires a problem implementing as_milp() -> MILPModel "
                f"or a MILPModel directly; got {{type(problem).__name__}}"
            )

        start = time.perf_counter()

        # ---- Replace this stub with a real solver call. ------------------
        # The trivial baseline below picks a solution that minimizes each
        # variable subject to its bounds (or a feasible default if bounds
        # are infinite). It is correct only for unconstrained models, so
        # the first thing you should do is wire in a real solver.
        x = []
        import math
        for i in range(milp.n_vars):
            lb = milp.lb[i] if math.isfinite(milp.lb[i]) else 0.0
            ub = milp.ub[i] if math.isfinite(milp.ub[i]) else 0.0
            x.append(float(lb if milp.sense == "min" else ub))
        # ------------------------------------------------------------------

        objective = milp.evaluate(x)
        best_value = objective if milp.sense == "min" else -objective
        runtime = time.perf_counter() - start

        return Result(
            best_value=float(best_value),
            best_solution=x,
            runtime_seconds=float(runtime),
            status="ok" if milp.is_feasible(x) else "infeasible_stub",
            metadata={{
                "solver": "stub",
                "objective": float(objective),
                "sense": milp.sense,
                "n_vars": milp.n_vars,
                "n_constraints": milp.n_constraints,
            }},
        )
'''


_OPTIMIZER_README = """# {name}

A qubots optimizer component (`type: optimizer`) generated by `qubots new`.

## Contract

This optimizer implements: **{flavor}**.

{flavor_notes}

## Run it

```bash
qubots validate {name}
```

```python
from qubots import AutoOptimizer, AutoProblem

problem = AutoProblem.from_repo("path/to/some_problem")
optimizer = AutoOptimizer.from_repo("{name}")
result = optimizer.optimize(problem)
print(result.best_value, result.best_solution)
```

## Editing checklist

- [ ] Replace the placeholder loop / solver call in `qubot.py`.
- [ ] Update `qubots.yaml` parameters and `tunable_parameters` for fine-tuning.
- [ ] Benchmark against the example optimizers on a shared dataset.
"""


_OPTIMIZER_FLAVOR_NOTES = {
    "blackbox": (
        "It calls `problem.random_solution()` + `problem.evaluate()` and "
        "works on any problem implementing the blackbox interface."
    ),
    "milp": (
        "It consumes `problem.as_milp()` (a `MILPModel`) and is intended "
        "for structured LP/MILP/CP-SAT-style backends."
    ),
}


_OPTIMIZER_FLAVOR_PARAMETERS = {
    "blackbox": {"iterations": 100, "seed": 0},
    "milp": {"time_limit_seconds": None},
}


_OPTIMIZER_FLAVOR_REQUIREMENTS: dict[str, list[str]] = {
    "blackbox": [],
    # MILP-flavor optimizers usually wrap a real solver; leave empty by
    # default so the author declares the right one (e.g. "highspy>=1.7"
    # or "ortools>=9.10") for their backend.
    "milp": [],
}


# ---------- manifest emit ----------------------------------------------------


def _emit_manifest(
    component_type: str,
    name: str,
    class_name: str,
    parameters: dict[str, object],
    requirements: list[str],
) -> str:
    lines = [
        "qubots_schema_version: 2",
        f"type: {component_type}",
        f"name: {name}",
        f"entrypoint: qubot.py:{class_name}",
    ]
    if requirements:
        lines.append("requirements:")
        for spec in requirements:
            lines.append(f"  - {_yaml_scalar(spec)}")
    else:
        # Empty list is meaningful: "I declare zero solver deps." Leaderboard
        # CI will install nothing extra for this submission.
        lines.append("requirements: []")
    if parameters:
        lines.append("parameters:")
        for key, default in parameters.items():
            lines.append(f"  {key}:")
            lines.append(f"    default: {_yaml_scalar(default)}")
    return "\n".join(lines) + "\n"


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return repr(str(value))


# ---------- public API -------------------------------------------------------


def scaffold_problem(
    name: str,
    out_dir: str | Path,
    *,
    flavor: str = "dual",
    force: bool = False,
) -> ScaffoldResult:
    """Create a fresh problem component repo at ``out_dir``."""
    if flavor not in PROBLEM_FLAVORS:
        raise ValueError(
            f"Unknown problem flavor {flavor!r}. Choose one of: "
            f"{', '.join(PROBLEM_FLAVORS)}"
        )

    slug = _slugify(name)
    if not _VALID_NAME_RE.match(slug):
        raise ValueError(
            f"Derived name {slug!r} is not a valid component name "
            "(must match [a-z][a-z0-9_]*)"
        )
    class_name = _class_name(slug)

    target = Path(out_dir).expanduser().resolve()
    _ensure_empty_dir(target, force=force)

    body_template = {
        "blackbox": _PROBLEM_BLACKBOX_PY,
        "milp": _PROBLEM_MILP_PY,
        "dual": _PROBLEM_DUAL_PY,
    }[flavor]
    body = body_template.format(name=slug, class_name=class_name)
    manifest = _emit_manifest(
        component_type="problem",
        name=slug,
        class_name=class_name,
        parameters=_PROBLEM_FLAVOR_PARAMETERS[flavor],
        requirements=_PROBLEM_FLAVOR_REQUIREMENTS[flavor],
    )
    readme = _PROBLEM_README.format(
        name=slug, flavor=flavor, flavor_notes=_PROBLEM_FLAVOR_NOTES[flavor]
    )

    files = [
        target / "qubots.yaml",
        target / "qubot.py",
        target / "README.md",
    ]
    _write(files[0], manifest)
    _write(files[1], body)
    _write(files[2], readme)

    return ScaffoldResult(path=target, files=files)


def scaffold_optimizer(
    name: str,
    out_dir: str | Path,
    *,
    flavor: str = "blackbox",
    force: bool = False,
) -> ScaffoldResult:
    """Create a fresh optimizer component repo at ``out_dir``."""
    if flavor not in OPTIMIZER_FLAVORS:
        raise ValueError(
            f"Unknown optimizer flavor {flavor!r}. Choose one of: "
            f"{', '.join(OPTIMIZER_FLAVORS)}"
        )

    slug = _slugify(name)
    if not _VALID_NAME_RE.match(slug):
        raise ValueError(
            f"Derived name {slug!r} is not a valid component name "
            "(must match [a-z][a-z0-9_]*)"
        )
    class_name = _class_name(slug)

    target = Path(out_dir).expanduser().resolve()
    _ensure_empty_dir(target, force=force)

    body_template = {
        "blackbox": _OPTIMIZER_BLACKBOX_PY,
        "milp": _OPTIMIZER_MILP_PY,
    }[flavor]
    body = body_template.format(name=slug, class_name=class_name)
    manifest = _emit_manifest(
        component_type="optimizer",
        name=slug,
        class_name=class_name,
        parameters=_OPTIMIZER_FLAVOR_PARAMETERS[flavor],
        requirements=_OPTIMIZER_FLAVOR_REQUIREMENTS[flavor],
    )
    readme = _OPTIMIZER_README.format(
        name=slug, flavor=flavor, flavor_notes=_OPTIMIZER_FLAVOR_NOTES[flavor]
    )

    files = [
        target / "qubots.yaml",
        target / "qubot.py",
        target / "README.md",
    ]
    _write(files[0], manifest)
    _write(files[1], body)
    _write(files[2], readme)

    return ScaffoldResult(path=target, files=files)
