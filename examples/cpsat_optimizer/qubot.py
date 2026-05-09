"""OR-Tools CP-SAT optimizer adapter.

Solves any problem implementing ``as_milp() -> MILPModel`` using Google's
CP-SAT solver via the ``ortools`` Python bindings. CP-SAT excels at
combinatorial / constraint-satisfaction problems (assignment, packing,
scheduling, etc.) and often beats MILP solvers on integer-only models.

Constraints (v1):
    - All variables must be marked integer (``integrality[i] = True``).
    - All coefficients (objective + constraints) must be integer-valued.
      Non-integer coefficients are rejected with a clear error rather than
      silently scaled.

Install with:

    pip install qubots[cpsat]

or directly:

    pip install ortools
"""

from __future__ import annotations

import math
import time
from typing import Any

from qubots.core.milp import MILPModel, SupportsMILP
from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


def _import_cp_model() -> Any:
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise ImportError(
            "CPSATOptimizer requires the 'ortools' package. "
            "Install with: pip install qubots[cpsat]"
        ) from exc
    return cp_model


def _coerce_int(value: float, *, label: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric, got {type(value).__name__}")
    if isinstance(value, int):
        return value
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value}")
    rounded = round(value)
    if abs(value - rounded) > 1e-9:
        raise ValueError(
            f"{label} = {value} is not integer-valued. CP-SAT requires integer "
            "coefficients; use the HiGHS optimizer for fractional models."
        )
    return int(rounded)


def _bound_to_cpsat(value: float, *, lower: bool, label: str) -> int:
    if not math.isfinite(value):
        # CP-SAT accepts very large bounds; clamp to its INT_MAX equivalent.
        cp_model = _import_cp_model()
        return -cp_model.INT32_MAX if lower else cp_model.INT32_MAX  # type: ignore[attr-defined]
    return _coerce_int(value, label=label)


class CPSATOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.time_limit_seconds: float | None = None
        self.num_workers: int | None = None
        self.log_to_console: bool = False
        self.random_seed: int | None = None

    def _build_model(self, milp: MILPModel) -> tuple[Any, list[Any], Any]:
        cp_model = _import_cp_model()

        if not all(milp.integrality):
            non_int = [
                milp.var_names[i]
                for i, flag in enumerate(milp.integrality)
                if not flag
            ]
            raise ValueError(
                "CP-SAT requires every variable to be integer; the following "
                f"are continuous: {non_int}. Use the HiGHS optimizer instead."
            )

        model = cp_model.CpModel()

        x = []
        for i in range(milp.n_vars):
            lb = _bound_to_cpsat(milp.lb[i], lower=True, label=f"lb[{milp.var_names[i]}]")
            ub = _bound_to_cpsat(milp.ub[i], lower=False, label=f"ub[{milp.var_names[i]}]")
            if lb == 0 and ub == 1:
                x.append(model.NewBoolVar(milp.var_names[i]))
            else:
                x.append(model.NewIntVar(lb, ub, milp.var_names[i]))

        for r, (row, b) in enumerate(zip(milp.A_ub, milp.b_ub)):
            coeffs = [_coerce_int(a, label=f"A_ub[{r}][{j}]") for j, a in enumerate(row)]
            rhs = _coerce_int(b, label=f"b_ub[{r}]")
            model.Add(sum(c * v for c, v in zip(coeffs, x)) <= rhs)

        for r, (row, b) in enumerate(zip(milp.A_eq, milp.b_eq)):
            coeffs = [_coerce_int(a, label=f"A_eq[{r}][{j}]") for j, a in enumerate(row)]
            rhs = _coerce_int(b, label=f"b_eq[{r}]")
            model.Add(sum(c * v for c, v in zip(coeffs, x)) == rhs)

        obj_coeffs = [_coerce_int(a, label=f"c[{j}]") for j, a in enumerate(milp.c)]
        objective_expr = sum(c * v for c, v in zip(obj_coeffs, x))
        if milp.sense == "max":
            model.Maximize(objective_expr)
        else:
            model.Minimize(objective_expr)

        return model, x, cp_model

    @staticmethod
    def _status_label(cp_model: Any, status: int) -> str:
        return {
            cp_model.OPTIMAL: "optimal",
            cp_model.FEASIBLE: "feasible",
            cp_model.INFEASIBLE: "infeasible",
            cp_model.MODEL_INVALID: "model_invalid",
            cp_model.UNKNOWN: "unknown",
        }.get(status, "unknown")

    def optimize(self, problem: Any) -> Result:
        if isinstance(problem, MILPModel):
            milp = problem
        elif isinstance(problem, SupportsMILP):
            milp = problem.as_milp()
        else:
            raise TypeError(
                "CPSATOptimizer requires a problem implementing 'as_milp() -> MILPModel' "
                f"or a MILPModel directly; got {type(problem).__name__}"
            )

        start = time.perf_counter()

        try:
            model, x, cp_model = self._build_model(milp)
        except (ImportError, ValueError, TypeError) as exc:
            return Result(
                best_value=float("inf"),
                best_solution=None,
                runtime_seconds=time.perf_counter() - start,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )

        solver = cp_model.CpSolver()
        if self.time_limit_seconds is not None:
            solver.parameters.max_time_in_seconds = float(self.time_limit_seconds)
        if self.num_workers is not None:
            solver.parameters.num_search_workers = int(self.num_workers)
        if self.random_seed is not None:
            solver.parameters.random_seed = int(self.random_seed)
        solver.parameters.log_search_progress = bool(self.log_to_console)

        status = solver.Solve(model)
        runtime = time.perf_counter() - start
        label = self._status_label(cp_model, status)

        feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        if feasible:
            solution = [float(solver.Value(v)) for v in x]
            objective = float(solver.ObjectiveValue())
            best_value = objective if milp.sense == "min" else -objective
            result_status = "ok" if status == cp_model.OPTIMAL else "feasible"
        else:
            solution = None
            objective = float("nan")
            best_value = float("inf")
            result_status = label

        metadata: dict[str, Any] = {
            "solver": "cpsat",
            "model_status": label,
            "objective": objective,
            "sense": milp.sense,
            "n_vars": milp.n_vars,
            "n_constraints": milp.n_constraints,
            "num_branches": int(solver.NumBranches()),
            "num_conflicts": int(solver.NumConflicts()),
            "wall_time_seconds": float(solver.WallTime()),
        }

        return Result(
            best_value=best_value,
            best_solution=solution,
            runtime_seconds=float(runtime),
            status=result_status,
            metadata=metadata,
        )
