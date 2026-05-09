"""HiGHS MILP/LP optimizer adapter.

Solves any problem implementing ``as_milp() -> MILPModel`` using the open-source
HiGHS solver via the ``highspy`` Python bindings.

Install with:

    pip install qubots[highs]

or directly:

    pip install highspy
"""

from __future__ import annotations

import math
import time
from typing import Any

from qubots.core.milp import MILPModel, SupportsMILP
from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


def _import_highspy() -> Any:
    try:
        import highspy
    except ImportError as exc:
        raise ImportError(
            "HiGHSOptimizer requires the 'highspy' package. "
            "Install with: pip install qubots[highs]"
        ) from exc
    return highspy


class HiGHSOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.time_limit_seconds: float | None = None
        self.mip_rel_gap: float | None = None
        self.threads: int | None = None
        self.log_to_console: bool = False

    def _build_model(self, milp: MILPModel) -> Any:
        highspy = _import_highspy()
        h = highspy.Highs()
        if not self.log_to_console:
            h.silent()

        if self.time_limit_seconds is not None:
            h.setOptionValue("time_limit", float(self.time_limit_seconds))
        if self.mip_rel_gap is not None:
            h.setOptionValue("mip_rel_gap", float(self.mip_rel_gap))
        if self.threads is not None:
            h.setOptionValue("threads", int(self.threads))

        sense = (
            highspy.ObjSense.kMinimize
            if milp.sense == "min"
            else highspy.ObjSense.kMaximize
        )
        h.changeObjectiveSense(sense)

        n = milp.n_vars
        for j in range(n):
            lb = milp.lb[j] if math.isfinite(milp.lb[j]) else -highspy.kHighsInf
            ub = milp.ub[j] if math.isfinite(milp.ub[j]) else highspy.kHighsInf
            h.addCol(float(milp.c[j]), float(lb), float(ub), 0, [], [])

        for j, is_int in enumerate(milp.integrality):
            if is_int:
                h.changeColIntegrality(j, highspy.HighsVarType.kInteger)

        for row, b in zip(milp.A_ub, milp.b_ub):
            indices = [i for i, a in enumerate(row) if a != 0.0]
            values = [float(row[i]) for i in indices]
            h.addRow(-highspy.kHighsInf, float(b), len(indices), indices, values)

        for row, b in zip(milp.A_eq, milp.b_eq):
            indices = [i for i, a in enumerate(row) if a != 0.0]
            values = [float(row[i]) for i in indices]
            h.addRow(float(b), float(b), len(indices), indices, values)

        return h, highspy

    @staticmethod
    def _model_status_to_result_status(highspy: Any, model_status: Any) -> str:
        S = highspy.HighsModelStatus
        mapping = [
            ("kOptimal", "optimal"),
            ("kInfeasible", "infeasible"),
            ("kUnbounded", "unbounded"),
            ("kUnboundedOrInfeasible", "unbounded_or_infeasible"),
            ("kTimeLimit", "time_limit"),
            ("kIterationLimit", "iteration_limit"),
            ("kSolutionLimit", "solution_limit"),
            ("kModelEmpty", "empty_model"),
        ]
        for attr, label in mapping:
            value = getattr(S, attr, None)
            if value is not None and value == model_status:
                return label
        return "unknown"

    def optimize(self, problem: Any) -> Result:
        if isinstance(problem, MILPModel):
            milp = problem
        elif isinstance(problem, SupportsMILP):
            milp = problem.as_milp()
        else:
            raise TypeError(
                "HiGHSOptimizer requires a problem implementing 'as_milp() -> MILPModel' "
                f"or a MILPModel directly; got {type(problem).__name__}"
            )

        start = time.perf_counter()

        try:
            h, highspy = self._build_model(milp)
            run_status = h.run()
            runtime = time.perf_counter() - start
        except ImportError:
            raise
        except Exception as exc:  # noqa: BLE001 - solver errors are surfaced
            return Result(
                best_value=float("inf"),
                best_solution=None,
                runtime_seconds=time.perf_counter() - start,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )

        model_status = h.getModelStatus()
        status_label = self._model_status_to_result_status(highspy, model_status)

        info = h.getInfo()
        solution = h.getSolution()
        x = list(solution.col_value) if solution.col_value is not None else []
        x = [float(v) for v in x]

        for j, is_int in enumerate(milp.integrality):
            if is_int and j < len(x):
                x[j] = float(round(x[j]))

        objective = float(h.getObjectiveValue()) if x else float("inf")
        # ``best_value`` is, by qubots convention, what the solver minimizes
        # internally; for ``max`` problems we negate so cross-optimizer
        # comparisons in benchmarks remain consistent.
        best_value = objective if milp.sense == "min" else -objective

        metadata: dict[str, Any] = {
            "solver": "highs",
            "model_status": str(model_status),
            "objective": objective,
            "sense": milp.sense,
            "n_vars": milp.n_vars,
            "n_constraints": milp.n_constraints,
        }
        for attr in (
            "mip_gap",
            "mip_node_count",
            "simplex_iteration_count",
            "ipm_iteration_count",
            "primal_solution_status",
            "dual_solution_status",
            "objective_function_value",
        ):
            value = getattr(info, attr, None)
            if value is not None:
                metadata[attr] = value

        return Result(
            best_value=best_value,
            best_solution=x if x else None,
            runtime_seconds=float(runtime),
            status="ok" if status_label == "optimal" else status_label,
            metadata=metadata,
        )
