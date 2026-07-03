"""PuLP MILP optimizer integration."""

from __future__ import annotations

import math
import time
from typing import Any

import pulp

from qubots.core.milp import MILPModel, SparseMILPModel, SupportsMILP
from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


class PuLPMILPOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.capabilities = ["milp_dense", "milp_sparse", "integer_only", "continuous"]
        self.time_limit_seconds: float | None = None
        self.log_to_console = False

    @staticmethod
    def _row_entries(
        milp: MILPModel | SparseMILPModel, row: Any
    ) -> list[tuple[int, float]]:
        if isinstance(milp, SparseMILPModel):
            return [(int(index), float(value)) for index, value in row]
        return [
            (index, float(value))
            for index, value in enumerate(row)
            if float(value) != 0.0
        ]

    @staticmethod
    def _bound(value: float) -> float | None:
        return float(value) if math.isfinite(value) else None

    def _milp_from_problem(self, problem: Any) -> MILPModel | SparseMILPModel:
        if isinstance(problem, (MILPModel, SparseMILPModel)):
            return problem
        if isinstance(problem, SupportsMILP):
            milp = problem.as_milp()
            if isinstance(milp, (MILPModel, SparseMILPModel)):
                return milp
        raise TypeError(
            "PuLPMILPOptimizer requires a problem implementing as_milp(), "
            "or a MILPModel/SparseMILPModel directly."
        )

    def optimize(self, problem: Any) -> Result:
        start = time.perf_counter()
        milp = self._milp_from_problem(problem)

        model_sense = pulp.LpMinimize if milp.sense == "min" else pulp.LpMaximize
        model = pulp.LpProblem("qubots_pulp_model", model_sense)

        variables = [
            pulp.LpVariable(
                name=milp.var_names[index],
                lowBound=self._bound(milp.lb[index]),
                upBound=self._bound(milp.ub[index]),
                cat=pulp.LpInteger if milp.integrality[index] else pulp.LpContinuous,
            )
            for index in range(milp.n_vars)
        ]

        model += pulp.lpSum(float(coef) * variables[i] for i, coef in enumerate(milp.c))

        for row_index, (row, rhs) in enumerate(zip(milp.A_ub, milp.b_ub)):
            model += (
                pulp.lpSum(value * variables[index] for index, value in self._row_entries(milp, row))
                <= float(rhs),
                f"ub_{row_index}",
            )
        for row_index, (row, rhs) in enumerate(zip(milp.A_eq, milp.b_eq)):
            model += (
                pulp.lpSum(value * variables[index] for index, value in self._row_entries(milp, row))
                == float(rhs),
                f"eq_{row_index}",
            )

        solver_kwargs: dict[str, Any] = {"msg": bool(self.log_to_console)}
        if self.time_limit_seconds is not None:
            solver_kwargs["timeLimit"] = float(self.time_limit_seconds)

        try:
            status_code = model.solve(pulp.PULP_CBC_CMD(**solver_kwargs))
        except Exception as exc:  # noqa: BLE001 - surface solver errors as result
            return Result(
                best_value=float("inf"),
                best_solution=None,
                runtime_seconds=float(time.perf_counter() - start),
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )

        solution = [float(var.value() if var.value() is not None else 0.0) for var in variables]
        for index, is_integer in enumerate(milp.integrality):
            if is_integer:
                solution[index] = float(round(solution[index]))

        objective = float(pulp.value(model.objective))
        best_value = objective if milp.sense == "min" else -objective
        status_label = str(pulp.LpStatus.get(status_code, "Unknown"))

        return Result(
            best_value=float(best_value),
            best_solution=solution,
            runtime_seconds=float(time.perf_counter() - start),
            status="ok" if status_label == "Optimal" else status_label.lower(),
            metadata={
                "framework": "pulp",
                "solver": "cbc",
                "objective": objective,
                "sense": milp.sense,
                "model_status": status_label,
                "n_vars": milp.n_vars,
                "n_constraints": milp.n_constraints,
                "model_format": "sparse" if isinstance(milp, SparseMILPModel) else "dense",
            },
        )
