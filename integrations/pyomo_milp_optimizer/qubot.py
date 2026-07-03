"""Pyomo MILP optimizer integration."""

from __future__ import annotations

import math
import time
from typing import Any

import pyomo.environ as pyo

from qubots.core.milp import MILPModel, SparseMILPModel, SupportsMILP
from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


class PyomoMILPOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.capabilities = ["milp_dense", "milp_sparse", "integer_only", "continuous"]
        self.solver = "appsi_highs"
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

    @staticmethod
    def _milp_from_problem(problem: Any) -> MILPModel | SparseMILPModel:
        if isinstance(problem, (MILPModel, SparseMILPModel)):
            return problem
        if isinstance(problem, SupportsMILP):
            milp = problem.as_milp()
            if isinstance(milp, (MILPModel, SparseMILPModel)):
                return milp
        raise TypeError(
            "PyomoMILPOptimizer requires a problem implementing as_milp(), "
            "or a MILPModel/SparseMILPModel directly."
        )

    def optimize(self, problem: Any) -> Result:
        start = time.perf_counter()
        milp = self._milp_from_problem(problem)

        model = pyo.ConcreteModel()
        model.I = pyo.RangeSet(0, milp.n_vars - 1)

        def bounds(_: Any, index: int) -> tuple[float | None, float | None]:
            return self._bound(milp.lb[index]), self._bound(milp.ub[index])

        def domain(_: Any, index: int) -> Any:
            return pyo.Integers if milp.integrality[index] else pyo.Reals

        model.x = pyo.Var(model.I, bounds=bounds, domain=domain)
        sense = pyo.minimize if milp.sense == "min" else pyo.maximize
        model.obj = pyo.Objective(
            expr=sum(float(milp.c[i]) * model.x[i] for i in range(milp.n_vars)),
            sense=sense,
        )

        model.constraints = pyo.ConstraintList()
        for row, rhs in zip(milp.A_ub, milp.b_ub):
            model.constraints.add(
                sum(value * model.x[index] for index, value in self._row_entries(milp, row))
                <= float(rhs)
            )
        for row, rhs in zip(milp.A_eq, milp.b_eq):
            model.constraints.add(
                sum(value * model.x[index] for index, value in self._row_entries(milp, row))
                == float(rhs)
            )

        solver = pyo.SolverFactory(str(self.solver))
        if not solver.available(False):
            return Result(
                best_value=float("inf"),
                best_solution=None,
                runtime_seconds=float(time.perf_counter() - start),
                status="error",
                error=f"Pyomo solver {self.solver!r} is not available.",
            )
        if self.time_limit_seconds is not None:
            solver.options["time_limit"] = float(self.time_limit_seconds)

        try:
            result = solver.solve(model, tee=bool(self.log_to_console))
        except Exception as exc:  # noqa: BLE001 - expose backend errors
            return Result(
                best_value=float("inf"),
                best_solution=None,
                runtime_seconds=float(time.perf_counter() - start),
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )

        solution = [float(pyo.value(model.x[i])) for i in range(milp.n_vars)]
        for index, is_integer in enumerate(milp.integrality):
            if is_integer:
                solution[index] = float(round(solution[index]))

        objective = float(pyo.value(model.obj))
        best_value = objective if milp.sense == "min" else -objective
        termination = str(result.solver.termination_condition).lower()

        return Result(
            best_value=float(best_value),
            best_solution=solution,
            runtime_seconds=float(time.perf_counter() - start),
            status="ok" if termination == "optimal" else termination,
            metadata={
                "framework": "pyomo",
                "solver": str(self.solver),
                "objective": objective,
                "sense": milp.sense,
                "model_status": termination,
                "n_vars": milp.n_vars,
                "n_constraints": milp.n_constraints,
            },
        )
