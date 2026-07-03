"""CVXPY continuous LP optimizer integration."""

from __future__ import annotations

import math
import time
from typing import Any

import cvxpy as cp

from qubots.core.milp import MILPModel, SparseMILPModel, SupportsMILP
from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


class CVXPYLPOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.capabilities = ["milp_dense", "milp_sparse", "continuous"]
        self.solver = "CLARABEL"
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
    def _linear_expr(
        variables: Any, entries: list[tuple[int, float]]
    ) -> Any:
        return cp.sum([value * variables[index] for index, value in entries])

    @staticmethod
    def _milp_from_problem(problem: Any) -> MILPModel | SparseMILPModel:
        if isinstance(problem, (MILPModel, SparseMILPModel)):
            return problem
        if isinstance(problem, SupportsMILP):
            milp = problem.as_milp()
            if isinstance(milp, (MILPModel, SparseMILPModel)):
                return milp
        raise TypeError(
            "CVXPYLPOptimizer requires a problem implementing as_milp(), "
            "or a MILPModel/SparseMILPModel directly."
        )

    def optimize(self, problem: Any) -> Result:
        start = time.perf_counter()
        milp = self._milp_from_problem(problem)

        if any(milp.integrality):
            return Result(
                best_value=float("inf"),
                best_solution=None,
                runtime_seconds=float(time.perf_counter() - start),
                status="unsupported",
                error="CVXPYLPOptimizer handles continuous LPs only.",
            )

        x = cp.Variable(milp.n_vars)
        constraints: list[Any] = []

        for index, lower in enumerate(milp.lb):
            if math.isfinite(lower):
                constraints.append(x[index] >= float(lower))
        for index, upper in enumerate(milp.ub):
            if math.isfinite(upper):
                constraints.append(x[index] <= float(upper))

        for row, rhs in zip(milp.A_ub, milp.b_ub):
            constraints.append(self._linear_expr(x, self._row_entries(milp, row)) <= float(rhs))
        for row, rhs in zip(milp.A_eq, milp.b_eq):
            constraints.append(self._linear_expr(x, self._row_entries(milp, row)) == float(rhs))

        objective_expr = cp.sum([float(coef) * x[index] for index, coef in enumerate(milp.c)])
        objective = cp.Minimize(objective_expr) if milp.sense == "min" else cp.Maximize(objective_expr)
        cvx_problem = cp.Problem(objective, constraints)

        try:
            cvx_problem.solve(
                solver=str(self.solver) if self.solver else None,
                verbose=bool(self.log_to_console),
            )
        except Exception as exc:  # noqa: BLE001 - expose backend errors
            return Result(
                best_value=float("inf"),
                best_solution=None,
                runtime_seconds=float(time.perf_counter() - start),
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )

        solution = [] if x.value is None else [float(value) for value in x.value.tolist()]
        objective_value = float(cvx_problem.value) if cvx_problem.value is not None else float("inf")
        best_value = objective_value if milp.sense == "min" else -objective_value
        status = str(cvx_problem.status)

        return Result(
            best_value=float(best_value),
            best_solution=solution if solution else None,
            runtime_seconds=float(time.perf_counter() - start),
            status="ok" if status in {"optimal", "optimal_inaccurate"} else status,
            metadata={
                "framework": "cvxpy",
                "solver": str(self.solver),
                "objective": objective_value,
                "sense": milp.sense,
                "model_status": status,
                "n_vars": milp.n_vars,
                "n_constraints": milp.n_constraints,
            },
        )
