"""SciPy linear-sum-assignment optimizer integration."""

from __future__ import annotations

import time
from typing import Any

from scipy.optimize import linear_sum_assignment

from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


class SciPyAssignmentOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.capabilities = ["blackbox", "integer_only"]

    @staticmethod
    def _problem_with_matrix(problem: Any) -> Any:
        if hasattr(problem, "_load_matrix"):
            return problem
        if hasattr(problem, "_problem"):
            loaded = problem._problem()
            if hasattr(loaded, "_load_matrix"):
                return loaded
        raise TypeError(
            "SciPyAssignmentOptimizer requires an assignment problem created "
            "by qubots detect/import, or another problem exposing _load_matrix()."
        )

    def optimize(self, problem: Any) -> Result:
        start = time.perf_counter()
        matrix_problem = self._problem_with_matrix(problem)
        matrix = matrix_problem._load_matrix()

        row_ind, col_ind = linear_sum_assignment(matrix)
        assignment = [-1] * len(matrix)
        for row, col in zip(row_ind, col_ind):
            assignment[int(row)] = int(col)

        best_value = float(matrix_problem.evaluate(assignment))
        return Result(
            best_value=best_value,
            best_solution=assignment,
            runtime_seconds=float(time.perf_counter() - start),
            status="ok",
            metadata={
                "framework": "scipy",
                "solver": "linear_sum_assignment",
                "objective": best_value,
                "n_workers": len(matrix),
                "n_tasks": len(matrix[0]) if matrix else 0,
            },
        )
