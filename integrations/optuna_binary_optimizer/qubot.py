"""Optuna-backed black-box binary optimizer integration."""

from __future__ import annotations

import time
from typing import Any

import optuna

from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


class OptunaBinaryOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.capabilities = ["blackbox"]
        self.n_trials = 64
        self.seed = 0

    def optimize(self, problem: Any) -> Result:
        start = time.perf_counter()
        initial = list(problem.random_solution())
        n = len(initial)
        if n == 0:
            return Result(0.0, [], 0.0, metadata={"framework": "optuna", "n_trials": 0})

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=int(self.seed))
        study = optuna.create_study(direction="minimize", sampler=sampler)

        trace: list[float] = []

        def objective(trial: optuna.Trial) -> float:
            solution = [trial.suggest_int(f"x_{index}", 0, 1) for index in range(n)]
            value = float(problem.evaluate(solution))
            best_so_far = value if not trace else min(trace[-1], value)
            trace.append(best_so_far)
            return value

        study.optimize(objective, n_trials=int(self.n_trials), show_progress_bar=False)
        best_solution = [
            int(study.best_params.get(f"x_{index}", 0))
            for index in range(n)
        ]

        return Result(
            best_value=float(study.best_value),
            best_solution=best_solution,
            runtime_seconds=float(time.perf_counter() - start),
            status="ok",
            trace=trace,
            metadata={
                "framework": "optuna",
                "sampler": "TPESampler",
                "n_trials": int(self.n_trials),
                "best_trial": int(study.best_trial.number),
            },
        )
