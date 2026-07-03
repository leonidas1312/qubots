"""Pandas-backed campaign-budget knapsack problem integration."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import pandas as pd

from qubots.core.milp import MILPModel
from qubots.core.problem import BaseProblem


class PandasKnapsackProblem(BaseProblem):
    def __init__(self) -> None:
        super().__init__()
        self.capabilities = ["blackbox", "milp_dense", "integer_only"]
        self.item_path = "data/campaign_budget.csv"
        self.seed = 0
        self._data_key: str | None = None
        self._frame: pd.DataFrame | None = None
        self._sample_count = 0

    def _resolve_path(self) -> Path:
        path = Path(str(self.item_path)).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        return path.resolve()

    def _load_frame(self) -> pd.DataFrame:
        path = self._resolve_path()
        key = str(path)
        if self._frame is not None and self._data_key == key:
            return self._frame

        frame = pd.read_csv(path)
        missing = {"value", "weight", "capacity"} - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        self._frame = frame
        self._data_key = key
        self._sample_count = 0
        return frame

    def _values_weights_capacity(self) -> tuple[list[float], list[float], float]:
        frame = self._load_frame()
        values = [float(value) for value in frame["value"].tolist()]
        weights = [float(value) for value in frame["weight"].tolist()]
        capacity = float(frame["capacity"].iloc[0])
        return values, weights, capacity

    def evaluate(self, solution: Any) -> float:
        values, weights, capacity = self._values_weights_capacity()
        x = [1 if int(value) else 0 for value in list(solution)[: len(values)]]
        x.extend([0] * max(0, len(values) - len(x)))
        total_value = sum(value * xi for value, xi in zip(values, x))
        total_weight = sum(weight * xi for weight, xi in zip(weights, x))
        penalty = max(0.0, total_weight - capacity) * 1_000_000.0
        return float(-total_value + penalty)

    def random_solution(self) -> list[int]:
        values, _, _ = self._values_weights_capacity()
        rng = random.Random(int(self.seed) + self._sample_count * 1_000_003)
        self._sample_count += 1
        return [rng.randint(0, 1) for _ in values]

    def as_milp(self) -> MILPModel:
        values, weights, capacity = self._values_weights_capacity()
        return MILPModel(
            sense="max",
            c=list(values),
            var_names=[f"take_{index}" for index in range(len(values))],
            integrality=[True] * len(values),
            lb=[0.0] * len(values),
            ub=[1.0] * len(values),
            A_ub=[list(weights)],
            b_ub=[float(capacity)],
            constraint_names=["capacity"],
        )
