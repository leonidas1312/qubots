"""Base interface for optimizers."""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qubots.core.types import Result

if TYPE_CHECKING:
    from qubots.tune.finetune import FineTuneResult


class BaseOptimizer(ABC):
    def __init__(self) -> None:
        self.parameters: dict[str, Any] = {}

    def set_parameters(self, **kwargs: Any) -> None:
        self.parameters.update(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def apply_trained(self, trained_path: str | Path) -> "BaseOptimizer":
        artifact_path = Path(trained_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Trained artifact not found: {artifact_path}")

        with artifact_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            raise ValueError("Trained artifact must be a JSON object")

        best_params = payload.get("best_params")
        if not isinstance(best_params, dict):
            raise ValueError("Trained artifact must contain object key 'best_params'")

        self.set_parameters(**best_params)
        setattr(self, "_qubots_trained_artifact_path", str(artifact_path.resolve()))
        setattr(self, "_qubots_trained_metadata", payload)
        return self

    def finetune(
        self,
        problem_repo: str | Path | None,
        dataset_path: str | Path,
        budget: int,
        metric: str = "mean_best_value",
        out_dir: str | Path | None = None,
        seed: int | None = None,
        optimizer_repo: str | Path | None = None,
    ) -> "FineTuneResult":
        from qubots.tune.finetune import finetune_optimizer

        return finetune_optimizer(
            optimizer=self,
            problem_repo=problem_repo,
            dataset_path=dataset_path,
            budget=budget,
            metric=metric,
            out_dir=out_dir,
            seed=seed,
            optimizer_repo=optimizer_repo,
        )

    @abstractmethod
    def optimize(self, problem: Any) -> Result:
        raise NotImplementedError
