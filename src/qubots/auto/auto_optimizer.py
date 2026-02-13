"""Auto-loader for optimizer components."""

import json
from pathlib import Path
from typing import Any

from qubots.core.optimizer import BaseOptimizer
from qubots.hub.loaders import load_component
from qubots.hub.resolver import derive_repo_name, resolve_repo_info


class AutoOptimizer:
    @staticmethod
    def _read_trained(path: str | Path) -> dict[str, Any]:
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Trained artifact not found: {artifact_path}")

        with artifact_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            raise ValueError("Trained artifact must be a JSON object")
        return payload

    @staticmethod
    def from_repo(path: str | Path) -> BaseOptimizer:
        resolved = resolve_repo_info(path)
        component = load_component(resolved.resolved_path, expected_type="optimizer")
        if not isinstance(component, BaseOptimizer):
            raise TypeError("Loaded component is not a BaseOptimizer")
        setattr(
            component,
            "_qubots_source",
            {
                "spec": str(path),
                "resolved_path": str(resolved.resolved_path),
                "ref": resolved.ref,
            },
        )
        setattr(
            component,
            "_qubots_source_name",
            derive_repo_name(path, resolved.resolved_path),
        )
        return component

    @staticmethod
    def from_trained(path_to_trained_json: str | Path) -> BaseOptimizer:
        payload = AutoOptimizer._read_trained(path_to_trained_json)
        optimizer_repo = payload.get("optimizer_repo")
        if not isinstance(optimizer_repo, str) or not optimizer_repo:
            raise ValueError(
                "Trained artifact must contain non-empty string key 'optimizer_repo'"
            )

        optimizer = AutoOptimizer.from_repo(optimizer_repo)
        optimizer.apply_trained(path_to_trained_json)
        return optimizer
