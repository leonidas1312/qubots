"""Utilities to export a trained artifact as a component repo."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
from typing import Any

import yaml


def _normalize_name(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw.strip()).strip("-")
    if not cleaned:
        return "trained-optimizer"
    return cleaned.lower()


def _load_trained_payload(trained_path: Path) -> dict[str, Any]:
    if not trained_path.exists():
        raise FileNotFoundError(f"Trained artifact not found: {trained_path}")

    with trained_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Trained artifact must be a JSON object")

    optimizer_repo = payload.get("optimizer_repo")
    if not isinstance(optimizer_repo, str) or not optimizer_repo:
        raise ValueError(
            "Trained artifact must contain non-empty string key 'optimizer_repo'"
        )

    best_params = payload.get("best_params")
    if not isinstance(best_params, dict):
        raise ValueError("Trained artifact must contain object key 'best_params'")

    return payload


def _build_qubot_py() -> str:
    return '''"""Exported trained optimizer component."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qubots.auto.auto_optimizer import AutoOptimizer
from qubots.core.optimizer import BaseOptimizer
from qubots.core.types import Result


class TrainedOptimizer(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        artifact_path = Path(__file__).with_name("trained.json")
        with artifact_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        optimizer_repo = payload.get("optimizer_repo")
        if not isinstance(optimizer_repo, str) or not optimizer_repo:
            raise ValueError(
                "Trained artifact must contain non-empty string key 'optimizer_repo'"
            )

        best_params = payload.get("best_params")
        if not isinstance(best_params, dict):
            raise ValueError("Trained artifact must contain object key 'best_params'")

        if optimizer_repo.startswith("github:"):
            repo_path = optimizer_repo
        else:
            local_path = Path(optimizer_repo)
            if not local_path.is_absolute():
                local_path = (artifact_path.parent / local_path).resolve()
            repo_path = str(local_path)

        self._base_optimizer = AutoOptimizer.from_repo(repo_path)
        self._trained_metadata = payload
        self.set_parameters(**best_params)

    def set_parameters(self, **kwargs: Any) -> None:
        super().set_parameters(**kwargs)
        if hasattr(self, "_base_optimizer"):
            self._base_optimizer.set_parameters(**kwargs)

    def optimize(self, problem: Any) -> Result:
        return self._base_optimizer.optimize(problem)
'''


def export_trained_optimizer(
    trained_path: str | Path,
    out_dir: str | Path,
    name: str | None = None,
) -> Path:
    trained = Path(trained_path).resolve()
    out = Path(out_dir).resolve()
    payload = _load_trained_payload(trained)

    base_name = name
    if base_name is None:
        base_name = f"{Path(str(payload['optimizer_repo'])).name}-trained"
    component_name = _normalize_name(base_name)

    out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(trained, out / "trained.json")

    best_params = payload.get("best_params", {})
    manifest = {
        "type": "optimizer",
        "name": component_name,
        "entrypoint": "qubot.py:TrainedOptimizer",
        "parameters": {
            key: {"default": value} for key, value in dict(best_params).items()
        },
    }

    with (out / "qubots.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    (out / "qubot.py").write_text(_build_qubot_py(), encoding="utf-8")

    readme = "\n".join(
        [
            f"# {component_name}",
            "",
            "This optimizer component was exported from a trained Qubots artifact.",
            "",
            f"- Source artifact: {trained}",
            f"- Base optimizer repo: {payload.get('optimizer_repo', 'unknown')}",
            f"- Training dataset: {payload.get('dataset_path', 'unknown')}",
            f"- Training timestamp: {payload.get('timestamp', 'unknown')}",
            "",
            "Load with:",
            "",
            "```python",
            "from qubots import AutoOptimizer",
            f'opt = AutoOptimizer.from_repo("{out}")',
            "```",
            "",
        ]
    )
    (out / "README.md").write_text(readme, encoding="utf-8")

    return out
