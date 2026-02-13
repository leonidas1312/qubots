"""Manifest parsing helpers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Manifest:
    type: str
    name: str
    entrypoint: str
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    tunable_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_manifest(repo_path: str | Path) -> Manifest:
    repo = Path(repo_path)
    manifest_path = repo / "qubots.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    missing = [key for key in ("type", "name", "entrypoint") if key not in raw]
    if missing:
        raise ValueError(f"Manifest missing required keys: {', '.join(missing)}")

    parameters = raw.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("Manifest 'parameters' must be a mapping")

    tunable_parameters = raw.get("tunable_parameters") or {}
    if not isinstance(tunable_parameters, dict):
        raise ValueError("Manifest 'tunable_parameters' must be a mapping")

    return Manifest(
        type=str(raw["type"]),
        name=str(raw["name"]),
        entrypoint=str(raw["entrypoint"]),
        parameters=parameters,
        tunable_parameters=tunable_parameters,
    )
