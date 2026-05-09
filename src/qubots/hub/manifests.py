"""Manifest parsing helpers.

Manifests are versioned via the top-level ``qubots_schema_version`` key.
A missing key is treated as version 1 for backward compatibility with
the original 0.1.x manifests (which had no version field). Future
incompatible schema changes increment ``CURRENT_SCHEMA_VERSION`` and add
to ``SUPPORTED_SCHEMA_VERSIONS``; loaders refuse versions they don't
understand with an explicit "upgrade qubots" message.

v2 introduces a top-level ``requirements:`` field — a list of
pip-installable specs (e.g. ``["highspy>=1.7"]``). The CI runner used by
the community leaderboard installs these into an isolated venv per
submission. v1 manifests have no ``requirements`` field and are loaded
with an empty list.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


CURRENT_SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1, 2})


@dataclass
class Manifest:
    type: str
    name: str
    entrypoint: str
    schema_version: int = CURRENT_SCHEMA_VERSION
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    tunable_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    requirements: list[str] = field(default_factory=list)


def _coerce_schema_version(raw: Any) -> int:
    if raw is None:
        return CURRENT_SCHEMA_VERSION
    if isinstance(raw, bool):
        raise ValueError("qubots_schema_version must be an integer, not a bool")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(
                f"qubots_schema_version must be an integer, got {raw!r}"
            ) from exc
    raise ValueError(
        f"qubots_schema_version must be an integer, got {type(raw).__name__}"
    )


def load_manifest(repo_path: str | Path) -> Manifest:
    repo = Path(repo_path)
    manifest_path = repo / "qubots.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    schema_version = _coerce_schema_version(raw.get("qubots_schema_version"))
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported qubots_schema_version={schema_version} in {manifest_path}. "
            f"This qubots installation supports versions: "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}. Upgrade or downgrade qubots."
        )

    missing = [key for key in ("type", "name", "entrypoint") if key not in raw]
    if missing:
        raise ValueError(f"Manifest missing required keys: {', '.join(missing)}")

    parameters = raw.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("Manifest 'parameters' must be a mapping")

    tunable_parameters = raw.get("tunable_parameters") or {}
    if not isinstance(tunable_parameters, dict):
        raise ValueError("Manifest 'tunable_parameters' must be a mapping")

    raw_requirements = raw.get("requirements")
    if raw_requirements is None:
        requirements: list[str] = []
    elif isinstance(raw_requirements, list) and all(
        isinstance(item, str) for item in raw_requirements
    ):
        requirements = list(raw_requirements)
    else:
        raise ValueError(
            "Manifest 'requirements' must be a list of pip-spec strings"
        )

    return Manifest(
        type=str(raw["type"]),
        name=str(raw["name"]),
        entrypoint=str(raw["entrypoint"]),
        schema_version=schema_version,
        parameters=parameters,
        tunable_parameters=tunable_parameters,
        requirements=requirements,
    )
