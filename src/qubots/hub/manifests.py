"""Manifest parsing helpers.

Manifests are versioned via the top-level ``qubots_schema_version`` key.
A missing key is treated as the current schema for backward compatibility with
the original manifests in this repo (which had no version field). Future
incompatible schema changes increment ``CURRENT_SCHEMA_VERSION`` and add
to ``SUPPORTED_SCHEMA_VERSIONS``; loaders refuse versions they don't
understand with an explicit "upgrade qubots" message.

v2 introduces a top-level ``requirements:`` field — a list of
pip-installable specs (e.g. ``["highspy>=1.7"]``). The CI runner used by
the community leaderboard installs these into an isolated venv per
submission. v1 manifests have no ``requirements`` field and are loaded
with an empty list.

v3 adds optional autodetection/Rastion metadata: ``capabilities``,
``problem_family``, ``data_schema``, ``metrics``, ``license``, ``citation``,
and ``rastion_card``. These fields are metadata only; v1/v2 repos remain
loadable and runnable.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


CURRENT_SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1, 2, 3})


@dataclass
class Manifest:
    type: str
    name: str
    entrypoint: str
    schema_version: int = CURRENT_SCHEMA_VERSION
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    tunable_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    requirements: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    problem_family: str | None = None
    data_schema: dict[str, Any] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=list)
    license: str | None = None
    citation: str | None = None
    rastion_card: str | None = None


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


def _coerce_optional_string_list(raw: Any, key: str) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return list(raw)
    raise ValueError(f"Manifest '{key}' must be a list of strings")


def _coerce_optional_string(raw: Any, key: str) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    raise ValueError(f"Manifest '{key}' must be a string")


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

    capabilities = _coerce_optional_string_list(raw.get("capabilities"), "capabilities")
    metrics = _coerce_optional_string_list(raw.get("metrics"), "metrics")

    data_schema_raw = raw.get("data_schema")
    if data_schema_raw is None:
        data_schema: dict[str, Any] = {}
    elif isinstance(data_schema_raw, dict):
        data_schema = dict(data_schema_raw)
    else:
        raise ValueError("Manifest 'data_schema' must be a mapping")

    return Manifest(
        type=str(raw["type"]),
        name=str(raw["name"]),
        entrypoint=str(raw["entrypoint"]),
        schema_version=schema_version,
        parameters=parameters,
        tunable_parameters=tunable_parameters,
        requirements=requirements,
        capabilities=capabilities,
        problem_family=_coerce_optional_string(raw.get("problem_family"), "problem_family"),
        data_schema=data_schema,
        metrics=metrics,
        license=_coerce_optional_string(raw.get("license"), "license"),
        citation=_coerce_optional_string(raw.get("citation"), "citation"),
        rastion_card=_coerce_optional_string(raw.get("rastion_card"), "rastion_card"),
    )
