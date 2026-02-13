"""Dataset loading for fine-tuning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetSpec:
    problem: str | None
    instances: list[dict[str, Any]]


def _validate_instances(raw_instances: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_instances, list):
        raise ValueError("Dataset instances must be a YAML list of parameter dictionaries")

    items: list[dict[str, Any]] = []
    for index, item in enumerate(raw_instances):
        if not isinstance(item, dict):
            raise ValueError(
                f"Dataset item at index {index} must be a mapping of problem parameters"
            )
        items.append(dict(item))

    if not items:
        raise ValueError("Dataset must contain at least one problem instance")

    return items


def load_dataset_spec(dataset_path: str | Path) -> DatasetSpec:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if isinstance(raw, list):
        return DatasetSpec(problem=None, instances=_validate_instances(raw))

    if isinstance(raw, dict):
        if "instances" not in raw:
            raise ValueError("Dataset mapping format requires an 'instances' key")
        problem = raw.get("problem")
        if problem is not None and not isinstance(problem, str):
            raise ValueError("Dataset 'problem' must be a string when provided")
        return DatasetSpec(
            problem=problem,
            instances=_validate_instances(raw["instances"]),
        )

    raise ValueError(
        "Dataset must be either a list of instances or a mapping with 'instances'"
    )


def load_dataset(dataset_path: str | Path) -> list[dict[str, Any]]:
    # Backward-compatible helper returning only instances.
    return load_dataset_spec(dataset_path).instances
