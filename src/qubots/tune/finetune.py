"""Random-search fine-tuning for optimizer parameters."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Any

from qubots.auto.auto_optimizer import AutoOptimizer
from qubots.auto.auto_problem import AutoProblem
from qubots.hub.manifests import load_manifest
from qubots.hub.resolver import is_github_spec, resolve_repo_info
from qubots.tune.dataset import load_dataset_spec


@dataclass
class FineTuneResult:
    artifact_path: Path
    best_params: dict[str, Any]
    score: float
    metric: str


def _qubots_version() -> str:
    try:
        return pkg_version("qubots")
    except PackageNotFoundError:
        return "0.0.0"


def _validate_search_space(
    tunable_parameters: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not tunable_parameters:
        raise ValueError("Manifest must define non-empty 'tunable_parameters' for finetune")

    normalized: dict[str, dict[str, Any]] = {}
    for name, config in tunable_parameters.items():
        if not isinstance(config, dict):
            raise ValueError(f"tunable_parameters.{name} must be a mapping")
        if "type" not in config:
            raise ValueError(f"tunable_parameters.{name}.type is required")

        param_type = config["type"]
        if param_type == "int":
            if "min" not in config or "max" not in config:
                raise ValueError(
                    f"tunable_parameters.{name} requires 'min' and 'max' for int"
                )
            min_value = config["min"]
            max_value = config["max"]
            if not isinstance(min_value, int) or isinstance(min_value, bool):
                raise ValueError(f"tunable_parameters.{name}.min must be an int")
            if not isinstance(max_value, int) or isinstance(max_value, bool):
                raise ValueError(f"tunable_parameters.{name}.max must be an int")
            if min_value > max_value:
                raise ValueError(f"tunable_parameters.{name}.min must be <= max")
            normalized[name] = {"type": "int", "min": min_value, "max": max_value}
            continue

        if param_type == "float":
            if "min" not in config or "max" not in config:
                raise ValueError(
                    f"tunable_parameters.{name} requires 'min' and 'max' for float"
                )
            min_value = config["min"]
            max_value = config["max"]
            if not isinstance(min_value, (int, float)) or isinstance(min_value, bool):
                raise ValueError(f"tunable_parameters.{name}.min must be numeric")
            if not isinstance(max_value, (int, float)) or isinstance(max_value, bool):
                raise ValueError(f"tunable_parameters.{name}.max must be numeric")
            if float(min_value) > float(max_value):
                raise ValueError(f"tunable_parameters.{name}.min must be <= max")
            normalized[name] = {
                "type": "float",
                "min": float(min_value),
                "max": float(max_value),
            }
            continue

        if param_type == "choice":
            if "values" not in config:
                raise ValueError(f"tunable_parameters.{name}.values is required for choice")
            values = config["values"]
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"tunable_parameters.{name}.values must be a non-empty list"
                )
            normalized[name] = {"type": "choice", "values": values}
            continue

        raise ValueError(
            f"Unsupported tunable parameter type '{param_type}' for '{name}'"
        )

    return normalized


def _sample_parameters(
    search_space: dict[str, dict[str, Any]],
    rng: random.Random,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, config in sorted(search_space.items()):
        param_type = config["type"]
        if param_type == "int":
            params[name] = rng.randint(int(config["min"]), int(config["max"]))
        elif param_type == "float":
            params[name] = rng.uniform(float(config["min"]), float(config["max"]))
        else:
            params[name] = rng.choice(config["values"])
    return params


def _score_trial(
    problem_repo: str | Path,
    optimizer_repo: str | Path,
    dataset_items: list[dict[str, Any]],
    params: dict[str, Any],
    metric: str,
    seed: int | None,
    trial_index: int,
) -> float:
    best_values: list[float] = []
    for item_index, problem_params in enumerate(dataset_items):
        if seed is not None:
            random.seed(seed + trial_index * 10_000 + item_index)

        optimizer = AutoOptimizer.from_repo(str(optimizer_repo))
        optimizer.set_parameters(**params)
        problem = AutoProblem.from_repo(str(problem_repo))
        if problem_params:
            problem.set_parameters(**problem_params)
        result = optimizer.optimize(problem)
        best_values.append(float(result.best_value))

    if metric == "mean_best_value":
        return float(sum(best_values) / len(best_values))

    raise ValueError(f"Unsupported metric '{metric}'. Use 'mean_best_value'.")


def finetune_optimizer(
    optimizer: Any,
    problem_repo: str | Path | None,
    dataset_path: str | Path,
    budget: int,
    metric: str = "mean_best_value",
    out_dir: str | Path | None = None,
    seed: int | None = None,
    optimizer_repo: str | Path | None = None,
) -> FineTuneResult:
    if budget <= 0:
        raise ValueError("budget must be a positive integer")

    optimizer_source = getattr(optimizer, "_qubots_source", {}) or {}
    optimizer_repo_spec = (
        optimizer_repo
        or optimizer_source.get("spec")
        or getattr(optimizer, "_qubots_repo_path", None)
    )
    if optimizer_repo_spec is None:
        raise ValueError(
            "optimizer_repo is required when optimizer was not loaded from a component repo"
        )

    dataset_file = Path(dataset_path).resolve()
    optimizer_repo_info = resolve_repo_info(str(optimizer_repo_spec))

    manifest = load_manifest(optimizer_repo_info.resolved_path)
    if manifest.type != "optimizer":
        raise ValueError("finetune requires an optimizer component repo")

    search_space = _validate_search_space(manifest.tunable_parameters)
    dataset_spec = load_dataset_spec(dataset_file)

    resolved_problem_repo: str
    if problem_repo is not None:
        resolved_problem_repo = str(problem_repo)
    elif dataset_spec.problem:
        if is_github_spec(dataset_spec.problem):
            resolved_problem_repo = dataset_spec.problem
        else:
            candidate = Path(dataset_spec.problem)
            if not candidate.is_absolute():
                candidate = (dataset_file.parent / candidate).resolve()
            resolved_problem_repo = str(candidate)
    else:
        raise ValueError(
            "problem_repo is required unless dataset provides a 'problem' value"
        )

    dataset_items = dataset_spec.instances

    sampler = random.Random(seed)
    best_params: dict[str, Any] = {}
    best_score = float("inf")

    for trial_index in range(budget):
        trial_params = _sample_parameters(search_space, sampler)
        trial_score = _score_trial(
            problem_repo=resolved_problem_repo,
            optimizer_repo=str(optimizer_repo_spec),
            dataset_items=dataset_items,
            params=trial_params,
            metric=metric,
            seed=seed,
            trial_index=trial_index,
        )
        if trial_score < best_score:
            best_score = trial_score
            best_params = trial_params

    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    timestamp_slug = now.strftime("%Y%m%dT%H%M%SZ")

    artifact_dir = (
        Path(out_dir)
        if out_dir is not None
        else Path("trained") / f"{manifest.name}-{timestamp_slug}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "trained.json"

    payload = {
        "best_params": best_params,
        "score": float(best_score),
        "metric": metric,
        "budget": int(budget),
        "dataset_path": str(dataset_file),
        "problem_repo": str(resolved_problem_repo),
        "optimizer_repo": str(optimizer_repo_spec),
        "qubots_version": _qubots_version(),
        "timestamp": timestamp,
    }

    with artifact_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return FineTuneResult(
        artifact_path=artifact_path,
        best_params=best_params,
        score=float(best_score),
        metric=metric,
    )
