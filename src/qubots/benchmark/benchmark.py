"""Benchmark runner for comparing optimizers on datasets."""

from __future__ import annotations

import json
from pathlib import Path
import random
import statistics
from typing import Any

from qubots.auto.auto_optimizer import AutoOptimizer
from qubots.auto.auto_problem import AutoProblem
from qubots.hub.resolver import derive_repo_name, is_github_spec
from qubots.tune.dataset import load_dataset_spec


def _resolve_dataset_problem_spec(
    problem_repo: str | Path | None,
    dataset_path: Path,
    dataset_problem: str | None,
) -> str:
    if problem_repo is not None:
        return str(problem_repo)

    if dataset_problem is None:
        raise ValueError(
            "problem_repo is required unless dataset provides a 'problem' value"
        )

    if is_github_spec(dataset_problem):
        return dataset_problem

    candidate = Path(dataset_problem)
    if not candidate.is_absolute():
        candidate = (dataset_path.parent / candidate).resolve()
    return str(candidate)


def _is_trained_artifact_spec(spec: str | Path) -> bool:
    text = str(spec)
    if is_github_spec(text):
        return False
    return Path(text).suffix.lower() == ".json"


def _load_optimizer_from_spec(spec: str | Path) -> tuple[Any, str]:
    if _is_trained_artifact_spec(spec):
        return AutoOptimizer.from_trained(spec), "trained"
    return AutoOptimizer.from_repo(spec), "repo"


def _display_name_for_optimizer(
    optimizer_spec: str | Path,
    source_type: str,
    optimizer: Any,
) -> str:
    if source_type == "trained":
        metadata = getattr(optimizer, "_qubots_trained_metadata", {})
        base_spec = metadata.get("optimizer_repo", str(optimizer_spec))
        return f"{derive_repo_name(base_spec)} (trained)"

    source = getattr(optimizer, "_qubots_source", {})
    resolved_path = source.get("resolved_path")
    return derive_repo_name(str(optimizer_spec), resolved_path=resolved_path)


def benchmark(
    problem_repo: str | Path | None,
    dataset_path: str | Path,
    optimizers: list[str | Path],
    repeats: int = 1,
    seed: int | None = None,
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    if not optimizers:
        raise ValueError("optimizers must contain at least one path")

    dataset_file = Path(dataset_path).resolve()
    dataset_spec = load_dataset_spec(dataset_file)
    problem_spec = _resolve_dataset_problem_spec(
        problem_repo=problem_repo,
        dataset_path=dataset_file,
        dataset_problem=dataset_spec.problem,
    )

    report: dict[str, Any] = {
        "dataset_path": str(dataset_file),
        "problem_repo": str(problem_spec),
        "repeats": int(repeats),
        "seed": seed,
        "results": [],
    }

    for optimizer_index, optimizer_input in enumerate(optimizers):
        optimizer_spec = str(optimizer_input)
        runs: list[dict[str, Any]] = []
        source_type = ""
        display_name = ""

        for repeat_index in range(repeats):
            for instance_index, instance_params in enumerate(dataset_spec.instances):
                if seed is not None:
                    random.seed(
                        seed
                        + optimizer_index * 1_000_000
                        + repeat_index * 10_000
                        + instance_index
                    )

                problem = AutoProblem.from_repo(problem_spec)
                if instance_params:
                    problem.set_parameters(**instance_params)

                optimizer, source_type = _load_optimizer_from_spec(optimizer_spec)
                display_name = _display_name_for_optimizer(
                    optimizer_spec=optimizer_spec,
                    source_type=source_type,
                    optimizer=optimizer,
                )

                result = optimizer.optimize(problem)

                runs.append(
                    {
                        "repeat": repeat_index,
                        "instance_index": instance_index,
                        "instance_params": dict(instance_params),
                        "best_value": float(result.best_value),
                        "runtime_seconds": float(result.runtime_seconds),
                        "status": str(result.status),
                    }
                )

        best_values = [entry["best_value"] for entry in runs]
        runtimes = [entry["runtime_seconds"] for entry in runs]
        success_count = sum(1 for entry in runs if entry["status"] == "ok")

        report["results"].append(
            {
                "optimizer": optimizer_spec,
                "display_name": display_name or optimizer_spec,
                "source_type": source_type,
                "num_runs": len(runs),
                "mean_best_value": float(statistics.fmean(best_values)),
                "mean_runtime_seconds": float(statistics.fmean(runtimes)),
                "success_rate": float(success_count / len(runs)),
                "runs": runs,
            }
        )

    return report


def report_to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "| optimizer | type | mean_best_value | mean_runtime_seconds | success_rate |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in report.get("results", []):
        lines.append(
            "| {optimizer} | {source_type} | {mean_best_value:.6f} | {mean_runtime_seconds:.6f} | {success_rate:.2%} |".format(
                optimizer=row.get("display_name", row.get("optimizer", "")),
                source_type=row.get("source_type", ""),
                mean_best_value=float(row.get("mean_best_value", 0.0)),
                mean_runtime_seconds=float(row.get("mean_runtime_seconds", 0.0)),
                success_rate=float(row.get("success_rate", 0.0)),
            )
        )

    return "\n".join(lines)


def write_report(report: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path
