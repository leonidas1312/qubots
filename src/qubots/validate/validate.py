"""Validation helpers for Qubots component repositories."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import re
from types import ModuleType
from typing import Any

import yaml

from qubots.core.optimizer import BaseOptimizer
from qubots.core.problem import BaseProblem
from qubots.hub.resolver import resolve_repo_info


_ENTRYPOINT_PATTERN = re.compile(
    r"^(?P<module>[^:]+\.py):(?P<class_name>[A-Za-z_][A-Za-z0-9_]*)$"
)


def _is_yaml_serializable(value: Any) -> bool:
    try:
        yaml.safe_dump(value)
    except Exception:
        return False
    return True


def _import_module(module_path: Path) -> ModuleType:
    module_name = f"qubots_validate_{module_path.stem}_{abs(hash(module_path))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_parameters_schema(raw: dict[str, Any], issues: list[str]) -> None:
    if "parameters" not in raw:
        return

    parameters = raw["parameters"]
    if not isinstance(parameters, dict):
        issues.append("Manifest 'parameters' must be a mapping")
        return

    for param_name, config in parameters.items():
        if not isinstance(config, dict):
            issues.append(f"parameters.{param_name} must be a mapping")
            continue
        if not _is_yaml_serializable(config):
            issues.append(f"parameters.{param_name} must be YAML-serializable")


_PIP_SPEC_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*"
    r"(?:\s*\[[A-Za-z0-9._,\s-]+\])?"
    r"(?:\s*[<>=!~]=?\s*[A-Za-z0-9._*+-]+(?:\s*,\s*[<>=!~]=?\s*[A-Za-z0-9._*+-]+)*)?$"
)


def _validate_requirements_schema(raw: dict[str, Any], issues: list[str]) -> None:
    if "requirements" not in raw:
        return

    requirements = raw["requirements"]
    if not isinstance(requirements, list):
        issues.append("Manifest 'requirements' must be a list of pip-spec strings")
        return

    for index, item in enumerate(requirements):
        if not isinstance(item, str) or not item.strip():
            issues.append(f"requirements[{index}] must be a non-empty string")
            continue
        if not _PIP_SPEC_RE.match(item.strip()):
            issues.append(
                f"requirements[{index}] is not a valid pip spec: {item!r}"
            )


def _validate_tunable_schema(raw: dict[str, Any], issues: list[str]) -> None:
    if "tunable_parameters" not in raw:
        return

    tunables = raw["tunable_parameters"]
    if not isinstance(tunables, dict):
        issues.append("Manifest 'tunable_parameters' must be a mapping")
        return

    for param_name, config in tunables.items():
        key = f"tunable_parameters.{param_name}"
        if not isinstance(config, dict):
            issues.append(f"{key} must be a mapping")
            continue
        if not _is_yaml_serializable(config):
            issues.append(f"{key} must be YAML-serializable")
            continue

        param_type = config.get("type")
        if param_type not in {"int", "float", "choice"}:
            issues.append(f"{key}.type must be one of: int, float, choice")
            continue

        if param_type == "int":
            if "min" not in config or "max" not in config:
                issues.append(f"{key} requires 'min' and 'max' for int")
                continue
            min_value = config["min"]
            max_value = config["max"]
            if not isinstance(min_value, int) or isinstance(min_value, bool):
                issues.append(f"{key}.min must be an int")
                continue
            if not isinstance(max_value, int) or isinstance(max_value, bool):
                issues.append(f"{key}.max must be an int")
                continue
            if min_value > max_value:
                issues.append(f"{key}.min must be <= max")
            continue

        if param_type == "float":
            if "min" not in config or "max" not in config:
                issues.append(f"{key} requires 'min' and 'max' for float")
                continue
            min_value = config["min"]
            max_value = config["max"]
            if not isinstance(min_value, (int, float)) or isinstance(min_value, bool):
                issues.append(f"{key}.min must be numeric")
                continue
            if not isinstance(max_value, (int, float)) or isinstance(max_value, bool):
                issues.append(f"{key}.max must be numeric")
                continue
            if float(min_value) > float(max_value):
                issues.append(f"{key}.min must be <= max")
            continue

        values = config.get("values")
        if not isinstance(values, list) or not values:
            issues.append(f"{key}.values must be a non-empty list for choice")


def _validate_entrypoint(
    repo: Path,
    component_type: str | None,
    entrypoint: Any,
    issues: list[str],
) -> None:
    if not isinstance(entrypoint, str):
        issues.append("Manifest 'entrypoint' must be a string in 'file.py:ClassName' format")
        return

    match = _ENTRYPOINT_PATTERN.match(entrypoint.strip())
    if not match:
        issues.append("Manifest 'entrypoint' must match 'file.py:ClassName'")
        return

    module_file = match.group("module")
    class_name = match.group("class_name")
    module_path = repo / module_file

    if not module_path.exists() or not module_path.is_file():
        issues.append(f"Entrypoint module not found: {module_file}")
        return

    try:
        module = _import_module(module_path)
    except Exception as exc:
        issues.append(f"Entrypoint import failed for {module_file}: {exc}")
        return

    if not hasattr(module, class_name):
        issues.append(f"Entrypoint class '{class_name}' not found in {module_file}")
        return

    cls = getattr(module, class_name)
    if not inspect.isclass(cls):
        issues.append(f"Entrypoint '{class_name}' in {module_file} is not a class")
        return

    if component_type == "problem":
        if not issubclass(cls, BaseProblem):
            issues.append(f"Entrypoint class '{class_name}' must subclass BaseProblem")
        else:
            has_blackbox = (
                getattr(cls, "evaluate", None) is not BaseProblem.evaluate
                and getattr(cls, "random_solution", None) is not BaseProblem.random_solution
            )
            has_milp = "as_milp" in cls.__mro__[0].__dict__ or any(
                "as_milp" in base.__dict__ for base in cls.__mro__
            )
            if not (has_blackbox or has_milp):
                issues.append(
                    f"Problem class '{class_name}' must implement either "
                    "evaluate()+random_solution() or as_milp()"
                )

    if component_type == "optimizer":
        if not issubclass(cls, BaseOptimizer):
            issues.append(f"Entrypoint class '{class_name}' must subclass BaseOptimizer")
        else:
            if getattr(cls, "optimize", None) is BaseOptimizer.optimize:
                issues.append(f"Optimizer class '{class_name}' must implement optimize()")

    try:
        cls()
    except Exception as exc:
        issues.append(f"Entrypoint class '{class_name}' must be instantiable with no args: {exc}")


def validate_repo(path: str | Path) -> list[str]:
    try:
        resolved = resolve_repo_info(path)
    except Exception as exc:
        return [str(exc)]

    repo = resolved.resolved_path
    issues: list[str] = []

    if not repo.exists() or not repo.is_dir():
        return [f"Repository path does not exist or is not a directory: {repo}"]

    manifest_path = repo / "qubots.yaml"
    if not manifest_path.exists() or not manifest_path.is_file():
        return [f"Manifest not found: {manifest_path}"]

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            raw_manifest = yaml.safe_load(f)
    except Exception as exc:
        return [f"Failed to parse manifest {manifest_path}: {exc}"]

    if raw_manifest is None:
        raw_manifest = {}

    if not isinstance(raw_manifest, dict):
        return ["Manifest must be a YAML mapping"]

    required = ["type", "name", "entrypoint"]
    missing = [key for key in required if key not in raw_manifest]
    if missing:
        issues.append(f"Manifest missing required keys: {', '.join(missing)}")

    component_type = raw_manifest.get("type")
    if component_type not in {"problem", "optimizer"}:
        issues.append("Manifest 'type' must be 'problem' or 'optimizer'")
        component_type = None

    if "name" in raw_manifest and not isinstance(raw_manifest.get("name"), str):
        issues.append("Manifest 'name' must be a string")

    _validate_parameters_schema(raw_manifest, issues)
    _validate_tunable_schema(raw_manifest, issues)
    _validate_requirements_schema(raw_manifest, issues)

    if "entrypoint" in raw_manifest:
        _validate_entrypoint(repo, component_type, raw_manifest.get("entrypoint"), issues)

    return issues


def validate_tree(path: str | Path) -> dict[str, list[str]]:
    try:
        resolved = resolve_repo_info(path)
    except Exception as exc:
        return {str(path): [str(exc)]}

    root = resolved.resolved_path
    if not root.exists() or not root.is_dir():
        return {str(root): [f"Path does not exist or is not a directory: {root}"]}

    repos: set[Path] = set()
    if (root / "qubots.yaml").exists():
        repos.add(root)

    for manifest_path in root.rglob("qubots.yaml"):
        repos.add(manifest_path.parent)

    return {str(repo): validate_repo(repo) for repo in sorted(repos)}
