"""Dynamic loading for local component repositories."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from qubots.hub.manifests import Manifest, load_manifest


def _ensure_local_path(path: str | Path) -> Path:
    text = str(path)
    if "://" in text:
        raise ValueError("Only local repository paths are supported in v0.1")
    return Path(path).resolve()


def _parse_entrypoint(entrypoint: str) -> tuple[str, str]:
    if ":" not in entrypoint:
        raise ValueError("Entrypoint must be in 'file.py:ClassName' format")
    module_file, class_name = entrypoint.split(":", maxsplit=1)
    return module_file, class_name


def _load_module(module_path: Path) -> ModuleType:
    if not module_path.exists():
        raise FileNotFoundError(f"Entrypoint module not found: {module_path}")

    module_name = f"qubots_component_{module_path.stem}_{abs(hash(module_path))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to build import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect_default_parameters(manifest: Manifest) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for name, config in manifest.parameters.items():
        if isinstance(config, dict) and "default" in config:
            defaults[name] = config["default"]
    return defaults


def load_component(repo_path: str | Path, expected_type: str) -> Any:
    repo = _ensure_local_path(repo_path)
    manifest = load_manifest(repo)

    if manifest.type != expected_type:
        raise ValueError(
            f"Component type mismatch: expected '{expected_type}', got '{manifest.type}'"
        )

    module_file, class_name = _parse_entrypoint(manifest.entrypoint)
    module = _load_module(repo / module_file)

    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in module '{module_file}'")

    cls = getattr(module, class_name)
    instance = cls()
    setattr(instance, "_qubots_repo_path", str(repo))
    setattr(instance, "_qubots_manifest", manifest)

    defaults = _collect_default_parameters(manifest)
    if defaults and hasattr(instance, "set_parameters"):
        instance.set_parameters(**defaults)

    return instance
