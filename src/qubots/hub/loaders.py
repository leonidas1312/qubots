"""Dynamic loading for local component repositories."""

import hashlib
import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Any

from qubots.hub.manifests import Manifest, load_manifest


_ENTRYPOINT_MODULE_RE = re.compile(r"^[A-Za-z0-9_./-]+\.py$")
_CLASS_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ensure_local_path(path: str | Path) -> Path:
    text = str(path)
    if "://" in text:
        raise ValueError("Only local repository paths are supported in v0.1")
    return Path(path).resolve()


def _parse_entrypoint(entrypoint: str) -> tuple[str, str]:
    if ":" not in entrypoint:
        raise ValueError("Entrypoint must be in 'file.py:ClassName' format")
    module_file, class_name = entrypoint.split(":", maxsplit=1)

    module_file = module_file.strip()
    class_name = class_name.strip()

    if not _ENTRYPOINT_MODULE_RE.match(module_file):
        raise ValueError(
            f"Entrypoint module path {module_file!r} contains disallowed "
            "characters; only [A-Za-z0-9_./-] are permitted in '<file>.py'."
        )
    if module_file.startswith("/") or module_file.startswith("\\"):
        raise ValueError(
            f"Entrypoint module path {module_file!r} must be relative to the repo."
        )
    parts = module_file.replace("\\", "/").split("/")
    if any(part == ".." for part in parts):
        raise ValueError(
            f"Entrypoint module path {module_file!r} must not contain '..' segments."
        )

    if not _CLASS_NAME_RE.match(class_name):
        raise ValueError(
            f"Entrypoint class name {class_name!r} is not a valid Python identifier."
        )

    return module_file, class_name


def _ensure_within_repo(repo: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and verify it stays under ``repo``."""
    repo_resolved = repo.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repo_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Entrypoint module path escapes the repository: "
            f"{resolved} is not under {repo_resolved}"
        ) from exc
    return resolved


def _load_module(module_path: Path) -> ModuleType:
    if not module_path.exists():
        raise FileNotFoundError(f"Entrypoint module not found: {module_path}")

    # Use a content-addressed module name so cached imports don't collide
    # across repos. ``abs(hash(...))`` was previously used but Python hash
    # randomization makes it nondeterministic across processes.
    digest = hashlib.sha1(str(module_path).encode("utf-8")).hexdigest()[:12]
    module_name = f"qubots_component_{module_path.stem}_{digest}"
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
    module_path = _ensure_within_repo(repo, repo / module_file)
    module = _load_module(module_path)

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
