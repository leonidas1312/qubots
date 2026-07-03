"""Rastion-facing publish readiness checks."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Any

import yaml

from qubots.detect.detectors import data_hash
from qubots.detect.models import ProblemSpec
from qubots.validate.validate import validate_repo


def _package_version() -> str:
    try:
        return pkg_version("qubots")
    except PackageNotFoundError:
        return "0.0.0"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return raw


def publish_check(repo_path: str | Path) -> dict[str, Any]:
    repo = Path(repo_path).expanduser().resolve()
    issues = validate_repo(repo)
    warnings: list[str] = []
    metadata: dict[str, Any] = {}

    manifest_path = repo / "qubots.yaml"
    if manifest_path.exists():
        try:
            manifest = _read_yaml(manifest_path)
            metadata["manifest"] = {
                "schema_version": manifest.get("qubots_schema_version", 1),
                "type": manifest.get("type"),
                "name": manifest.get("name"),
                "problem_family": manifest.get("problem_family"),
                "capabilities": manifest.get("capabilities") or [],
            }
            if manifest.get("qubots_schema_version") != 3:
                warnings.append("Manifest is not schema v3.")
            for key in ("capabilities", "problem_family", "data_schema", "metrics"):
                if key not in manifest:
                    warnings.append(f"Manifest missing optional v3 field: {key}")
            if "license" not in manifest or manifest.get("license") in {None, ""}:
                warnings.append("License is not declared.")
            if "citation" not in manifest or manifest.get("citation") in {None, ""}:
                warnings.append("Citation is not declared.")
        except Exception as exc:
            issues.append(f"Failed to inspect manifest metadata: {exc}")
    else:
        issues.append(f"Manifest not found: {manifest_path}")

    spec_path = repo / "problem_spec.yaml"
    if spec_path.exists():
        try:
            spec = ProblemSpec.from_yaml(spec_path)
            source = Path(spec.source_path)
            if not source.is_absolute():
                source = repo / source
            metadata["problem_spec"] = spec.to_dict()
            if not source.exists():
                issues.append(f"ProblemSpec source data not found: {source}")
            else:
                actual_hash = data_hash(source)
                if spec.data_hash and actual_hash != spec.data_hash:
                    issues.append(
                        "ProblemSpec data_hash mismatch: "
                        f"expected {spec.data_hash}, got {actual_hash}"
                    )
        except Exception as exc:
            issues.append(f"Failed to inspect problem_spec.yaml: {exc}")
    else:
        warnings.append("No problem_spec.yaml found; imported-data metadata unavailable.")

    status = "ok" if not issues else "fail"
    return {
        "artifact_type": "qubots.publish_check",
        "qubots_version": _package_version(),
        "repo": str(repo),
        "status": status,
        "rastion_ready": status == "ok",
        "issues": issues,
        "warnings": warnings,
        "metadata": metadata,
    }


def write_publish_check(report: dict[str, Any], out_path: str | Path) -> Path:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    return output
