"""Import detected data into a reusable qubots problem repository."""

from __future__ import annotations

from dataclasses import dataclass
import json
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
import re
import shutil
from typing import Any

import yaml

from qubots.detect.detectors import (
    build_problem_card,
    build_problem_spec,
    select_detection,
)
from qubots.detect.models import ProblemCard, ProblemDetection, ProblemSpec
from qubots.detect.problems import capabilities_for_family
from qubots.validate.validate import validate_repo


@dataclass
class ImportResult:
    path: Path
    detection: ProblemDetection
    spec: ProblemSpec
    card: ProblemCard
    validation_issues: list[str]
    files: list[Path]


def _package_version() -> str:
    try:
        return pkg_version("qubots")
    except PackageNotFoundError:
        return "0.0.0"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip().lower()).strip("_")
    return slug or "imported_problem"


def _prepare_out_dir(out: Path, *, force: bool) -> None:
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)


def _copy_source(source: Path, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / source.name
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(
            source,
            target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
        )
    else:
        shutil.copy2(source, target)
    return target


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _manifest_for_import(
    *,
    name: str,
    spec: ProblemSpec,
    card: ProblemCard,
) -> dict[str, Any]:
    requirements: list[str] = []
    if spec.family == "milp":
        requirements.append("highspy>=1.7")

    manifest: dict[str, Any] = {
        "qubots_schema_version": 3,
        "type": "problem",
        "name": name,
        "entrypoint": "qubot.py:ImportedDataProblem",
        "capabilities": capabilities_for_family(spec.family),
        "problem_family": spec.family,
        "data_schema": {
            "source_format": spec.source_format,
            "detector": spec.detector,
            "data_hash": spec.data_hash,
        },
        "metrics": list(card.metrics),
        "license": None,
        "citation": None,
        "rastion_card": "problem_card.yaml",
        "parameters": {},
    }
    if requirements:
        manifest["requirements"] = requirements
    return manifest


def import_problem(
    source_path: str | Path,
    out: str | Path,
    *,
    family: str | None = None,
    detector: str | None = None,
    confidence_threshold: float = 0.75,
    name: str | None = None,
    force: bool = False,
    copy_data: bool = True,
) -> ImportResult:
    source = Path(source_path).expanduser().resolve()
    out_dir = Path(out).expanduser().resolve()
    detection = select_detection(
        source,
        family=family,
        detector=detector,
        confidence_threshold=confidence_threshold,
    )
    if detection.family in {"qubots_manifest", "qubots_dataset"}:
        raise ValueError(
            f"{detection.family} is already structured metadata and is not "
            "converted into an imported single problem repo."
        )

    component_name = _slugify(name or f"{source.stem}_{detection.family}")
    _prepare_out_dir(out_dir, force=force)

    if copy_data:
        stored_source = _copy_source(source, out_dir / "data")
        spec_source_path = str(stored_source.relative_to(out_dir))
    else:
        stored_source = source
        spec_source_path = str(source)

    spec = build_problem_spec(
        source,
        detection,
        stored_source_path=spec_source_path,
    )
    # Keep the runtime parameters aligned with the stored data path.
    for key in ("mps_path", "tsp_path", "edge_path", "matrix_path", "item_path"):
        if key in spec.parameters:
            spec.parameters[key] = spec_source_path

    card = build_problem_card(component_name, spec, detection)
    files: list[Path] = []

    manifest = _manifest_for_import(name=component_name, spec=spec, card=card)
    manifest_path = out_dir / "qubots.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    files.append(manifest_path)

    spec_path = spec.write_yaml(out_dir / "problem_spec.yaml")
    files.append(spec_path)

    detection_path = _write_json(
        out_dir / "detection.json",
        {
            "artifact_type": "qubots.problem_detection",
            "qubots_version": _package_version(),
            "source_path": str(source),
            "stored_source_path": spec_source_path,
            "data_hash": spec.data_hash,
            "detection": detection.to_dict(),
        },
    )
    files.append(detection_path)

    qubot_path = out_dir / "qubot.py"
    qubot_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "from qubots.detect.problems import ProblemSpecBackedProblem",
                "",
                "",
                "class ImportedDataProblem(ProblemSpecBackedProblem):",
                "    def __init__(self):",
                "        super().__init__(Path(__file__).with_name('problem_spec.yaml'))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    files.append(qubot_path)

    validation_issues = validate_repo(out_dir)
    card.validation_status = "valid" if not validation_issues else "invalid"
    card_path = card.write_yaml(out_dir / "problem_card.yaml")
    files.append(card_path)

    return ImportResult(
        path=out_dir,
        detection=detection,
        spec=spec,
        card=card,
        validation_issues=validation_issues,
        files=files,
    )
