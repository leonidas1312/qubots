"""Deterministic data/file detectors for v3 problem import."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import yaml

from qubots.detect.models import ProblemCard, ProblemDetection, ProblemSpec


DetectorFn = Callable[[Path], list[ProblemDetection]]


def data_hash(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    if target.is_file():
        digest.update(target.read_bytes())
        return f"sha256:{digest.hexdigest()}"

    if target.is_dir():
        for child in sorted(p for p in target.rglob("*") if p.is_file()):
            digest.update(str(child.relative_to(target)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(child.read_bytes())
            digest.update(b"\0")
        return f"sha256:{digest.hexdigest()}"

    raise FileNotFoundError(f"Data path not found: {target}")


def _read_text_sample(path: Path, limit: int = 65536) -> str:
    with path.open("rb") as f:
        raw = f.read(limit)
    return raw.decode("utf-8", errors="ignore")


def _is_number(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [
            [cell.strip() for cell in row]
            for row in csv.reader(f)
            if any(cell.strip() for cell in row)
        ]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _manifest_detector(path: Path) -> list[ProblemDetection]:
    candidates: list[ProblemDetection] = []
    manifest_path = path / "qubots.yaml" if path.is_dir() else path
    if manifest_path.name == "qubots.yaml" and manifest_path.is_file():
        try:
            raw = _load_yaml(manifest_path) or {}
        except Exception:
            raw = {}
        if isinstance(raw, dict) and {"type", "name", "entrypoint"}.issubset(raw):
            candidates.append(
                ProblemDetection(
                    family="qubots_manifest",
                    confidence=1.0,
                    detector="manifest",
                    evidence=["Found qubots.yaml with type/name/entrypoint."],
                    parameters={"repo_path": str(manifest_path.parent)},
                    path=str(path),
                )
            )
        return candidates

    if path.suffix.lower() not in {".yaml", ".yml"} or not path.is_file():
        return candidates

    try:
        raw = _load_yaml(path) or {}
    except Exception:
        return candidates

    if not isinstance(raw, dict):
        return candidates

    if {"type", "name", "entrypoint"}.issubset(raw):
        candidates.append(
            ProblemDetection(
                family="qubots_manifest",
                confidence=1.0,
                detector="manifest",
                evidence=["YAML contains qubots component manifest keys."],
                parameters={"repo_path": str(path.parent)},
                path=str(path),
            )
        )
    elif "instances" in raw and ("problem" in raw or isinstance(raw.get("instances"), list)):
        candidates.append(
            ProblemDetection(
                family="qubots_dataset",
                confidence=0.98,
                detector="dataset_manifest",
                evidence=["YAML looks like a qubots benchmark dataset."],
                warnings=[
                    "Dataset YAML describes multiple instances; use qubots benchmark "
                    "or import an individual source file."
                ],
                parameters={"dataset_path": str(path)},
                path=str(path),
            )
        )
    return candidates


def _mps_lp_detector(path: Path) -> list[ProblemDetection]:
    if not path.is_file():
        return []
    suffix = path.suffix.lower()
    sample = _read_text_sample(path).upper()
    evidence: list[str] = []
    if suffix in {".mps", ".lp"}:
        evidence.append(f"File extension is {suffix}.")
    if "ROWS" in sample and "COLUMNS" in sample and "RHS" in sample:
        evidence.append("Text contains MPS sections ROWS/COLUMNS/RHS.")
    if "\\ BOUNDS" in sample or "MINIMIZE" in sample or "MAXIMIZE" in sample:
        evidence.append("Text contains LP-style optimization keywords.")

    if not evidence:
        return []

    confidence = 0.96 if suffix in {".mps", ".lp"} else 0.82
    return [
        ProblemDetection(
            family="milp",
            confidence=confidence,
            detector="mps_lp",
            evidence=evidence,
            required_files=[path.name],
            parameters={"mps_path": str(path), "sparse": True},
            path=str(path),
        )
    ]


def _tsplib_detector(path: Path) -> list[ProblemDetection]:
    if not path.is_file():
        return []
    sample = _read_text_sample(path)
    upper = sample.upper()
    suffix = path.suffix.lower()
    if suffix != ".tsp" and "NODE_COORD_SECTION" not in upper:
        return []

    evidence: list[str] = []
    if suffix == ".tsp":
        evidence.append("File extension is .tsp.")
    if "NODE_COORD_SECTION" in upper:
        evidence.append("Found TSPLIB NODE_COORD_SECTION.")
    if "DIMENSION" in upper:
        evidence.append("Found TSPLIB DIMENSION header.")

    if not evidence:
        return []

    return [
        ProblemDetection(
            family="tsp",
            confidence=0.93,
            detector="tsplib",
            evidence=evidence,
            required_files=[path.name],
            parameters={"tsp_path": str(path)},
            path=str(path),
        )
    ]


def _edge_list_rows(path: Path) -> tuple[int, int, bool] | None:
    rows: list[tuple[str, ...]] = []
    try:
        if path.suffix.lower() == ".csv":
            raw_rows = _read_csv_rows(path)
            if not raw_rows:
                return None
            first = [cell.lower() for cell in raw_rows[0]]
            start = 1 if {"source", "target"}.issubset(set(first)) else 0
            rows = [tuple(row) for row in raw_rows[start:]]
        else:
            for line in _read_text_sample(path).splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", "%", "//")):
                    continue
                parts = tuple(stripped.split())
                if parts[0].lower() in {"c", "p"}:
                    continue
                if parts[0].lower() == "e":
                    parts = parts[1:]
                rows.append(parts)
    except UnicodeDecodeError:
        return None

    nodes: set[str] = set()
    edge_count = 0
    weighted = False
    for row in rows:
        if len(row) not in {2, 3}:
            return None
        u, v = row[0], row[1]
        if not (_is_number(u) or u) or not (_is_number(v) or v):
            return None
        if len(row) == 3:
            if not _is_number(row[2]):
                return None
            weighted = True
        nodes.update((u, v))
        edge_count += 1

    if edge_count == 0:
        return None
    return len(nodes), edge_count, weighted


def _edge_list_detector(path: Path) -> list[ProblemDetection]:
    if not path.is_file() or path.suffix.lower() not in {
        ".txt",
        ".edges",
        ".edgelist",
        ".gph",
        ".graph",
        ".csv",
    }:
        return []

    parsed = _edge_list_rows(path)
    if parsed is None:
        return []
    n_nodes, n_edges, weighted = parsed
    confidence = 0.86 if path.suffix.lower() in {".edges", ".edgelist", ".gph"} else 0.78
    evidence = [
        f"Parsed {n_edges} edge rows over {n_nodes} node labels.",
        "Rows have source/target[/weight] shape.",
    ]
    if weighted:
        evidence.append("Detected edge weights.")
    return [
        ProblemDetection(
            family="maxcut",
            confidence=confidence,
            detector="edge_list",
            evidence=evidence,
            required_files=[path.name],
            parameters={
                "edge_path": str(path),
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "weighted": weighted,
            },
            path=str(path),
        )
    ]


def _matrix_from_csv(path: Path) -> list[list[float]] | None:
    rows = _read_csv_rows(path)
    if not rows:
        return None
    parsed: list[list[float]] = []
    for row in rows:
        if not row or not all(_is_number(cell) for cell in row):
            return None
        parsed.append([float(cell) for cell in row])
    width = len(parsed[0])
    if width == 0 or any(len(row) != width for row in parsed):
        return None
    return parsed


def _matrix_from_json(path: Path) -> list[list[float]] | None:
    raw = _load_json(path)
    if isinstance(raw, dict):
        raw = raw.get("cost_matrix") or raw.get("matrix") or raw.get("costs")
    if not isinstance(raw, list) or not raw:
        return None
    matrix: list[list[float]] = []
    for row in raw:
        if not isinstance(row, list) or not row:
            return None
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in row):
            return None
        matrix.append([float(value) for value in row])
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        return None
    return matrix


def _cost_matrix_detector(path: Path) -> list[ProblemDetection]:
    if not path.is_file() or path.suffix.lower() not in {".csv", ".json"}:
        return []
    try:
        matrix = _matrix_from_csv(path) if path.suffix.lower() == ".csv" else _matrix_from_json(path)
    except Exception:
        return []
    if matrix is None:
        return []

    rows = len(matrix)
    cols = len(matrix[0])
    evidence = [f"Parsed numeric {rows}x{cols} cost matrix."]
    if rows == cols:
        evidence.append("Matrix is square, a common assignment format.")

    return [
        ProblemDetection(
            family="assignment",
            confidence=0.87,
            detector="cost_matrix",
            evidence=evidence,
            required_files=[path.name],
            parameters={"matrix_path": str(path), "n_workers": rows, "n_tasks": cols},
            path=str(path),
        )
    ]


def _knapsack_detector(path: Path) -> list[ProblemDetection]:
    if not path.is_file() or path.suffix.lower() not in {".csv", ".json"}:
        return []

    try:
        if path.suffix.lower() == ".json":
            raw = _load_json(path)
            if not isinstance(raw, dict):
                return []
            capacity = raw.get("capacity")
            items = raw.get("items")
            if not isinstance(items, list) or capacity is None:
                return []
            valid_items = [
                item
                for item in items
                if isinstance(item, dict) and "value" in item and "weight" in item
            ]
            if len(valid_items) != len(items) or not valid_items:
                return []
            n_items = len(valid_items)
        else:
            rows = _read_csv_rows(path)
            if len(rows) < 2:
                return []
            header = [cell.lower().strip() for cell in rows[0]]
            header_set = set(header)
            value_col = "value" if "value" in header_set else "profit"
            weight_col = "weight" if "weight" in header_set else "weights"
            if value_col not in header_set or weight_col not in header_set:
                return []
            if "capacity" not in header_set:
                return []
            value_idx = header.index(value_col)
            weight_idx = header.index(weight_col)
            capacity_idx = header.index("capacity")
            n_items = 0
            for row in rows[1:]:
                if len(row) <= max(value_idx, weight_idx, capacity_idx):
                    return []
                if not (_is_number(row[value_idx]) and _is_number(row[weight_idx])):
                    return []
                if not _is_number(row[capacity_idx]):
                    return []
                n_items += 1
            if n_items == 0:
                return []
    except Exception:
        return []

    return [
        ProblemDetection(
            family="knapsack",
            confidence=0.91,
            detector="item_table",
            evidence=[f"Found value/weight/capacity item table with {n_items} items."],
            required_columns=["value", "weight", "capacity"],
            required_files=[path.name],
            parameters={"item_path": str(path), "n_items": n_items},
            path=str(path),
        )
    ]


DETECTORS: dict[str, DetectorFn] = {
    "manifest": _manifest_detector,
    "dataset_manifest": _manifest_detector,
    "mps_lp": _mps_lp_detector,
    "tsplib": _tsplib_detector,
    "edge_list": _edge_list_detector,
    "cost_matrix": _cost_matrix_detector,
    "item_table": _knapsack_detector,
}

DETECTOR_ORDER = [
    "manifest",
    "mps_lp",
    "tsplib",
    "edge_list",
    "cost_matrix",
    "item_table",
]


def _family_matches(candidate: str, requested: str | None) -> bool:
    if requested is None:
        return True
    normalized = requested.strip().lower().replace("-", "_")
    aliases = {
        "mps": "milp",
        "lp": "milp",
        "mps_lp": "milp",
        "graph": "maxcut",
        "cost_matrix": "assignment",
        "item_table": "knapsack",
    }
    normalized = aliases.get(normalized, normalized)
    return candidate == normalized


def detect(
    path: str | Path,
    *,
    family: str | None = None,
    detector: str | None = None,
) -> list[ProblemDetection]:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Detection path not found: {target}")

    names = [detector] if detector is not None else DETECTOR_ORDER
    detections: list[ProblemDetection] = []
    for name in names:
        if name not in DETECTORS:
            raise ValueError(
                f"Unknown detector {name!r}; available detectors: {', '.join(DETECTORS)}"
            )
        for detection in DETECTORS[name](target):
            detection.path = detection.path or str(target)
            if _family_matches(detection.family, family):
                detections.append(detection)

    detections.sort(key=lambda item: (-item.confidence, item.family, item.detector))
    if len(detections) > 1:
        alternatives = ", ".join(
            f"{item.family}:{item.confidence:.2f}" for item in detections[1:]
        )
        detections[0].warnings.append(f"Ambiguous input; also matched {alternatives}.")
    return detections


def select_detection(
    path: str | Path,
    *,
    family: str | None = None,
    detector: str | None = None,
    confidence_threshold: float = 0.75,
) -> ProblemDetection:
    detections = detect(path, family=family, detector=detector)
    if not detections:
        raise ValueError(f"No supported optimization problem detected at {path}")
    best = detections[0]
    if best.confidence < confidence_threshold:
        raise ValueError(
            f"Best detection confidence {best.confidence:.2f} is below threshold "
            f"{confidence_threshold:.2f}: {best.family}"
        )
    return best


def build_problem_spec(
    source_path: str | Path,
    detection: ProblemDetection,
    *,
    stored_source_path: str | None = None,
) -> ProblemSpec:
    source = Path(source_path).expanduser().resolve()
    source_format = source.suffix.lower().lstrip(".") or ("directory" if source.is_dir() else "")
    return ProblemSpec(
        family=detection.family,
        detector=detection.detector,
        source_path=stored_source_path or str(source),
        source_format=source_format,
        data_hash=data_hash(source),
        parameters=dict(detection.parameters),
        detection=detection.to_dict(),
    )


def build_problem_card(name: str, spec: ProblemSpec, detection: ProblemDetection) -> ProblemCard:
    family = spec.family
    domain = {
        "milp": "mathematical-programming",
        "tsp": "routing",
        "maxcut": "graph-optimization",
        "assignment": "assignment",
        "knapsack": "packing",
    }.get(family, "optimization")
    objective_sense = {
        "milp": "source-defined",
        "tsp": "min",
        "assignment": "min",
        "maxcut": "max",
        "knapsack": "max",
    }.get(family, "unknown")
    metrics = {
        "tsp": ["tour_length"],
        "assignment": ["total_cost"],
        "maxcut": ["cut_weight"],
        "knapsack": ["total_value", "total_weight"],
        "milp": ["objective", "feasibility"],
    }.get(family, ["objective"])

    variables: dict[str, Any] = {}
    constraints: dict[str, Any] = {}
    params = spec.parameters
    if family == "assignment":
        variables = {"workers": params.get("n_workers"), "tasks": params.get("n_tasks")}
        constraints = {"assignment": "each worker and task used at most once"}
    elif family == "maxcut":
        variables = {"nodes": params.get("n_nodes")}
        constraints = {"edges": params.get("n_edges")}
    elif family == "knapsack":
        variables = {"items": params.get("n_items")}
        constraints = {"capacity": "single knapsack capacity"}
    elif family == "tsp":
        variables = {"nodes": params.get("dimension")}
        constraints = {"tour": "Hamiltonian cycle"}

    return ProblemCard(
        name=name,
        family=family,
        domain=domain,
        objective_sense=objective_sense,
        variables=variables,
        constraints=constraints,
        data_files=[spec.source_path],
        metrics=metrics,
        validation_status="unchecked",
        warnings=list(detection.warnings),
    )
