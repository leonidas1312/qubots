"""Data models for deterministic problem detection and import."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DETECT_SCHEMA_VERSION = 3


@dataclass
class ProblemDetection:
    family: str
    confidence: float
    detector: str
    evidence: list[str] = field(default_factory=list)
    required_columns: list[str] = field(default_factory=list)
    required_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DETECT_SCHEMA_VERSION,
            "family": self.family,
            "confidence": float(self.confidence),
            "detector": self.detector,
            "evidence": list(self.evidence),
            "required_columns": list(self.required_columns),
            "required_files": list(self.required_files),
            "warnings": list(self.warnings),
            "parameters": dict(self.parameters),
            "path": self.path,
        }


@dataclass
class ProblemSpec:
    family: str
    detector: str
    source_path: str
    source_format: str
    data_hash: str
    parameters: dict[str, Any] = field(default_factory=dict)
    detection: dict[str, Any] = field(default_factory=dict)
    schema_version: int = DETECT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "family": self.family,
            "detector": self.detector,
            "source_path": self.source_path,
            "source_format": self.source_format,
            "data_hash": self.data_hash,
            "parameters": dict(self.parameters),
            "detection": dict(self.detection),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ProblemSpec":
        return cls(
            schema_version=int(raw.get("schema_version", DETECT_SCHEMA_VERSION)),
            family=str(raw["family"]),
            detector=str(raw["detector"]),
            source_path=str(raw["source_path"]),
            source_format=str(raw.get("source_format", "")),
            data_hash=str(raw.get("data_hash", "")),
            parameters=dict(raw.get("parameters") or {}),
            detection=dict(raw.get("detection") or {}),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProblemSpec":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("ProblemSpec YAML must contain a mapping")
        return cls.from_dict(raw)

    def write_yaml(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        return output


@dataclass
class ProblemCard:
    name: str
    family: str
    domain: str
    objective_sense: str
    variables: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    data_files: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    license: str | None = None
    citation: str | None = None
    validation_status: str = "unchecked"
    warnings: list[str] = field(default_factory=list)
    schema_version: int = DETECT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "name": self.name,
            "family": self.family,
            "domain": self.domain,
            "objective_sense": self.objective_sense,
            "variables": dict(self.variables),
            "constraints": dict(self.constraints),
            "data_files": list(self.data_files),
            "metrics": list(self.metrics),
            "license": self.license,
            "citation": self.citation,
            "validation_status": self.validation_status,
            "warnings": list(self.warnings),
        }

    def write_yaml(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        return output
