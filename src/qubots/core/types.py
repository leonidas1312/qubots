"""Core result types."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Result:
    best_value: float
    best_solution: Any
    runtime_seconds: float
    status: str = "ok"
    trace: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
