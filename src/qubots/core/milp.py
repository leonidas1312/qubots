"""MILP model representation and capability protocol.

Problems that can be expressed as a (mixed-integer) linear program may
implement ``as_milp()`` to expose structure to solvers like HiGHS, OR-Tools,
SCIP, Gurobi, etc. Blackbox optimizers (random search, hill climb, SA) keep
working through ``evaluate()`` on the same problem object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class MILPModel:
    """Standard-form (mixed-integer) linear program.

    Constraints are expressed as either inequality rows ``A_ub @ x <= b_ub``
    or equality rows ``A_eq @ x == b_eq``. Variable bounds are per-column
    via ``lb`` / ``ub``; use ``-math.inf`` / ``math.inf`` for unbounded sides.
    Integer / binary variables are indicated via ``integrality[i] = True``;
    binary variables additionally have ``lb=0, ub=1``.
    """

    sense: str  # "min" or "max"
    c: list[float]
    var_names: list[str] = field(default_factory=list)
    integrality: list[bool] = field(default_factory=list)
    lb: list[float] = field(default_factory=list)
    ub: list[float] = field(default_factory=list)
    A_ub: list[list[float]] = field(default_factory=list)
    b_ub: list[float] = field(default_factory=list)
    A_eq: list[list[float]] = field(default_factory=list)
    b_eq: list[float] = field(default_factory=list)
    constraint_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.sense not in ("min", "max"):
            raise ValueError(f"sense must be 'min' or 'max', got {self.sense!r}")

        n = len(self.c)
        if not self.var_names:
            self.var_names = [f"x{i}" for i in range(n)]
        if not self.integrality:
            self.integrality = [False] * n
        if not self.lb:
            self.lb = [-math.inf] * n
        if not self.ub:
            self.ub = [math.inf] * n

        for label, vec in (("var_names", self.var_names), ("integrality", self.integrality), ("lb", self.lb), ("ub", self.ub)):
            if len(vec) != n:
                raise ValueError(f"{label} length {len(vec)} does not match c length {n}")

        if len(self.A_ub) != len(self.b_ub):
            raise ValueError("A_ub and b_ub must have the same number of rows")
        for row in self.A_ub:
            if len(row) != n:
                raise ValueError(f"A_ub row width {len(row)} does not match n_vars {n}")

        if len(self.A_eq) != len(self.b_eq):
            raise ValueError("A_eq and b_eq must have the same number of rows")
        for row in self.A_eq:
            if len(row) != n:
                raise ValueError(f"A_eq row width {len(row)} does not match n_vars {n}")

    @property
    def n_vars(self) -> int:
        return len(self.c)

    @property
    def n_constraints(self) -> int:
        return len(self.A_ub) + len(self.A_eq)

    def evaluate(self, solution: list[float]) -> float:
        """Compute the objective value of an assignment (no feasibility check)."""
        if len(solution) != self.n_vars:
            raise ValueError(
                f"solution length {len(solution)} does not match n_vars {self.n_vars}"
            )
        return float(sum(ci * xi for ci, xi in zip(self.c, solution)))

    def is_feasible(self, solution: list[float], tol: float = 1e-6) -> bool:
        if len(solution) != self.n_vars:
            return False
        for i, x in enumerate(solution):
            if x < self.lb[i] - tol or x > self.ub[i] + tol:
                return False
            if self.integrality[i] and abs(x - round(x)) > tol:
                return False
        for row, b in zip(self.A_ub, self.b_ub):
            if sum(a * x for a, x in zip(row, solution)) > b + tol:
                return False
        for row, b in zip(self.A_eq, self.b_eq):
            if abs(sum(a * x for a, x in zip(row, solution)) - b) > tol:
                return False
        return True


@runtime_checkable
class SupportsMILP(Protocol):
    """Problems that can be expressed as a (mixed-integer) linear program."""

    def as_milp(self) -> MILPModel: ...
