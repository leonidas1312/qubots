"""MILP model representation and capability protocol.

Problems that can be expressed as a (mixed-integer) linear program may
implement ``as_milp()`` to expose structure to solvers like HiGHS, OR-Tools,
SCIP, Gurobi, etc. Blackbox optimizers (random search, hill climb, SA) keep
working through ``evaluate()`` on the same problem object.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


SparseRow = list[tuple[int, float]]


def _normalize_sparse_row(row: Any, n_vars: int) -> SparseRow:
    if isinstance(row, Mapping):
        items = row.items()
    else:
        items = row

    out: dict[int, float] = {}
    for item in items:
        if not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError("Sparse rows must contain (index, value) pairs")
        index_raw, value_raw = item
        if isinstance(index_raw, bool) or not isinstance(index_raw, int):
            raise ValueError(f"Sparse row index must be an int, got {index_raw!r}")
        if index_raw < 0 or index_raw >= n_vars:
            raise ValueError(
                f"Sparse row index {index_raw} is outside variable range 0..{n_vars - 1}"
            )
        value = float(value_raw)
        if value != 0.0:
            out[index_raw] = out.get(index_raw, 0.0) + value

    return sorted((index, value) for index, value in out.items() if value != 0.0)


def sparse_row_dot(row: SparseRow, solution: list[float]) -> float:
    return float(sum(value * solution[index] for index, value in row))


def sparse_row_to_dense(row: SparseRow, n_vars: int) -> list[float]:
    dense = [0.0] * n_vars
    for index, value in row:
        dense[index] = float(value)
    return dense


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


@dataclass
class SparseMILPModel:
    """Sparse mixed-integer linear program.

    Sparse rows are represented as ``[(column_index, coefficient), ...]`` pairs
    or as ``{column_index: coefficient}`` mappings. The rest of the contract is
    identical to :class:`MILPModel`, so dense examples and sparse industrial
    instances can share solver adapters.
    """

    sense: str
    c: list[float]
    var_names: list[str] = field(default_factory=list)
    integrality: list[bool] = field(default_factory=list)
    lb: list[float] = field(default_factory=list)
    ub: list[float] = field(default_factory=list)
    A_ub: list[Any] = field(default_factory=list)
    b_ub: list[float] = field(default_factory=list)
    A_eq: list[Any] = field(default_factory=list)
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

        for label, vec in (
            ("var_names", self.var_names),
            ("integrality", self.integrality),
            ("lb", self.lb),
            ("ub", self.ub),
        ):
            if len(vec) != n:
                raise ValueError(f"{label} length {len(vec)} does not match c length {n}")

        if len(self.A_ub) != len(self.b_ub):
            raise ValueError("A_ub and b_ub must have the same number of rows")
        if len(self.A_eq) != len(self.b_eq):
            raise ValueError("A_eq and b_eq must have the same number of rows")

        self.A_ub = [_normalize_sparse_row(row, n) for row in self.A_ub]
        self.A_eq = [_normalize_sparse_row(row, n) for row in self.A_eq]

    @property
    def n_vars(self) -> int:
        return len(self.c)

    @property
    def n_constraints(self) -> int:
        return len(self.A_ub) + len(self.A_eq)

    @property
    def nnz(self) -> int:
        return sum(len(row) for row in self.A_ub) + sum(len(row) for row in self.A_eq)

    def to_dense(self) -> MILPModel:
        return MILPModel(
            sense=self.sense,
            c=list(self.c),
            var_names=list(self.var_names),
            integrality=list(self.integrality),
            lb=list(self.lb),
            ub=list(self.ub),
            A_ub=[sparse_row_to_dense(row, self.n_vars) for row in self.A_ub],
            b_ub=list(self.b_ub),
            A_eq=[sparse_row_to_dense(row, self.n_vars) for row in self.A_eq],
            b_eq=list(self.b_eq),
            constraint_names=list(self.constraint_names),
        )

    def evaluate(self, solution: list[float]) -> float:
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
            if sparse_row_dot(row, solution) > b + tol:
                return False
        for row, b in zip(self.A_eq, self.b_eq):
            if abs(sparse_row_dot(row, solution) - b) > tol:
                return False
        return True


@runtime_checkable
class SupportsMILP(Protocol):
    """Problems that can be expressed as a (mixed-integer) linear program."""

    def as_milp(self) -> MILPModel | SparseMILPModel: ...
