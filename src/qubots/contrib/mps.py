"""Read industry-standard MPS / LP files into a qubots MILPModel.

Uses HiGHS's built-in MPS reader (via ``highspy``) and converts the resulting
``HighsLp`` into our ``MILPModel``. Range constraints (``rl <= a*x <= ru``
where ``rl != ru``) are encoded as paired ``A_ub`` rows.

The resulting MILPModel is **dense**, so very large MIPLIB instances
(millions of nonzeros) will exhaust memory. Use this for small-to-medium
benchmarks (≲ 5000 vars × 5000 rows). A future sparse representation will
lift this limit.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from qubots.core.milp import MILPModel
from qubots.core.problem import BaseProblem


def _import_highspy() -> Any:
    try:
        import highspy
    except ImportError as exc:
        raise ImportError(
            "Reading MPS files requires the 'highspy' package. "
            "Install with: pip install qubots[highs]"
        ) from exc
    return highspy


def _to_finite(value: float, infinity: float) -> float:
    if value <= -infinity:
        return -math.inf
    if value >= infinity:
        return math.inf
    return float(value)


def _build_dense_rows(matrix: Any, n_row: int, n_col: int, highspy: Any) -> list[list[float]]:
    starts = list(matrix.start_)
    indices = list(matrix.index_)
    values = list(matrix.value_)

    rows: list[list[float]] = [[0.0] * n_col for _ in range(n_row)]
    if matrix.format_ == highspy.MatrixFormat.kRowwise:
        for r in range(n_row):
            for k in range(starts[r], starts[r + 1]):
                rows[r][indices[k]] = float(values[k])
    else:  # column-wise
        for c in range(n_col):
            for k in range(starts[c], starts[c + 1]):
                rows[indices[k]][c] = float(values[k])
    return rows


def read_mps(path: str | Path) -> MILPModel:
    """Parse an MPS or LP file and return a :class:`MILPModel`."""
    highspy = _import_highspy()

    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"MPS file not found: {file_path}")

    h = highspy.Highs()
    h.silent()
    status = h.readModel(str(file_path))
    if status != highspy.HighsStatus.kOk:
        raise ValueError(f"HiGHS failed to read MPS file: {file_path}")

    lp = h.getLp()
    n_col = int(lp.num_col_)
    n_row = int(lp.num_row_)

    sense = "min" if lp.sense_ == highspy.ObjSense.kMinimize else "max"
    c = [float(v) for v in lp.col_cost_]

    inf = float(highspy.kHighsInf)
    lb = [_to_finite(v, inf) for v in lp.col_lower_]
    ub = [_to_finite(v, inf) for v in lp.col_upper_]

    integrality_raw = list(lp.integrality_) if lp.integrality_ else []
    if integrality_raw:
        integrality = [v == highspy.HighsVarType.kInteger for v in integrality_raw]
    else:
        integrality = [False] * n_col

    var_names_raw = list(lp.col_names_) if lp.col_names_ else []
    var_names = (
        [str(n) for n in var_names_raw]
        if len(var_names_raw) == n_col
        else [f"x{i}" for i in range(n_col)]
    )

    row_names_raw = list(lp.row_names_) if lp.row_names_ else []
    row_names = (
        [str(n) for n in row_names_raw]
        if len(row_names_raw) == n_row
        else [f"r{i}" for i in range(n_row)]
    )

    rows = _build_dense_rows(lp.a_matrix_, n_row, n_col, highspy)

    A_ub: list[list[float]] = []
    b_ub: list[float] = []
    A_eq: list[list[float]] = []
    b_eq: list[float] = []
    constraint_names: list[str] = []

    for r in range(n_row):
        rl = float(lp.row_lower_[r])
        ru = float(lp.row_upper_[r])
        row = rows[r]
        name = row_names[r]

        if rl == ru:
            A_eq.append(row)
            b_eq.append(rl)
            constraint_names.append(name)
            continue

        if ru < inf:
            A_ub.append(list(row))
            b_ub.append(ru)
            constraint_names.append(name + ("<=" if rl > -inf else ""))
        if rl > -inf:
            # >= rl  ⇔  -row · x <= -rl
            A_ub.append([-v for v in row])
            b_ub.append(-rl)
            constraint_names.append(name + (">=" if ru < inf else ""))

    return MILPModel(
        sense=sense,
        c=c,
        var_names=var_names,
        integrality=integrality,
        lb=lb,
        ub=ub,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        constraint_names=constraint_names,
    )


class MPSProblem(BaseProblem):
    """qubots problem that wraps any MPS/LP file as a :class:`MILPModel`.

    Set ``mps_path`` either at construction (for in-process use) or via
    ``set_parameters`` / a manifest ``parameters.mps_path.default``. The MPS is
    parsed lazily on first call to ``as_milp()`` and cached.
    """

    def __init__(self, mps_path: str | Path | None = None) -> None:
        super().__init__()
        self.mps_path: str | None = str(mps_path) if mps_path is not None else None
        self._cached_milp: MILPModel | None = None
        self._cached_for_path: str | None = None

    def as_milp(self) -> MILPModel:
        if self.mps_path is None:
            raise ValueError(
                "MPSProblem requires mps_path. Set it via set_parameters(mps_path=...) "
                "or in the manifest as parameters.mps_path.default."
            )
        path_str = str(Path(self.mps_path).expanduser().resolve())
        if self._cached_milp is None or self._cached_for_path != path_str:
            self._cached_milp = read_mps(path_str)
            self._cached_for_path = path_str
        return self._cached_milp
