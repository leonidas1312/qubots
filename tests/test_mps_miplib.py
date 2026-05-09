"""Tests for MPS reading and the MIPLIB fetch helper."""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("highspy")

from qubots import AutoOptimizer, AutoProblem  # noqa: E402
from qubots.contrib.miplib import fetch_miplib, is_miplib_cached  # noqa: E402
from qubots.contrib.mps import MPSProblem, read_mps  # noqa: E402


# Tiny binary knapsack as MPS:
#   max 10 x1 + 15 x2 + 7 x3
#   s.t.  2 x1 + 3 x2 + x3 <= 5
#         x_i in {0, 1}
# Optimum: x1=x2=1, x3=0  → value = 25, weight = 5
#
# HiGHS minimizes by convention, so the objective coefficients are negated.
TINY_MPS = """NAME          TINYKNAP
ROWS
 N  COST
 L  WEIGHT
COLUMNS
    X1        COST          -10   WEIGHT         2
    X2        COST          -15   WEIGHT         3
    X3        COST           -7   WEIGHT         1
RHS
    RHS       WEIGHT          5
BOUNDS
 BV BND       X1
 BV BND       X2
 BV BND       X3
ENDATA
"""


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_mps_path(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.mps"
    p.write_text(TINY_MPS, encoding="utf-8")
    return p


def test_read_mps_parses_tiny_knapsack(tiny_mps_path: Path) -> None:
    milp = read_mps(tiny_mps_path)
    assert milp.sense == "min"
    assert milp.n_vars == 3
    assert milp.c == [-10.0, -15.0, -7.0]
    assert milp.integrality == [True, True, True]
    assert milp.lb == [0.0, 0.0, 0.0]
    assert milp.ub == [1.0, 1.0, 1.0]
    # One <= constraint: 2 x1 + 3 x2 + x3 <= 5
    assert len(milp.A_ub) == 1
    assert len(milp.A_eq) == 0
    assert milp.A_ub[0] == [2.0, 3.0, 1.0]
    assert milp.b_ub == [5.0]


def test_mps_problem_solves_to_known_optimum(tiny_mps_path: Path) -> None:
    problem = MPSProblem(mps_path=tiny_mps_path)
    solver = AutoOptimizer.from_repo(ROOT / "examples" / "highs_optimizer")
    result = solver.optimize(problem)
    assert result.status == "ok"
    # min sense, optimum = -25
    assert result.best_value == pytest.approx(-25.0)
    sol = [round(v) for v in result.best_solution]
    assert sol == [1, 1, 0]


def test_mps_problem_repo_loads_via_set_parameters(tiny_mps_path: Path) -> None:
    problem = AutoProblem.from_repo(ROOT / "examples" / "mps_problem")
    problem.set_parameters(mps_path=str(tiny_mps_path))
    solver = AutoOptimizer.from_repo(ROOT / "examples" / "highs_optimizer")
    result = solver.optimize(problem)
    assert result.status == "ok"
    assert result.best_value == pytest.approx(-25.0)


def test_mps_problem_caches_per_path(tiny_mps_path: Path, tmp_path: Path) -> None:
    p = MPSProblem(mps_path=tiny_mps_path)
    first = p.as_milp()
    second = p.as_milp()
    assert first is second  # cache hit on identical path

    other = tmp_path / "tiny2.mps"
    other.write_text(TINY_MPS, encoding="utf-8")
    p.set_parameters(mps_path=str(other))
    third = p.as_milp()
    assert third is not first  # cache invalidated on path change


def test_read_mps_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_mps(tmp_path / "does_not_exist.mps")


def test_mps_problem_validates_clean() -> None:
    from qubots.validate.validate import validate_repo

    assert validate_repo(ROOT / "examples" / "mps_problem") == []


# ------------- MIPLIB fetcher (no network in CI) -----------------------------


def _make_fake_urlopen(payload_bytes: bytes):
    """urlopen that returns a context manager streaming a gz payload."""

    class _Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    def fake(url, timeout=None, context=None):  # type: ignore[no-untyped-def]
        return _Resp(payload_bytes)

    return fake


def test_fetch_miplib_decompresses_into_cache(tmp_path: Path) -> None:
    raw = b"NAME tiny\nROWS\n N COST\nCOLUMNS\nRHS\nENDATA\n"
    gz = gzip.compress(raw)

    cache = tmp_path / "miplib_cache"
    assert not is_miplib_cached("tiny", cache_dir=cache)
    with patch(
        "qubots.contrib.miplib.urllib.request.urlopen",
        side_effect=_make_fake_urlopen(gz),
    ):
        path = fetch_miplib("tiny", cache_dir=cache)

    assert path == cache / "tiny.mps"
    assert path.read_bytes() == raw
    assert is_miplib_cached("tiny", cache_dir=cache)
    # gz file is removed after extraction
    assert not (cache / "tiny.mps.gz").exists()


def test_fetch_miplib_reuses_cache(tmp_path: Path) -> None:
    cache = tmp_path / "miplib_cache"
    cache.mkdir(parents=True)
    cached = cache / "already.mps"
    cached.write_text("CACHED CONTENT", encoding="utf-8")

    with patch(
        "qubots.contrib.miplib.urllib.request.urlopen",
        side_effect=AssertionError("urlopen should not be called for cached instance"),
    ):
        path = fetch_miplib("already", cache_dir=cache)

    assert path == cached
    assert path.read_text(encoding="utf-8") == "CACHED CONTENT"


def test_fetch_miplib_rejects_path_traversal(tmp_path: Path) -> None:
    cache = tmp_path / "miplib_cache"
    with pytest.raises(ValueError):
        fetch_miplib("../escape", cache_dir=cache)
    with pytest.raises(ValueError):
        fetch_miplib(".secret", cache_dir=cache)


# ------------- Bundled MIPLIB-style benchmark dataset ------------------------


def test_bundled_tiny_mps_dataset_solves_via_benchmark() -> None:
    from qubots import benchmark

    dataset = ROOT / "examples" / "datasets" / "mps_tiny.yaml"
    report = benchmark(
        problem_repo=None,
        dataset_path=dataset,
        optimizers=[ROOT / "examples" / "highs_optimizer"],
        repeats=1,
    )
    assert len(report["results"]) == 1
    runs = report["results"][0]["runs"]
    assert len(runs) == 1
    assert runs[0]["status"] == "ok"
    # tiny.mps has known optimum -25 (max value 25)
    assert runs[0]["best_value"] == pytest.approx(-25.0)
