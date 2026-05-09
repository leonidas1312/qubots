"""Tests for the CP-SAT optimizer adapter."""

from __future__ import annotations

from pathlib import Path

import pytest


pytest.importorskip("ortools.sat.python")


from qubots import AutoOptimizer, AutoProblem, MILPModel  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CPSAT_OPTIMIZER = ROOT / "examples" / "cpsat_optimizer"
KNAPSACK_MILP = ROOT / "examples" / "knapsack_milp_problem"


def test_cpsat_solves_small_knapsack_to_optimum() -> None:
    problem = AutoProblem.from_repo(KNAPSACK_MILP)
    problem.set_parameters(n_items=10, capacity_ratio=0.4, seed=42)

    optimizer = AutoOptimizer.from_repo(CPSAT_OPTIMIZER)
    result = optimizer.optimize(problem)

    assert result.status in ("ok", "feasible")
    assert result.best_solution is not None
    assert len(result.best_solution) == 10
    for x in result.best_solution:
        assert x in (0.0, 1.0)

    milp = problem.as_milp()
    assert milp.is_feasible(result.best_solution)


def test_cpsat_matches_highs_on_same_instance() -> None:
    pytest.importorskip("highspy")

    problem_a = AutoProblem.from_repo(KNAPSACK_MILP)
    problem_a.set_parameters(n_items=20, capacity_ratio=0.4, seed=7)
    problem_b = AutoProblem.from_repo(KNAPSACK_MILP)
    problem_b.set_parameters(n_items=20, capacity_ratio=0.4, seed=7)

    cpsat = AutoOptimizer.from_repo(CPSAT_OPTIMIZER)
    highs = AutoOptimizer.from_repo(ROOT / "examples" / "highs_optimizer")

    cpsat_result = cpsat.optimize(problem_a)
    highs_result = highs.optimize(problem_b)

    # Both should find the same optimum on this small instance.
    assert cpsat_result.best_value == pytest.approx(highs_result.best_value, abs=1e-6)


def test_cpsat_rejects_continuous_variables() -> None:
    optimizer = AutoOptimizer.from_repo(CPSAT_OPTIMIZER)
    lp = MILPModel(
        sense="min",
        c=[1.0, 1.0],
        integrality=[True, False],
        lb=[0.0, 0.0],
        ub=[10.0, 10.0],
        A_ub=[[1.0, 1.0]],
        b_ub=[5.0],
    )
    result = optimizer.optimize(lp)
    assert result.status == "error"
    assert "continuous" in (result.error or "").lower() or "integer" in (result.error or "").lower()


def test_cpsat_rejects_fractional_coefficients() -> None:
    optimizer = AutoOptimizer.from_repo(CPSAT_OPTIMIZER)
    lp = MILPModel(
        sense="max",
        c=[1.5, 2.0],  # fractional
        integrality=[True, True],
        lb=[0.0, 0.0],
        ub=[1.0, 1.0],
        A_ub=[[1.0, 1.0]],
        b_ub=[1.0],
    )
    result = optimizer.optimize(lp)
    assert result.status == "error"
    assert "integer" in (result.error or "").lower()


def test_cpsat_handles_infeasible_problem() -> None:
    optimizer = AutoOptimizer.from_repo(CPSAT_OPTIMIZER)
    infeasible = MILPModel(
        sense="min",
        c=[1.0],
        integrality=[True],
        lb=[0.0],
        ub=[1.0],
        A_ub=[[1.0]],
        b_ub=[-1.0],
    )
    result = optimizer.optimize(infeasible)
    assert result.status == "infeasible"


def test_cpsat_metadata_includes_solver_stats() -> None:
    problem = AutoProblem.from_repo(KNAPSACK_MILP)
    problem.set_parameters(n_items=10, capacity_ratio=0.4, seed=42)
    optimizer = AutoOptimizer.from_repo(CPSAT_OPTIMIZER)
    result = optimizer.optimize(problem)

    assert result.metadata.get("solver") == "cpsat"
    assert "num_branches" in result.metadata
    assert "num_conflicts" in result.metadata
    assert "wall_time_seconds" in result.metadata


def test_cpsat_validates_clean() -> None:
    from qubots.validate.validate import validate_repo
    assert validate_repo(CPSAT_OPTIMIZER) == []
