from pathlib import Path

import pytest

from qubots import AutoOptimizer, AutoProblem, MILPModel


ROOT = Path(__file__).resolve().parents[1]
HIGHS_OPTIMIZER = ROOT / "examples" / "highs_optimizer"
KNAPSACK_MILP = ROOT / "examples" / "knapsack_milp_problem"


highspy = pytest.importorskip("highspy")


def test_highs_solves_small_knapsack_to_optimality() -> None:
    problem = AutoProblem.from_repo(KNAPSACK_MILP)
    problem.set_parameters(n_items=10, capacity_ratio=0.4, seed=42)

    optimizer = AutoOptimizer.from_repo(HIGHS_OPTIMIZER)
    result = optimizer.optimize(problem)

    assert result.status == "ok"
    assert result.best_solution is not None
    assert len(result.best_solution) == 10
    for x in result.best_solution:
        assert x in (0.0, 1.0)

    milp = problem.as_milp()
    assert milp.is_feasible(result.best_solution)

    # qubots convention: lower is better; for max problems best_value = -objective.
    assert result.best_value == pytest.approx(-milp.evaluate(result.best_solution))


def test_highs_beats_random_search_on_same_instance() -> None:
    problem_milp = AutoProblem.from_repo(KNAPSACK_MILP)
    problem_milp.set_parameters(n_items=20, capacity_ratio=0.4, seed=7)

    problem_blackbox = AutoProblem.from_repo(KNAPSACK_MILP)
    problem_blackbox.set_parameters(n_items=20, capacity_ratio=0.4, seed=7)

    highs = AutoOptimizer.from_repo(HIGHS_OPTIMIZER)
    rs = AutoOptimizer.from_repo(ROOT / "examples" / "random_search_optimizer")
    rs.set_parameters(iterations=50)

    highs_result = highs.optimize(problem_milp)
    rs_result = rs.optimize(problem_blackbox)

    # HiGHS reaches optimum; random search at 50 iters generally won't.
    assert highs_result.status == "ok"
    assert highs_result.best_value <= rs_result.best_value + 1e-6


def test_highs_handles_infeasible_problem() -> None:
    optimizer = AutoOptimizer.from_repo(HIGHS_OPTIMIZER)
    infeasible = MILPModel(
        sense="min",
        c=[1.0],
        lb=[0.0],
        ub=[1.0],
        A_ub=[[1.0]],
        b_ub=[-1.0],
    )
    result = optimizer.optimize(infeasible)
    assert result.status == "infeasible"


def test_highs_optimizer_validates_clean() -> None:
    from qubots.validate.validate import validate_repo

    assert validate_repo(HIGHS_OPTIMIZER) == []
    assert validate_repo(KNAPSACK_MILP) == []
