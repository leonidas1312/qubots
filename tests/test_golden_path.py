from pathlib import Path

from qubots import AutoOptimizer, AutoProblem


ROOT = Path(__file__).resolve().parents[1]


def test_golden_path_end_to_end() -> None:
    problem = AutoProblem.from_repo(ROOT / "examples" / "one_max_problem")
    optimizer = AutoOptimizer.from_repo(ROOT / "examples" / "random_search_optimizer")

    problem.set_parameters(n_bits=16)
    optimizer.set_parameters(iterations=50)

    result = optimizer.optimize(problem)

    assert result.best_solution is not None
    assert isinstance(result.best_value, float)
