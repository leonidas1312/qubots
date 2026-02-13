from pathlib import Path

from qubots import AutoOptimizer, AutoProblem, benchmark
from qubots.benchmark.benchmark import report_to_markdown


ROOT = Path(__file__).resolve().parents[1]


def _problem_settings(problem_name: str) -> dict[str, float | int]:
    if problem_name == "knapsack_problem":
        return {"n_items": 16, "capacity_ratio": 0.4, "seed": 7}
    if problem_name == "maxcut_problem":
        return {"n_nodes": 12, "edge_prob": 0.25, "seed": 7}
    raise ValueError(f"Unknown problem example: {problem_name}")


def _optimizer_settings(optimizer_name: str) -> dict[str, float | int]:
    if optimizer_name == "hill_climb_optimizer":
        return {"steps": 20, "restarts": 1}
    if optimizer_name == "simulated_annealing_optimizer":
        return {"steps": 40, "t0": 1.0, "cooling": 0.98}
    raise ValueError(f"Unknown optimizer example: {optimizer_name}")


def test_new_examples_run_all_pairs() -> None:
    problem_names = ["knapsack_problem", "maxcut_problem"]
    optimizer_names = ["hill_climb_optimizer", "simulated_annealing_optimizer"]

    for problem_name in problem_names:
        for optimizer_name in optimizer_names:
            problem = AutoProblem.from_repo(ROOT / "examples" / problem_name)
            optimizer = AutoOptimizer.from_repo(ROOT / "examples" / optimizer_name)

            problem.set_parameters(**_problem_settings(problem_name))
            optimizer.set_parameters(**_optimizer_settings(optimizer_name))

            result = optimizer.optimize(problem)

            assert result.best_solution is not None
            assert isinstance(result.best_value, float)
            assert result.status == "ok"


def test_benchmark_two_optimizers_on_knapsack_dataset() -> None:
    report = benchmark(
        problem_repo=None,
        dataset_path=ROOT / "examples" / "datasets" / "knapsack_small.yaml",
        optimizers=[
            ROOT / "examples" / "hill_climb_optimizer",
            ROOT / "examples" / "simulated_annealing_optimizer",
        ],
        repeats=1,
        seed=123,
    )

    assert isinstance(report, dict)
    assert len(report.get("results", [])) == 2

    table = report_to_markdown(report)
    assert "hill_climb_optimizer" in table
    assert "simulated_annealing_optimizer" in table
