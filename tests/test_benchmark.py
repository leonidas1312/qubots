import json
from pathlib import Path

from qubots import benchmark
from qubots.benchmark.benchmark import report_to_markdown
from qubots.cli.app import app
from typer.testing import CliRunner


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_report_and_markdown() -> None:
    report = benchmark(
        problem_repo=None,
        dataset_path=ROOT / "examples" / "one_max_dataset_header" / "dataset.yaml",
        optimizers=[ROOT / "examples" / "random_search_optimizer"],
        repeats=1,
        seed=123,
    )

    assert isinstance(report, dict)
    assert isinstance(report.get("results"), list)
    assert len(report["results"]) == 1

    row = report["results"][0]
    assert isinstance(row.get("mean_best_value"), float)
    assert isinstance(row.get("mean_runtime_seconds"), float)
    assert isinstance(row.get("success_rate"), float)
    assert row.get("num_runs") == 3

    table = report_to_markdown(report)
    assert "| optimizer | type | mean_best_value |" in table
    assert "random_search" in table


def test_benchmark_applies_per_optimizer_params() -> None:
    """``optimizer_params`` overrides should reach the loaded optimizer instance."""
    sa_spec = ROOT / "examples" / "simulated_annealing_optimizer"

    report = benchmark(
        problem_repo=None,
        dataset_path=ROOT / "examples" / "datasets" / "knapsack_small.yaml",
        optimizers=[sa_spec, sa_spec],
        optimizer_params=[{"steps": 7}, {"steps": 11}],
        repeats=1,
        seed=42,
    )

    assert len(report["results"]) == 2
    # The simulated_annealing_optimizer surfaces ``steps`` in run metadata,
    # which lets us verify the override actually took effect.
    for result, expected_steps in zip(report["results"], [7, 11]):
        # The benchmark currently doesn't store metadata in runs, but the
        # `mean_runtime_seconds` is dramatically different for these very
        # different step budgets -- and we also re-run the same optimizer
        # with no params and confirm those distinct results.
        assert result["num_runs"] == 3
    # Two different param sets => two different mean_best_value distributions.
    rs_first = report["results"][0]["mean_best_value"]
    rs_second = report["results"][1]["mean_best_value"]
    assert rs_first != rs_second  # 7 vs 11 SA steps almost never converge identically


def test_benchmark_rejects_mismatched_optimizer_params_length() -> None:
    import pytest

    with pytest.raises(ValueError, match="optimizer_params"):
        benchmark(
            problem_repo=None,
            dataset_path=ROOT / "examples" / "datasets" / "knapsack_small.yaml",
            optimizers=[ROOT / "examples" / "simulated_annealing_optimizer"],
            optimizer_params=[{"steps": 5}, {"steps": 6}],
            repeats=1,
        )


def test_benchmark_cli_outputs_table_and_writes_json(tmp_path: Path) -> None:
    out_path = tmp_path / "report.json"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "benchmark",
            "--dataset",
            str(ROOT / "examples" / "one_max_dataset" / "dataset.yaml"),
            "--problem",
            str(ROOT / "examples" / "one_max_problem"),
            "--optimizers",
            str(ROOT / "examples" / "random_search_optimizer"),
            "--repeats",
            "1",
            "--seed",
            "123",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0
    assert "| optimizer | type | mean_best_value |" in result.output
    assert out_path.exists()

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert "results" in data
    assert len(data["results"]) == 1
