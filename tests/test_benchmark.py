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
