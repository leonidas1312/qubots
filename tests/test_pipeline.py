from pathlib import Path

from qubots import AutoOptimizer, pipeline
from qubots.cli.app import app
from typer.testing import CliRunner


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_without_trained() -> None:
    run = pipeline(
        problem=ROOT / "examples" / "one_max_problem",
        optimizer=ROOT / "examples" / "random_search_optimizer",
    )

    result = run(
        problem_params={"n_bits": 16},
        optimizer_params={"iterations": 20},
    )

    assert result.best_solution is not None
    assert isinstance(result.best_value, float)


def test_pipeline_with_trained_artifact(tmp_path: Path) -> None:
    optimizer = AutoOptimizer.from_repo(ROOT / "examples" / "random_search_optimizer")
    tune_run = optimizer.finetune(
        problem_repo=ROOT / "examples" / "one_max_problem",
        dataset_path=ROOT / "examples" / "one_max_dataset" / "dataset.yaml",
        budget=2,
        metric="mean_best_value",
        out_dir=tmp_path / "trained",
        seed=123,
    )

    trained_optimizer = AutoOptimizer.from_trained(tune_run.artifact_path)
    assert hasattr(trained_optimizer, "_qubots_trained_metadata")

    run = pipeline(
        problem=ROOT / "examples" / "one_max_problem",
        optimizer=ROOT / "examples" / "random_search_optimizer",
        trained=tune_run.artifact_path,
    )
    result = run(problem_params={"n_bits": 16})

    assert result.best_solution is not None
    assert isinstance(result.best_value, float)


def test_run_cli_command() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--problem",
            str(ROOT / "examples" / "one_max_problem"),
            "--optimizer",
            str(ROOT / "examples" / "random_search_optimizer"),
            "--problem-params",
            '{"n_bits": 16}',
            "--optimizer-params",
            '{"iterations": 20}',
        ],
    )
    assert result.exit_code == 0
    assert "Best value:" in result.output
