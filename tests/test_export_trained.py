from pathlib import Path

from qubots import AutoOptimizer, AutoProblem
from qubots.cli.app import app
from typer.testing import CliRunner


ROOT = Path(__file__).resolve().parents[1]


def test_export_trained_roundtrip(tmp_path: Path) -> None:
    base_optimizer = AutoOptimizer.from_repo(ROOT / "examples" / "random_search_optimizer")
    tune_run = base_optimizer.finetune(
        problem_repo=ROOT / "examples" / "one_max_problem",
        dataset_path=ROOT / "examples" / "one_max_dataset" / "dataset.yaml",
        budget=2,
        metric="mean_best_value",
        out_dir=tmp_path / "trained_run",
        seed=123,
    )

    exported_repo = tmp_path / "exported_opt"
    runner = CliRunner()
    cli_result = runner.invoke(
        app,
        [
            "export-trained",
            "--trained",
            str(tune_run.artifact_path),
            "--out",
            str(exported_repo),
            "--name",
            "exported-random-search",
        ],
    )

    assert cli_result.exit_code == 0
    assert (exported_repo / "qubots.yaml").exists()
    assert (exported_repo / "qubot.py").exists()
    assert (exported_repo / "trained.json").exists()
    assert (exported_repo / "README.md").exists()

    exported_optimizer = AutoOptimizer.from_repo(exported_repo)
    problem = AutoProblem.from_repo(ROOT / "examples" / "one_max_problem")
    problem.set_parameters(n_bits=16)

    result = exported_optimizer.optimize(problem)
    assert result.best_solution is not None
    assert isinstance(result.best_value, float)
