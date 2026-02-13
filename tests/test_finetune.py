import json
from pathlib import Path

from qubots import AutoOptimizer
from qubots.cli.app import app
from qubots.tune.dataset import load_dataset, load_dataset_spec
from typer.testing import CliRunner


ROOT = Path(__file__).resolve().parents[1]


def test_finetune_creates_artifact(tmp_path: Path) -> None:
    optimizer = AutoOptimizer.from_repo(ROOT / "examples" / "random_search_optimizer")

    run = optimizer.finetune(
        problem_repo=ROOT / "examples" / "one_max_problem",
        dataset_path=ROOT / "examples" / "one_max_dataset" / "dataset.yaml",
        budget=5,
        metric="mean_best_value",
        out_dir=tmp_path,
        seed=123,
    )

    artifact_path = tmp_path / "trained.json"
    assert run.artifact_path == artifact_path
    assert artifact_path.exists()

    with artifact_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data.get("best_params"), dict)
    assert isinstance(data.get("score"), float)


def test_dataset_loader_supports_old_and_header_formats() -> None:
    old_dataset = ROOT / "examples" / "one_max_dataset" / "dataset.yaml"
    old_spec = load_dataset_spec(old_dataset)
    assert old_spec.problem is None
    assert len(old_spec.instances) == 3
    assert len(load_dataset(old_dataset)) == 3

    header_dataset = ROOT / "examples" / "one_max_dataset_header" / "dataset.yaml"
    header_spec = load_dataset_spec(header_dataset)
    assert header_spec.problem == "../one_max_problem"
    assert len(header_spec.instances) == 3


def test_finetune_accepts_problem_from_dataset_header(tmp_path: Path) -> None:
    optimizer = AutoOptimizer.from_repo(ROOT / "examples" / "random_search_optimizer")
    header_dataset = ROOT / "examples" / "one_max_dataset_header" / "dataset.yaml"

    run = optimizer.finetune(
        problem_repo=None,
        dataset_path=header_dataset,
        budget=2,
        metric="mean_best_value",
        out_dir=tmp_path / "out",
        seed=123,
    )

    assert run.artifact_path.exists()


def test_finetune_cli_accepts_dataset_header_problem(tmp_path: Path) -> None:
    out_dir = tmp_path / "trained_output"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "finetune",
            "--optimizer",
            str(ROOT / "examples" / "random_search_optimizer"),
            "--dataset",
            str(ROOT / "examples" / "one_max_dataset_header" / "dataset.yaml"),
            "--budget",
            "2",
            "--out",
            str(out_dir),
            "--seed",
            "123",
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "trained.json").exists()
