from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qubots import AutoProblem
from qubots.cli.app import app
from qubots.detect import detect, import_problem, publish_check
from qubots.validate.validate import validate_repo


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "detect"


def test_detectors_rank_supported_fixture_families() -> None:
    cases = {
        "tiny.mps": "milp",
        "square.tsp": "tsp",
        "graph.edgelist": "maxcut",
        "assignment.csv": "assignment",
        "knapsack.csv": "knapsack",
    }
    for filename, family in cases.items():
        detections = detect(FIXTURES / filename)
        assert detections, filename
        assert detections[0].family == family
        assert detections[0].confidence >= 0.75
        assert detections[0].evidence


def test_ambiguity_reports_ranked_candidates_and_warning() -> None:
    detections = detect(FIXTURES / "ambiguous.csv")
    families = [item.family for item in detections]
    assert families[:2] == ["assignment", "maxcut"]
    assert any("Ambiguous" in warning for warning in detections[0].warnings)


def test_auto_problem_from_data_loads_runnable_detected_problem() -> None:
    tsp = AutoProblem.from_data(FIXTURES / "square.tsp")
    assert tsp.evaluate([0, 1, 2, 3]) == pytest.approx(4.0)
    assert getattr(tsp, "problem_family") == "tsp"

    knapsack = AutoProblem.from_data(FIXTURES / "knapsack.csv")
    assert knapsack.evaluate([1, 1, 0]) == pytest.approx(-25.0)
    milp = knapsack.as_milp()
    assert milp.sense == "max"
    assert milp.n_vars == 3


def test_import_creates_valid_problem_repo(tmp_path: Path) -> None:
    out = tmp_path / "imported_knapsack"
    result = import_problem(FIXTURES / "knapsack.csv", out=out)
    assert result.validation_issues == []
    assert (out / "qubots.yaml").exists()
    assert (out / "problem_spec.yaml").exists()
    assert (out / "problem_card.yaml").exists()
    assert (out / "detection.json").exists()
    assert validate_repo(out) == []

    problem = AutoProblem.from_repo(out)
    assert problem.evaluate([1, 1, 0]) == pytest.approx(-25.0)
    assert problem.as_milp().n_vars == 3


def test_imported_problem_spec_can_be_benchmarked(tmp_path: Path) -> None:
    out = tmp_path / "imported_assignment"
    import_problem(FIXTURES / "assignment.csv", out=out)

    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        "\n".join(
            [
                f"problem: {out / 'problem_spec.yaml'}",
                "instances:",
                "  - {}",
            ]
        ),
        encoding="utf-8",
    )

    from qubots import benchmark

    report = benchmark(
        problem_repo=None,
        dataset_path=dataset,
        optimizers=[ROOT / "examples" / "random_search_optimizer"],
        repeats=1,
    )
    run = report["results"][0]["runs"][0]
    assert run["artifact_type"] == "qubots.benchmark_run"
    assert run["data_hash"]
    assert run["detector"] == "cost_matrix"
    assert "solver_parameters" in run


def test_publish_check_reports_imported_repo_ready(tmp_path: Path) -> None:
    out = tmp_path / "imported_graph"
    import_problem(FIXTURES / "graph.edgelist", out=out)
    report = publish_check(out)
    assert report["status"] == "ok"
    assert report["rastion_ready"] is True
    assert report["metadata"]["manifest"]["problem_family"] == "maxcut"


def test_detect_import_publish_cli(tmp_path: Path) -> None:
    runner = CliRunner()
    detect_result = runner.invoke(app, ["detect", str(FIXTURES / "square.tsp"), "--json"])
    assert detect_result.exit_code == 0
    payload = json.loads(detect_result.stdout)
    assert payload["detections"][0]["family"] == "tsp"

    out = tmp_path / "cli_import"
    import_result = runner.invoke(
        app,
        ["import", str(FIXTURES / "knapsack.csv"), "--out", str(out)],
    )
    assert import_result.exit_code == 0, import_result.stdout
    assert validate_repo(out) == []

    publish_result = runner.invoke(app, ["publish-check", str(out), "--json"])
    assert publish_result.exit_code == 0
    publish_payload = json.loads(publish_result.stdout)
    assert publish_payload["status"] == "ok"
