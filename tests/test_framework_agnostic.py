from __future__ import annotations

from pathlib import Path

import pytest

from qubots import AutoOptimizer, AutoProblem, benchmark


ROOT = Path(__file__).resolve().parents[1]


pytest.importorskip("networkx")


def test_networkx_optimizer_runs_on_detected_graph_data() -> None:
    problem = AutoProblem.from_data(
        ROOT / "examples" / "pilots" / "qoblib_karate_edges.gph",
        family="maxcut",
    )
    optimizer = AutoOptimizer.from_repo(
        ROOT / "integrations" / "networkx_maxcut_optimizer"
    )
    optimizer.set_parameters(max_passes=5, seed=7)

    result = optimizer.optimize(problem)

    assert result.status == "ok"
    assert result.best_solution is not None
    assert result.best_value <= 0.0
    assert result.metadata["framework"] == "networkx"
    assert result.metadata["nodes"] == 34
    assert result.metadata["edges"] == 78


def test_networkx_optimizer_benchmarks_imported_problem_spec(tmp_path: Path) -> None:
    from qubots.detect import import_problem

    imported = tmp_path / "karate_maxcut"
    import_problem(
        ROOT / "examples" / "pilots" / "qoblib_karate_edges.gph",
        out=imported,
        family="maxcut",
    )

    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        "\n".join(
            [
                f"problem: {imported / 'problem_spec.yaml'}",
                "instances:",
                "  - {}",
            ]
        ),
        encoding="utf-8",
    )

    report = benchmark(
        problem_repo=None,
        dataset_path=dataset,
        optimizers=[ROOT / "integrations" / "networkx_maxcut_optimizer"],
        repeats=1,
        seed=7,
    )

    run = report["results"][0]["runs"][0]
    assert run["status"] == "ok"
    assert run["detector"] == "edge_list"
    assert run["best_value"] <= 0.0
