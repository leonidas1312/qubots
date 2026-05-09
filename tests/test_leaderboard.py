"""End-to-end tests for the qubots leaderboard primitive.

Uses the bundled tiny knapsack dataset and the example optimizers that ship
with the repo, so the suite runs in <1s without external network or solvers
beyond what the rest of the test suite already requires.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qubots.cli.app import app
from qubots.leaderboard import (
    BenchmarkResult,
    load_submission,
    load_submissions_from_dir,
    load_suite,
    report_to_markdown,
    run_leaderboard,
    write_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_suite(tmp_path: Path) -> Path:
    suite = tmp_path / "suite.yaml"
    knapsack_dataset = ROOT / "examples" / "datasets" / "knapsack_small.yaml"
    suite.write_text(
        "\n".join(
            [
                "qubots_leaderboard_schema_version: 1",
                "name: tiny-suite",
                "description: small smoke test",
                "benchmarks:",
                "  - name: knapsack_small",
                f"    dataset: {knapsack_dataset}",
            ]
        ),
        encoding="utf-8",
    )
    return suite


def _write_submissions(tmp_path: Path) -> Path:
    sub_dir = tmp_path / "submissions"
    sub_dir.mkdir()

    (sub_dir / "alice_random.yaml").write_text(
        "\n".join(
            [
                "qubots_submission_schema_version: 1",
                f"spec: {ROOT / 'examples' / 'random_search_optimizer'}",
                "submitter: alice",
                "display_name: random_search",
            ]
        ),
        encoding="utf-8",
    )
    (sub_dir / "bob_hill_climb.yaml").write_text(
        "\n".join(
            [
                "qubots_submission_schema_version: 1",
                f"spec: {ROOT / 'examples' / 'hill_climb_optimizer'}",
                "submitter: bob",
                "display_name: hill_climb",
            ]
        ),
        encoding="utf-8",
    )
    return sub_dir


def test_load_suite_and_submissions(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    sub_dir = _write_submissions(tmp_path)

    suite = load_suite(suite_path)
    assert suite.name == "tiny-suite"
    assert len(suite.benchmarks) == 1
    assert suite.benchmarks[0]["name"] == "knapsack_small"
    assert suite.benchmarks[0]["dataset_path"].exists()

    submissions = load_submissions_from_dir(sub_dir)
    assert len(submissions) == 2
    submitters = {s.submitter for s in submissions}
    assert submitters == {"alice", "bob"}


def test_load_submission_resolves_relative_spec(tmp_path: Path) -> None:
    target = tmp_path / "fake_optimizer"
    target.mkdir()
    (target / "qubots.yaml").write_text("type: optimizer\n", encoding="utf-8")

    sub_path = tmp_path / "sub.yaml"
    sub_path.write_text(
        "\n".join(
            [
                "qubots_submission_schema_version: 1",
                "spec: fake_optimizer",
                "submitter: alice",
                "display_name: fake",
            ]
        ),
        encoding="utf-8",
    )

    sub = load_submission(sub_path)
    assert Path(sub.spec).resolve() == target.resolve()


def test_load_suite_rejects_unknown_schema_version(tmp_path: Path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "\n".join(
            [
                "qubots_leaderboard_schema_version: 99",
                "name: weird",
                "benchmarks:",
                "  - name: x",
                "    dataset: /nope.yaml",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported"):
        load_suite(suite)


def test_load_suite_rejects_duplicate_benchmark_names(tmp_path: Path) -> None:
    suite = tmp_path / "suite.yaml"
    dataset = ROOT / "examples" / "datasets" / "knapsack_small.yaml"
    suite.write_text(
        "\n".join(
            [
                "qubots_leaderboard_schema_version: 1",
                "name: dup",
                "benchmarks:",
                "  - name: same",
                f"    dataset: {dataset}",
                "  - name: same",
                f"    dataset: {dataset}",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_suite(suite)


def test_run_leaderboard_end_to_end(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    sub_dir = _write_submissions(tmp_path)

    report = run_leaderboard(suite_path, sub_dir, repeats=1, seed=123)

    assert report.suite_name == "tiny-suite"
    assert report.benchmarks == ["knapsack_small"]
    assert len(report.submissions) == 2
    # 2 submissions × 1 benchmark
    assert len(report.results) == 2
    for entry in report.results:
        assert isinstance(entry, BenchmarkResult)
        assert entry.benchmark_name == "knapsack_small"
        assert entry.success_rate == pytest.approx(1.0)
        assert entry.num_runs >= 1
        assert isinstance(entry.mean_best_value, float)


def test_report_to_markdown_has_per_benchmark_and_summary(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    sub_dir = _write_submissions(tmp_path)

    report = run_leaderboard(suite_path, sub_dir, repeats=1, seed=0)
    md = report_to_markdown(report)

    assert "# tiny-suite" in md
    assert "## knapsack_small" in md
    assert "## Summary" in md
    assert "alice" in md
    assert "bob" in md


def test_write_report_emits_json_and_markdown(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    sub_dir = _write_submissions(tmp_path)

    report = run_leaderboard(suite_path, sub_dir, repeats=1, seed=0)

    json_path = tmp_path / "leaderboard.json"
    md_path = tmp_path / "LEADERBOARD.md"
    written_json, written_md = write_report(
        report, json_path=json_path, markdown_path=md_path
    )

    assert written_json == json_path
    assert written_md == md_path
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["suite_name"] == "tiny-suite"
    assert len(payload["results"]) == 2


def test_run_leaderboard_applies_per_submission_parameters(tmp_path: Path) -> None:
    """Two submissions that wrap the same optimizer with different
    ``parameters:`` blocks should produce different mean_best_value."""
    suite_path = _write_suite(tmp_path)
    sub_dir = tmp_path / "param_subs"
    sub_dir.mkdir()

    sa_repo = ROOT / "examples" / "simulated_annealing_optimizer"
    (sub_dir / "sa_few.yaml").write_text(
        "\n".join(
            [
                "qubots_submission_schema_version: 1",
                f"spec: {sa_repo}",
                "submitter: alice",
                "display_name: SA-few",
                "parameters:",
                "  steps: 1",
            ]
        ),
        encoding="utf-8",
    )
    (sub_dir / "sa_many.yaml").write_text(
        "\n".join(
            [
                "qubots_submission_schema_version: 1",
                f"spec: {sa_repo}",
                "submitter: alice",
                "display_name: SA-many",
                "parameters:",
                "  steps: 500",
            ]
        ),
        encoding="utf-8",
    )

    report = run_leaderboard(suite_path, sub_dir, repeats=1, seed=0)
    by_name = {r.display_name: r for r in report.results}
    assert "SA-few" in by_name and "SA-many" in by_name
    # SA with 500 steps should reach a (weakly) better mean than SA with 1 step.
    assert by_name["SA-many"].mean_best_value <= by_name["SA-few"].mean_best_value
    # And the two should not be identical -- if they were, the parameter
    # passthrough is silently broken.
    assert by_name["SA-many"].mean_best_value != by_name["SA-few"].mean_best_value


def test_cli_leaderboard_runs_and_prints_table(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    sub_dir = _write_submissions(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "leaderboard",
            "--suite",
            str(suite_path),
            "--submissions",
            str(sub_dir),
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "tiny-suite" in result.output
    assert "knapsack_small" in result.output
    assert "alice" in result.output
    assert "bob" in result.output


def test_cli_leaderboard_writes_outputs(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path)
    sub_dir = _write_submissions(tmp_path)
    json_path = tmp_path / "out.json"
    md_path = tmp_path / "OUT.md"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "leaderboard",
            "--suite",
            str(suite_path),
            "--submissions",
            str(sub_dir),
            "--out",
            str(md_path),
            "--json",
            str(json_path),
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["suite_name"] == "tiny-suite"
