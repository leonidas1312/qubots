"""Tests for the qubots new scaffolder.

Every scaffolded component must:
- exist on disk with the expected file set
- pass validate_repo() with no issues
- be loadable via AutoProblem / AutoOptimizer (i.e. instantiable)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from qubots import AutoOptimizer, AutoProblem
from qubots.cli.app import app
from qubots.scaffold import (
    OPTIMIZER_FLAVORS,
    PROBLEM_FLAVORS,
    scaffold_optimizer,
    scaffold_problem,
)
from qubots.validate.validate import validate_repo


@pytest.mark.parametrize("flavor", PROBLEM_FLAVORS)
def test_scaffold_problem_validates_clean(flavor: str, tmp_path: Path) -> None:
    target = tmp_path / f"my_{flavor}_problem"
    result = scaffold_problem("my_problem", target, flavor=flavor)

    assert result.path == target.resolve()
    assert (target / "qubots.yaml").exists()
    assert (target / "qubot.py").exists()
    assert (target / "README.md").exists()

    issues = validate_repo(target)
    assert issues == [], f"Scaffolded {flavor} problem has issues: {issues}"


@pytest.mark.parametrize("flavor", OPTIMIZER_FLAVORS)
def test_scaffold_optimizer_validates_clean(flavor: str, tmp_path: Path) -> None:
    target = tmp_path / f"my_{flavor}_optimizer"
    result = scaffold_optimizer("my_optimizer", target, flavor=flavor)

    assert result.path == target.resolve()
    assert (target / "qubots.yaml").exists()
    assert (target / "qubot.py").exists()
    assert (target / "README.md").exists()

    issues = validate_repo(target)
    assert issues == [], f"Scaffolded {flavor} optimizer has issues: {issues}"


def test_scaffold_problem_blackbox_runs_under_blackbox_optimizer(tmp_path: Path) -> None:
    """Round-trip: scaffold a problem and optimizer, then run them together."""
    problem_dir = tmp_path / "round_trip_problem"
    optimizer_dir = tmp_path / "round_trip_optimizer"

    scaffold_problem("round_trip_problem", problem_dir, flavor="blackbox")
    scaffold_optimizer("round_trip_optimizer", optimizer_dir, flavor="blackbox")

    problem = AutoProblem.from_repo(problem_dir)
    optimizer = AutoOptimizer.from_repo(optimizer_dir)
    optimizer.set_parameters(iterations=10)

    result = optimizer.optimize(problem)
    assert result.status == "ok"
    assert result.best_solution is not None
    assert isinstance(result.best_value, float)


def test_scaffold_problem_dual_exposes_both_interfaces(tmp_path: Path) -> None:
    target = tmp_path / "dual_problem"
    scaffold_problem("dual_problem", target, flavor="dual")

    problem = AutoProblem.from_repo(target)
    # Blackbox interface
    sample = problem.random_solution()
    score = problem.evaluate(sample)
    assert isinstance(score, float)
    # MILP interface
    milp = problem.as_milp()
    assert milp.n_vars > 0


def test_scaffold_problem_milp_default_evaluate_works(tmp_path: Path) -> None:
    """A MILP-only problem should still expose a default evaluate via BaseProblem."""
    target = tmp_path / "milp_only_problem"
    scaffold_problem("milp_only_problem", target, flavor="milp")

    problem = AutoProblem.from_repo(target)
    sample = problem.random_solution()
    score = problem.evaluate(sample)
    assert isinstance(score, float)


def test_scaffold_refuses_existing_dir_without_force(tmp_path: Path) -> None:
    target = tmp_path / "preexisting"
    target.mkdir(parents=True)
    (target / "marker").write_text("x", encoding="utf-8")

    with pytest.raises(FileExistsError):
        scaffold_problem("preexisting", target, flavor="dual")

    # With force=True it should overwrite the manifest/entrypoint cleanly.
    result = scaffold_problem("preexisting", target, flavor="dual", force=True)
    assert (result.path / "qubots.yaml").exists()
    assert validate_repo(target) == []


def test_scaffold_rejects_unknown_flavor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="flavor"):
        scaffold_problem("x", tmp_path / "x", flavor="not_a_real_flavor")
    with pytest.raises(ValueError, match="flavor"):
        scaffold_optimizer("y", tmp_path / "y", flavor="not_a_real_flavor")


def test_scaffold_slugifies_name_and_class(tmp_path: Path) -> None:
    target = tmp_path / "slug_check"
    scaffold_problem("Shift Scheduling Problem!", target, flavor="blackbox")

    manifest = (target / "qubots.yaml").read_text(encoding="utf-8")
    assert "name: shift_scheduling_problem" in manifest
    assert "qubot.py:ShiftSchedulingProblem" in manifest
    assert validate_repo(target) == []


# ---------- CLI tests --------------------------------------------------------


def test_cli_new_problem_default_flavor_validates(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "cli_problem"
    result = runner.invoke(
        app,
        [
            "new",
            "problem",
            "--name",
            "cli_problem",
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "[OK]" in result.output
    assert validate_repo(out_dir) == []


def test_cli_new_optimizer_milp_flavor_validates(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "cli_optimizer"
    result = runner.invoke(
        app,
        [
            "new",
            "optimizer",
            "--name",
            "cli_optimizer",
            "--flavor",
            "milp",
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert validate_repo(out_dir) == []


def test_cli_new_rejects_bad_kind(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "new",
            "rocket",
            "--name",
            "x",
            "--out",
            str(tmp_path / "x"),
        ],
    )
    assert result.exit_code != 0


def test_cli_new_rejects_bad_flavor(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "new",
            "problem",
            "--name",
            "x",
            "--flavor",
            "fictional_flavor",
            "--out",
            str(tmp_path / "x"),
        ],
    )
    assert result.exit_code != 0
