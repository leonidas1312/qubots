import os
import pytest
import subprocess
import tempfile
from pathlib import Path
from typer.testing import CliRunner
from rastion_cli.cli import app

runner = CliRunner()

@pytest.fixture(scope="module")
def check_github_token():
    """
    Ensure GITHUB_TOKEN is available, otherwise skip these tests.
    """
    token = os.getenv("GITHUB_TOKEN", None)
    if not token:
        pytest.skip("GITHUB_TOKEN not set. Skipping Git-based CLI tests.")
    return token

def test_create_and_push_solver(check_github_token):
    """
    1) create a new solver repo
    2) push local .py + solver_config.json
    3) potentially run run-solver (if we have a problem or do a trivial run).
    """
    token = check_github_token
    org = "Rastion"
    solver_repo_name = "test-solver-repo-pytest"

    # Step 1) create repo
    result_create = runner.invoke(app, [
        "create_repo",
        solver_repo_name,
        "--org", org,
        "--private", "False",
        "--github-token", token
    ])
    assert result_create.exit_code == 0, f"create_repo failed: {result_create.stdout}"

    # Step 2) push local solver code
    # We'll create a temporary local solver file & config
    with tempfile.TemporaryDirectory(prefix="solver_") as tmpdir:
        solver_py = Path(tmpdir) / "my_solver.py"
        solver_py.write_text(
            """\
from rastion_core.base_optimizer import BaseOptimizer

class MyTestSolver(BaseOptimizer):
    def __init__(self, param=123):
        self.param = param

    def optimize(self, problem, **kwargs):
        return [0,1], 999
""")

        solver_cfg = Path(tmpdir) / "solver_config.json"
        solver_cfg.write_text(
            """{
  "entry_point": "my_solver:MyTestSolver",
  "default_params": {"param": 456}
}
""")

        # now call push_solver
        result_push = runner.invoke(app, [
            "push_solver",
            solver_repo_name,
            "--file", str(solver_py),
            "--config", str(solver_cfg),
            "--org", org,
            "--github-token", token
        ])
        assert result_push.exit_code == 0, f"push_solver failed: {result_push.stdout}"

    # optionally, run-solver to ensure we can clone & run
    # We'll skip a problem for now
    result_run = runner.invoke(app, [
        "run_solver",
        f"{org}/{solver_repo_name}",
        "--solver-rev", "main"
    ])
    # Should succeed but say "No problem repo provided"
    assert result_run.exit_code == 0, f"run_solver failed: {result_run.stdout}"
    assert "No problem repo provided" in result_run.stdout

def test_create_and_push_problem(check_github_token):
    """
    Similar test for a problem. We'll create a problem repo, push a single .py + problem_config.json.
    """
    token = check_github_token
    org = "Rastion"
    problem_repo_name = "test-problem-repo-pytest"

    # create the repo
    result_create = runner.invoke(app, [
        "create_repo",
        problem_repo_name,
        "--org", org,
        "--private", "False",
        "--github-token", token
    ])
    assert result_create.exit_code == 0, f"create_repo failed: {result_create.stdout}"

    # create local problem code
    with tempfile.TemporaryDirectory(prefix="problem_") as tmpdir:
        problem_py = Path(tmpdir) / "my_problem.py"
        problem_py.write_text(
            """\
from rastion_core.base_problem import BaseProblem

class MyTestProblem(BaseProblem):
    def evaluate_solution(self, solution):
        return sum(solution)  # trivial

    def random_solution(self):
        return [1,2,3]
""")

        problem_cfg = Path(tmpdir) / "problem_config.json"
        problem_cfg.write_text(
            """{
  "entry_point": "my_problem:MyTestProblem"
}
""")

        # push_problem
        result_push = runner.invoke(app, [
            "push_problem",
            problem_repo_name,
            "--file", str(problem_py),
            "--config", str(problem_cfg),
            "--org", org,
            "--github-token", token
        ])
        assert result_push.exit_code == 0, f"push_problem failed: {result_push.stdout}"

    # Now let's do run_solver with solver_repo + problem_repo
    solver_repo_name = "test-solver-repo-pytest"  # from the previous test
    result_run = runner.invoke(app, [
        "run_solver",
        f"{org}/{solver_repo_name}",
        "--problem-repo", f"{org}/{problem_repo_name}",
        "--solver-rev", "main",
        "--problem-rev", "main"
    ])
    assert result_run.exit_code == 0, f"run_solver with problem failed: {result_run.stdout}"
    assert "Optimization completed: best_sol=" in result_run.stdout

    # This means the solver's .optimize() was called with MyTestProblem's random_solution = [1,2,3].
    # The solver we pushed earlier returns [0,1], 999 - so the 'problem' sum doesn't matter, 
    # but we see that the code is combining them successfully.


def test_clone_repo(check_github_token):
    """
    Test the 'clone_repo' command to confirm it runs 'git clone' properly.
    """
    token = check_github_token
    org = "Rastion"
    solver_repo_name = "test-solver-repo-pytest"  # from above

    # We'll clone to a temp dir
    with tempfile.TemporaryDirectory(prefix="clone_") as tmpdir:
        result_clone = runner.invoke(app, [
            "clone_repo",
            solver_repo_name,
            "--org", org,
            "--branch", "main",
            "--dest", tmpdir
        ])
        assert result_clone.exit_code == 0, f"clone_repo failed: {result_clone.stdout}"
        # check that folder exists
        cloned_folder = Path(tmpdir) / solver_repo_name
        assert cloned_folder.is_dir(), "Expected cloned repo folder not found"

        # We can check if the file we pushed is present
        solver_cfg = cloned_folder / "solver_config.json"
        assert solver_cfg.exists(), "Expected solver_config.json not found after clone"

