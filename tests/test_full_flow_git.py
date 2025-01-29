import os
import pytest
import tempfile
from pathlib import Path
from typer.testing import CliRunner
from rastion_cli.cli import app

# We'll use these to load and run after pushing
from rastion_hub.auto_optimizer import AutoOptimizer
from rastion_hub.auto_problem import AutoProblem

runner = CliRunner()

@pytest.fixture(scope="module")
def github_token_check():
    token = os.getenv("GITHUB_TOKEN", None)
    if not token:
        pytest.skip("GITHUB_TOKEN not set. Skipping Git-based CLI tests.")
    return token

def test_full_combined_repo_flow(github_token_check):
    """
    End-to-end test:
      1) Create a single GitHub repo, 'my-combined-repo'
      2) Locally create my_solver.py, my_problem.py, solver_config.json, problem_config.json
      3) Push them all in one commit
      4) Use AutoOptimizer.from_repo(...) and AutoProblem.from_repo(...) from that same repo
      5) solver.optimize(problem) => check we get a result
    """
    token = github_token_check
    org = "Rastion"
    repo_name = "test-combined-repo-pytest"

    # Step 1) create the repo
    create_result = runner.invoke(app, [
        "create_repo",
        repo_name,
        "--org", org,
        "--private", "False",
        "--github-token", token
    ])
    assert create_result.exit_code == 0, f"create_repo failed: {create_result.stdout}"

    # Step 2) prepare local files
    with tempfile.TemporaryDirectory(prefix="combined_") as tmpdir:
        # solver code: referencing an existing algorithm from rastion_core,
        # or you can copy a partial logic. We'll do direct reference to GeneticAlgorithm
        solver_py = Path(tmpdir) / "my_solver.py"
        solver_py.write_text(
            """\
# bridging reference: we rely on installed rastion_core
# so the entry point can be "my_solver:MySolver"
from rastion_core.algorithms.genetic_algorithm import GeneticAlgorithm

class MySolver(GeneticAlgorithm):
    # just inherit to confirm it's recognized
    pass
""")

        solver_cfg = Path(tmpdir) / "solver_config.json"
        solver_cfg.write_text("""{
  "entry_point": "my_solver:MySolver",
  "default_params": {
    "population_size": 10,
    "max_generations": 20,
    "verbose": false
  }
}
""")

        # problem code: referencing TSPProblem from rastion_core, or we can do direct code
        problem_py = Path(tmpdir) / "my_problem.py"
        problem_py.write_text(
            """\
# bridging reference
from rastion_core.problems.traveling_salesman import TSPProblem

class MyProblem(TSPProblem):
    pass
""")
        problem_cfg = Path(tmpdir) / "problem_config.json"
        problem_cfg.write_text("""{
  "entry_point": "my_problem:MyProblem"
}
""")

        # Step 3) push them with 'push_solver'? We only have push_solver/push_problem individually in the CLI. 
        # We'll do 2 calls. Or do a single manual push. 
        # Let's do a single manual approach with 'clone_repo + copy + commit + push' for simplicity.

        # a) clone
        from subprocess import run
        clone_result = runner.invoke(app, [
            "clone_repo",
            repo_name,
            "--org", org,
            "--branch", "main",
            "--dest", tmpdir
        ])
        assert clone_result.exit_code == 0, f"clone_repo failed: {clone_result.stdout}"

        local_repo_dir = Path(tmpdir) / repo_name
        # b) copy all 4 files
        for f in [solver_py, solver_cfg, problem_py, problem_cfg]:
            (local_repo_dir / f.name).write_text(f.read_text())

        # c) commit & push
        run(["git", "add", "."], cwd=local_repo_dir, check=True)
        run(["git", "commit", "-m", "Add solver & problem code & config"], cwd=local_repo_dir, check=True)
        run(["git", "push", "origin", "main"], cwd=local_repo_dir, check=True)

    # Step 4) now we do AutoOptimizer & AutoProblem from that single repo
    solver = AutoOptimizer.from_repo(f"{org}/{repo_name}", revision="main")
    problem = AutoProblem.from_repo(f"{org}/{repo_name}", revision="main")

    # Step 5) run .optimize()
    best_sol, best_cost = solver.optimize(problem)
    print(f"BEST_SOL={best_sol}, BEST_COST={best_cost}")
    # We'll just assert the run didn't crash
    assert best_sol is not None
    assert isinstance(best_cost, float) or isinstance(best_cost, int)
