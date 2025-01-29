import os
import shutil
import time
import pytest
import tempfile
from pathlib import Path
from typer.testing import CliRunner
from rastion_cli.cli import app
import requests
from subprocess import run

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

runner = CliRunner()

@pytest.fixture(scope="module")
def github_token_check():
    token = os.getenv("GITHUB_TOKEN", None)
    if not token:
        pytest.skip("GITHUB_TOKEN not set. Skipping Git-based CLI tests.")
    return token

def check_repo_exists(org, repo_name, token):
    url = f"https://api.github.com/repos/{org}/{repo_name}"
    headers = {"Authorization": f"token {token}"}
    response = requests.get(url, headers=headers)
    return response.status_code == 200  # If status is 200, repo exists

def delete_temp_dir_with_retries(temp_dir, retries=3, delay=2):
    """Attempt to delete the temporary directory with retries"""
    for _ in range(retries):
        try:
            shutil.rmtree(temp_dir)
            break
        except PermissionError:
            print(f"PermissionError while trying to delete {temp_dir}. Retrying...")
            time.sleep(delay)  # Wait before retrying
    else:
        print(f"Failed to delete {temp_dir} after {retries} retries.")

def test_full_combined_repo_flow(github_token_check):
    token = github_token_check
    org = "Rastion"
    repo_name = "test-combined-repo-pytest"

    # Step 1) Check if the repo already exists
    if check_repo_exists(org, repo_name, token):
        print(f"Repository '{repo_name}' already exists. Skipping creation.")
    else:
        create_result = runner.invoke(app, [
            "create_repo",
            repo_name,
            "--org", org,
            "--github-token", token
        ])
        assert create_result.exit_code == 0, f"create_repo failed: {create_result.stdout}"

    # Step 2) prepare local files (this stays the same)
    temp_dir = tempfile.mkdtemp(prefix="combined_")  # create temp dir manually
    try:
        # solver code: referencing an existing algorithm from rastion_core
        solver_py = Path(temp_dir) / "my_solver.py"
        solver_py.write_text(
            """\
from rastion_core.algorithms.genetic_algorithm import GeneticAlgorithm

class MySolver(GeneticAlgorithm):
    pass
""")

        solver_cfg = Path(temp_dir) / "solver_config.json"
        solver_cfg.write_text("""{
  "entry_point": "my_solver:MySolver",
  "default_params": {
    "population_size": 10,
    "max_generations": 20,
    "verbose": false
  }
}
""")

        # problem code: referencing TSPProblem from rastion_core
        problem_py = Path(temp_dir) / "my_problem.py"
        problem_py.write_text(
            """\
from rastion_core.problems.traveling_salesman import TSPProblem

class MyProblem(TSPProblem):
    pass
""")
        problem_cfg = Path(temp_dir) / "problem_config.json"
        problem_cfg.write_text("""{
  "entry_point": "my_problem:MyProblem"
}
""")

        # Step 3) push them with 'push_solver'? (this part goes outside the `with` block)
        # a) clone
        clone_result = runner.invoke(app, [
            "clone_repo",
            repo_name,
            "--org", org,
            "--branch", "main",  # Ensure the correct branch is used
            "--dest", temp_dir
        ])
        assert clone_result.exit_code == 0, f"clone_repo failed: {clone_result.stdout}"

        local_repo_dir = Path(temp_dir) / repo_name
        # b) copy all 4 files
        for f in [solver_py, solver_cfg, problem_py, problem_cfg]:
            (local_repo_dir / f.name).write_text(f.read_text())

        # c) commit & push
        run(["git", "add", "."], cwd=local_repo_dir, check=True)
        run(["git", "commit", "-m", "Add solver & problem code & config"], cwd=local_repo_dir, check=True)
        run(["git", "push", "origin", "main"], cwd=local_repo_dir, check=True)

    finally:
        # Clean up the temporary directory after pushing files
        delete_temp_dir_with_retries(temp_dir)

    # Step 4) now we do AutoOptimizer & AutoProblem from that single repo
    solver = AutoOptimizer.from_repo(f"{org}/{repo_name}", revision="main")
    problem = AutoProblem.from_repo(f"{org}/{repo_name}", revision="main")

    # Step 5) run .optimize()
    best_sol, best_cost = solver.optimize(problem)
    print(f"BEST_SOL={best_sol}, BEST_COST={best_cost}")
    assert best_sol is not None
    assert isinstance(best_cost, float) or isinstance(best_cost, int)
