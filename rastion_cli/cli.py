# rastion_cli/cli.py

import typer
import os
import requests
import subprocess
import shutil
import json
from pathlib import Path
from typing import Optional
from rastion_hub.auto_optimizer import AutoOptimizer

app = typer.Typer(help="Rastion CLI - A tool for Git-based solver/problem repos.")

GITHUB_API = "https://api.github.com"


@app.command()
def create_repo(
    repo_name: str = typer.Argument(..., help="Name of the new GitHub repo"),
    org: str = typer.Option("Rastion", "--org", help="GitHub organization name (default 'Rastion')"),
    private: bool = typer.Option(False, "--private", help="Make the repo private?"),
    github_token: Optional[str] = typer.Option(None, envvar="GITHUB_TOKEN", help="Your GitHub personal access token")
):
    """
    Create a new GitHub repo under the specified org (default 'Rastion').
    """
    if not github_token:
        typer.echo("ERROR: GitHub token not provided. Use --github-token or set GITHUB_TOKEN env var.")
        raise typer.Exit(1)

    url = f"{GITHUB_API}/orgs/{org}/repos"
    headers = {"Authorization": f"token {github_token}"}
    payload = {
        "name": repo_name,
        "private": private,
        "auto_init": False
    }
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code >= 300:
        typer.echo(f"ERROR: Failed to create repo: {resp.status_code} {resp.text}")
        raise typer.Exit(1)
    data = resp.json()
    clone_url = data["clone_url"]
    typer.echo(f"Repo created successfully under org='{org}': {clone_url}")


@app.command()
def clone_repo(
    repo_name: str = typer.Argument(..., help="Name of the repo to clone"),
    org: str = typer.Option("Rastion", "--org", help="GitHub org (default 'Rastion')"),
    branch: str = typer.Option("main", "--branch", help="Branch name to checkout"),
    dest: str = typer.Option(".", "--dest", help="Destination folder to clone into")
):
    """
    Clone a GitHub repo locally (like git clone).
    """
    repo_url = f"https://github.com/{org}/{repo_name}.git"
    typer.echo(f"Cloning from {repo_url} into {dest}")

    subprocess.run(["git", "clone", "--branch", branch, repo_url], cwd=dest, check=True)
    typer.echo("Clone completed.")


@app.command()
def push_solver(
    repo_name: str = typer.Argument(..., help="Name of the solver repo (must already exist)"),
    local_file: str = typer.Option(..., "--file", help="Path to the local .py solver file"),
    solver_config: str = typer.Option(..., "--config", help="Path to solver_config.json"),
    org: str = typer.Option("Rastion", "--org", help="GitHub org (default 'Rastion')"),
    github_token: Optional[str] = typer.Option(None, envvar="GITHUB_TOKEN", help="Your GitHub personal access token"),
    branch: str = typer.Option("main", "--branch", help="Branch name to push to")
):
    """
    Push a local solver .py file + solver_config.json to an existing GitHub repo.
    We'll clone the repo locally, copy the files, commit, and push.
    """
    if not github_token:
        typer.echo("ERROR: GitHub token not provided.")
        raise typer.Exit(1)

    repo_url = f"https://github.com/{org}/{repo_name}.git"
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="rastion_")
    typer.echo(f"Cloning {repo_url} into temp dir: {tmp_dir}")

    subprocess.run(["git", "clone", repo_url, "--branch", branch], cwd=tmp_dir, check=True)
    local_repo_dir = Path(tmp_dir) / repo_name

    solver_py = Path(local_file)
    config_json = Path(solver_config)
    if not solver_py.is_file():
        typer.echo(f"ERROR: {solver_py} not found.")
        raise typer.Exit(1)
    if not config_json.is_file():
        typer.echo(f"ERROR: {config_json} not found.")
        raise typer.Exit(1)

    shutil.copy(str(solver_py), str(local_repo_dir / solver_py.name))
    shutil.copy(str(config_json), str(local_repo_dir / "solver_config.json"))

    subprocess.run(["git", "add", "."], cwd=local_repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Add solver code & config"], cwd=local_repo_dir, check=True)
    subprocess.run(["git", "push", "origin", branch], cwd=local_repo_dir, check=True)

    typer.echo("Solver pushed to GitHub successfully!")


@app.command()
def push_problem(
    repo_name: str = typer.Argument(..., help="Name of the problem repo (must already exist)"),
    local_file: str = typer.Option(..., "--file", help="Path to the local .py problem file"),
    problem_config: str = typer.Option(..., "--config", help="Path to problem_config.json"),
    org: str = typer.Option("Rastion", "--org", help="GitHub org (default 'Rastion')"),
    github_token: Optional[str] = typer.Option(None, envvar="GITHUB_TOKEN", help="Your GitHub personal access token"),
    branch: str = typer.Option("main", "--branch", help="Branch name to push to")
):
    """
    Push a local problem .py file + problem_config.json to an existing GitHub repo.
    Similar approach to push_solver.
    """
    if not github_token:
        typer.echo("ERROR: GitHub token not provided.")
        raise typer.Exit(1)

    repo_url = f"https://github.com/{org}/{repo_name}.git"
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="rastion_")
    typer.echo(f"Cloning {repo_url} into temp dir: {tmp_dir}")

    subprocess.run(["git", "clone", repo_url, "--branch", branch], cwd=tmp_dir, check=True)
    local_repo_dir = Path(tmp_dir) / repo_name

    prob_py = Path(local_file)
    config_json = Path(problem_config)
    if not prob_py.is_file():
        typer.echo(f"ERROR: {prob_py} not found.")
        raise typer.Exit(1)
    if not config_json.is_file():
        typer.echo(f"ERROR: {config_json} not found.")
        raise typer.Exit(1)

    shutil.copy(str(prob_py), str(local_repo_dir / prob_py.name))
    shutil.copy(str(config_json), str(local_repo_dir / "problem_config.json"))

    subprocess.run(["git", "add", "."], cwd=local_repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Add problem code & config"], cwd=local_repo_dir, check=True)
    subprocess.run(["git", "push", "origin", branch], cwd=local_repo_dir, check=True)

    typer.echo("Problem pushed to GitHub successfully!")


@app.command()
def run_solver(
    solver_repo: str = typer.Argument(..., help="e.g. 'Rastion/my-solver-repo' on GitHub"),
    solver_revision: str = typer.Option("main", "--solver-rev", help="Solver branch or tag"),
    problem_repo: Optional[str] = typer.Option(None, "--problem-repo", help="e.g. 'Rastion/my-problem-repo'"),
    problem_revision: str = typer.Option("main", "--problem-rev", help="Problem branch or tag"),
):
    """
    Clone or pull the solver repo from GitHub, load the solver, 
    optionally also load a problem from another repo, 
    then call solver.optimize(problem).
    """

    # 1) Load solver from Git-based approach
    typer.echo(f"Loading solver from: {solver_repo}@{solver_revision}")
    solver = AutoOptimizer.from_repo(solver_repo, revision=solver_revision)

    # 2) If problem_repo is provided, do the same approach for problem
    if problem_repo:
        typer.echo(f"Loading problem from: {problem_repo}@{problem_revision}")
        from rastion_hub.auto_problem import AutoProblem
        problem = AutoProblem.from_repo(problem_repo, revision=problem_revision)
    else:
        # if no problem, we can just do a dummy or None
        typer.echo("No problem repo provided, not optimizing.")
        return

    # 3) We have solver + problem, run optimization
    best_sol, best_cost = solver.optimize(problem)
    typer.echo(f"Optimization completed: best_sol={best_sol}, best_cost={best_cost}")


def main():
    app()


if __name__ == "__main__":
    main()
