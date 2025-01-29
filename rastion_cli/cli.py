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
from rastion_core.problems.traveling_salesman import TSPProblem

app = typer.Typer(help="Rastion CLI - A tool for Git-based solver repos.")

GITHUB_API = "https://api.github.com"


@app.command()
def create_repo(
    repo_name: str = typer.Argument(..., help="Name of the new GitHub repo"),
    org: str = typer.Option("myorg", "--org", help="GitHub organization name"),
    private: bool = typer.Option(False, "--private", help="Make the repo private?"),
    github_token: Optional[str] = typer.Option(None, envvar="GITHUB_TOKEN", help="Your GitHub personal access token")
):
    """
    Create a new GitHub repo under the specified org (default is 'myorg').
    """
    if not github_token:
        typer.echo("ERROR: GitHub token not provided (set GITHUB_TOKEN env var or use --github-token).")
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
    typer.echo(f"Repo created successfully: {clone_url}")


@app.command()
def push_solver(
    repo_name: str = typer.Argument(..., help="Name of the repo to push to (must already exist)"),
    local_file: str = typer.Option(..., "--file", help="Path to the local .py solver file"),
    solver_config: str = typer.Option(..., "--config", help="Path to solver_config.json"),
    org: str = typer.Option("myorg", "--org", help="GitHub org"),
    github_token: Optional[str] = typer.Option(None, envvar="GITHUB_TOKEN", help="Your GitHub personal access token"),
    branch: str = typer.Option("main", help="Branch name to push to")
):
    """
    Push a local solver .py file + solver_config.json to an existing GitHub repo.
    We'll clone the repo locally, copy the files, commit, and push.
    """
    if not github_token:
        typer.echo("ERROR: GitHub token not provided (set GITHUB_TOKEN env var or use --github-token).")
        raise typer.Exit(1)

    repo_url = f"https://github.com/{org}/{repo_name}.git"
    # local temp dir
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="rastion_")
    typer.echo(f"Cloning {repo_url} into {tmp_dir}")

    # clone
    subprocess.run(["git", "clone", repo_url, "--branch", branch], cwd=tmp_dir, check=True)
    local_repo_dir = Path(tmp_dir) / repo_name

    # copy local_file and solver_config
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

    # commit & push
    subprocess.run(["git", "add", "."], cwd=local_repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Add solver code & config"], cwd=local_repo_dir, check=True)
    subprocess.run(["git", "push", "origin", branch], cwd=local_repo_dir, check=True)

    typer.echo("Solver pushed to GitHub successfully!")


@app.command()
def clone_repo(
    repo_name: str = typer.Argument(..., help="Name of the repo to clone"),
    org: str = typer.Option("myorg", "--org", help="GitHub org"),
    branch: str = typer.Option("main", help="Branch name to checkout"),
    dest: str = typer.Option(".", help="Destination folder to clone into")
):
    """
    Clone a GitHub repo locally (like git clone).
    """
    repo_url = f"https://github.com/{org}/{repo_name}.git"
    typer.echo(f"Cloning from {repo_url} into {dest}")

    subprocess.run(["git", "clone", "--branch", branch, repo_url], cwd=dest, check=True)
    typer.echo("Clone completed.")


@app.command()
def run_solver(
    repo_id: str = typer.Argument(..., help="'org_name/repo_name' on GitHub"),
    revision: str = typer.Option("main", "--revision", help="Branch or tag"),
    distance_matrix: str = typer.Option(None, help="Comma-delimited string for TSP distances, e.g. '0,2,9;2,0,6;9,6,0'")
):
    """
    Example command that uses AutoOptimizerGit to clone/pull a solver repo, 
    then run it on a TSP problem if distance_matrix is given, else no problem.
    """
    from rastion_hub.hub_integration.auto_optimizer_github import AutoOptimizerGit

    solver = AutoOptimizerGit.from_repo(repo_id, revision=revision)

    if distance_matrix:
        # parse the matrix
        rows = distance_matrix.split(";")
        dist_matrix = []
        for row in rows:
            dist_matrix.append([float(x) for x in row.split(",")])
        from rastion_core.problems.traveling_salesman import TSPProblem
        problem = TSPProblem(dist_matrix)
        best_sol, best_cost = solver.optimize(problem)
        typer.echo(f"Best TSP solution: {best_sol}, cost: {best_cost}")
    else:
        # no problem, just show we can instantiate
        typer.echo("No distance_matrix provided. Instantiated solver but did not run optimize().")


def main():
    app()


if __name__ == "__main__":
    main()
