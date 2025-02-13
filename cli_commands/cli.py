import typer
import os
import requests
import subprocess
import shutil
import json
from pathlib import Path
from typing import Optional
import tempfile
from subprocess import run
import re
from functools import wraps

from qubots.auto_optimizer import AutoOptimizer

app = typer.Typer(help="Rastion CLI - A tool for Git-based solver/problem repos.")

GITHUB_API = "https://api.github.com"
org = "Rastion"


def verify_github_token(token: str) -> str:
    """
    Verifies that the provided GitHub token is valid and that the user
    is a member of the 'Rastion' organization.
    Returns the GitHub username if the token is valid.
    """
    # Validate token by fetching user data
    user_response = requests.get(
        f"{GITHUB_API}/user", headers={"Authorization": f"token {token}"}
    )
    if user_response.status_code != 200:
        typer.echo("ERROR: Invalid GitHub token provided.")
        raise typer.Exit(1)
    user_data = user_response.json()
    username = user_data.get("login")
    if not username:
        typer.echo("ERROR: Could not determine the username from the token.")
        raise typer.Exit(1)

    # Check if the user is a member of the organization
    membership_url = f"{GITHUB_API}/orgs/{org}/people/{username}"
    membership_response = requests.get(
        membership_url, headers={"Authorization": f"token {token}"}
    )
    
    return username


def require_valid_token(func):
    """
    Decorator to ensure that a valid GitHub token is provided and that
    the token belongs to a member of the Rastion organization.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        github_token = kwargs.get("github_token")
        if not github_token:
            typer.echo("ERROR: GitHub token not provided. Use --github-token or set GITHUB_TOKEN env var.")
            raise typer.Exit(1)
        # Verify the token and org membership
        verify_github_token(github_token)
        return func(*args, **kwargs)
    return wrapper


def generate_requirements_for_repo(repo_dir: Path):
    """
    Uses pipreqs to automatically generate a requirements.txt file for the given repo directory.
    It will force overwrite any existing file and ignore the .git folder.
    """
    subprocess.run(["pipreqs", str(repo_dir), "--force", "--ignore", ".git", "--mode", "no-pin"], check=True)


@app.command(name="create_repo")
@require_valid_token
def create_repo(
    repo_name: str = typer.Argument(..., help="Name of the new GitHub repo"),
    private: bool = typer.Option(False, "--private", is_flag=True, help="Omit for public repo, use for private repo"),
    github_token: str = typer.Option(..., envvar="GITHUB_TOKEN", help="Your GitHub personal access token")
):
    """
    Create a new GitHub repo under the 'Rastion' org.
    """
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
    
    with tempfile.TemporaryDirectory() as temp_repo_dir:
        run(["git", "init"], cwd=temp_repo_dir, check=True)
        run(["git", "remote", "add", "origin", f"https://github.com/{org}/{repo_name}.git"], cwd=temp_repo_dir, check=True)
        with open(Path(temp_repo_dir) / "README.md", "w") as readme:
            readme.write("# Initialized repository\n")
        run(["git", "add", "."], cwd=temp_repo_dir, check=True)
        run(["git", "commit", "-m", "Initial commit to create main branch"], cwd=temp_repo_dir, check=True)
        run(["git", "branch", "-M", "main"], cwd=temp_repo_dir, check=True)  # Rename to 'main'
        run(["git", "push", "-u", "origin", "main"], cwd=temp_repo_dir, check=True)

    data = resp.json()
    clone_url = data["clone_url"]
    typer.echo(f"Repo created successfully under org='{org}': {clone_url}")


@app.command(name="update_repo")
@require_valid_token
def update_repo(
    repo_name: str = typer.Argument(..., help="Name of the repository to update"),
    local_dir: str = typer.Option(".", "--local-dir", help="Local directory with updated files"),
    branch: str = typer.Option("main", "--branch", help="Branch to update (default 'main')"),
    github_token: str = typer.Option(..., envvar="GITHUB_TOKEN", help="Your GitHub personal access token")
):
    """
    Update an existing GitHub repo with new local changes.
    """
    repo_url = f"https://github.com/{org}/{repo_name}.git"
    typer.echo(f"Updating repository '{repo_name}' from organization '{org}' using branch '{branch}'...")
    
    tmp_dir = tempfile.mkdtemp(prefix="rastion_update_")
    try:
        clone_cmd = ["git", "clone", "--branch", branch, repo_url, tmp_dir]
        typer.echo("Cloning repository...")
        subprocess.run(clone_cmd, cwd=".", check=True)
        
        local_path = Path(local_dir).resolve()
        repo_path = Path(tmp_dir)
        typer.echo(f"Copying files from {local_path} to repository clone...")
        for item in local_path.iterdir():
            if item.name == ".git":
                continue
            dest = repo_path / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy(item, dest)
        
        typer.echo("Staging changes...")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        commit_message = "Update repository with local changes"
        typer.echo("Committing changes...")
        subprocess.run(["git", "commit", "-m", commit_message], cwd=repo_path, check=True)
        typer.echo("Pushing changes...")
        subprocess.run(["git", "push", "origin", branch], cwd=repo_path, check=True)
        typer.echo("Repository updated successfully!")
    except subprocess.CalledProcessError as e:
        typer.echo(f"ERROR: {e}")
        raise typer.Exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.command(name="delete_repo")
@require_valid_token
def delete_repo(
    repo_name: str = typer.Argument(..., help="Name of the repository to delete"),
    github_token: str = typer.Option(..., envvar="GITHUB_TOKEN", help="Your GitHub personal access token")
):
    """
    Delete a GitHub repository.
    """
    url = f"{GITHUB_API}/repos/{org}/{repo_name}"
    headers = {"Authorization": f"token {github_token}"}
    typer.echo(f"Deleting repository '{repo_name}' from organization '{org}'...")
    response = requests.delete(url, headers=headers)
    if response.status_code == 204:
        typer.echo(f"Repository '{repo_name}' deleted successfully.")
    else:
        typer.echo(f"Failed to delete repository: {response.status_code} {response.text}")
        raise typer.Exit(1)


@app.command(name="clone_repo")
def clone_repo(
    repo_name: str = typer.Argument(..., help="Name of the repo to clone"),
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


@app.command(name="push_solver")
@require_valid_token
def push_solver(
    repo_name: str = typer.Argument(..., help="Name of the solver repo (must already exist)"),
    local_file: str = typer.Option(..., "--file", help="Path to the local .py solver file"),
    solver_config: str = typer.Option(..., "--config", help="Path to solver_config.json"),
    github_token: str = typer.Option(..., envvar="GITHUB_TOKEN", help="Your GitHub personal access token"),
    branch: str = typer.Option("main", "--branch", help="Branch name to push to")
):
    """
    Push a local solver .py file + solver_config.json to an existing GitHub repo.
    """
    repo_url = f"https://github.com/{org}/{repo_name}.git"
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

    generate_requirements_for_repo(local_repo_dir)

    subprocess.run(["git", "add", "."], cwd=local_repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Add solver code & config"], cwd=local_repo_dir, check=True)
    subprocess.run(["git", "push", "origin", branch], cwd=local_repo_dir, check=True)

    typer.echo("Solver pushed to GitHub successfully!")


@app.command(name="push_problem")
@require_valid_token
def push_problem(
    repo_name: str = typer.Argument(..., help="Name of the problem repo (must already exist)"),
    local_file: str = typer.Option(..., "--file", help="Path to the local .py problem file"),
    problem_config: str = typer.Option(..., "--config", help="Path to problem_config.json"),
    github_token: str = typer.Option(..., envvar="GITHUB_TOKEN", help="Your GitHub personal access token"),
    branch: str = typer.Option("main", "--branch", help="Branch name to push to")
):
    """
    Push a local problem .py file + problem_config.json to an existing GitHub repo.
    """
    repo_url = f"https://github.com/{org}/{repo_name}.git"
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

    generate_requirements_for_repo(local_repo_dir)

    subprocess.run(["git", "add", "."], cwd=local_repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Add problem code & config"], cwd=local_repo_dir, check=True)
    subprocess.run(["git", "push", "origin", branch], cwd=local_repo_dir, check=True)

    typer.echo("Problem pushed to GitHub successfully!")


@app.command(name="run_solver")
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
    typer.echo(f"Loading solver from: {solver_repo}@{solver_revision}")
    solver = AutoOptimizer.from_repo(solver_repo, revision=solver_revision)

    if problem_repo:
        typer.echo(f"Loading problem from: {problem_repo}@{problem_revision}")
        from qubots.auto_problem import AutoProblem
        problem = AutoProblem.from_repo(problem_repo, revision=problem_revision)
    else:
        typer.echo("No problem repo provided, not optimizing.")
        return

    best_sol, best_cost = solver.optimize(problem)
    typer.echo(f"Optimization completed: best_sol={best_sol}, best_cost={best_cost}")


def main():
    app()


if __name__ == "__main__":
    main()
