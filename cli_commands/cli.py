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
    is a member of the 'Rastion' org.
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

    membership_url = f"{GITHUB_API}/orgs/{org}/members/{username}"
    membership_response = requests.get(
        membership_url, headers={"Authorization": f"token {token}"}
    )

    return username


def require_valid_token(func):
    """
    Decorator to ensure that a valid GitHub token is provided and that
    the token belongs to a member of the Rastion org.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        github_token = kwargs.get("github_token")
        if not github_token:
            typer.echo("ERROR: GitHub token not provided. Use --github-token or set GITHUB_TOKEN env var.")
            raise typer.Exit(1)
        verify_github_token(github_token)
        return func(*args, **kwargs)
    return wrapper


def generate_requirements_for_repo(repo_dir: Path):
    """
    Uses pipreqs to automatically generate a requirements.txt file for the given repo directory.
    It will force overwrite any existing file and ignore the .git folder.
    """
    try:
        subprocess.run(
            ["pipreqs", str(repo_dir), "--force", "--ignore", ".git", "--mode", "no-pin"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        typer.echo("Error running pipreqs. Please ensure pipreqs is installed (pip install pipreqs) and try again.")
        raise



def copy_source(source: Path, destination: Path):
    """
    Copies the contents of the source (file or directory) into the destination directory.
    If source is a directory, its entire content is copied (skipping .git).
    If source is a file, it is copied directly.
    """
    if source.is_dir():
        for item in source.iterdir():
            if item.name == ".git":
                continue
            dest = destination / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy(item, dest)
    else:
        shutil.copy(source, destination / source.name)


@app.command(name="create_repo")
@require_valid_token
def create_repo(
    repo_name: str = typer.Argument(..., help="Name of the new GitHub repo"),
    private: bool = typer.Option(False, "--private", is_flag=True, help="Use for private repo; omit for public repo"),
    github_token: str = typer.Option(..., envvar="GITHUB_TOKEN", help="Your GitHub personal access token")
):
    """
    Create a new GitHub repo under the 'Rastion' organization.
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
        run(["git", "branch", "-M", "main"], cwd=temp_repo_dir, check=True)
        run(["git", "push", "-u", "origin", "main"], cwd=temp_repo_dir, check=True)

    data = resp.json()
    clone_url = data["clone_url"]
    typer.echo(f"Repo created successfully under org='{org}': {clone_url}")


@app.command(name="update_repo")
@require_valid_token
def update_repo(
    repo_name: str = typer.Argument(..., help="Name of the repository to update"),
    source: str = typer.Option(".", "--source", help="Local directory with updated files"),
    branch: str = typer.Option("main", "--branch", help="Branch to update (default 'main')"),
    github_token: str = typer.Option(..., envvar="GITHUB_TOKEN", help="Your GitHub personal access token")
):
    """
    Update an existing GitHub repo with new local changes from a source directory.
    """
    repo_url = f"https://github.com/{org}/{repo_name}.git"
    typer.echo(f"Updating repository '{repo_name}' from organization '{org}' using branch '{branch}'...")
    
    tmp_dir = tempfile.mkdtemp(prefix="rastion_update_")
    try:
        clone_cmd = ["git", "clone", "--branch", branch, repo_url, tmp_dir]
        typer.echo("Cloning repository...")
        subprocess.run(clone_cmd, cwd=".", check=True)
        
        source_path = Path(source).resolve()
        repo_path = Path(tmp_dir)
        typer.echo(f"Copying files from {source_path} to repository clone...")
        copy_source(source_path, repo_path)
        
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
    source: str = typer.Option(..., "--source", help="Path to the local directory containing solver files"),
    github_token: str = typer.Option(..., envvar="GITHUB_TOKEN", help="Your GitHub personal access token"),
    branch: str = typer.Option("main", "--branch", help="Branch name to push to")
):
    """
    Push the contents of a local source directory (containing solver code, config, etc.)
    to an existing GitHub repo.
    """
    repo_url = f"https://github.com/{org}/{repo_name}.git"
    tmp_dir = tempfile.mkdtemp(prefix="rastion_")
    typer.echo(f"Cloning {repo_url} into temporary directory: {tmp_dir}")
    subprocess.run(["git", "clone", repo_url, "--branch", branch], cwd=tmp_dir, check=True)
    local_repo_dir = Path(tmp_dir) / repo_name

    source_path = Path(source)
    if not source_path.exists():
        typer.echo(f"ERROR: The source path '{source_path}' does not exist.")
        raise typer.Exit(1)

    copy_source(source_path, local_repo_dir)

    # Automatically generate/update requirements.txt using pipreqs.
    generate_requirements_for_repo(local_repo_dir)

    subprocess.run(["git", "add", "."], cwd=local_repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Add/Update solver code & config"], cwd=local_repo_dir, check=True)
    subprocess.run(["git", "push", "origin", branch], cwd=local_repo_dir, check=True)

    typer.echo("Solver pushed to GitHub successfully!")


@app.command(name="push_problem")
@require_valid_token
def push_problem(
    repo_name: str = typer.Argument(..., help="Name of the problem repo (must already exist)"),
    source: str = typer.Option(..., "--source", help="Path to the local directory containing problem files"),
    github_token: str = typer.Option(..., envvar="GITHUB_TOKEN", help="Your GitHub personal access token"),
    branch: str = typer.Option("main", "--branch", help="Branch name to push to")
):
    """
    Push the contents of a local source directory (containing problem code, config, etc.)
    to an existing GitHub repo.
    """
    repo_url = f"https://github.com/{org}/{repo_name}.git"
    tmp_dir = tempfile.mkdtemp(prefix="rastion_")
    typer.echo(f"Cloning {repo_url} into temporary directory: {tmp_dir}")
    subprocess.run(["git", "clone", repo_url, "--branch", branch], cwd=tmp_dir, check=True)
    local_repo_dir = Path(tmp_dir) / repo_name

    source_path = Path(source)
    if not source_path.exists():
        typer.echo(f"ERROR: The source path '{source_path}' does not exist.")
        raise typer.Exit(1)

    copy_source(source_path, local_repo_dir)

    generate_requirements_for_repo(local_repo_dir)

    subprocess.run(["git", "add", "."], cwd=local_repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Add/Update problem code & config"], cwd=local_repo_dir, check=True)
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
    Clone (or pull) the solver repo from GitHub, load the solver, optionally also load a problem from another repo,
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
