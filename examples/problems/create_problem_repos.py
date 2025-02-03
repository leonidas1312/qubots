#!/usr/bin/env python
import os
import subprocess
import tempfile
import shutil
import requests
from pathlib import Path
from dotenv import load_dotenv
from typer.testing import CliRunner
from rastion_cli.cli import app

# Load environment variables from .env
load_dotenv()

# Get GitHub token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise EnvironmentError("GITHUB_TOKEN is not defined in your .env file.")

ORG = "Rastion"

# List of problem repository names (with hyphens)
repos = [
    "vehicle-routing",
    "job-scheduling",
    "facility-location",
    "portfolio-optimization",
    "network-flow",
    "graph-coloring",
    "bin-packing",
    "resource-allocation",
    "inventory-management",
    "optimal-control",
    "energy-management",
    "sensor-placement",
    "supply-chain",
    "data-center-optimization",
]

runner = CliRunner()

def check_repo_exists(org, repo_name, token):
    url = f"https://api.github.com/repos/{org}/{repo_name}"
    headers = {"Authorization": f"token {token}"}
    response = requests.get(url, headers=headers)
    return response.status_code == 200

def create_initial_commit(org, repo_name):
    """
    Create an initial commit (with an empty README) so that the main branch is initialized.
    This uses direct git commands in a temporary directory.
    """
    with tempfile.TemporaryDirectory() as temp_repo_dir:
        # Initialize repository locally
        subprocess.run(["git", "init"], cwd=temp_repo_dir, check=True)
        subprocess.run(["git", "remote", "add", "origin", f"https://github.com/{org}/{repo_name}.git"], cwd=temp_repo_dir, check=True)
        readme_path = Path(temp_repo_dir) / "README.md"
        with open(readme_path, "w") as readme:
            readme.write("# Initialized repository\n")
        subprocess.run(["git", "add", "."], cwd=temp_repo_dir, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit to create main branch"], cwd=temp_repo_dir, check=True)
        subprocess.run(["git", "branch", "-M", "main"], cwd=temp_repo_dir, check=True)
        subprocess.run(["git", "push", "-u", "origin", "main"], cwd=temp_repo_dir, check=True)

def push_local_files(org, repo_name, local_dir):
    """
    In the cloned repo (located in local_dir/repo_name), copy local files (the problem .py and config)
    from a local folder (./{repo_name}/) into the repository, then commit and push.
    """
    local_repo_dir = Path(local_dir) / repo_name

    # For problem repos, the Python file name is derived by replacing hyphens with underscores.
    file_name = repo_name.replace("-", "_")  # e.g. vehicle-routing -> vehicle_routing
    problem_file = Path(f"./{repo_name}/{file_name}.py")
    config_file = Path(f"./{repo_name}/problem_config.json")

    if not problem_file.exists():
        raise FileNotFoundError(f"File {problem_file} not found.")
    if not config_file.exists():
        raise FileNotFoundError(f"File {config_file} not found.")

    # Copy files into the cloned repository folder
    shutil.copy(problem_file, local_repo_dir / problem_file.name)
    shutil.copy(config_file, local_repo_dir / config_file.name)

    # Commit and push changes
    subprocess.run(["git", "add", "."], cwd=local_repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Add problem code & config"], cwd=local_repo_dir, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=local_repo_dir, check=True)

for repo in repos:
    print(f"\n=== Processing problem repository: {repo} ===")
    if check_repo_exists(ORG, repo, GITHUB_TOKEN):
        print(f"Repository '{repo}' already exists. Skipping creation step.")
    else:
        # Create repository via Rastion CLI
        result = runner.invoke(app, [
            "create_repo", repo,
            "--org", ORG,
            "--github-token", GITHUB_TOKEN
        ])
        assert result.exit_code == 0, f"create_repo failed: {result.stdout}"

    # Create initial commit so that main branch exists
    create_initial_commit(ORG, repo)
    
    # Clone the repository using the CLI to a temporary directory
    temp_dir = tempfile.mkdtemp(prefix="combined_")
    try:
        clone_result = runner.invoke(app, [
            "clone_repo", repo,
            "--org", ORG,
            "--branch", "main",
            "--dest", temp_dir
        ])
        assert clone_result.exit_code == 0, f"clone_repo failed: {clone_result.stdout}"
        # Push local files (copy problem .py and problem_config.json) into the repo
        push_local_files(ORG, repo, temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

print("\nAll problem repositories have been created and pushed using Rastion CLI.")
