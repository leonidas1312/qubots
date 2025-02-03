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

# Retrieve the GitHub token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise EnvironmentError("GITHUB_TOKEN is not defined in your .env file.")

ORG = "Rastion"

# List of optimizer repository names (with hyphens)
optimizers = [
    "particle-swarm",
    "ant-colony",
    "differential-evolution",
    "tabu-search",
    "bayesian-optimization",
    "evolution-strategies",
]

runner = CliRunner()

def check_repo_exists(org, repo_name, token):
    url = f"https://api.github.com/repos/{org}/{repo_name}"
    headers = {"Authorization": f"token {token}"}
    response = requests.get(url, headers=headers)
    return response.status_code == 200

def create_initial_commit(org, repo_name):
    """Create an initial commit (with a README) to initialize the main branch."""
    with tempfile.TemporaryDirectory() as temp_repo_dir:
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
    In the cloned repository (located in local_dir/repo_name), copy the optimizer file and configuration file,
    then commit and push.
    """
    local_repo_dir = Path(local_dir) / repo_name

    # For optimizer repos, the file name is created by replacing hyphens with underscores and appending _optimizer.py.
    # For example, "particle-swarm" becomes "particle_swarm_optimizer.py".
    base_file = repo_name.replace("-", "_")
    file_name = f"{base_file}_optimizer.py"
    optimizer_file = Path(f"./{repo_name}/{file_name}")
    config_file = Path(f"./{repo_name}/solver_config.json")

    if not optimizer_file.exists():
        raise FileNotFoundError(f"File {optimizer_file} not found.")
    if not config_file.exists():
        raise FileNotFoundError(f"File {config_file} not found.")

    shutil.copy(optimizer_file, local_repo_dir / optimizer_file.name)
    shutil.copy(config_file, local_repo_dir / config_file.name)

    subprocess.run(["git", "add", "."], cwd=local_repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Add optimizer code & config"], cwd=local_repo_dir, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=local_repo_dir, check=True)

for repo in optimizers:
    print(f"\n=== Processing optimizer repository: {repo} ===")
    if check_repo_exists(ORG, repo, GITHUB_TOKEN):
        print(f"Repository '{repo}' already exists. Updating it.")
    else:
        result = runner.invoke(app, [
            "create_repo", repo,
            "--org", ORG,
            "--github-token", GITHUB_TOKEN
        ])
        assert result.exit_code == 0, f"create_repo failed: {result.stdout}"
    
    create_initial_commit(ORG, repo)
    
    # Clone the repository using the CLI
    temp_dir = tempfile.mkdtemp(prefix="combined_")
    try:
        clone_result = runner.invoke(app, [
            "clone_repo", repo,
            "--org", ORG,
            "--branch", "main",
            "--dest", temp_dir
        ])
        assert clone_result.exit_code == 0, f"clone_repo failed: {clone_result.stdout}"
        push_local_files(ORG, repo, temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

print("\nAll optimizer repositories have been created and pushed using Rastion CLI.")
