import os
import sys
import subprocess
import json
import importlib
from pathlib import Path

class AutoProblem:
    """
    Similar approach as AutoOptimizerGit, but for 'problem_config.json'
    which references 'entry_point': 'module:ClassName'.
    """

    @classmethod
    def from_repo(
        cls,
        repo_id: str,
        revision: str = "main",
        cache_dir: str = "~/.cache/rastion_hub"
    ):
        cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        local_repo_path = cls._clone_or_pull(repo_id, revision, cache_dir)

        config_path = Path(local_repo_path) / "problem_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"No problem_config.json found in {config_path}")

        with open(config_path, "r") as f:
            config_data = json.load(f)

        entry_point = config_data.get("entry_point")
        if not entry_point or ":" not in entry_point:
            raise ValueError("Invalid 'entry_point' in problem_config.json. Must be 'module:ClassName'.")

        module_path, class_name = entry_point.split(":")
        if str(local_repo_path) not in sys.path:
            sys.path.insert(0, str(local_repo_path))

        mod = importlib.import_module(module_path)
        ProblemClass = getattr(mod, class_name)

        # Optionally parse 'default_params' for the problem, if you want
        # or the user might have data references, etc. For now, we just instantiate.
        problem_instance = ProblemClass()
        return problem_instance

    @staticmethod
    def _clone_or_pull(repo_id: str, revision: str, cache_dir: str) -> str:
        owner, repo_name = repo_id.split("/")
        repo_url = f"https://github.com/{owner}/{repo_name}.git"
        local_repo_path = os.path.join(cache_dir, repo_name)

        if not os.path.isdir(local_repo_path):
            subprocess.run(["git", "clone", "--branch", revision, repo_url, local_repo_path], check=True)
        else:
            subprocess.run(["git", "fetch"], cwd=local_repo_path, check=True)
            subprocess.run(["git", "checkout", revision], cwd=local_repo_path, check=True)
            subprocess.run(["git", "pull", "--allow-unrelated-histories"], cwd=local_repo_path, check=True)

        return local_repo_path
