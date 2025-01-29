import os
import sys
import subprocess
import json
import importlib
from pathlib import Path
from typing import Optional

class AutoOptimizer:
    """
    A loader that clones/pulls a GitHub repo containing
    solver code + a solver_config.json with an `entry_point` and `default_params`.
    """

    @classmethod
    def from_repo(
        cls,
        repo_id: str,              # "org_name/repo_name"
        revision: str = "main",    # branch or tag
        cache_dir: str = "~/.cache/rastion_hub",
        override_params: Optional[dict] = None
    ):
        """
        :param repo_id: e.g. "rastion-hub/GeneticTSPv1"
        :param revision: e.g. "main" or "v1.0.0"
        :param cache_dir: where to clone/pull repos
        :param override_params: extra hyperparams to override the solver's default
        """
        cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        local_repo_path = cls._clone_or_pull(repo_id, revision, cache_dir)

        # read solver_config.json
        config_path = Path(local_repo_path) / "solver_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"No solver_config.json found in {config_path}")
        with open(config_path, "r") as f:
            config_data = json.load(f)

        entry_point = config_data.get("entry_point")
        if not entry_point or ":" not in entry_point:
            raise ValueError("Invalid 'entry_point' in solver_config.json. Expected 'module.path:ClassName'.")
        default_params = config_data.get("default_params", {})

        if override_params:
            final_params = {**default_params, **override_params}
        else:
            final_params = default_params

        # dynamic import
        module_path, class_name = entry_point.split(":")
        if str(local_repo_path) not in sys.path:
            sys.path.insert(0, str(local_repo_path))

        mod = importlib.import_module(module_path)
        SolverClass = getattr(mod, class_name)

        solver_instance = SolverClass(**final_params)
        return solver_instance

    @staticmethod
    def _clone_or_pull(repo_id: str, revision: str, cache_dir: str) -> str:
        """
        Clone or pull the GitHub repo. Return the local path.
        """
        owner, repo_name = repo_id.split("/")
        repo_url = f"https://github.com/{owner}/{repo_name}.git"
        local_repo_path = os.path.join(cache_dir, repo_name)

        if not os.path.isdir(local_repo_path):
            # clone
            subprocess.run(["git", "clone", "--branch", revision, repo_url, local_repo_path], check=True)
        else:
            # pull
            subprocess.run(["git", "fetch"], cwd=local_repo_path, check=True)
            subprocess.run(["git", "checkout", revision], cwd=local_repo_path, check=True)
            subprocess.run(["git", "pull"], cwd=local_repo_path, check=True)

        return local_repo_path
