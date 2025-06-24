"""
Problem loader with registry integration, caching, and validation.
Provides dynamic loading of problems from GitHub repositories.
"""

import os
import sys
import subprocess
import json
import importlib
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from .base_problem import BaseProblem

class AutoProblem:
    """
    Enhanced problem loader with caching, and validation.
    Clones/pulls repos from hub.rastion.com and instantiates problem classes.
    """

    @classmethod
    def from_repo(
        cls,
        repo_id: str,
        revision: str = "main",
        cache_dir: str = "~/.cache/rastion_hub",
        override_params: Optional[dict] = None,
        validate_metadata: bool = True,
        dataset: Optional[str] = None,
    ) -> BaseProblem:
        cache = os.path.expanduser(cache_dir)
        os.makedirs(cache, exist_ok=True)

        path = cls._clone_or_pull(repo_id, revision, cache)

        # 1) Install requirements if any
        req = Path(path) / "requirements.txt"
        if req.is_file():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--user", "-r", str(req)],
                check=True
            )

            # Force refresh of Python path to ensure newly installed packages are available
            import site
            import importlib
            site.main()  # Refresh site-packages
            importlib.invalidate_caches()  # Clear import caches

            # Ensure user site-packages is in Python path
            user_site = site.getusersitepackages()
            if user_site not in sys.path:
                sys.path.insert(0, user_site)

        # 2) Load config.json
        cfg_file = Path(path) / "config.json"
        if not cfg_file.is_file():
            raise FileNotFoundError(f"No config.json in {path}")
        cfg = json.loads(cfg_file.read_text())

        if cfg.get("type") != "problem":
            raise ValueError(f"Failed to load model '{repo_id}': Expected type='problem' in config.json, got '{cfg.get('type')}'. "
                           f"This model appears to be a '{cfg.get('type')}', not a problem. "
                           f"Please check that you're loading the correct model type.")

        entry_mod = cfg["entry_point"]       # e.g. "my_problem_module"
        class_name = cfg["class_name"]       # e.g. "MyProblem"
        params = cfg.get("default_params", {})

        if override_params:
            params.update(override_params)

        # Add dataset to params if provided (modular flow support)
        if dataset is not None:
            params["dataset"] = dataset

        # 3) Dynamic import with module cache handling
        sys.path.insert(0, str(path))

        # Handle module name conflicts by clearing cache if module already exists
        # This prevents conflicts when both problems and optimizers use the same module name (e.g., "qubot")
        # Also force fresh import after requirements installation to ensure imports work correctly
        if entry_mod in sys.modules:
            # Remove from cache to force fresh import from the correct path
            del sys.modules[entry_mod]

        # Force fresh import to ensure any newly installed packages are available
        module = importlib.import_module(entry_mod)
        ProblemCls = getattr(module, class_name)

        # 4) Create problem instance
        problem_instance = ProblemCls(**params)

        # 5) Enhanced validation and registry integration
        if validate_metadata and hasattr(problem_instance, 'metadata'):
            cls._validate_problem_metadata(problem_instance)

        return problem_instance

    @staticmethod
    def _validate_problem_metadata(problem: BaseProblem):
        """Validate problem metadata for enhanced compatibility."""
        if not hasattr(problem, 'metadata'):
            raise ValueError("Problem must have metadata attribute for enhanced features")

        metadata = problem.metadata
        required_fields = ['name', 'description', 'problem_type']

        for field in required_fields:
            if not hasattr(metadata, field) or not getattr(metadata, field):
                raise ValueError(f"Problem metadata missing required field: {field}")

    @staticmethod
    def _get_commit_hash(repo_path: str) -> str:
        """Get current commit hash of the repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return ""

    @staticmethod
    def _clone_or_pull(repo_id: str, revision: str, cache_dir: str) -> str:
        import os
        owner, name = repo_id.split("/")
        base = "https://hub.rastion.com"
        url  = f"{base.rstrip('/')}/{owner}/{name}.git"
        dest = os.path.join(cache_dir, name)

        if not os.path.isdir(dest):
            subprocess.run(["git", "clone", "--branch", revision, url, dest], check=True)
        else:
            subprocess.run(["git", "fetch", "--all"], cwd=dest, check=True)
            subprocess.run(["git", "checkout", "-f", revision], cwd=dest, check=True)
            subprocess.run(["git", "reset", "--hard", f"origin/{revision}"], cwd=dest, check=True)

        return dest
