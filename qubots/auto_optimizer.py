import os
import sys
import subprocess
import json
import importlib
from pathlib import Path
from typing import Optional

class AutoOptimizer:
    """
    Clones/pulls a repo from hub.rastion.com and
    instantiates the optimizer class named in config.json.
    """

    @classmethod
    def from_repo(
        cls,
        repo_id: str,
        revision: str = "main",
        cache_dir: str = "~/.cache/rastion_hub",
        override_params: Optional[dict] = None
    ):
        cache = os.path.expanduser(cache_dir)
        os.makedirs(cache, exist_ok=True)

        path = cls._clone_or_pull(repo_id, revision, cache)

        # 1) Install requirements if any
        req = Path(path) / "requirements.txt"
        if req.is_file():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req)],
                check=True
            )

        # 2) Load config.json
        cfg_file = Path(path) / "config.json"
        if not cfg_file.is_file():
            raise FileNotFoundError(f"No config.json in {path}")
        cfg = json.loads(cfg_file.read_text())

        if cfg.get("type") != "optimizer":
            raise ValueError(f"Expected type='optimizer' in config.json, got {cfg.get('type')}")

        entry_mod = cfg["entry_point"]        # e.g. "my_optimizer_module"
        class_name = cfg["class_name"]        # e.g. "MyOptimizer"
        params = cfg.get("default_params", {})

        if override_params:
            params.update(override_params)

        # 3) Dynamic import
        sys.path.insert(0, str(path))
        module = importlib.import_module(entry_mod)
        OptimizerCls = getattr(module, class_name)
        return OptimizerCls(**params)

    @staticmethod
    def _clone_or_pull(repo_id: str, revision: str, cache_dir: str) -> str:
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
