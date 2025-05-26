"""
Optimizer loader with registry integration, caching, and validation.
Provides dynamic loading of optimizers from GitHub repositories.
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
from .base_optimizer import BaseOptimizer
from .registry import get_global_registry, RegistryType

class AutoOptimizer:
    """
    Enhanced optimizer loader with registry integration, caching, and validation.
    Clones/pulls repos from hub.rastion.com and instantiates optimizer classes.
    """

    @classmethod
    def from_repo(
        cls,
        repo_id: str,
        revision: str = "main",
        cache_dir: str = "~/.cache/rastion_hub",
        override_params: Optional[dict] = None,
        validate_metadata: bool = True,
        register_in_registry: bool = True
    ) -> BaseOptimizer:
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
            raise ValueError(f"Failed to load model '{repo_id}': Expected type='optimizer' in config.json, got '{cfg.get('type')}'. "
                           f"This model appears to be a '{cfg.get('type')}', not an optimizer. "
                           f"Please check that you're loading the correct model type.")

        entry_mod = cfg["entry_point"]        # e.g. "my_optimizer_module"
        class_name = cfg["class_name"]        # e.g. "MyOptimizer"
        params = cfg.get("default_params", {})

        if override_params:
            params.update(override_params)

        # 3) Dynamic import with module cache handling
        sys.path.insert(0, str(path))

        # Handle module name conflicts by clearing cache if module already exists
        # This prevents conflicts when both problems and optimizers use the same module name (e.g., "qubot")
        if entry_mod in sys.modules:
            # Remove from cache to force fresh import from the correct path
            del sys.modules[entry_mod]

        module = importlib.import_module(entry_mod)
        OptimizerCls = getattr(module, class_name)

        # 4) Create optimizer instance
        optimizer_instance = OptimizerCls(**params)

        # 5) Enhanced validation and registry integration
        if validate_metadata and hasattr(optimizer_instance, 'metadata'):
            cls._validate_optimizer_metadata(optimizer_instance)

        if register_in_registry:
            try:
                registry = get_global_registry()
                repository_info = {
                    "url": f"https://hub.rastion.com/{repo_id}.git",
                    "path": repo_id,
                    "commit": cls._get_commit_hash(path)
                }
                registry.register_optimizer(optimizer_instance, repository_info)
                registry.increment_download_count(
                    cls._generate_registry_id(repo_id, cfg.get("version", "1.0.0"))
                )
            except Exception as e:
                # Don't fail if registry operations fail
                print(f"Warning: Registry operation failed: {e}")

        return optimizer_instance

    @classmethod
    def from_registry(cls, entry_id: str, override_params: Optional[dict] = None) -> BaseOptimizer:
        """
        Load an optimizer directly from the registry.

        Args:
            entry_id: Registry entry ID
            override_params: Parameters to override

        Returns:
            Optimizer instance
        """
        registry = get_global_registry()
        entry = registry.get_entry(entry_id)

        if not entry:
            raise ValueError(f"Entry {entry_id} not found in registry")

        if entry.registry_type != RegistryType.OPTIMIZER:
            raise ValueError(f"Entry {entry_id} is not an optimizer")

        # Extract repo_id from repository_path
        repo_id = entry.repository_path
        if not repo_id:
            raise ValueError(f"No repository path found for entry {entry_id}")

        return cls.from_repo(repo_id, override_params=override_params, register_in_registry=False)

    @classmethod
    def search_optimizers(cls, query: str = "", tags: Optional[List[str]] = None,
                         author: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for optimizers in the registry.

        Args:
            query: Text query
            tags: Filter by tags
            author: Filter by author

        Returns:
            List of matching optimizer entries
        """
        registry = get_global_registry()
        entries = registry.search(query, RegistryType.OPTIMIZER, tags, author)
        return [entry.to_dict() for entry in entries]

    @classmethod
    def get_compatible_optimizers(cls, problem_id: str) -> List[Dict[str, Any]]:
        """
        Get optimizers compatible with a specific problem.

        Args:
            problem_id: ID of the problem

        Returns:
            List of compatible optimizer entries
        """
        registry = get_global_registry()
        entries = registry.get_compatible_optimizers(problem_id)
        return [entry.to_dict() for entry in entries]

    @classmethod
    def get_recommendations(cls, optimizer_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get optimizer recommendations based on an optimizer.

        Args:
            optimizer_id: ID of the base optimizer
            limit: Maximum number of recommendations

        Returns:
            List of recommended optimizer entries
        """
        registry = get_global_registry()
        entries = registry.get_recommendations(optimizer_id, limit)
        return [entry.to_dict() for entry in entries]

    @staticmethod
    def _validate_optimizer_metadata(optimizer: BaseOptimizer):
        """Validate optimizer metadata for enhanced compatibility."""
        if not hasattr(optimizer, 'metadata'):
            raise ValueError("Optimizer must have metadata attribute for enhanced features")

        metadata = optimizer.metadata
        required_fields = ['name', 'description', 'optimizer_type']

        for field in required_fields:
            if not hasattr(metadata, field) or not getattr(metadata, field):
                raise ValueError(f"Optimizer metadata missing required field: {field}")

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
    def _generate_registry_id(repo_id: str, version: str) -> str:
        """Generate registry ID for tracking."""
        content = f"{repo_id}_{version}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

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
