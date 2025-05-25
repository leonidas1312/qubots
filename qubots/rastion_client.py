"""
Rastion Platform Client Integration for Qubots
Provides seamless upload, download, and management of optimization models.
"""

import os
import json
import requests
import tempfile
import shutil
import inspect
import pickle
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from .base_problem import BaseProblem
from .base_optimizer import BaseOptimizer
from .auto_problem import AutoProblem
from .auto_optimizer import AutoOptimizer
from .registry import get_global_registry, RegistryType


@dataclass
class ModelMetadata:
    """Metadata for uploaded qubots models."""
    name: str
    description: str
    author: str
    version: str
    model_type: str  # 'problem' or 'optimizer'
    tags: List[str]
    dependencies: List[str]
    python_requirements: List[str]
    created_at: datetime
    repository_url: str = ""
    repository_path: str = ""


class RastionClient:
    """
    Enhanced client for interacting with the Rastion platform.
    Provides seamless upload, download, and management of qubots models.
    """

    def __init__(self, api_base: str = "https://hub.rastion.com/api/v1",
                 config_path: str = "~/.rastion/config.json"):
        """
        Initialize the Rastion client.

        Args:
            api_base: Base URL for the Rastion API
            config_path: Path to configuration file
        """
        self.api_base = api_base
        self.config_path = Path(config_path).expanduser()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {}

    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(config, indent=2))

    def authenticate(self, token: str) -> bool:
        """
        Authenticate with the Rastion platform.

        Args:
            token: Gitea personal access token

        Returns:
            True if authentication successful
        """
        headers = {"Authorization": f"token {token}"}
        response = requests.get(f"{self.api_base}/user", headers=headers)
        if response.status_code == 200:
            user_data = response.json()
            self.config = {
                "gitea_token": token,
                "gitea_username": user_data["login"],
                "authenticated": True
            }
            self._save_config(self.config)
            return True
        return False

    def is_authenticated(self) -> bool:
        """Check if client is authenticated."""
        return self.config.get("authenticated", False) and "gitea_token" in self.config

    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        if not self.is_authenticated():
            raise ValueError("Not authenticated. Please call authenticate() first.")
        return {"Authorization": f"token {self.config['gitea_token']}"}

    def create_repository(self, repo_name: str, private: bool = False) -> Dict[str, Any]:
        """
        Create a new repository on the Rastion platform.

        Args:
            repo_name: Name of the repository
            private: Whether the repository should be private

        Returns:
            Repository information
        """
        headers = self._get_headers()
        payload = {
            "name": repo_name,
            "private": private,
            "auto_init": True,
            "default_branch": "main"
        }

        response = requests.post(f"{self.api_base}/user/repos",
                               headers=headers, json=payload)

        if response.status_code >= 300:
            raise RuntimeError(f"Failed to create repository: {response.text}")

        return response.json()

    def upload_file_to_repo(self, owner: str, repo: str, file_path: str,
                           content: str, message: str = "Upload file") -> Dict[str, Any]:
        """
        Upload a file to a repository.

        Args:
            owner: Repository owner
            repo: Repository name
            file_path: Path within the repository
            content: File content
            message: Commit message

        Returns:
            Upload response
        """
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"

        payload = {
            "content": base64.b64encode(content.encode()).decode(),
            "message": message,
            "branch": "main"
        }

        url = f"{self.api_base}/repos/{owner}/{repo}/contents/{file_path}"
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code >= 300:
            raise RuntimeError(f"Failed to upload file: {response.text}")

        return response.json()

    def list_repositories(self, username: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List repositories for a user.

        Args:
            username: Username (defaults to authenticated user)

        Returns:
            List of repositories
        """
        if username is None:
            username = self.config.get("gitea_username")
            if not username:
                raise ValueError("No username provided and not authenticated")

        headers = self._get_headers() if self.is_authenticated() else {}
        response = requests.get(f"{self.api_base}/users/{username}/repos", headers=headers)

        if response.status_code >= 300:
            raise RuntimeError(f"Failed to list repositories: {response.text}")

        return response.json()

    def search_repositories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for repositories.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching repositories
        """
        params = {"q": query, "limit": limit}
        response = requests.get(f"{self.api_base}/repos/search", params=params)

        if response.status_code >= 300:
            raise RuntimeError(f"Failed to search repositories: {response.text}")

        return response.json().get("data", [])


class QubotPackager:
    """
    Utility class for packaging qubots models for upload to the Rastion platform.
    """

    @staticmethod
    def package_model(model: Union[BaseProblem, BaseOptimizer],
                     name: str, description: str,
                     requirements: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Package a qubots model for upload.

        Args:
            model: The model instance to package
            name: Name for the packaged model
            description: Description of the model
            requirements: Python requirements

        Returns:
            Dictionary containing packaged files
        """
        if requirements is None:
            requirements = ["qubots"]

        # Determine model type
        model_type = "problem" if isinstance(model, BaseProblem) else "optimizer"

        # Extract class information
        model_class = model.__class__
        module_name = model_class.__module__
        class_name = model_class.__name__

        # Get source code
        try:
            source_code = inspect.getsource(model_class)
        except OSError:
            raise ValueError(f"Cannot extract source code for {class_name}")

        # Create config.json
        config = {
            "type": model_type,
            "entry_point": "qubot",
            "class_name": class_name,
            "default_params": {},
            "metadata": {
                "name": name,
                "description": description,
                "author": getattr(model.metadata, 'author', 'Unknown') if hasattr(model, 'metadata') else 'Unknown',
                "version": getattr(model.metadata, 'version', '1.0.0') if hasattr(model, 'metadata') else '1.0.0',
                "tags": list(getattr(model.metadata, 'tags', set())) if hasattr(model, 'metadata') else []
            }
        }

        # Create requirements.txt
        requirements_txt = "\n".join(requirements)

        

        return {
            "qubot.py": source_code,
            "config.json": json.dumps(config, indent=2),
            "requirements.txt": requirements_txt,
        }


# Global client instance
_global_client = None


def get_global_client() -> RastionClient:
    """Get the global Rastion client instance."""
    global _global_client
    if _global_client is None:
        _global_client = RastionClient()
    return _global_client


def upload_qubots_model(model: Union[BaseProblem, BaseOptimizer],
                       name: str, description: str,
                       requirements: Optional[List[str]] = None,
                       private: bool = False,
                       client: Optional[RastionClient] = None) -> str:
    """
    Upload a qubots model to the Rastion platform.

    Args:
        model: The model instance to upload
        name: Name for the model repository
        description: Description of the model
        requirements: Python requirements
        private: Whether the repository should be private
        client: Rastion client instance (uses global if None)

    Returns:
        Repository URL
    """
    if client is None:
        client = get_global_client()

    if not client.is_authenticated():
        raise ValueError("Client not authenticated. Please authenticate first.")

    # Package the model
    packaged_files = QubotPackager.package_model(model, name, description, requirements)

    # Create repository
    username = client.config["gitea_username"]
    repo_info = client.create_repository(name, private=private)

    # Upload files
    for file_path, content in packaged_files.items():
        client.upload_file_to_repo(username, name, file_path, content,
                                 f"Add {file_path}")

    # Register in local registry
    try:
        registry = get_global_registry()
        repository_info = {
            "url": repo_info["clone_url"],
            "path": f"{username}/{name}",
            "commit": "main"
        }

        if isinstance(model, BaseProblem):
            registry.register_problem(model, repository_info)
        else:
            registry.register_optimizer(model, repository_info)
    except Exception as e:
        print(f"Warning: Failed to register in local registry: {e}")

    return repo_info["clone_url"]


def load_qubots_model(model_name: str,
                     username: Optional[str] = None,
                     revision: str = "main",
                     client: Optional[RastionClient] = None) -> Union[BaseProblem, BaseOptimizer]:
    """
    Load a qubots model from the Rastion platform with one line of code.

    Args:
        model_name: Name of the model repository
        username: Repository owner (auto-detected if None)
        revision: Git revision to load
        client: Rastion client instance (uses global if None)

    Returns:
        Loaded model instance
    """
    if client is None:
        client = get_global_client()

    # If username not provided, try to find the model
    if username is None:
        # Search for the model
        search_results = client.search_repositories(model_name)

        if not search_results:
            raise ValueError(f"Model '{model_name}' not found")

        # Use the first result
        repo = search_results[0]
        username = repo["owner"]["login"]
        model_name = repo["name"]

    repo_id = f"{username}/{model_name}"

    # Try to determine if it's a problem or optimizer by checking config
    try:
        # First, try to load as a problem
        return AutoProblem.from_repo(repo_id, revision=revision)
    except Exception:
        try:
            # If that fails, try as an optimizer
            return AutoOptimizer.from_repo(repo_id, revision=revision)
        except Exception as e:
            raise ValueError(f"Failed to load model '{repo_id}': {e}")


def list_available_models(username: Optional[str] = None,
                         model_type: Optional[str] = None,
                         client: Optional[RastionClient] = None) -> List[Dict[str, Any]]:
    """
    List available qubots models on the Rastion platform.

    Args:
        username: Filter by username (None for all users)
        model_type: Filter by model type ('problem' or 'optimizer')
        client: Rastion client instance (uses global if None)

    Returns:
        List of available models with metadata
    """
    if client is None:
        client = get_global_client()

    if username:
        repos = client.list_repositories(username)
    else:
        # Search for qubots repositories
        repos = client.search_repositories("qubots", limit=100)

    models = []
    for repo in repos:
        # Try to get config.json to determine if it's a qubots model
        try:
            # This is a simplified check - in a real implementation,
            # you'd fetch the config.json file from the repository
            model_info = {
                "name": repo["name"],
                "description": repo.get("description", ""),
                "owner": repo["owner"]["login"],
                "url": repo["clone_url"],
                "updated_at": repo.get("updated_at"),
                "stars": repo.get("stars_count", 0)
            }

            # Add type if filtering is requested
            if model_type is None:
                models.append(model_info)
            # Note: In a real implementation, you'd fetch and parse config.json
            # to determine the actual type

        except Exception:
            continue

    return models


def search_models(query: str,
                 model_type: Optional[str] = None,
                 limit: int = 10,
                 client: Optional[RastionClient] = None) -> List[Dict[str, Any]]:
    """
    Search for qubots models on the Rastion platform.

    Args:
        query: Search query
        model_type: Filter by model type ('problem' or 'optimizer')
        limit: Maximum number of results
        client: Rastion client instance (uses global if None)

    Returns:
        List of matching models
    """
    if client is None:
        client = get_global_client()

    # Enhance query to include qubots-specific terms
    enhanced_query = f"{query} qubots"

    repos = client.search_repositories(enhanced_query, limit=limit)

    models = []
    for repo in repos:
        model_info = {
            "name": repo["name"],
            "description": repo.get("description", ""),
            "owner": repo["owner"]["login"],
            "url": repo["clone_url"],
            "updated_at": repo.get("updated_at"),
            "stars": repo.get("stars_count", 0)
        }
        models.append(model_info)

    return models
