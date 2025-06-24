"""
Unified Rastion Platform Client
Handles both Gitea operations and dataset operations with appropriate endpoints
"""

import os
import json
import requests
import tempfile
import shutil
import inspect
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from .base_problem import BaseProblem
from .base_optimizer import BaseOptimizer
from .auto_problem import AutoProblem
from .auto_optimizer import AutoOptimizer


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
    Unified client for Rastion platform operations.
    
    Handles two different API endpoints:
    - Gitea operations (repos, auth, upload): hub.rastion.com/api/v1
    - Dataset operations (download, metadata): rastion.com/api
    """

    def __init__(self, 
                 gitea_api_base: str = "https://hub.rastion.com/api/v1",
                 dataset_api_base: str = "https://rastion.com/api",
                 config_path: str = "~/.rastion/config.json"):
        """
        Initialize the unified Rastion client.

        Args:
            gitea_api_base: Base URL for Gitea operations (repos, auth, upload)
            dataset_api_base: Base URL for dataset operations (download, metadata)
            config_path: Path to store authentication config
        """
        self.gitea_api_base = gitea_api_base.rstrip('/')
        self.dataset_api_base = dataset_api_base.rstrip('/')
        self.config_path = Path(config_path).expanduser()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(config, indent=2))

    # ==================== AUTHENTICATION ====================

    def authenticate(self, token: str) -> bool:
        """
        Authenticate with the Rastion platform using Gitea endpoint.

        Args:
            token: Gitea personal access token

        Returns:
            True if authentication successful
        """
        headers = {"Authorization": f"token {token}"}
        response = requests.get(f"{self.gitea_api_base}/user", headers=headers)
        
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
        return self.config.get("authenticated", False)

    def get_token(self) -> Optional[str]:
        """Get the stored authentication token."""
        return self.config.get("gitea_token")

    def get_username(self) -> Optional[str]:
        """Get the authenticated username."""
        return self.config.get("gitea_username")

    # ==================== DATASET OPERATIONS ====================

    def load_dataset(self, dataset_id: str, token: Optional[str] = None) -> str:
        """
        Load dataset content from the platform backend.

        Args:
            dataset_id: Dataset ID from Rastion platform
            token: Authentication token (uses stored token if not provided)

        Returns:
            Dataset content as string

        Raises:
            ValueError: If dataset_id is empty or no token available
            requests.RequestException: If API request fails
        """
        if not dataset_id:
            raise ValueError("dataset_id cannot be empty")
        
        auth_token = token or self.get_token()
        if not auth_token:
            raise ValueError("No authentication token available")

        # Use dataset API endpoint
        dataset_url = f"{self.dataset_api_base}/datasets/{dataset_id}/download"
        headers = {
            "Authorization": f"token {auth_token}",
            "Accept": "application/octet-stream"
        }

        try:
            response = requests.get(dataset_url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to load dataset {dataset_id}: {str(e)}")

    def list_datasets(self, token: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available datasets from the platform backend.

        Args:
            token: Authentication token (uses stored token if not provided)

        Returns:
            List of dataset metadata dictionaries
        """
        auth_token = token or self.get_token()
        if not auth_token:
            raise ValueError("No authentication token available")

        headers = {"Authorization": f"token {auth_token}"}
        
        try:
            response = requests.get(f"{self.dataset_api_base}/datasets", headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('datasets', [])
        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to list datasets: {str(e)}")

    # ==================== GITEA REPOSITORY OPERATIONS ====================

    def upload_model_from_path(self,
                              path: str,
                              repository_name: str,
                              description: str = "",
                              requirements: Optional[List[str]] = None,
                              private: bool = False,
                              overwrite: bool = False) -> str:
        """
        Upload a qubots model from a local directory to Gitea.

        Args:
            path: Path to the model directory
            repository_name: Name for the repository
            description: Repository description
            requirements: Python requirements list
            private: Whether to make repository private
            overwrite: Whether to overwrite existing repository

        Returns:
            Repository URL

        Raises:
            ValueError: If not authenticated or invalid parameters
            requests.RequestException: If upload fails
        """
        if not self.is_authenticated():
            raise ValueError("Not authenticated. Please call authenticate(token) first.")

        username = self.get_username()

        # Check if repository already exists
        repo_exists = self._repository_exists(username, repository_name)

        if repo_exists and not overwrite:
            raise ValueError(f"Repository '{repository_name}' already exists. Use overwrite=True to replace it.")

        # Create or recreate repository
        if not repo_exists:
            print(f"📦 Creating repository '{repository_name}'...")
            self._create_repository(repository_name, private)
        else:
            print(f"🔄 Repository '{repository_name}' exists, overwriting...")

        # Package the model from path
        print(f"📁 Packaging model from {path}...")
        packaged_files = self._package_model_from_path(path, repository_name, description, requirements)

        # Upload all files
        print(f"⬆️ Uploading {len(packaged_files)} files...")
        for file_path, content in packaged_files.items():
            print(f"   Uploading {file_path}...")
            self._upload_file_to_repo(username, repository_name, file_path, content)

        repo_url = f"{self.gitea_api_base.replace('/api/v1', '')}/{username}/{repository_name}"
        print(f"✅ Model uploaded successfully to {repo_url}")

        return repo_url

    def _repository_exists(self, owner: str, repo_name: str) -> bool:
        """Check if a repository exists."""
        headers = {"Authorization": f"token {self.get_token()}"}
        response = requests.get(f"{self.gitea_api_base}/repos/{owner}/{repo_name}", headers=headers)
        return response.status_code == 200

    def _create_repository(self, repo_name: str, private: bool = False) -> Dict[str, Any]:
        """Create a new repository."""
        headers = {"Authorization": f"token {self.get_token()}"}
        payload = {
            "name": repo_name,
            "private": private,
            "auto_init": True,
            "default_branch": "main"
        }

        response = requests.post(f"{self.gitea_api_base}/user/repos", headers=headers, json=payload)

        if response.status_code >= 300:
            raise RuntimeError(f"Failed to create repository: {response.text}")

        return response.json()

    def _upload_file_to_repo(self, owner: str, repo: str, file_path: str, content: str) -> Dict[str, Any]:
        """Upload a file to a repository."""
        headers = {
            "Authorization": f"token {self.get_token()}",
            "Content-Type": "application/json"
        }

        url = f"{self.gitea_api_base}/repos/{owner}/{repo}/contents/{file_path}"

        # Check if file exists
        get_response = requests.get(url, headers=headers)

        # Handle binary content (base64 encoded)
        if content.startswith("__BINARY_BASE64__:"):
            base64_content = content[18:]  # Remove prefix
        else:
            base64_content = base64.b64encode(content.encode()).decode()

        payload = {
            "content": base64_content,
            "message": f"Upload {file_path}",
            "branch": "main"
        }

        if get_response.status_code == 200:
            # File exists, update it
            existing_file = get_response.json()
            payload["sha"] = existing_file["sha"]
            response = requests.put(url, headers=headers, json=payload)
        else:
            # File doesn't exist, create it
            response = requests.post(url, headers=headers, json=payload)

        if response.status_code >= 300:
            raise RuntimeError(f"Failed to upload file {file_path}: {response.text}")

        return response.json()

    def _package_model_from_path(self, model_path: str, name: str, description: str,
                                requirements: Optional[List[str]] = None) -> Dict[str, str]:
        """Package a qubots model from a directory path for upload."""
        if requirements is None:
            requirements = ["qubots"]

        model_path = Path(model_path)
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        # Read existing config.json
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise ValueError(f"config.json not found in {model_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Update metadata
        if "metadata" not in config:
            config["metadata"] = {}
        config["metadata"]["name"] = name
        config["metadata"]["description"] = description

        # Read qubot.py
        qubot_path = model_path / "qubot.py"
        if not qubot_path.exists():
            raise ValueError(f"qubot.py not found in {model_path}")

        with open(qubot_path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        # Read or create requirements.txt
        requirements_path = model_path / "requirements.txt"
        if requirements_path.exists():
            with open(requirements_path, 'r', encoding='utf-8') as f:
                existing_requirements = [line.strip() for line in f.readlines() if line.strip()]
            all_requirements = list(set(existing_requirements + requirements))
        else:
            all_requirements = requirements

        # Start with core files
        packaged_files = {
            "qubot.py": source_code,
            "config.json": json.dumps(config, indent=2),
            "requirements.txt": "\n".join(all_requirements)
        }

        # Add README.md if it exists
        readme_path = model_path / "README.md"
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                packaged_files["README.md"] = f.read()

        # Add additional files and directories
        additional_dirs = ["instances", "data", "datasets", "examples", "tests"]
        additional_files = [".gitignore", "LICENSE", "CHANGELOG.md"]

        # Add directories if they exist
        for dir_name in additional_dirs:
            dir_path = model_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(model_path)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                            packaged_files[str(relative_path).replace('\\', '/')] = file_content
                        except UnicodeDecodeError:
                            # Handle binary files
                            with open(file_path, 'rb') as f:
                                binary_content = f.read()
                            file_content = f"__BINARY_BASE64__:{base64.b64encode(binary_content).decode('ascii')}"
                            packaged_files[str(relative_path).replace('\\', '/')] = file_content
                        except Exception as e:
                            print(f"Warning: Could not read file {file_path}: {e}")

        # Add individual additional files
        for file_name in additional_files:
            file_path = model_path / file_name
            if file_path.exists() and file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        packaged_files[file_name] = f.read()
                except Exception as e:
                    print(f"Warning: Could not read file {file_path}: {e}")

        return packaged_files

    def load_model(self, model_name: str, username: Optional[str] = None) -> Union[BaseProblem, BaseOptimizer]:
        """
        Load a qubots model from Gitea repository.

        Args:
            model_name: Name of the model repository
            username: Repository owner (uses authenticated user if not provided)

        Returns:
            Loaded model instance
        """
        # Implementation would go here
        # This involves downloading from Gitea and instantiating the model
        pass


# ==================== GLOBAL CLIENT INSTANCE ====================

_global_client: Optional[RastionClient] = None

def get_global_client() -> RastionClient:
    """Get or create the global Rastion client instance."""
    global _global_client
    if _global_client is None:
        _global_client = RastionClient()
    return _global_client


# ==================== SIMPLIFIED API FUNCTIONS ====================

def authenticate(token: str) -> bool:
    """Authenticate with the Rastion platform."""
    return get_global_client().authenticate(token)

def is_authenticated() -> bool:
    """Check if authenticated with the Rastion platform."""
    return get_global_client().is_authenticated()

def load_dataset_from_platform(token: str, dataset_id: str) -> str:
    """Load dataset content from Rastion platform."""
    client = get_global_client()
    return client.load_dataset(dataset_id, token)

def autoLoad(dataset_id: str, rastion_token: str) -> str:
    """Auto-load dataset content (legacy function name)."""
    return load_dataset_from_platform(rastion_token, dataset_id)

def upload_qubots_model(path: str = None,
                       model: Union[BaseProblem, BaseOptimizer] = None,
                       repository_name: str = None,
                       name: str = None,
                       description: str = "",
                       requirements: Optional[List[str]] = None,
                       private: bool = False,
                       overwrite: bool = False) -> str:
    """Upload a qubots model to the platform."""
    client = get_global_client()
    if path:
        return client.upload_model_from_path(
            path=path,
            repository_name=repository_name or name,
            description=description,
            requirements=requirements,
            private=private,
            overwrite=overwrite
        )
    else:
        # Handle model instance upload
        raise NotImplementedError("Model instance upload not yet implemented")

def load_qubots_model(model_name: str,
                     username: Optional[str] = None,
                     override_params: Optional[Dict[str, Any]] = None) -> Union[BaseProblem, BaseOptimizer]:
    """Load a qubots model from the platform."""
    client = get_global_client()
    return client.load_model(model_name, username)


# ==================== DATASET CLASS ====================

class Dataset:
    """Dataset wrapper class for qubots framework."""
    
    def __init__(self, content: str, dataset_id: Optional[str] = None):
        self.content = content
        self.dataset_id = dataset_id
    
    @classmethod
    def from_rastion(cls, dataset_id: str, rastion_token: str) -> 'Dataset':
        """Create Dataset instance by loading from Rastion platform."""
        content = load_dataset_from_platform(rastion_token, dataset_id)
        return cls(content=content, dataset_id=dataset_id)
    
    @classmethod
    def from_file(cls, file_path: str) -> 'Dataset':
        """Create Dataset instance from local file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return cls(content=content)
    
    def __str__(self) -> str:
        return self.content
    
    def __len__(self) -> int:
        return len(self.content)
