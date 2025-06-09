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

        url = f"{self.api_base}/repos/{owner}/{repo}/contents/{file_path}"

        # First, try to get the existing file to check if it exists
        get_response = requests.get(url, headers=headers)

        payload = {
            "content": base64.b64encode(content.encode()).decode(),
            "message": message,
            "branch": "main"
        }

        if get_response.status_code == 200:
            # File exists, use PUT to update it
            existing_file = get_response.json()
            payload["sha"] = existing_file["sha"]  # Required for updates
            response = requests.put(url, headers=headers, json=payload)
        else:
            # File doesn't exist, use POST to create it
            response = requests.post(url, headers=headers, json=payload)

        if response.status_code >= 300:
            raise RuntimeError(f"Failed to upload file: {response.text}")

        return response.json()

    def upload_binary_file_to_repo(self, owner: str, repo: str, file_path: str,
                                  base64_content: str, message: str = "Upload binary file") -> Dict[str, Any]:
        """
        Upload a binary file to a repository using pre-encoded base64 content.

        Args:
            owner: Repository owner
            repo: Repository name
            file_path: Path within the repository
            base64_content: Pre-encoded base64 content
            message: Commit message

        Returns:
            Upload response
        """
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"

        url = f"{self.api_base}/repos/{owner}/{repo}/contents/{file_path}"

        # First, try to get the existing file to check if it exists
        get_response = requests.get(url, headers=headers)

        payload = {
            "content": base64_content,  # Already base64 encoded
            "message": message,
            "branch": "main"
        }

        if get_response.status_code == 200:
            # File exists, use PUT to update it
            existing_file = get_response.json()
            payload["sha"] = existing_file["sha"]  # Required for updates
            response = requests.put(url, headers=headers, json=payload)
        else:
            # File doesn't exist, use POST to create it
            response = requests.post(url, headers=headers, json=payload)

        if response.status_code >= 300:
            raise RuntimeError(f"Failed to upload binary file: {response.text}")

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

        # Get complete source code with all dependencies
        try:
            class_source = inspect.getsource(model_class)

            # Get the module source to extract complete dependencies
            module = inspect.getmodule(model_class)
            if module and hasattr(module, '__file__') and module.__file__:
                try:
                    module_source = inspect.getsource(module)

                    # Use comprehensive dependency extraction for rich, heuristic problems
                    dependencies = QubotPackager._extract_complete_module_dependencies(module_source)

                    # Combine dependencies with the main class
                    if dependencies.strip():
                        source_code = dependencies + "\n\n" + class_source
                    else:
                        # Fallback to basic imports if no dependencies found
                        basic_imports = QubotPackager._get_basic_qubots_imports(model_type)
                        source_code = basic_imports + "\n\n" + class_source

                except (OSError, TypeError):
                    # Fallback: add basic qubots imports
                    basic_imports = QubotPackager._get_basic_qubots_imports(model_type)
                    source_code = basic_imports + "\n\n" + class_source
            else:
                # Fallback: add basic qubots imports
                basic_imports = QubotPackager._get_basic_qubots_imports(model_type)
                source_code = basic_imports + "\n\n" + class_source

        except OSError:
            raise ValueError(f"Cannot extract source code for {class_name}")

        # Create config.json
        config = {
            "type": model_type,
            "entry_point": "qubot",
            "class_name": class_name,
            "framework": "qubots",
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

    @staticmethod
    def package_model_from_path(model_path: str, name: str, description: str,
                               requirements: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Package a qubots model from a directory path for upload.

        This method reads existing files from the directory and packages them,
        preserving the existing config.json structure while updating metadata.

        Args:
            model_path: Path to the model directory
            name: Name for the packaged model
            description: Description of the model
            requirements: Python requirements

        Returns:
            Dictionary containing packaged files
        """
        from pathlib import Path
        import json

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

        # Validate required fields
        required_fields = ["type", "entry_point", "class_name"]
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"Missing required config fields: {missing_fields}")

        # Update metadata while preserving existing config structure
        if "metadata" not in config:
            config["metadata"] = {}

        config["metadata"]["name"] = name
        config["metadata"]["description"] = description

        # Ensure framework field is set
        if "framework" not in config:
            config["framework"] = "qubots"

        # Read qubot.py
        qubot_path = model_path / "qubot.py"
        if not qubot_path.exists():
            raise ValueError(f"qubot.py not found in {model_path}")

        # Try different encodings to handle various file formats
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
        source_code = None

        for encoding in encodings_to_try:
            try:
                with open(qubot_path, 'r', encoding=encoding) as f:
                    source_code = f.read()
                break
            except UnicodeDecodeError:
                continue

        if source_code is None:
            # Last resort: read as binary and decode with error handling
            with open(qubot_path, 'rb') as f:
                raw_content = f.read()
            source_code = raw_content.decode('utf-8', errors='replace')

        # Read or create requirements.txt
        requirements_path = model_path / "requirements.txt"
        if requirements_path.exists():
            with open(requirements_path, 'r', encoding='utf-8') as f:
                existing_requirements = [line.strip() for line in f.readlines() if line.strip()]
            # Merge with provided requirements, avoiding duplicates
            all_requirements = list(set(existing_requirements + requirements))
        else:
            all_requirements = requirements

        requirements_txt = "\n".join(all_requirements)

        # Create or read README.md
        readme_path = model_path / "README.md"
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()
        else:
            # Generate basic README
            model_type = config.get("type", "model")
            readme_content = f"""# {name}

{description}

## Model Information
- **Type**: {model_type}
- **Class**: {config.get("class_name", "Unknown")}
- **Framework**: qubots
```

## Requirements
{chr(10).join(f"- {req}" for req in all_requirements)}
"""

        # Start with core files
        packaged_files = {
            "qubot.py": source_code,
            "config.json": json.dumps(config, indent=2),
            "requirements.txt": requirements_txt,
            "README.md": readme_content
        }

        # Include additional directories and files
        additional_dirs = ["instances", "data", "datasets", "examples", "tests"]
        additional_files = [".gitignore", "LICENSE", "CHANGELOG.md"]

        # Add directories if they exist
        for dir_name in additional_dirs:
            dir_path = model_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                # Recursively add all files in the directory
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file():
                        # Create relative path for the packaged file
                        relative_path = file_path.relative_to(model_path)

                        # Read file content
                        try:
                            # Try different encodings for text files
                            encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
                            file_content = None

                            for encoding in encodings_to_try:
                                try:
                                    with open(file_path, 'r', encoding=encoding) as f:
                                        file_content = f.read()
                                    break
                                except UnicodeDecodeError:
                                    continue

                            if file_content is None:
                                # If all text encodings fail, read as binary and encode as base64
                                with open(file_path, 'rb') as f:
                                    binary_content = f.read()
                                # For binary files, we'll store them as base64 with a special prefix
                                import base64
                                file_content = f"__BINARY_BASE64__:{base64.b64encode(binary_content).decode('ascii')}"

                            packaged_files[str(relative_path).replace('\\', '/')] = file_content

                        except Exception as e:
                            print(f"Warning: Could not read file {file_path}: {e}")
                            continue

        # Add individual additional files if they exist
        for file_name in additional_files:
            file_path = model_path / file_name
            if file_path.exists() and file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        packaged_files[file_name] = f.read()
                except Exception as e:
                    print(f"Warning: Could not read file {file_path}: {e}")
                    continue

        return packaged_files

    @staticmethod
    def _extract_complete_module_dependencies(module_source: str) -> str:
        """
        Extract complete module dependencies including custom classes, imports, and helper functions.
        This approach ensures maximum compatibility with rich, heuristic problems and optimizers.

        Args:
            module_source: Source code of the module

        Returns:
            String containing all necessary dependencies for standalone execution
        """
        lines = module_source.split('\n')

        # Strategy: Extract everything except the main problem/optimizer class
        # This ensures all dependencies, custom classes, and helper functions are included

        result_lines = []
        main_class_found = False
        main_class_name = None
        main_class_indent = 0
        in_main_class = False

        # First pass: identify the main class (inherits from BaseProblem or BaseOptimizer)
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('class ') and
                ('BaseProblem' in stripped or 'BaseOptimizer' in stripped or
                 'ContinuousProblem' in stripped or 'CombinatorialProblem' in stripped or
                 'PopulationBasedOptimizer' in stripped or 'LocalSearchOptimizer' in stripped)):
                main_class_name = stripped.split('(')[0].replace('class ', '').strip()
                break

        # Second pass: extract everything except the main class
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check if this is the main class definition
            if (stripped.startswith('class ') and main_class_name and
                main_class_name in stripped and
                ('BaseProblem' in stripped or 'BaseOptimizer' in stripped or
                 'ContinuousProblem' in stripped or 'CombinatorialProblem' in stripped or
                 'PopulationBasedOptimizer' in stripped or 'LocalSearchOptimizer' in stripped)):
                # Skip the main class - it will be added separately
                main_class_indent = len(line) - len(line.lstrip())
                i += 1

                # Skip all lines that belong to the main class
                while i < len(lines):
                    next_line = lines[i]
                    if not next_line.strip():
                        i += 1
                        continue

                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_indent <= main_class_indent and next_line.strip():
                        # End of main class
                        break
                    i += 1
                continue

            # Include everything else (imports, helper classes, functions, constants)
            result_lines.append(line)
            i += 1

        return '\n'.join(result_lines)

    @staticmethod
    def _extract_imports(module_source: str) -> str:
        """
        Extract import statements and supporting classes from module source code.

        Args:
            module_source: Source code of the module

        Returns:
            String containing all import statements and supporting classes
        """
        # Check if the module contains dataclasses or supporting classes
        has_dataclass = '@dataclass' in module_source
        has_supporting_classes = any(pattern in module_source for pattern in [
            'class Customer', 'class Vehicle', 'class Node', 'class Edge',
            'class Task', 'class Resource'
        ])

        # If the module has dataclasses or supporting classes, extract the entire relevant content
        if has_dataclass or has_supporting_classes:
            lines = module_source.split('\n')
            result_lines = []

            # Add basic imports that we know work
            basic_imports = [
                'import numpy as np',
                'import random',
                'import json',
                'from typing import List, Tuple, Dict, Any, Optional',
                'from dataclasses import dataclass, asdict',
                'from datetime import datetime'
            ]

            # Check if qubots imports exist in the original module
            has_qubots_import = any('from qubots import' in line or 'import qubots' in line for line in lines)
            if has_qubots_import:
                # Determine the model type from the module content
                if 'BaseProblem' in module_source:
                    basic_imports.extend([
                        'from qubots import (',
                        '    BaseProblem, ProblemMetadata, ProblemType,',
                        '    ObjectiveType, DifficultyLevel',
                        ')'
                    ])
                elif 'BaseOptimizer' in module_source:
                    basic_imports.extend([
                        'from qubots import (',
                        '    BaseOptimizer, OptimizerMetadata, OptimizerType,',
                        '    OptimizerFamily, OptimizationResult, BaseProblem',
                        ')'
                    ])

            result_lines.extend(basic_imports)
            result_lines.append('')

            # Extract all classes and their decorators
            i = 0
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()

                # Check for @dataclass decorator or relevant class
                if (stripped.startswith('@dataclass') or
                    (stripped.startswith('class ') and
                     any(name in stripped for name in ['Customer', 'Vehicle', 'Node', 'Edge', 'Task', 'Resource',
                                                      'Problem', 'Optimizer', 'Routing', 'Scheduling', 'Portfolio']))):

                    # Start collecting the class (including decorators)
                    class_lines = []
                    class_indent = None

                    # If this is a decorator, start from here
                    if stripped.startswith('@'):
                        class_lines.append(line)
                        i += 1

                        # Look for the class definition
                        while i < len(lines):
                            next_line = lines[i]
                            next_stripped = next_line.strip()

                            if next_stripped.startswith('class '):
                                class_lines.append(next_line)
                                class_indent = len(next_line) - len(next_line.lstrip())
                                i += 1
                                break
                            elif next_stripped == '' or next_stripped.startswith('#') or next_stripped.startswith('@'):
                                class_lines.append(next_line)
                                i += 1
                            else:
                                break
                    else:
                        # This is a class definition
                        class_lines.append(line)
                        class_indent = len(line) - len(line.lstrip())
                        i += 1

                    # Now collect the class body
                    if class_indent is not None:
                        while i < len(lines):
                            next_line = lines[i]
                            next_stripped = next_line.strip()

                            if not next_stripped:
                                # Empty line
                                class_lines.append(next_line)
                                i += 1
                            else:
                                next_indent = len(next_line) - len(next_line.lstrip())
                                if next_indent > class_indent:
                                    # Part of the class
                                    class_lines.append(next_line)
                                    i += 1
                                else:
                                    # End of class
                                    break

                    # Add the collected class to results
                    if class_lines:
                        result_lines.extend(class_lines)
                        result_lines.append('')  # Add separator
                else:
                    i += 1

            return '\n'.join(result_lines)

        else:
            # Fallback: just extract imports
            import_lines = []
            for line in module_source.split('\n'):
                stripped = line.strip()
                if (stripped.startswith('import ') or stripped.startswith('from ')) and not stripped.startswith('#'):
                    if any(keyword in line.lower() for keyword in [
                        'qubots', 'numpy', 'random', 'typing', 'dataclass', 'datetime'
                    ]):
                        import_lines.append(line)

            return '\n'.join(import_lines)

    @staticmethod
    def _get_basic_qubots_imports(model_type: str) -> str:
        """
        Get basic qubots imports based on model type.

        Args:
            model_type: Type of model ('problem' or 'optimizer')

        Returns:
            String containing basic import statements
        """
        common_imports = [
            "import random",
            "import numpy as np",
            "from typing import Optional, List, Dict, Any, Tuple, Union"
        ]

        if model_type == "problem":
            qubots_imports = [
                "from qubots import (",
                "    BaseProblem, ProblemMetadata, ProblemType,",
                "    ObjectiveType, DifficultyLevel",
                ")"
            ]
        else:  # optimizer
            qubots_imports = [
                "from qubots import (",
                "    BaseOptimizer, OptimizerMetadata, OptimizerType,",
                "    OptimizerFamily, OptimizationResult, BaseProblem",
                ")"
            ]

        # Add fallback handling
        fallback_section = [
            "",
            "# Qubots imports with fallback",
            "try:",
            "    " + "\n    ".join(qubots_imports),
            "    QUBOTS_AVAILABLE = True",
            "except ImportError:",
            "    # Fallback for environments where qubots is not available",
            "    QUBOTS_AVAILABLE = False",
            "    print(\"Warning: qubots not available. Using fallback implementation.\")",
            "",
            "    # Minimal fallback classes",
            "    class BaseProblem:" if model_type == "problem" else "    class BaseOptimizer:",
            "        def __init__(self, metadata=None):",
            "            self.metadata = metadata",
            "",
            "    class ProblemMetadata:" if model_type == "problem" else "    class OptimizerMetadata:",
            "        def __init__(self, **kwargs):",
            "            for k, v in kwargs.items():",
            "                setattr(self, k, v)"
        ]

        all_imports = common_imports + fallback_section
        return '\n'.join(all_imports)


# Global client instance
_global_client = None


def get_global_client() -> RastionClient:
    """Get the global Rastion client instance."""
    global _global_client
    if _global_client is None:
        _global_client = RastionClient()
    return _global_client


def upload_qubots_model(model: Union[BaseProblem, BaseOptimizer] = None,
                       name: str = None, description: str = None,
                       requirements: Optional[List[str]] = None,
                       private: bool = False,
                       client: Optional[RastionClient] = None,
                       # Path-based upload parameters
                       path: Optional[str] = None,
                       repository_name: Optional[str] = None,
                       overwrite: bool = False) -> str:
    """
    Upload a qubots model to the Rastion platform.

    Supports two modes:
    1. Instance-based upload: Pass model, name, description
    2. Path-based upload: Pass path, repository_name, description

    Args:
        model: The model instance to upload (instance-based mode)
        name: Name for the model repository (instance-based mode)
        description: Description of the model
        requirements: Python requirements
        private: Whether the repository should be private
        client: Rastion client instance (uses global if None)
        path: Path to model directory (path-based mode)
        repository_name: Repository name (path-based mode)
        overwrite: Whether to overwrite existing repository (path-based mode)

    Returns:
        Repository URL
    """
    if client is None:
        client = get_global_client()

    if not client.is_authenticated():
        raise ValueError("Client not authenticated. Please authenticate first.")

    # Determine upload mode
    if path is not None:
        # Path-based upload mode
        if repository_name is None:
            raise ValueError("repository_name is required for path-based upload")
        if description is None:
            description = f"Qubots model uploaded from {path}"

        # Package from path
        packaged_files = QubotPackager.package_model_from_path(
            path, repository_name, description, requirements
        )
        repo_name = repository_name

    elif model is not None:
        # Instance-based upload mode
        if name is None:
            raise ValueError("name is required for instance-based upload")
        if description is None:
            raise ValueError("description is required for instance-based upload")

        # Package from model instance
        packaged_files = QubotPackager.package_model(model, name, description, requirements)
        repo_name = name

    else:
        raise ValueError("Either 'model' or 'path' must be provided")

    # Create repository
    username = client.config["gitea_username"]

    # Handle overwrite for path-based uploads
    if overwrite and path is not None:
        try:
            # Try to delete existing repository
            delete_url = f"{client.api_base}/repos/{username}/{repo_name}"
            headers = client._get_headers()
            response = requests.delete(delete_url, headers=headers)
            # Don't fail if repository doesn't exist
        except Exception:
            pass

    repo_info = client.create_repository(repo_name, private=private)

    # Upload files
    for file_path, content in packaged_files.items():
        # Handle binary files that were encoded with special prefix
        if content.startswith("__BINARY_BASE64__:"):
            # Extract the base64 content and upload as binary
            base64_content = content[len("__BINARY_BASE64__:"):]
            client.upload_binary_file_to_repo(username, repo_name, file_path, base64_content,
                                            f"Add {file_path}")
        else:
            # Regular text file
            client.upload_file_to_repo(username, repo_name, file_path, content,
                                     f"Add {file_path}")

    return repo_info["clone_url"]


def load_qubots_model(model_name: str,
                     username: Optional[str] = None,
                     revision: str = "main",
                     client: Optional[RastionClient] = None,
                     override_params: Optional[Dict[str, Any]] = None) -> Union[BaseProblem, BaseOptimizer]:
    """
    Load a qubots model from the Rastion platform with one line of code.

    Args:
        model_name: Name of the model repository
        username: Repository owner (auto-detected if None)
        revision: Git revision to load
        client: Rastion client instance (uses global if None)
        override_params: Parameters to override during model instantiation

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
        return AutoProblem.from_repo(repo_id, revision=revision, override_params=override_params)
    except Exception:
        try:
            # If that fails, try as an optimizer
            return AutoOptimizer.from_repo(repo_id, revision=revision, override_params=override_params)
        except Exception as e:
            raise ValueError(f"Failed to load model '{repo_id}': {e}")