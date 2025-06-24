"""
Simplified Rastion interface for seamless qubots model management.
"""

import requests
from typing import Union, Optional, List, Dict, Any
from .base_problem import BaseProblem
from .base_optimizer import BaseOptimizer
from .rastion_client import (
    get_global_client,
    load_qubots_model as _load_qubots_model,
    upload_qubots_model as _upload_qubots_model,
)


# Global authentication state
_authenticated = False


def authenticate(token: str) -> bool:
    """
    Authenticate with the Rastion platform.

    Args:
        token: Gitea personal access token

    Returns:
        True if authentication successful
    """
    global _authenticated
    client = get_global_client()
    success = client.authenticate(token)
    _authenticated = success
    return success


def is_authenticated() -> bool:
    """Check if authenticated with the Rastion platform."""
    client = get_global_client()
    return client.is_authenticated()


def load_qubots_model(model_name: str,
                     username: Optional[str] = None,
                     override_params: Optional[Dict[str, Any]] = None) -> Union[BaseProblem, BaseOptimizer]:
    """
    Load a qubots model with one line of code.

    This is the main interface for loading models from the Rastion platform.

    Args:
        model_name: Name of the model repository
        username: Repository owner (auto-detected if None)
        override_params: Parameters to override during model instantiation

    Returns:
        Loaded model instance (BaseProblem or BaseOptimizer)

    Example:
        >>> import qubots.rastion as rastion
        >>> model = rastion.load_qubots_model("traveling_salesman_problem")
        >>> # or with specific username
        >>> model = rastion.load_qubots_model("tsp_solver", username="Rastion")
        >>> # or with parameter overrides
        >>> model = rastion.load_qubots_model("maxcut_problem", override_params={"n_vertices": 80})
    """
    return _load_qubots_model(model_name, username, override_params=override_params)


def upload_model(model: Union[BaseProblem, BaseOptimizer],
                name: str, description: str,
                requirements: Optional[List[str]] = None,
                private: bool = False) -> str:
    """
    Upload a qubots model to the Rastion platform.

    Args:
        model: The model instance to upload
        name: Name for the model repository
        description: Description of the model
        requirements: Python requirements (defaults to ["qubots"])
        private: Whether the repository should be private

    Returns:
        Repository URL

    Example:
        >>> import qubots.rastion as rastion
        >>> # Assuming you have a model instance
        >>> url = rastion.upload_model(my_optimizer, "my_genetic_algorithm",
        ...                           "A custom genetic algorithm for TSP")
    """
    if not is_authenticated():
        raise ValueError("Not authenticated. Please call rastion.authenticate(token) first.")

    return _upload_qubots_model(model=model, name=name, description=description,
                               requirements=requirements, private=private)


def upload_model_from_path(path: str, repository_name: str, description: str,
                          requirements: Optional[List[str]] = None,
                          private: bool = False, overwrite: bool = False) -> str:
    """
    Upload a qubots model from a directory path to the Rastion platform.

    Args:
        path: Path to the model directory containing qubot.py and config.json
        repository_name: Name for the model repository
        description: Description of the model
        requirements: Python requirements (defaults to ["qubots"])
        private: Whether the repository should be private
        overwrite: Whether to overwrite existing repository

    Returns:
        Repository URL

    Example:
        >>> import qubots.rastion as rastion
        >>> # Upload from a directory
        >>> url = rastion.upload_model_from_path("./my_vrp_problem",
        ...                                     "my_vrp_problem",
        ...                                     "My VRP Problem")
    """
    if not is_authenticated():
        raise ValueError("Not authenticated. Please call rastion.authenticate(token) first.")

    return _upload_qubots_model(path=path, repository_name=repository_name,
                               description=description, requirements=requirements,
                               private=private, overwrite=overwrite)

# Convenience aliases for backward compatibility
load_model = load_qubots_model
upload = upload_model
upload_from_path = upload_model_from_path

# Support for legacy upload_qubots_model calls with path parameter
upload_qubots_model = _upload_qubots_model


def autoLoad(dataset_id: str, rastion_token: str, api_base: str = "https://hub.rastion.com/api/v1") -> str:
    """
    Auto-load dataset content from Rastion platform using dataset ID and token.

    This function provides a simple interface for loading datasets that can be passed
    directly to problems that support dataset input.

    Args:
        dataset_id: The dataset ID from Rastion platform
        rastion_token: User's Rastion authentication token
        api_base: Base URL for the Rastion API (default: https://hub.rastion.com/api/v1)

    Returns:
        Dataset content as string

    Raises:
        ValueError: If dataset_id or rastion_token is empty
        requests.RequestException: If API request fails

    Example:
        >>> import qubots.rastion as rastion
        >>> dataset_content = rastion.autoLoad("dataset_123", "your_token")
        >>> problem = TSPProblem(dataset=dataset_content)
    """
    if not dataset_id:
        raise ValueError("dataset_id cannot be empty")
    if not rastion_token:
        raise ValueError("rastion_token cannot be empty")

    # Construct API endpoint for dataset download
    dataset_url = f"{api_base.rstrip('/')}/datasets/{dataset_id}/download"

    # Set up headers with authentication
    headers = {
        "Authorization": f"token {rastion_token}",
        "Accept": "application/octet-stream"
    }

    try:
        # Make API request to download dataset
        response = requests.get(dataset_url, headers=headers, timeout=30)
        response.raise_for_status()

        # Return dataset content as string
        return response.text

    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to load dataset {dataset_id}: {str(e)}")


class Dataset:
    """
    Dataset wrapper class for modular dataset handling.

    This class provides a clean interface for working with datasets in the
    modular qubots architecture: datasets -> problems -> optimizers -> results.
    """

    def __init__(self, content: str, dataset_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize dataset with content and optional metadata.

        Args:
            content: Dataset content as string
            dataset_id: Optional dataset ID for reference
            metadata: Optional metadata dictionary
        """
        self.content = content
        self.dataset_id = dataset_id
        self.metadata = metadata or {}

    @classmethod
    def from_rastion(cls, dataset_id: str, rastion_token: str, api_base: str = "https://hub.rastion.com/api/v1") -> 'Dataset':
        """
        Create Dataset instance by loading from Rastion platform.

        Args:
            dataset_id: The dataset ID from Rastion platform
            rastion_token: User's Rastion authentication token
            api_base: Base URL for the Rastion API

        Returns:
            Dataset instance with loaded content
        """
        content = autoLoad(dataset_id, rastion_token, api_base)
        return cls(content=content, dataset_id=dataset_id)

    @classmethod
    def from_file(cls, file_path: str) -> 'Dataset':
        """
        Create Dataset instance from local file.

        Args:
            file_path: Path to local dataset file

        Returns:
            Dataset instance with file content
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return cls(content=content)

    def __str__(self) -> str:
        """Return dataset content as string."""
        return self.content

    def __len__(self) -> int:
        """Return length of dataset content."""
        return len(self.content)