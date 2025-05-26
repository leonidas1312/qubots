"""
Simplified Rastion interface for seamless qubots model management.
Provides the one-line interface: rastion.load_qubots_model("model_name")
"""

from typing import Union, Optional, List, Dict, Any
from .base_problem import BaseProblem
from .base_optimizer import BaseOptimizer
from .rastion_client import (
    get_global_client,
    load_qubots_model as _load_qubots_model,
    upload_qubots_model as _upload_qubots_model,
    list_available_models as _list_available_models,
    search_models as _search_models,
    RastionClient
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
                     username: Optional[str] = None) -> Union[BaseProblem, BaseOptimizer]:
    """
    Load a qubots model with one line of code.

    This is the main interface for loading models from the Rastion platform.

    Args:
        model_name: Name of the model repository
        username: Repository owner (auto-detected if None)

    Returns:
        Loaded model instance (BaseProblem or BaseOptimizer)

    Example:
        >>> import qubots.rastion as rastion
        >>> model = rastion.load_qubots_model("traveling_salesman_problem")
        >>> # or with specific username
        >>> model = rastion.load_qubots_model("tsp_solver", username="Rastion")
    """
    return _load_qubots_model(model_name, username)


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


def discover_models(query: Optional[str] = None,
                   username: Optional[str] = None,
                   limit: int = 10) -> List[Dict[str, Any]]:
    """
    Discover available qubots models on the platform.

    Args:
        query: Search query (None to list all)
        username: Filter by username
        limit: Maximum number of results

    Returns:
        List of available models with metadata

    Example:
        >>> import qubots.rastion as rastion
        >>> models = rastion.discover_models("genetic algorithm")
        >>> for model in models:
        ...     print(f"{model['name']}: {model['description']}")
    """
    if query:
        return _search_models(query, limit=limit)
    else:
        return _list_available_models(username)


def search_models(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search for qubots models on the platform.

    Args:
        query: Search query
        limit: Maximum number of results

    Returns:
        List of matching models
    """
    return _search_models(query, limit=limit)


def list_my_models() -> List[Dict[str, Any]]:
    """
    List models uploaded by the authenticated user.

    Returns:
        List of user's models
    """
    if not is_authenticated():
        raise ValueError("Not authenticated. Please call rastion.authenticate(token) first.")

    client = get_global_client()
    username = client.config.get("gitea_username")
    return _list_available_models(username)


# Convenience aliases for backward compatibility
load_model = load_qubots_model
upload = upload_model
upload_from_path = upload_model_from_path
search = search_models
discover = discover_models

# Support for legacy upload_qubots_model calls with path parameter
upload_qubots_model = _upload_qubots_model


# Example usage documentation
__doc__ += """

## Quick Start

```python
import qubots.rastion as rastion

# Authenticate (one time setup)
rastion.authenticate("your_gitea_token")

# Load a model with one line
model = rastion.load_qubots_model("traveling_salesman_problem")

# Upload your own model
url = rastion.upload_model(my_optimizer, "my_algorithm", "Description here")

# Discover available models
models = rastion.discover_models("genetic algorithm")
```

## Main Functions

- `load_qubots_model(name)`: Load a model with one line
- `upload_model(model, name, description)`: Upload your model
- `discover_models(query)`: Find available models
- `authenticate(token)`: One-time authentication setup
"""
