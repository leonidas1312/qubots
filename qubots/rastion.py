"""
Simplified Rastion interface for seamless qubots model management.
"""

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