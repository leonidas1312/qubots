"""
Qubots Playground Integration Module

Provides standardized interfaces for integrating qubots with the Rastion platform playground.
Handles result formatting, progress reporting, and error management for web-based optimization.
"""

import json
import time
import traceback
import inspect
from typing import Dict, Any, Optional, Union, List, Callable
from datetime import datetime
from dataclasses import dataclass, asdict

from .base_problem import BaseProblem
from .base_optimizer import BaseOptimizer
from .rastion_client import get_global_client
from .rastion import load_qubots_model
from .dashboard import QubotsAutoDashboard, DashboardResult


@dataclass
class PlaygroundResult:
    """Standardized result format for playground optimization runs."""
    success: bool
    problem_name: str
    optimizer_name: str
    problem_username: str
    optimizer_username: str
    execution_time: float
    timestamp: str
    best_solution: Optional[List[float]] = None
    best_value: Optional[float] = None
    iterations: Optional[int] = None
    history: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ModelInfo:
    """Information about a qubots model for playground display."""
    name: str
    username: str
    description: str
    model_type: str  # 'problem' or 'optimizer'
    repository_url: str
    last_updated: str
    tags: List[str]
    metadata: Dict[str, Any]


class PlaygroundExecutor:
    """
    Handles execution of qubots optimizations for the playground interface.
    Provides standardized result formatting and error handling.
    """

    def __init__(self, progress_callback: Optional[Callable[[str, float], None]] = None):
        """
        Initialize the playground executor.

        Args:
            progress_callback: Optional callback for progress updates (message, progress_percent)
        """
        self.progress_callback = progress_callback
        self.client = get_global_client()

    def execute_optimization(self,
                           problem_name: str,
                           optimizer_name: str,
                           problem_username: Optional[str] = None,
                           optimizer_username: Optional[str] = None,
                           problem_params: Optional[Dict[str, Any]] = None,
                           optimizer_params: Optional[Dict[str, Any]] = None) -> PlaygroundResult:
        """
        Execute an optimization using qubots models from the Rastion platform.

        Args:
            problem_name: Name of the problem repository
            optimizer_name: Name of the optimizer repository
            problem_username: Username of problem owner (auto-detected if None)
            optimizer_username: Username of optimizer owner (auto-detected if None)
            problem_params: Optional parameters to override problem defaults
            optimizer_params: Optional parameters to override optimizer defaults

        Returns:
            PlaygroundResult with execution details and results
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()

        try:
            # Report progress
            self._report_progress("Loading problem model...", 10)

            # Load problem
            problem = load_qubots_model(problem_name, problem_username)
            if not isinstance(problem, BaseProblem):
                raise ValueError(f"Model {problem_name} is not a valid problem")

            # Apply problem parameter overrides if provided
            if problem_params:
                for key, value in problem_params.items():
                    if hasattr(problem, key):
                        setattr(problem, key, value)

            self._report_progress("Loading optimizer model...", 30)

            # Load optimizer
            optimizer = load_qubots_model(optimizer_name, optimizer_username)
            if not isinstance(optimizer, BaseOptimizer):
                raise ValueError(f"Model {optimizer_name} is not a valid optimizer")

            # Apply optimizer parameter overrides if provided
            if optimizer_params:
                for key, value in optimizer_params.items():
                    if hasattr(optimizer, key):
                        setattr(optimizer, key, value)

            self._report_progress("Running optimization...", 50)

            # Execute optimization
            result = optimizer.optimize(problem)

            self._report_progress("Processing results...", 90)

            # Extract results in standardized format
            execution_time = time.time() - start_time

            # Handle different result formats
            if hasattr(result, 'best_solution'):
                best_solution = result.best_solution
                best_value = getattr(result, 'best_value', None) or getattr(result, 'best_fitness', None)
            elif isinstance(result, dict):
                best_solution = result.get('best_solution')
                best_value = result.get('best_value') or result.get('best_fitness')
            else:
                best_solution = None
                best_value = None

            # Extract iteration history if available
            history = None
            iterations = None
            if hasattr(result, 'history'):
                history = result.history
                iterations = len(history) if history else None
            elif isinstance(result, dict) and 'history' in result:
                history = result['history']
                iterations = len(history) if history else None

            # Collect metadata
            metadata = {
                'problem_class': problem.__class__.__name__,
                'optimizer_class': optimizer.__class__.__name__,
                'problem_metadata': getattr(problem, 'metadata', {}),
                'optimizer_metadata': getattr(optimizer, 'metadata', {}),
                'result_type': type(result).__name__
            }

            self._report_progress("Complete!", 100)

            return PlaygroundResult(
                success=True,
                problem_name=problem_name,
                optimizer_name=optimizer_name,
                problem_username=problem_username or "unknown",
                optimizer_username=optimizer_username or "unknown",
                execution_time=execution_time,
                timestamp=timestamp,
                best_solution=best_solution,
                best_value=best_value,
                iterations=iterations,
                history=history,
                metadata=metadata
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            error_type = type(e).__name__

            # Log full traceback for debugging
            print(f"Playground execution error: {traceback.format_exc()}")

            return PlaygroundResult(
                success=False,
                problem_name=problem_name,
                optimizer_name=optimizer_name,
                problem_username=problem_username or "unknown",
                optimizer_username=optimizer_username or "unknown",
                execution_time=execution_time,
                timestamp=timestamp,
                error_message=error_message,
                error_type=error_type
            )

    def _report_progress(self, message: str, progress: float):
        """Report progress if callback is available."""
        if self.progress_callback:
            self.progress_callback(message, progress)


class ModelDiscovery:
    """
    Handles discovery and listing of available qubots models for the playground.
    """

    def __init__(self):
        self.client = get_global_client()

    def get_user_problems(self, username: Optional[str] = None) -> List[ModelInfo]:
        """
        Get list of problem models available to the user.

        Args:
            username: Username to filter by (current user if None)

        Returns:
            List of ModelInfo objects for problems
        """
        try:
            # Use the existing search functionality
            repositories = self.client.search_repositories("type:problem", limit=100)

            problems = []
            for repo in repositories:
                # Filter by username if specified
                if username and repo.get('owner', {}).get('login') != username:
                    continue

                # Extract model information
                model_info = ModelInfo(
                    name=repo['name'],
                    username=repo['owner']['login'],
                    description=repo.get('description', ''),
                    model_type='problem',
                    repository_url=repo['html_url'],
                    last_updated=repo.get('updated_at', ''),
                    tags=repo.get('topics', []),
                    metadata={
                        'stars': repo.get('stars_count', 0),
                        'forks': repo.get('forks_count', 0),
                        'size': repo.get('size', 0)
                    }
                )
                problems.append(model_info)

            return problems

        except Exception as e:
            print(f"Error fetching user problems: {e}")
            return []

    def get_user_optimizers(self, username: Optional[str] = None) -> List[ModelInfo]:
        """
        Get list of optimizer models available to the user.

        Args:
            username: Username to filter by (current user if None)

        Returns:
            List of ModelInfo objects for optimizers
        """
        try:
            # Use the existing search functionality
            repositories = self.client.search_repositories("type:optimizer", limit=100)

            optimizers = []
            for repo in repositories:
                # Filter by username if specified
                if username and repo.get('owner', {}).get('login') != username:
                    continue

                # Extract model information
                model_info = ModelInfo(
                    name=repo['name'],
                    username=repo['owner']['login'],
                    description=repo.get('description', ''),
                    model_type='optimizer',
                    repository_url=repo['html_url'],
                    last_updated=repo.get('updated_at', ''),
                    tags=repo.get('topics', []),
                    metadata={
                        'stars': repo.get('stars_count', 0),
                        'forks': repo.get('forks_count', 0),
                        'size': repo.get('size', 0)
                    }
                )
                optimizers.append(model_info)

            return optimizers

        except Exception as e:
            print(f"Error fetching user optimizers: {e}")
            return []

    def search_models(self, query: str, model_type: Optional[str] = None) -> List[ModelInfo]:
        """
        Search for models by query.

        Args:
            query: Search query
            model_type: Filter by 'problem' or 'optimizer' (None for both)

        Returns:
            List of matching ModelInfo objects
        """
        try:
            # Construct search query
            search_query = query
            if model_type:
                search_query += f" type:{model_type}"

            repositories = self.client.search_repositories(search_query, limit=50)

            models = []
            for repo in repositories:
                # Determine model type from repository metadata or topics
                repo_type = 'unknown'
                topics = repo.get('topics', [])
                if 'problem' in topics or 'qubots-problem' in topics:
                    repo_type = 'problem'
                elif 'optimizer' in topics or 'qubots-optimizer' in topics:
                    repo_type = 'optimizer'

                # Skip if type filter doesn't match
                if model_type and repo_type != model_type:
                    continue

                model_info = ModelInfo(
                    name=repo['name'],
                    username=repo['owner']['login'],
                    description=repo.get('description', ''),
                    model_type=repo_type,
                    repository_url=repo['html_url'],
                    last_updated=repo.get('updated_at', ''),
                    tags=topics,
                    metadata={
                        'stars': repo.get('stars_count', 0),
                        'forks': repo.get('forks_count', 0),
                        'size': repo.get('size', 0)
                    }
                )
                models.append(model_info)

            return models

        except Exception as e:
            print(f"Error searching models: {e}")
            return []


# Convenience functions for direct use
def execute_playground_optimization(problem_name: str = None,
                                  optimizer_name: str = None,
                                  problem_username: Optional[str] = None,
                                  optimizer_username: Optional[str] = None,
                                  # Directory-based execution parameters
                                  problem_dir: Optional[str] = None,
                                  optimizer_dir: Optional[str] = None,
                                  **kwargs) -> Dict[str, Any]:
    """
    Convenience function to execute optimization and return dashboard result as dictionary.
    Uses qubots built-in dashboard and visualization capabilities.

    Supports two modes:
    1. Name-based: Load models by name from Rastion platform
    2. Directory-based: Load models from local directories
    """
    try:
        # Determine execution mode
        if problem_dir is not None and optimizer_dir is not None:
            # Directory-based execution mode
            problem = _load_model_from_directory(problem_dir, "problem")
            optimizer = _load_model_from_directory(optimizer_dir, "optimizer")

            # Use directory names as display names if not provided
            if problem_name is None:
                from pathlib import Path
                problem_name = Path(problem_dir).name
            if optimizer_name is None:
                from pathlib import Path
                optimizer_name = Path(optimizer_dir).name

        elif problem_name is not None and optimizer_name is not None:
            # Name-based execution mode
            problem = load_qubots_model(problem_name, problem_username)
            optimizer = load_qubots_model(optimizer_name, optimizer_username)

        else:
            raise ValueError("Either provide (problem_name, optimizer_name) or (problem_dir, optimizer_dir)")

        # Apply parameters if provided
        problem_params = kwargs.get('problem_params', {})
        optimizer_params = kwargs.get('optimizer_params', {})

        # Configure problem and optimizer with parameters
        if problem_params:
            for key, value in problem_params.items():
                if hasattr(problem, key):
                    setattr(problem, key, value)

        if optimizer_params:
            for key, value in optimizer_params.items():
                if hasattr(optimizer, key):
                    setattr(optimizer, key, value)

        # Run optimization with automatic dashboard generation
        dashboard_result = QubotsAutoDashboard.auto_optimize_with_dashboard(
            problem=problem,
            optimizer=optimizer,
            problem_name=problem_name,
            optimizer_name=optimizer_name
        )

        return dashboard_result.to_dict()

    except Exception as e:
        # Return error dashboard result
        import traceback
        error_result = DashboardResult(
            success=False,
            problem_name=problem_name or "unknown",
            optimizer_name=optimizer_name or "unknown",
            execution_time=0.0,
            error_message=str(e)
        )
        result_dict = error_result.to_dict()
        result_dict["traceback"] = traceback.format_exc()
        return result_dict


def _load_model_from_directory(directory: str, expected_type: str):
    """
    Load a qubots model from a directory containing qubot.py and config.json.

    Args:
        directory: Path to the directory
        expected_type: Expected model type ("problem" or "optimizer")

    Returns:
        Loaded model instance
    """
    import sys
    import json
    import importlib.util
    from pathlib import Path

    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    # Load config.json
    config_path = directory / "config.json"
    if not config_path.exists():
        raise ValueError(f"config.json not found in {directory}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Validate type
    model_type = config.get("type")
    if model_type != expected_type:
        raise ValueError(f"Expected type='{expected_type}' in config.json, got '{model_type}'")

    # Load the module
    qubot_path = directory / "qubot.py"
    if not qubot_path.exists():
        raise ValueError(f"qubot.py not found in {directory}")

    # Add directory to Python path temporarily
    sys.path.insert(0, str(directory))
    try:
        # Import the module
        spec = importlib.util.spec_from_file_location("qubot", qubot_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the class
        class_name = config.get("class_name")
        if not class_name:
            raise ValueError("class_name not found in config.json")

        if not hasattr(module, class_name):
            raise ValueError(f"Class '{class_name}' not found in qubot.py")

        model_class = getattr(module, class_name)

        # Create instance with default parameters
        default_params = config.get("default_params", {})
        model_instance = model_class(**default_params)

        return model_instance

    finally:
        # Remove directory from Python path
        if str(directory) in sys.path:
            sys.path.remove(str(directory))


def get_available_models(username: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience function to get all available models for a user.
    """
    discovery = ModelDiscovery()
    problems = discovery.get_user_problems(username)
    optimizers = discovery.get_user_optimizers(username)

    return {
        'problems': [model.__dict__ for model in problems],
        'optimizers': [model.__dict__ for model in optimizers]
    }


def extract_parameter_schema(model: Union[BaseProblem, BaseOptimizer]) -> Dict[str, Any]:
    """
    Extract parameter schema from a qubots model for dynamic UI generation.

    Args:
        model: BaseProblem or BaseOptimizer instance

    Returns:
        Dictionary containing parameter schema information
    """
    schema = {
        "model_type": "problem" if isinstance(model, BaseProblem) else "optimizer",
        "model_name": getattr(model.metadata, 'name', model.__class__.__name__),
        "parameters": {}
    }

    if isinstance(model, BaseOptimizer):
        # Extract from optimizer metadata
        metadata = model._metadata
        param_info = model.get_parameter_info()

        # Process required parameters
        for param in metadata.required_parameters:
            param_schema = {
                "required": True,
                "type": "string",  # Default type
                "description": f"Required parameter: {param}"
            }

            # Add range information if available
            if param in metadata.parameter_ranges:
                min_val, max_val = metadata.parameter_ranges[param]
                param_schema.update({
                    "type": "number",
                    "minimum": min_val,
                    "maximum": max_val
                })

            # Add current value if available
            if param in param_info.get("current_values", {}):
                param_schema["default"] = param_info["current_values"][param]

            schema["parameters"][param] = param_schema

        # Process optional parameters
        for param in metadata.optional_parameters:
            param_schema = {
                "required": False,
                "type": "string",  # Default type
                "description": f"Optional parameter: {param}"
            }

            # Add range information if available
            if param in metadata.parameter_ranges:
                min_val, max_val = metadata.parameter_ranges[param]
                param_schema.update({
                    "type": "number",
                    "minimum": min_val,
                    "maximum": max_val
                })

            # Add current value if available
            if param in param_info.get("current_values", {}):
                param_schema["default"] = param_info["current_values"][param]

            schema["parameters"][param] = param_schema

    elif isinstance(model, BaseProblem):
        # Extract from problem metadata and constructor
        metadata = model._metadata

        # Try to extract parameters from constructor signature
        try:
            sig = inspect.signature(model.__class__.__init__)
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue

                param_schema = {
                    "required": param.default == inspect.Parameter.empty,
                    "type": "string",  # Default type
                    "description": f"Problem parameter: {param_name}"
                }

                # Try to infer type from default value
                if param.default != inspect.Parameter.empty:
                    param_schema["default"] = param.default
                    if isinstance(param.default, (int, float)):
                        param_schema["type"] = "number"
                    elif isinstance(param.default, bool):
                        param_schema["type"] = "boolean"
                    elif isinstance(param.default, list):
                        param_schema["type"] = "array"

                # Add bounds information if available in metadata
                if metadata.variable_bounds and param_name in metadata.variable_bounds:
                    min_val, max_val = metadata.variable_bounds[param_name]
                    param_schema.update({
                        "type": "number",
                        "minimum": min_val,
                        "maximum": max_val
                    })

                schema["parameters"][param_name] = param_schema

        except Exception as e:
            # Fallback: add common problem parameters
            schema["parameters"]["dimension"] = {
                "required": False,
                "type": "number",
                "description": "Problem dimension",
                "minimum": 1,
                "default": metadata.dimension or 10
            }

    return schema


def get_model_parameter_schema(model_name: str,
                              username: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a model and extract its parameter schema.

    Args:
        model_name: Name of the model repository
        username: Repository owner (auto-detected if None)

    Returns:
        Parameter schema dictionary
    """
    try:
        model = load_qubots_model(model_name, username)
        return extract_parameter_schema(model)
    except Exception as e:
        return {
            "error": str(e),
            "model_name": model_name,
            "parameters": {}
        }