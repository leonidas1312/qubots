"""
Qubots Playground Integration Module

Provides standardized interfaces for integrating qubots with the Rastion platform playground.
Handles result formatting, progress reporting, and error management for web-based optimization.
"""

import json
import time
import traceback
from typing import Dict, Any, Optional, Union, List, Callable
from datetime import datetime
from dataclasses import dataclass, asdict

from .base_problem import BaseProblem
from .base_optimizer import BaseOptimizer
from .rastion_client import get_global_client
from .rastion import load_qubots_model


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
def execute_playground_optimization(problem_name: str,
                                  optimizer_name: str,
                                  problem_username: Optional[str] = None,
                                  optimizer_username: Optional[str] = None,
                                  **kwargs) -> Dict[str, Any]:
    """
    Convenience function to execute optimization and return result as dictionary.
    """
    executor = PlaygroundExecutor()
    result = executor.execute_optimization(
        problem_name, optimizer_name, problem_username, optimizer_username, **kwargs
    )
    return result.to_dict()


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