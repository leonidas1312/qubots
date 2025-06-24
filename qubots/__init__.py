"""
Qubots: Modular Optimization Framework
QUBO + Bot = Modular optimization components like lego blocks

Modular architecture: datasets -> problems -> optimizers -> results
Provides clean interfaces for optimization workflows.
"""

# Core base classes
from .base_problem import (
    BaseProblem,
    ProblemMetadata,
    ProblemType,
    ObjectiveType,
    DifficultyLevel,
    EvaluationResult
)
from .base_optimizer import (
    BaseOptimizer,
    OptimizerMetadata,
    OptimizerType,
    OptimizerFamily,
    OptimizationResult
)



# Auto-loading functionality
from .auto_problem import AutoProblem
from .auto_optimizer import AutoOptimizer

# Registry system has been removed - focusing on AutoProblem and AutoOptimizer

# Rastion platform integration
from .rastion_client import (
    RastionClient,
    load_qubots_model,
    upload_qubots_model
)

# Import rastion module for convenience
from . import rastion
from .rastion import autoLoad, Dataset

__version__ = "1.1.5"

__all__ = [
    # Core classes
    "BaseProblem",
    "BaseOptimizer",
    "ProblemMetadata",
    "OptimizerMetadata",
    "ProblemType",
    "ObjectiveType",
    "DifficultyLevel",
    "OptimizerType",
    "OptimizerFamily",
    "EvaluationResult",
    "OptimizationResult",

    # Auto-loading
    "AutoProblem",
    "AutoOptimizer",

    # Dataset functionality
    "autoLoad",
    "Dataset",

    # Rastion platform integration
    "RastionClient",
    "load_qubots_model",
    "upload_qubots_model",
    "rastion",
]
