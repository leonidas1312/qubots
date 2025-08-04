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

# Configuration management
from .config_manager import (
    get_config,
    set_profile,
    get_active_git_config,
    is_local_mode
)

# Local Gitea integration
from .local_gitea import (
    LocalGiteaClient,
    get_local_gitea_client,
    setup_local_gitea
)

# Code generation and export
from .code_generator import (
    CodeGenerator,
    WorkflowDefinition,
    WorkflowNode,
    WorkflowEdge
)

from .workflow_tester import (
    WorkflowTester,
    TestResult,
    TestReport
)

from .workflow_exporter import (
    WorkflowExporter
)

# Registry system has been removed - focusing on AutoProblem and AutoOptimizer

# Rastion platform integration (unified)
from .rastion_unified import (
    RastionClient,
    Dataset,
    authenticate,
    is_authenticated,
    load_dataset_from_platform,
    autoLoad,
    get_global_client,
    upload_qubots_model,
    load_qubots_model
)

# Import unified rastion module for convenience
from . import rastion_unified as rastion

__version__ = "1.1.6"

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

    # Configuration management
    "get_config",
    "set_profile",
    "get_active_git_config",
    "is_local_mode",

    # Local Gitea integration
    "LocalGiteaClient",
    "get_local_gitea_client",
    "setup_local_gitea",

    # Code generation and export
    "CodeGenerator",
    "WorkflowDefinition",
    "WorkflowNode",
    "WorkflowEdge",
    "WorkflowTester",
    "TestResult",
    "TestReport",
    "WorkflowExporter",

    # Dataset functionality
    "autoLoad",
    "Dataset",
    "load_dataset_from_platform",

    # Rastion platform integration
    "RastionClient",
    "authenticate",
    "is_authenticated",
    "get_global_client",
    "upload_qubots_model",
    "load_qubots_model",
    "rastion",
]
