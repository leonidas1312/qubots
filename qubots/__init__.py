"""
Enhanced __init__.py with comprehensive exports and documentation.
Provides easy access to all qubots components.
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

# Specialized base classes
from .specialized_problems import (
    ContinuousProblem,
    DiscreteProblem,
    CombinatorialProblem,
    ConstrainedProblem,
    MultiObjectiveProblem
)
from .specialized_optimizers import (
    PopulationBasedOptimizer,
    LocalSearchOptimizer,
    GradientBasedOptimizer,
    SwarmOptimizer,
    HybridOptimizer
)

# Auto-loading functionality
from .auto_problem import AutoProblem
from .auto_optimizer import AutoOptimizer

# Benchmarking and evaluation
from .benchmarking import (
    BenchmarkSuite,
    BenchmarkResult,
    BenchmarkMetrics,
    BenchmarkType
)

# Registry and discovery
from .registry import (
    QubotRegistry,
    RegistryEntry,
    RegistryType,
    get_global_registry
)

# Rastion platform integration
from .rastion_client import (
    RastionClient,
    QubotPackager,
    load_qubots_model,
    upload_qubots_model,
    list_available_models,
    search_models
)

# Import rastion module for convenience
from . import rastion

# Playground integration
from .playground_integration import (
    PlaygroundExecutor,
    ModelDiscovery,
    PlaygroundResult,
    ModelInfo,
    execute_playground_optimization,
    get_available_models
)

# Dashboard and visualization
from .dashboard import (
    DashboardResult,
    VisualizationData,
    QubotsVisualizer,
    QubotsAutoDashboard
)

# Testing and debugging framework
from .testing import (
    ModelTester,
    UploadValidator,
    LoadValidator,
    ExecutionValidator,
    DebuggingTools,
    TestResult,
    ValidationError,
    run_comprehensive_test,
    validate_model_upload,
    validate_model_loading,
    validate_model_execution
)

__version__ = "1.0.2"

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

    # Specialized classes
    "ContinuousProblem",
    "DiscreteProblem",
    "CombinatorialProblem",
    "ConstrainedProblem",
    "MultiObjectiveProblem",
    "PopulationBasedOptimizer",
    "LocalSearchOptimizer",
    "GradientBasedOptimizer",
    "SwarmOptimizer",
    "HybridOptimizer",

    # Auto-loading
    "AutoProblem",
    "AutoOptimizer",

    # Benchmarking
    "BenchmarkSuite",
    "BenchmarkResult",
    "BenchmarkMetrics",
    "BenchmarkType",

    # Registry
    "QubotRegistry",
    "RegistryEntry",
    "RegistryType",
    "get_global_registry",

    # Rastion platform integration
    "RastionClient",
    "QubotPackager",
    "load_qubots_model",
    "upload_qubots_model",
    "list_available_models",
    "search_models",
    "rastion",

    # Playground integration
    "PlaygroundExecutor",
    "ModelDiscovery",
    "PlaygroundResult",
    "ModelInfo",
    "execute_playground_optimization",
    "get_available_models",

    # Dashboard and visualization
    "DashboardResult",
    "VisualizationData",
    "QubotsVisualizer",
    "QubotsAutoDashboard",

    # Testing and debugging framework
    "ModelTester",
    "UploadValidator",
    "LoadValidator",
    "ExecutionValidator",
    "DebuggingTools",
    "TestResult",
    "ValidationError",
    "run_comprehensive_test",
    "validate_model_upload",
    "validate_model_loading",
    "validate_model_execution",
]
