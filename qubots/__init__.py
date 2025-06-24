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

# Registry system has been removed - focusing on AutoProblem and AutoOptimizer

# Rastion platform integration
from .rastion_client import (
    RastionClient,
    load_qubots_model,
    upload_qubots_model
)

# Import rastion module for convenience
from . import rastion

# Playground integration
from .playground_integration import (
    PlaygroundExecutor,
    PlaygroundResult,
    ModelInfo,
    execute_playground_optimization,
)

# Leaderboard integration
from .leaderboard import (
    LeaderboardClient,
    LeaderboardIntegration,
    LeaderboardSubmission,
    StandardizedProblem,
    submit_to_leaderboard,
    get_problem_leaderboard,
    get_standardized_problems
)

# Standardized benchmarks
from .standardized_benchmarks import (
    StandardizedTSPProblem,
    StandardizedMaxCutProblem,
    StandardizedBenchmarkRegistry,
    BenchmarkProblemSpec
)

# Dashboard and visualization
from .dashboard import (
    DashboardResult,
    VisualizationData,
    QubotsVisualizer,
    QubotsAutoDashboard
)



__version__ = "1.1.4"

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

    # Rastion platform integration
    "RastionClient",
    "load_qubots_model",
    "upload_qubots_model",
    "rastion",

    # Playground integration
    "PlaygroundExecutor",
    "PlaygroundResult",
    "ModelInfo",
    "execute_playground_optimization",

    # Leaderboard integration
    "LeaderboardClient",
    "LeaderboardIntegration",
    "LeaderboardSubmission",
    "StandardizedProblem",
    "submit_to_leaderboard",
    "get_problem_leaderboard",
    "get_standardized_problems",

    # Standardized benchmarks
    "StandardizedTSPProblem",
    "StandardizedMaxCutProblem",
    "StandardizedBenchmarkRegistry",
    "BenchmarkProblemSpec",

    # Dashboard and visualization
    "DashboardResult",
    "VisualizationData",
    "QubotsVisualizer",
    "QubotsAutoDashboard",

    
]
