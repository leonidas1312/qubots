"""Qubots public API."""

from importlib.metadata import PackageNotFoundError, version

from qubots.auto.auto_optimizer import AutoOptimizer
from qubots.auto.auto_problem import AutoProblem
from qubots.benchmark.benchmark import benchmark
from qubots.core.milp import MILPModel, SupportsMILP
from qubots.core.optimizer import BaseOptimizer
from qubots.core.problem import BaseProblem
from qubots.core.types import Result
from qubots.leaderboard import (
    LeaderboardReport,
    Submission,
    SuiteSpec,
    run_leaderboard,
)
from qubots.pipeline.pipeline import Pipeline, pipeline
from qubots.validate.validate import validate_repo, validate_tree

try:
    __version__ = version("qubots")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "AutoProblem",
    "AutoOptimizer",
    "BaseOptimizer",
    "BaseProblem",
    "benchmark",
    "LeaderboardReport",
    "MILPModel",
    "Pipeline",
    "pipeline",
    "Result",
    "run_leaderboard",
    "Submission",
    "SuiteSpec",
    "SupportsMILP",
    "validate_repo",
    "validate_tree",
    "__version__",
]
