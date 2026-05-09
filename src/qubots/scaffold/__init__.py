"""Scaffolders for new qubots component repositories.

Used by the ``qubots new`` CLI command to generate a fresh component
directory (manifest + entrypoint module + README) that immediately passes
``qubots validate``. Authors then fill in the body of ``evaluate``,
``random_solution``, ``as_milp``, or ``optimize``.
"""

from qubots.scaffold.scaffold import (
    PROBLEM_FLAVORS,
    OPTIMIZER_FLAVORS,
    ScaffoldResult,
    scaffold_optimizer,
    scaffold_problem,
)

__all__ = [
    "PROBLEM_FLAVORS",
    "OPTIMIZER_FLAVORS",
    "ScaffoldResult",
    "scaffold_optimizer",
    "scaffold_problem",
]
