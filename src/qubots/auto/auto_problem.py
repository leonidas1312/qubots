"""Auto-loader for problem components."""

from pathlib import Path

from qubots.core.problem import BaseProblem
from qubots.hub.loaders import load_component
from qubots.hub.resolver import derive_repo_name, resolve_repo_info


class AutoProblem:
    @staticmethod
    def from_repo(path: str | Path) -> BaseProblem:
        resolved = resolve_repo_info(path)
        component = load_component(resolved.resolved_path, expected_type="problem")
        if not isinstance(component, BaseProblem):
            raise TypeError("Loaded component is not a BaseProblem")
        setattr(
            component,
            "_qubots_source",
            {
                "spec": str(path),
                "resolved_path": str(resolved.resolved_path),
                "ref": resolved.ref,
            },
        )
        setattr(
            component,
            "_qubots_source_name",
            derive_repo_name(path, resolved.resolved_path),
        )
        return component
