"""Auto-loader for problem components."""

from pathlib import Path

from qubots.core.problem import BaseProblem
from qubots.detect.detectors import build_problem_spec, select_detection
from qubots.detect.problems import problem_from_spec
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

    @staticmethod
    def from_data(
        path: str | Path,
        *,
        family: str | None = None,
        detector: str | None = None,
        confidence_threshold: float = 0.75,
    ) -> BaseProblem:
        detection = select_detection(
            path,
            family=family,
            detector=detector,
            confidence_threshold=confidence_threshold,
        )

        if detection.family == "qubots_manifest":
            repo_path = detection.parameters.get("repo_path")
            if not isinstance(repo_path, str) or not repo_path:
                raise ValueError("Manifest detection did not include repo_path")
            return AutoProblem.from_repo(repo_path)

        if detection.family == "qubots_dataset":
            raise ValueError(
                "Detected a qubots dataset YAML, not a single runnable problem. "
                "Use qubots benchmark --dataset or import an individual source file."
            )

        spec = build_problem_spec(path, detection)
        problem = problem_from_spec(spec)
        setattr(
            problem,
            "_qubots_source",
            {
                "spec": str(path),
                "resolved_path": str(Path(path).expanduser().resolve()),
                "ref": None,
                "detector": detection.detector,
                "data_hash": spec.data_hash,
            },
        )
        setattr(problem, "_qubots_source_name", Path(path).stem)
        return problem
