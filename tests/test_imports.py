from qubots import (
    AutoOptimizer,
    AutoProblem,
    benchmark,
    pipeline,
    validate_repo,
    validate_tree,
)


def test_public_imports() -> None:
    assert AutoProblem is not None
    assert AutoOptimizer is not None
    assert benchmark is not None
    assert pipeline is not None
    assert validate_repo is not None
    assert validate_tree is not None
