from __future__ import annotations

from pathlib import Path

import pytest

from qubots import AutoOptimizer, AutoProblem, benchmark


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "detect"
INTEGRATIONS = ROOT / "integrations"


def test_scipy_assignment_optimizer_solves_detected_matrix() -> None:
    pytest.importorskip("scipy", exc_type=ImportError)

    problem = AutoProblem.from_data(FIXTURES / "assignment.csv", family="assignment")
    optimizer = AutoOptimizer.from_repo(INTEGRATIONS / "scipy_assignment_optimizer")

    result = optimizer.optimize(problem)

    assert result.status == "ok"
    assert result.best_value == pytest.approx(5.0)
    assert result.best_solution == [1, 0, 2]
    assert result.metadata["framework"] == "scipy"


def test_cvxpy_optimizer_solves_continuous_blending_problem() -> None:
    pytest.importorskip("cvxpy", exc_type=ImportError)

    problem = AutoProblem.from_repo(INTEGRATIONS / "continuous_blending_problem")
    optimizer = AutoOptimizer.from_repo(INTEGRATIONS / "cvxpy_lp_optimizer")

    result = optimizer.optimize(problem)

    assert result.status == "ok"
    assert result.metadata["framework"] == "cvxpy"
    assert result.metadata["objective"] == pytest.approx(13.0, abs=1e-5)
    assert result.best_solution == pytest.approx([2.0, 3.0], abs=1e-4)
    assert problem.as_milp().is_feasible(list(result.best_solution), tol=1e-4)


def test_pulp_optimizer_solves_pandas_knapsack_problem() -> None:
    pytest.importorskip("pulp", exc_type=ImportError)
    pytest.importorskip("pandas", exc_type=ImportError)

    problem = AutoProblem.from_repo(INTEGRATIONS / "pandas_knapsack_problem")
    optimizer = AutoOptimizer.from_repo(INTEGRATIONS / "pulp_milp_optimizer")

    result = optimizer.optimize(problem)

    assert result.status == "ok"
    assert result.metadata["framework"] == "pulp"
    assert result.metadata["objective"] == pytest.approx(122.0)
    assert result.best_value == pytest.approx(-122.0)
    assert problem.as_milp().is_feasible(list(result.best_solution))


def test_pyomo_optimizer_solves_pandas_knapsack_problem() -> None:
    pytest.importorskip("pyomo", exc_type=ImportError)
    pytest.importorskip("pandas", exc_type=ImportError)

    problem = AutoProblem.from_repo(INTEGRATIONS / "pandas_knapsack_problem")
    optimizer = AutoOptimizer.from_repo(INTEGRATIONS / "pyomo_milp_optimizer")

    result = optimizer.optimize(problem)
    if result.status == "error" and "not available" in (result.error or ""):
        pytest.skip(result.error)

    assert result.status == "ok"
    assert result.metadata["framework"] == "pyomo"
    assert result.metadata["objective"] == pytest.approx(122.0)
    assert result.best_value == pytest.approx(-122.0)
    assert problem.as_milp().is_feasible(list(result.best_solution))


def test_optuna_binary_optimizer_improves_one_max() -> None:
    pytest.importorskip("optuna", exc_type=ImportError)

    problem = AutoProblem.from_repo(ROOT / "examples" / "one_max_problem")
    problem.set_parameters(n_bits=8)
    optimizer = AutoOptimizer.from_repo(INTEGRATIONS / "optuna_binary_optimizer")
    optimizer.set_parameters(n_trials=96, seed=7)

    result = optimizer.optimize(problem)

    assert result.status == "ok"
    assert result.metadata["framework"] == "optuna"
    assert result.best_value <= -6.0
    assert len(result.best_solution) == 8


def test_networkx_integration_solves_detected_graph() -> None:
    pytest.importorskip("networkx", exc_type=ImportError)

    problem = AutoProblem.from_data(
        ROOT / "examples" / "pilots" / "qoblib_karate_edges.gph",
        family="maxcut",
    )
    optimizer = AutoOptimizer.from_repo(INTEGRATIONS / "networkx_maxcut_optimizer")
    optimizer.set_parameters(max_passes=5, seed=7)

    result = optimizer.optimize(problem)

    assert result.status == "ok"
    assert result.metadata["framework"] == "networkx"
    assert result.metadata["nodes"] == 34
    assert result.metadata["edges"] == 78
    assert result.best_value <= 0.0


def test_jax_integration_solves_detected_graph() -> None:
    pytest.importorskip("jax", exc_type=ImportError)

    problem = AutoProblem.from_data(
        ROOT / "examples" / "pilots" / "qoblib_karate_edges.gph",
        family="maxcut",
    )
    optimizer = AutoOptimizer.from_repo(INTEGRATIONS / "jax_maxcut_optimizer")
    optimizer.set_parameters(max_passes=5, seed=7)

    result = optimizer.optimize(problem)

    assert result.status == "ok"
    assert result.metadata["framework"] == "jax"
    assert result.metadata["nodes"] == 34
    assert result.metadata["edges"] == 78
    assert result.best_value <= -45.0


def test_dwave_neal_integration_solves_detected_graph() -> None:
    pytest.importorskip("dimod", exc_type=ImportError)
    pytest.importorskip("neal", exc_type=ImportError)

    problem = AutoProblem.from_data(
        ROOT / "examples" / "pilots" / "qoblib_karate_edges.gph",
        family="maxcut",
    )
    optimizer = AutoOptimizer.from_repo(INTEGRATIONS / "dwave_neal_maxcut_optimizer")
    optimizer.set_parameters(num_reads=20, sweeps=50, seed=7)

    result = optimizer.optimize(problem)

    assert result.status == "ok"
    assert result.metadata["framework"] == "dwave-neal"
    assert result.metadata["nodes"] == 34
    assert result.metadata["edges"] == 78
    assert result.best_value <= -45.0


def test_qiskit_qaoa_integration_solves_tiny_graph() -> None:
    pytest.importorskip("qiskit", exc_type=ImportError)

    problem = AutoProblem.from_data(FIXTURES / "graph.edgelist", family="maxcut")
    optimizer = AutoOptimizer.from_repo(INTEGRATIONS / "qiskit_qaoa_maxcut_optimizer")
    optimizer.set_parameters(gamma_grid=5, beta_grid=5, shots=128, seed=7)

    result = optimizer.optimize(problem)

    assert result.status == "ok"
    assert result.metadata["framework"] == "qiskit"
    assert result.metadata["algorithm"] == "qaoa_grid_statevector"
    assert result.metadata["n_qubits"] == 4
    assert result.best_value == pytest.approx(-7.0)


def test_qiskit_vqe_integration_solves_tiny_graph() -> None:
    pytest.importorskip("qiskit", exc_type=ImportError)

    problem = AutoProblem.from_data(FIXTURES / "graph.edgelist", family="maxcut")
    optimizer = AutoOptimizer.from_repo(INTEGRATIONS / "qiskit_vqe_maxcut_optimizer")
    optimizer.set_parameters(trials=24, shots=128, seed=7)

    result = optimizer.optimize(problem)

    assert result.status == "ok"
    assert result.metadata["framework"] == "qiskit"
    assert result.metadata["algorithm"] == "vqe_random_statevector"
    assert result.metadata["n_qubits"] == 4
    assert result.best_value == pytest.approx(-7.0)


def test_integration_dataset_benchmarks() -> None:
    pytest.importorskip("pulp", exc_type=ImportError)
    pytest.importorskip("pandas", exc_type=ImportError)

    report = benchmark(
        problem_repo=None,
        dataset_path=INTEGRATIONS / "datasets" / "pandas_knapsack.yaml",
        optimizers=[INTEGRATIONS / "pulp_milp_optimizer"],
        repeats=1,
        seed=7,
    )

    run = report["results"][0]["runs"][0]
    assert run["status"] == "ok"
    assert run["objective"] == pytest.approx(122.0)


def test_continuous_integration_dataset_benchmarks() -> None:
    pytest.importorskip("cvxpy", exc_type=ImportError)

    report = benchmark(
        problem_repo=None,
        dataset_path=INTEGRATIONS / "datasets" / "continuous_blending.yaml",
        optimizers=[INTEGRATIONS / "cvxpy_lp_optimizer"],
        repeats=1,
        seed=7,
    )

    run = report["results"][0]["runs"][0]
    assert run["status"] == "ok"
    assert run["objective"] == pytest.approx(13.0, abs=1e-5)


def test_qiskit_integration_dataset_benchmarks() -> None:
    pytest.importorskip("qiskit", exc_type=ImportError)

    report = benchmark(
        problem_repo=None,
        dataset_path=INTEGRATIONS / "datasets" / "tiny_maxcut.yaml",
        optimizers=[
            INTEGRATIONS / "qiskit_qaoa_maxcut_optimizer",
            INTEGRATIONS / "qiskit_vqe_maxcut_optimizer",
        ],
        repeats=1,
        seed=7,
    )

    assert len(report["results"]) == 2
    for row in report["results"]:
        run = row["runs"][0]
        assert run["status"] == "ok"
        assert run["best_value"] == pytest.approx(-7.0)
