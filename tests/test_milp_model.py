import pytest

from qubots import MILPModel, SupportsMILP
from qubots.core.problem import BaseProblem


def test_milp_model_validates_shape() -> None:
    with pytest.raises(ValueError):
        MILPModel(sense="invalid", c=[1.0])

    with pytest.raises(ValueError):
        MILPModel(sense="min", c=[1.0, 2.0], A_ub=[[1.0]], b_ub=[5.0])

    with pytest.raises(ValueError):
        MILPModel(sense="min", c=[1.0], A_eq=[[1.0]], b_eq=[1.0, 2.0])


def test_milp_model_defaults_fill_in() -> None:
    milp = MILPModel(sense="min", c=[1.0, 2.0])
    assert milp.var_names == ["x0", "x1"]
    assert milp.integrality == [False, False]
    assert milp.lb[0] == float("-inf")
    assert milp.ub[0] == float("inf")
    assert milp.n_vars == 2
    assert milp.n_constraints == 0


def test_milp_model_evaluate_and_feasibility() -> None:
    milp = MILPModel(
        sense="max",
        c=[3.0, 2.0],
        integrality=[True, True],
        lb=[0.0, 0.0],
        ub=[1.0, 1.0],
        A_ub=[[2.0, 1.0]],
        b_ub=[2.0],
    )
    assert milp.evaluate([1.0, 0.0]) == pytest.approx(3.0)
    assert milp.is_feasible([1.0, 0.0])
    assert not milp.is_feasible([1.0, 1.0])
    assert not milp.is_feasible([0.5, 0.0])


def test_supports_milp_protocol_runtime_check() -> None:
    class MILPish(BaseProblem):
        def evaluate(self, solution):  # type: ignore[no-untyped-def]
            return 0.0

        def random_solution(self):  # type: ignore[no-untyped-def]
            return []

        def as_milp(self) -> MILPModel:
            return MILPModel(sense="min", c=[1.0])

    class Plain(BaseProblem):
        def evaluate(self, solution):  # type: ignore[no-untyped-def]
            return 0.0

        def random_solution(self):  # type: ignore[no-untyped-def]
            return []

    assert isinstance(MILPish(), SupportsMILP)
    assert not isinstance(Plain(), SupportsMILP)
