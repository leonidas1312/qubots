"""Small continuous blending LP used by CVXPY/Pyomo integrations."""

from __future__ import annotations

from qubots.core.milp import MILPModel
from qubots.core.problem import BaseProblem


class ContinuousBlendingProblem(BaseProblem):
    def __init__(self) -> None:
        super().__init__()
        self.capabilities = ["blackbox", "milp_dense", "continuous"]

    def as_milp(self) -> MILPModel:
        # Minimize 2x + 3y
        # s.t. x + 2y >= 8, 3x + y >= 9, x,y >= 0.
        # The optimal solution is x=2, y=3, objective=13.
        return MILPModel(
            sense="min",
            c=[2.0, 3.0],
            var_names=["blend_a", "blend_b"],
            integrality=[False, False],
            lb=[0.0, 0.0],
            ub=[float("inf"), float("inf")],
            A_ub=[[-1.0, -2.0], [-3.0, -1.0]],
            b_ub=[-8.0, -9.0],
            constraint_names=["nutrient_1_min", "nutrient_2_min"],
        )
