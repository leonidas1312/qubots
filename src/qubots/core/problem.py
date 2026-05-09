"""Base interface for optimization problems.

A problem subclass must implement at least one of:

- ``evaluate(solution)`` plus ``random_solution()``: the blackbox interface
  used by metaheuristic optimizers (random search, hill climb, simulated
  annealing).
- ``as_milp()``: structured MILP form used by linear/integer solvers like
  HiGHS, OR-Tools, SCIP, Gurobi. When this is the only thing implemented,
  the base class provides sensible defaults for ``evaluate()`` and
  ``random_solution()`` that delegate to the MILP.

Implementing both is encouraged when feasible: the same problem object then
runs under any optimizer in the hub.
"""

from typing import Any


class BaseProblem:
    def __init__(self) -> None:
        self.parameters: dict[str, Any] = {}

    def set_parameters(self, **kwargs: Any) -> None:
        self.parameters.update(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def evaluate(self, solution: Any) -> float:
        """Default blackbox evaluation derived from ``as_milp()``.

        For maximization problems the returned value is ``-objective + penalty``
        (lower is better, qubots convention); for minimization it's
        ``objective + penalty``. Constraint violations contribute a positive
        penalty so feasible solutions always score below infeasible ones.
        Override this for tighter / problem-specific penalties.
        """
        if not hasattr(self, "as_milp"):
            raise NotImplementedError(
                f"{type(self).__name__} must implement evaluate() or as_milp()"
            )
        from qubots.core.milp import MILPModel

        milp = self.as_milp()
        if not isinstance(milp, MILPModel):
            raise NotImplementedError(
                "as_milp() did not return MILPModel; cannot derive default evaluate()"
            )

        n = milp.n_vars
        x = list(solution[:n]) + [0.0] * max(0, n - len(solution))
        x = [float(v) for v in x]

        objective = sum(c * xi for c, xi in zip(milp.c, x))
        penalty = 0.0

        for row, b in zip(milp.A_ub, milp.b_ub):
            slack = sum(a * xi for a, xi in zip(row, x)) - b
            if slack > 0:
                penalty += slack
        for row, b in zip(milp.A_eq, milp.b_eq):
            penalty += abs(sum(a * xi for a, xi in zip(row, x)) - b)
        for i, xi in enumerate(x):
            if xi < milp.lb[i]:
                penalty += milp.lb[i] - xi
            if xi > milp.ub[i]:
                penalty += xi - milp.ub[i]

        signed_objective = -objective if milp.sense == "max" else objective
        return float(signed_objective + 100.0 * penalty)

    def random_solution(self) -> Any:
        """Default sampler derived from ``as_milp()`` bounds."""
        if not hasattr(self, "as_milp"):
            raise NotImplementedError(
                f"{type(self).__name__} must implement random_solution() or as_milp()"
            )
        import math
        import random as _random

        from qubots.core.milp import MILPModel

        milp = self.as_milp()
        if not isinstance(milp, MILPModel):
            raise NotImplementedError(
                "as_milp() did not return MILPModel; cannot derive default random_solution()"
            )

        out: list[float] = []
        for i in range(milp.n_vars):
            lb = milp.lb[i] if math.isfinite(milp.lb[i]) else -1.0
            ub = milp.ub[i] if math.isfinite(milp.ub[i]) else 1.0
            if milp.integrality[i]:
                lo = int(math.ceil(lb))
                hi = int(math.floor(ub))
                if lo > hi:
                    lo, hi = hi, lo
                out.append(float(_random.randint(lo, hi)))
            else:
                out.append(_random.uniform(float(lb), float(ub)))
        return out
