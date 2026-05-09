"""qubots problem: load any MPS or LP file as a MILPModel.

Usage:

    from qubots import AutoProblem
    p = AutoProblem.from_repo("examples/mps_problem")
    p.set_parameters(mps_path="/path/to/instance.mps")

Or with MIPLIB on demand:

    from qubots.contrib.miplib import fetch_miplib
    p = AutoProblem.from_repo("examples/mps_problem")
    p.set_parameters(mps_path=str(fetch_miplib("gen-ip002")))
"""

from __future__ import annotations

from qubots.contrib.mps import MPSProblem


__all__ = ["MPSProblem"]
