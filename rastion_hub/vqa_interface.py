# File: rastion_hub/vqa_interface.py

import copy
import numpy as np
import random
import pennylane as qml
from rastion_core.base_problem import BaseProblem

###############################################################################
# Helper Class: Wrap a cost function (over θ) into a BaseProblem.
###############################################################################
class QuantumParameterProblem(BaseProblem):
    """
    Wraps a cost function f(θ) into a BaseProblem so that any classical optimizer
    (which expects a BaseProblem instance) can optimize the variational parameters.
    """
    def __init__(self, cost_function, dim, lower_bound=0, upper_bound=2*np.pi):
        """
        :param cost_function: Function f(θ) that returns a scalar cost.
        :param dim: Dimension of the parameter vector θ.
        :param lower_bound: Lower bound for each element of θ.
        :param upper_bound: Upper bound for each element of θ.
        """
        self.cost_function = cost_function
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def evaluate_solution(self, solution) -> float:
        return self.cost_function(solution)

    def random_solution(self):
        return [random.uniform(self.lower_bound, self.upper_bound) for _ in range(self.dim)]

###############################################################################
# The VQA Cycle Interface
###############################################################################
class VQACycleInterface:
    """
    A modular interface for a VQA cycle. It encapsulates:
      - A classical optimizer for optimizing the quantum-circuit parameters.
      - A quantum circuit function (ansatz).
      - A cost function (that evaluates the quantum circuit output given θ).
      - A draw function to sample candidate bitstrings from the circuit output.
      - Optionally, a simulator function to build a QNode (if needed).
      
    All necessary functions are passed as keyword arguments.
    """
    def __init__(self, qubo_matrix, qubo_constant, classical_optimizer, **quantum_kwargs):
        """
        :param qubo_matrix: A numpy array representing the QUBO matrix.
        :param qubo_constant: A constant offset for the QUBO.
        :param classical_optimizer: An instance of a classical optimizer (conforming to BaseOptimizer).
        :param quantum_kwargs: Additional keyword arguments needed for the quantum program.
               Expected keys:
                 - num_layers: Number of layers in the quantum circuit.
                 - max_iters: Maximum iterations for optimizing θ.
                 - nbitstrings: Number of candidate bitstrings to draw.
                 - quantum_circuit_fn: Function defining the quantum circuit (ansatz);
                                       signature: quantum_circuit_fn(θ) -> output.
                 - cost_function: Function that maps a parameter vector θ to a scalar cost.
                 - draw_fn: Function that, given θ and number of bitstrings, returns candidate bitstrings.
               Optionally:
                 - simulator_fn: A function that, given quantum_circuit_fn and number of qubits,
                                 returns a QNode. (If not provided, the default Pennylane simulator is used.)
        """
        self.qubo_matrix = qubo_matrix
        self.qubo_constant = qubo_constant
        self.classical_optimizer = classical_optimizer
        self.quantum_kwargs = quantum_kwargs

    def optimize(self):
        # Unpack quantum_kwargs.
        num_layers = self.quantum_kwargs["num_layers"]
        max_iters = self.quantum_kwargs["max_iters"]
        nbitstrings = self.quantum_kwargs["nbitstrings"]
        quantum_circuit_fn = self.quantum_kwargs["quantum_circuit_fn"]
        cost_function = self.quantum_kwargs["cost_function"]
        draw_fn = self.quantum_kwargs["draw_fn"]
        simulator_fn = self.quantum_kwargs.get("simulator_fn", None)

        # Determine the dimension of θ.
        n_qubo = self.qubo_matrix.shape[0]
        nqq = int(np.ceil(np.log2(n_qubo))) + 1
        dim = nqq * num_layers

        # Set up the simulator.
        if simulator_fn is None:
            num_shots = 10000
            dev = qml.device("lightning.qubit", wires=nqq, shots=num_shots)
            my_qnode = qml.QNode(quantum_circuit_fn, dev, diff_method="parameter-shift")
        else:
            my_qnode = simulator_fn(quantum_circuit_fn, nqq)

        # To make our quantum functions modular, we import our quantum circuit helper module
        # (for example, from qubit_eff.py) and set its global variables.
        import rastion_hub.qubit_eff as qe
        qe.QUBO_matrix = self.qubo_matrix
        qe.const = self.qubo_constant
        qe.my_qnode = my_qnode
        qe.n_layers = num_layers
        qe.nqq = nqq

        # Wrap the cost function into a BaseProblem.
        qp = QuantumParameterProblem(cost_function=cost_function, dim=dim)

        # Optimize θ using the provided classical optimizer.
        best_theta, best_theta_cost = self.classical_optimizer.optimize(qp, **{"cost_function": cost_function})
        print("Optimized quantum parameters (θ) obtained with cost:", best_theta_cost)

        # Draw candidate bitstrings using the optimized parameters.
        candidates = draw_fn(best_theta, nbitstrings)

        # Evaluate each candidate using the QUBO objective.
        best_candidate = None
        best_candidate_cost = float('inf')
        for candidate in candidates:
            candidate = np.array(candidate)
            candidate_cost = float(candidate.T @ self.qubo_matrix @ candidate + self.qubo_constant)
            if candidate_cost < best_candidate_cost:
                best_candidate = candidate
                best_candidate_cost = candidate_cost
        return best_candidate, best_candidate_cost
