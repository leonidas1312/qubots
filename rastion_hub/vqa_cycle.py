# File: rastion_hub/vqa_cycle.py

import copy
import numpy as np
import random
from rastion_core.base_problem import BaseProblem

###############################################################################
# Helper Class: Wrap a cost function (over θ) into a BaseProblem.
###############################################################################
class QuantumParameterProblem(BaseProblem):
    """
    Wraps a cost function f(theta) into a BaseProblem so that any classical optimizer
    (which expects a BaseProblem instance) can optimize the variational parameters.
    """
    def __init__(self, cost_function, dim, lower_bound=0, upper_bound=2*np.pi):
        """
        :param cost_function: Function f(theta) that returns a scalar cost.
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
# The VQA Cycle Function
###############################################################################
def vqa_cycle(qubo_matrix, qubo_constant, num_layers, max_iters, nbitstrings,
              classical_optimizer, quantum_circuit_fn, simulator_fn, cost_function, draw_fn):
    """
    A generic VQA cycle that:
      1. Wraps the quantum circuit’s cost function (provided by the user) into a problem over θ.
      2. Uses a classical optimizer (e.g. TorchAdamOptimizer) to optimize the variational parameters.
      3. Uses a user-supplied draw function to sample candidate bitstrings from the circuit output.
      4. Evaluates the candidate bitstrings against the QUBO objective and returns the best one.
    
    Parameters:
      - qubo_matrix: A numpy array representing the QUBO matrix.
      - qubo_constant: A constant offset for the QUBO.
      - num_layers: Number of layers in the quantum circuit.
      - max_iters: Maximum number of iterations for the classical optimizer.
      - nbitstrings: Number of candidate bitstrings to sample.
      - classical_optimizer: An instance of a classical optimizer (conforming to BaseOptimizer).
      - quantum_circuit_fn: A function defining the quantum circuit (ansatz); signature: quantum_circuit_fn(angles) -> output.
      - simulator_fn: (if needed) a function that sets up the quantum simulator and returns a callable QNode.
                      If not provided (i.e. None), the default simulator is used:
                        dev = qml.device("lightning.qubit", wires=nqq, shots=num_shots)
                        my_qnode = qml.QNode(quantum_circuit_fn, dev, diff_method="parameter-shift")
      - cost_function: A function that takes a parameter vector (θ) and returns a scalar cost
                       by evaluating the quantum circuit output (using the simulator and quantum_circuit_fn).
      - draw_fn: A function that, given θ and a number of bitstrings, returns candidate bitstrings.
                 Signature: draw_fn(theta, nbitstrings) -> list of candidate bitstrings.
    
    Returns:
      - best_candidate: The candidate bitstring (as a numpy array or list) with the lowest QUBO cost.
      - best_candidate_cost: The QUBO cost (float) of that candidate.
    """
    # Determine the dimension of θ.
    n_qubo = qubo_matrix.shape[0]
    # For example, we choose the number of qubits (nqq) to be enough to represent n_qubo.
    nqq = int(np.ceil(np.log2(n_qubo))) + 1
    dim = nqq * num_layers

    global my_qnode

    # Set up the simulator (if the user provides one, use it; else, use the default Pennylane simulator).
    if simulator_fn is None:
        import pennylane as qml
        num_shots = 10000
        dev = qml.device("lightning.qubit", wires=nqq, shots=num_shots)
        # Create a QNode wrapping the provided quantum circuit function.
        my_qnode = qml.QNode(quantum_circuit_fn, dev, diff_method="parameter-shift")
    else:
        # Assume simulator_fn returns a QNode when given quantum_circuit_fn and nqq.
        my_qnode = simulator_fn(quantum_circuit_fn, nqq)

    # Make these objects globally available so that cost_function and draw_fn can use them.
    global QUBO_matrix, const
    QUBO_matrix = qubo_matrix
    const = qubo_constant

    # Wrap the user-supplied cost function into a BaseProblem.
    qp = QuantumParameterProblem(cost_function=cost_function, dim=dim)
    
    # Use the provided classical optimizer to optimize θ.
    best_theta, best_theta_cost = classical_optimizer.optimize(qp, **{"cost_function": cost_function})
    print("Optimized quantum parameters (θ) obtained with cost:", best_theta_cost)
    
    # Draw candidate solutions (bitstrings) from the quantum circuit using best_theta.
    candidates = draw_fn(best_theta, nbitstrings)
    
    # Evaluate each candidate using the QUBO objective.
    best_candidate = None
    best_candidate_cost = float('inf')
    for candidate in candidates:
        candidate = np.array(candidate)
        candidate_cost = float(candidate.T @ np.array(qubo_matrix) @ candidate + qubo_constant)
        if candidate_cost < best_candidate_cost:
            best_candidate = candidate
            best_candidate_cost = candidate_cost
    return best_candidate, best_candidate_cost
