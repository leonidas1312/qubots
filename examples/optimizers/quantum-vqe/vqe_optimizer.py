# File: examples/optimizers/vqe/vqe_optimizer.py

from qubots.base_optimizer import BaseOptimizer
import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from collections import defaultdict

def from_Q_to_Ising(Q, offset=0):
    """
    Convert the QUBO matrix Q into Ising parameters h and J.
    
    For a QUBO defined as f(x) = x^T Q x + const and using the mapping
      x_i = (1 - Z_i) / 2,
    the corresponding Ising parameters are computed.
    Returns dictionaries h and J, an updated offset, and the list of edges.
    """
    n_qubits = len(Q)
    h = defaultdict(float)
    J = defaultdict(float)
    edges = []
    for i in range(n_qubits):
        h[(i,)] -= Q[i, i] / 2.0
        offset += Q[i, i] / 2.0
        for j in range(i + 1, n_qubits):
            if Q[i, j] != 0:
                edges.append((i, j))
            J[(i, j)] += Q[i, j] / 4.0
            h[(i,)] -= Q[i, j] / 4.0
            h[(j,)] -= Q[i, j] / 4.0
            offset += Q[i, j] / 4.0
    return h, J, offset, edges

def energy_Ising(z, h, J, offset):
    """
    Calculate the energy of an Ising model given a spin configuration z.
    The convention is that if z is provided as a string,
    '0' is mapped to +1 and '1' to -1.
    """
    if isinstance(z, str):
        z = [1 if ch == '0' else -1 for ch in z]
    energy = offset
    for key, value in h.items():
        energy += value * z[key[0]]
    for (i, j), value in J.items():
        energy += value * z[i] * z[j]
    return energy

class VQEOptimizer(BaseOptimizer):
    """
    VQE Optimizer for QUBO problems.
    
    This optimizer uses a hardware-efficient ansatz to approximate the ground state
    of the cost Hamiltonian obtained via a QUBO-to-Ising transformation and normalization.
    It then samples the circuit to choose a candidate bitstring which is scored classically.
    """
    def __init__(self, num_layers=2, max_iters=100, learning_rate=0.1, n_shots=1000, verbose=False):
        self.num_layers = num_layers      # Number of layers in the ansatz.
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.n_shots = n_shots
        self.verbose = verbose

    def optimize(self, problem, initial_solution=None, **kwargs):
        # Retrieve the QUBO matrix and constant.
        Q_matrix, qubo_const = problem.get_qubo()
        # Unpack Q_matrix as our QUBO; here we choose to use an initial offset of 0.
        Q = Q_matrix
        offset = qubo_const
        n = Q.shape[0]

        # Convert the QUBO matrix into Ising parameters.
        h_dict, J_dict, offset, edges = from_Q_to_Ising(Q, offset)

        # Build the normalized cost Hamiltonian H_C from the Ising parameters.
        coeffs = []
        obs = []
        for i in range(n):
            if (i,) in h_dict and abs(h_dict[(i,)]) > 1e-8:
                coeffs.append(h_dict[(i,)])
                obs.append(qml.PauliZ(i))
        for (i, j) in edges:
            if (i, j) in J_dict and abs(J_dict[(i, j)]) > 1e-8:
                coeffs.append(J_dict[(i, j)])
                # Note: Using qml.operation.Tensor to form the tensor product.
                obs.append(qml.operation.Tensor(qml.PauliZ(i), qml.PauliZ(j)))
        H_C = qml.Hamiltonian(coeffs, obs)

        # Define the hardware-efficient ansatz.
        p = self.num_layers
        dev = qml.device("lightning.qubit", wires=n, shots=self.n_shots)

        def vqe_ansatz(params):
            # Reshape params to (p, n)
            params = pnp.reshape(params, (p, n))
            # Start in the uniform superposition |+>^n (apply Hadamard gates).
            for i in range(n):
                qml.Hadamard(wires=i)
            # For each layer, apply single-qubit RY rotations and a chain of CNOTs.
            for layer in range(p):
                for i in range(n):
                    qml.RY(params[layer, i], wires=i)
                for i in range(n - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(H_C)

        vqe_qnode = qml.QNode(vqe_ansatz, dev)
        num_params = p * n
        # Initialize parameters as a PennyLane numpy array with gradient support.
        params = pnp.array(np.random.uniform(0, 2*np.pi, num_params), requires_grad=True)

        # Use PennyLane's Adam optimizer.
        opt = qml.AdamOptimizer(stepsize=self.learning_rate)

        # Optimization loop using step_and_cost.
        for it in range(self.max_iters):
            params, cost = opt.step_and_cost(vqe_qnode, params)
            if self.verbose:
                print(f"VQE Iteration {it}: cost = {cost}")

        best_params = params

        # Define a sampling QNode to obtain probabilities.
        @qml.qnode(dev)
        def sample_vqe(params):
            params = pnp.reshape(params, (p, n))
            for i in range(n):
                qml.Hadamard(wires=i)
            for layer in range(p):
                for i in range(n):
                    qml.RY(params[layer, i], wires=i)
                for i in range(n - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.probs(wires=range(n))

        probs = sample_vqe(best_params)
        state_index = int(np.argmax(probs))
        bitstring = np.array(list(np.binary_repr(state_index, width=n))).astype(np.int8)
        final_cost = problem.evaluate_solution(bitstring)

        # (Optional) You can compute the classical Ising energy using energy_Ising:
        # classical_energy = energy_Ising(bitstring, h_dict, J_dict, offset)
        # and compare it with final_cost if desired.

        return bitstring.tolist(), final_cost
