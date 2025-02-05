import copy
import time
import pennylane as qml
from pennylane import numpy as np
import numpy as cnp

from rastion_core.base_optimizer import BaseOptimizer


def pennylane_HEcirc(angles):
    """
    Hardware Efficient Ansatz.
    Uses global variables nqq and n_layers.
    """
    for nq_ in range(nqq):
        qml.Hadamard(wires=nq_)
    for l_ in range(n_layers):
        for i_ in range(nqq):
            qml.RY(angles[i_ + l_ * nqq], wires=i_)
        for j in range(0, nqq, 2):
            if j >= nqq - 1:
                continue
            else:
                qml.CNOT(wires=[j, j + 1])
        for k in range(1, nqq, 2):
            if k >= nqq - 1:
                continue
            else:
                qml.CNOT(wires=[k, k + 1])
    return qml.probs(wires=list(range(nqq)))


def calmecf(angles):
    """
    Calculate the cost value based on the output of the quantum circuit.
    Relies on global QUBO_matrix, const, and a QNode (my_qnode).
    """
    nc = len(QUBO_matrix)
    data = my_qnode(angles)
    clist = data[::2] + data[1::2]
    blist = []
    for p in range(nc):
        if clist[p] == 0:
            blist.append(0.5)
        else:
            blist.append(data[2 * p + 1] / clist[p])

    blist = copy.deepcopy(np.array(blist))
    prob_matrix = np.outer(blist, blist)
    prob_diag = np.diag(prob_matrix)
    mat_diag = np.diag(QUBO_matrix)
    totcost = np.multiply(prob_matrix, QUBO_matrix).sum()
    subtract_cost = np.multiply(prob_diag, mat_diag).sum()
    add_cost = np.multiply(blist, mat_diag).sum()
    quantum_cost = totcost - subtract_cost + add_cost + const
    return quantum_cost


def draw_bitstrings_minenc(angles, nbitstrings):
    nc = len(QUBO_matrix)
    data = my_qnode(angles)
    clist = data[::2] + data[1::2]
    blist = []
    for p in range(nc):
        if clist[p] == 0:
            blist.append(0.5)
        else:
            blist.append(data[2 * p + 1] / clist[p])
    list_of_bitstrings = set()
    rz1 = cnp.random.RandomState()
    while len(list_of_bitstrings) < nbitstrings:
        bitstring = tuple(rz1.choice(2, p=[1 - bitprob, bitprob]) for bitprob in blist)
        list_of_bitstrings.add(bitstring)
    return [np.array(bitstring) for bitstring in list_of_bitstrings]


def OPT_step(opt, theta):
    theta, adam_cost = opt.step_and_cost(calmecf, theta)
    return theta, adam_cost


def quantum_opt(QUBO_m, c, num_layers, max_iters, nbitstrings):
    """
    Main VQA routine: Uses a variational quantum circuit (via Pennylane's ADAM optimizer)
    to optimize variational parameters.
    """
    global nqq, n_layers, QUBO_matrix, const, dev, my_qnode
    nqq = int(np.ceil(np.log2(len(QUBO_m)))) + 1
    n_layers = num_layers
    QUBO_matrix = QUBO_m
    const = c
    num_shots = 10000
    dev = qml.device("lightning.qubit", wires=nqq, shots=num_shots)
    my_qnode = qml.QNode(pennylane_HEcirc, dev, diff_method="parameter-shift")

    print("Number of classical variables = " + str(len(QUBO_m)))
    print("Number of qubits for the quantum circuit = " + str(nqq))
    print("Number of layers for the quantum circuit = " + str(num_layers))
    print("Number of shots = " + str(num_shots))

    opt = qml.AdamOptimizer(stepsize=0.01)
    theta = np.array([2 * np.pi * np.random.rand() for _ in range(nqq * n_layers)], requires_grad=True)

    best_theta = []
    best_cost = float('inf')
    best_cost_opt = float('inf')
    best_bitstring = None
    progress_opt_costs = []

    for iter in range(max_iters):
        theta, opt_cost = OPT_step(opt, theta)
        print("Optimizer iteration : " + str(iter) + ", Cost = " + str(opt_cost))
        if opt_cost < best_cost_opt:
            best_cost_opt = opt_cost
            best_theta = theta
        progress_opt_costs.append(best_cost_opt)

    possible_solutions = draw_bitstrings_minenc(best_theta, nbitstrings)


    for solution in possible_solutions:
        cost = float(solution.T @ QUBO_matrix @ solution + c)

        if cost < best_cost:
            best_cost = cost

    
    return best_bitstring, best_cost, progress_opt_costs




class VQAOptimizer(BaseOptimizer):
    """
    A Rastion solver that implements a variational quantum algorithm (VQA) for QUBO problems.
    
    It calls the quantum_opt function internally.
    """
    def __init__(self, num_layers=3, max_iters=10, nbitstrings=10):
        self.num_layers = num_layers
        self.max_iters = max_iters
        self.nbitstrings = nbitstrings

    def optimize(self, problem, **kwargs):
        """
        The problem argument is expected to be a QUBO problem (with a get_qubo() method).
        This function extracts the QUBO matrix and constant from the problem and then calls quantum_opt.
        """
        # Extract QUBO data from the problem instance.
        QUBO_matrix, qubo_constant = problem.get_qubo()
        solution, cost, progress_opt_costs = quantum_opt(
            QUBO_matrix,
            qubo_constant,
            self.num_layers,
            self.max_iters,
            self.nbitstrings
        )
        return solution, cost
