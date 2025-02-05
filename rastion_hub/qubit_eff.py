import copy
import time
import pennylane as qml
from pennylane import numpy as np
import numpy as cnp

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
    rz1 = cnp.random.RandomState()
    list_of_bitstrings = []
    for _ in range(nbitstrings):
        bitstring = tuple(
            rz1.choice(2, p=[1 - bitprob, bitprob]) 
            for bitprob in blist
        )
        list_of_bitstrings.append(bitstring)

    return [np.array(bs) for bs in list_of_bitstrings]
