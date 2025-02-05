# File: tests/test_vqa_cycle_maxcut.py

import time
import numpy as np
from rastion_hub.auto_problem import AutoProblem
from rastion_hub.auto_optimizer import AutoOptimizer
from rastion_hub.vqa_cycle import vqa_cycle

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

def main():
    # 1. Load the problem instance (MaxCut QUBO problem)
    problem = AutoProblem.from_repo("Rastion/max-cut", revision="main")
    qubo_matrix, qubo_constant = problem.get_qubo()
    
    # 2. Load the TorchAdamOptimizer as our classical optimizer.
    classical_optimizer = AutoOptimizer.from_repo("Rastion/torch-adam-optimizer", revision="main")
    
    # 3. Run the vqa_cycle.
    # For the simulator_fn we pass None (to use the default Pennylane simulator)
    best_candidate, best_candidate_cost = vqa_cycle(
        qubo_matrix,
        qubo_constant,
        num_layers=4,
        max_iters=100,
        nbitstrings=10,
        classical_optimizer=classical_optimizer,
        quantum_circuit_fn=pennylane_HEcirc,
        simulator_fn=None,  # Using default simulator setup (modular design allows future expansion)
        cost_function=calmecf,
        draw_fn=draw_bitstrings_minenc
    )
    
    print("Best candidate bitstring:", best_candidate)
    print("Candidate QUBO cost:", best_candidate_cost)

if __name__ == "__main__":
    main()
