# File: vqa_optimizer.py

import copy
import time
import pennylane as qml
from pennylane import numpy as np
import numpy as cnp
import torch

from rastion_core.base_optimizer import BaseOptimizer

# === Helper Functions for the VQA (as provided) ===

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
    start_time1 = time.time()
    nc = len(QUBO_matrix)
    data = my_qnode(angles)
    clist = data[::2] + data[1::2]
    blist = []
    for p in range(nc):
        if clist[p] == 0:
            blist.append(0.5)
        else:
            blist.append(data[2 * p + 1] / clist[p])
    end_time1 = time.time()
    print("Time running the qc and getting back results : " + str(end_time1 - start_time1))
    start_time_opt = time.time()
    blist = copy.deepcopy(np.array(blist))
    prob_matrix = np.outer(blist, blist)
    prob_diag = np.diag(prob_matrix)
    mat_diag = np.diag(QUBO_matrix)
    totcost = np.multiply(prob_matrix, QUBO_matrix).sum()
    subtract_cost = np.multiply(prob_diag, mat_diag).sum()
    add_cost = np.multiply(blist, mat_diag).sum()
    quantum_cost = totcost - subtract_cost + add_cost + const
    end_time_opt = time.time()
    print("Time to estimate cost : " + str(end_time_opt - start_time_opt))
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


def softmax(x, temperature=1.0):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()


def calculate_percentile(input_values, q):
    return np.percentile(input_values, q)


def rescaled_rank_rewards(current_value, previous_values, q=1):
    Cq = calculate_percentile(previous_values, q)
    print(f"Percentile cost: {Cq}")
    if current_value < Cq:
        return -(q / 100)
    elif current_value > Cq:
        return 1 - q / 100
    else:
        return 1 if torch.rand(1).item() > 0.5 else -1


def simulated_annealing(current_cost, new_cost, temperature):
    if new_cost < current_cost:
        return True
    else:
        prob_accept = np.exp(-(new_cost - current_cost) / temperature)
        return np.random.rand() < prob_accept
        #return True  


def simplified_rl_search(bitstring, QUBO_matrix, const, time_limit, temperature=10, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    QUBO_matrix = torch.tensor(QUBO_matrix, dtype=torch.float32, device=device)
    const = torch.tensor(const, dtype=torch.float32, device=device)
    initial_bitstring = torch.tensor(bitstring, dtype=torch.float32, device=device)
    print("initial bs : ", initial_bitstring)
    best_state = initial_bitstring.clone()
    state = initial_bitstring.clone()
    num_bits = len(bitstring)
    bit_flip_rewards = np.zeros(num_bits)
    learning_rate = 0.1
    best_cost = torch.matmul(state, torch.matmul(QUBO_matrix, state)) + const
    progress_costs = [best_cost.item()]
    print("Initial cost : ", best_cost.item())
    cut_values = [best_cost.item()]
    P = 100
    total_iterations = 0
    bit_flip_counts = np.zeros(num_bits)
    bit_flip_total_rewards = np.zeros(num_bits)
    start_time = time.time()
    while not time_limit or (time.time() - start_time) < time_limit:
        iteration_start_time = time.time()
        remaining_time = time_limit - (time.time() - start_time)
        total_iterations += 1
        total_actions = np.sum(bit_flip_counts)
        if total_actions > 0:
            print("UCB rewards : ", bit_flip_total_rewards)
            ucb_scores = bit_flip_total_rewards / (bit_flip_counts + 1e-5)
            ucb_scores += np.sqrt(2 * np.log(total_actions) / (bit_flip_counts + 1e-5))
            print("UCB scores : ", ucb_scores)
            bit_to_flip = np.argmax(ucb_scores)
            print("UCB Bit to flip : ", bit_to_flip)
        else:
            bit_to_flip = np.random.choice(num_bits)
        new_state = state.clone()
        new_state[bit_to_flip] = 1 - new_state[bit_to_flip]
        new_cost = torch.matmul(new_state, torch.matmul(QUBO_matrix, new_state)) + const
        progress_costs.append(new_cost.item())
        print("New state : ", new_state)
        print("New cost : ", new_cost)
        if simulated_annealing(best_cost.item(), new_cost.item(), temperature):
            if new_cost.item() < best_cost.item():
                best_state = new_state
                best_cost = new_cost
            state = new_state
            cut_values.append(new_cost.item())
            cut_values = cut_values[-P:]
        bit_flip_total_rewards[bit_to_flip] += rescaled_rank_rewards(new_cost.item(), cut_values)
        bit_flip_counts[bit_to_flip] += 1
        iteration_time = time.time() - iteration_start_time
        if verbose:
            print(f"Current cost: {new_cost.item()}, Global cost: {best_cost.item()}, Time per iteration: {iteration_time:.4f} sec, Time remaining: {remaining_time:.4f} sec")
    return best_state.cpu().numpy(), best_cost.cpu().numpy(), progress_costs


def OPT_step(opt, theta):
    theta, adam_cost = opt.step_and_cost(calmecf, theta)
    return theta, adam_cost


def quantum_opt(QUBO_m, c, num_layers, max_iters, nbitstrings, opt_time, rl_time, initial_temperature, verbose=False):
    """
    Main VQA routine: first uses a variational quantum circuit (via Pennylane's ADAM optimizer)
    to optimize variational parameters and then performs an RL-based branching search.
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
    cost_values = []
    iter_counter_opt = 0
    progress_opt_costs = []

    for iteration in range(max_iters):
        start_time_loop = time.time()
        end_time = time.time() + opt_time

        while time.time() < end_time:
            start_time_opt = time.time()
            theta, opt_cost = OPT_step(opt, theta)
            end_time_opt = time.time()
            iter_counter_opt += 1
            time_per_adam = end_time_opt - start_time_opt
            print("Optimizer iteration : " + str(iter_counter_opt) + ", Cost = " + str(opt_cost) + ", Time : " + str(time_per_adam) + " sec")
            if opt_cost < best_cost_opt:
                best_cost_opt = opt_cost
                best_theta = theta
            progress_opt_costs.append(best_cost_opt)

        iter_counter_branch = 0
        drawn_bitstrings = draw_bitstrings_minenc(best_theta, nbitstrings)
        start_time_branching = time.time()
        for draw_bs in drawn_bitstrings:
            best_bs_bb, current_cost, progress_rl_costs = simplified_rl_search(
                draw_bs, QUBO_matrix, const, rl_time / len(drawn_bitstrings),
                temperature=initial_temperature, verbose=verbose
            )
            if current_cost < best_cost:
                best_cost = current_cost
                best_bitstring = best_bs_bb
            iter_counter_branch += 1
            print("RL iteration : " + str(iter_counter_branch) + ", Cost = " + str(current_cost))
        cost_values.append(best_cost)
        end_time_loop = time.time()
        time_per_iteration = end_time_loop - start_time_loop
        print("Cycle no. " + str(iteration + 1))
        print("Time per ADAM/RL cycle = " + str(time_per_iteration))
        print("Overall minimum cost found = " + str(best_cost))
    return best_bitstring, best_cost, cost_values, time_per_adam + time_per_iteration, progress_rl_costs, progress_opt_costs

# === End Helper Functions ===

# === VQAOptimizer Class ===

class VQAOptimizer(BaseOptimizer):
    """
    A Rastion solver that implements a variational quantum algorithm (VQA) for QUBO problems.
    
    It calls the quantum_opt function internally.
    """
    def __init__(self, num_layers=3, max_iters=10, nbitstrings=10,
                 opt_time=5, rl_time=5, initial_temperature=10, verbose=False):
        self.num_layers = num_layers
        self.max_iters = max_iters
        self.nbitstrings = nbitstrings
        self.opt_time = opt_time
        self.rl_time = rl_time
        self.initial_temperature = initial_temperature
        self.verbose = verbose

    def optimize(self, problem, **kwargs):
        """
        The problem argument is expected to be a QUBO problem (with a get_qubo() method).
        This function extracts the QUBO matrix and constant from the problem and then calls quantum_opt.
        """
        # Extract QUBO data from the problem instance.
        QUBO_matrix, qubo_constant = problem.get_qubo()
        solution, cost, cost_values, cycle_time, progress_rl, progress_opt = quantum_opt(
            QUBO_matrix,
            qubo_constant,
            self.num_layers,
            self.max_iters,
            self.nbitstrings,
            self.opt_time,
            self.rl_time,
            self.initial_temperature,
            self.verbose
        )
        return solution, cost
