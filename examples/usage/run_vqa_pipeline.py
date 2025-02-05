from rastion_hub.auto_problem import AutoProblem
from rastion_hub.auto_optimizer import AutoOptimizer
from rastion_hub.vqa_interface import VQACycleInterface

# Import quantum functions from qubit_eff (our module containing the ansatz, cost, and draw functions)
from rastion_hub.qubit_eff import pennylane_HEcirc, calmecf, draw_bitstrings_minenc

def main():
    # 1. Load the MaxCut problem instance (assumed to be a QUBO problem with get_qubo())
    problem = AutoProblem.from_repo("Rastion/max-cut", revision="main")
    qubo_matrix, qubo_constant = problem.get_qubo()

    # 2. Load the TorchAdamOptimizer as the classical optimizer.
    classical_optimizer = AutoOptimizer.from_repo("Rastion/torch-adam-optimizer", 
                                                  revision="main",
                                                  override_params={
                                                      "max_steps":200
                                                  })
    
    # 3. Define the kwargs for the quantum program.
    quantum_kwargs = {
        "num_layers": 4,
        "nbitstrings": 10,
        "quantum_circuit_fn": pennylane_HEcirc,  # the quantum ansatz
        "cost_function": calmecf,                # cost function (which uses globals from qubit_eff)
        "draw_fn": draw_bitstrings_minenc,       # function to draw candidate bitstrings
        "simulator_fn": None                     # use default simulator (Pennylane device)
    }
    
    # 4. Create the VQA Cycle Interface instance.
    vqa_interface = VQACycleInterface(
        qubo_matrix=qubo_matrix,
        qubo_constant=qubo_constant,
        classical_optimizer=classical_optimizer,
        **quantum_kwargs
    )
    
    # 5. Run the VQA cycle.
    best_candidate, best_candidate_cost = vqa_interface.optimize()
    
    print("Best candidate bitstring:", best_candidate)
    print("Candidate QUBO cost:", best_candidate_cost)

if __name__ == "__main__":
    main()
