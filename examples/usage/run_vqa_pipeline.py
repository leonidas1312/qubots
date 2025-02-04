# File: run_vqa_pipeline.py

from rastion_hub.auto_problem import AutoProblem
from rastion_hub.auto_optimizer import AutoOptimizer
from rastion_hub.vqa_pipeline import create_vqa

def run_vqa_pipeline():
    org = "Rastion"
    
    # Load the gate assignment QUBO problem (which provides get_qubo())
    problem = AutoProblem.from_repo(f"{org}/max-cut", revision="main")
    
    # Load the quantum optimizer (vqa-optimizer) from the Rastion hub.
    quantum_optimizer = AutoOptimizer.from_repo(
        f"{org}/vqa-optimizer",
        revision="main",
        override_params={
            "num_layers": 4,
            "max_iters": 1,
            "nbitstrings": 1,
            "opt_time": 5,
            "rl_time": 5,
            "initial_temperature": 10,
            "verbose": True
        }
    )
    
    # Load a classical optimizer that will refine the quantum result.
    classical_optimizer = AutoOptimizer.from_repo(
        f"{org}/particle-swarm",
        revision="main",
        override_params={
            "max_iters": 100
        }
    )
    
    # Create the VQA pipeline by combining quantum and classical routines.
    vqa = create_vqa(quantum_routine=quantum_optimizer, classical_optimizer=classical_optimizer)
    
    # Run the pipeline on the problem.
    solution, cost = vqa.optimize(problem)
    
    print("Final solution from VQA pipeline:", solution)
    print("Final cost from VQA pipeline:", cost)

if __name__ == "__main__":
    run_vqa_pipeline()
