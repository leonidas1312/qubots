"""
This script benchmarks the performance of the VQA pipeline (a hybrid quantum-classical solver)
against the optimal solution obtained by an exhaustive search solver for small QUBO problems.
It runs benchmarks for three problems: MaxCut, Graph Coloring, and Knapsack.
The VQA pipeline is built by composing a quantum optimizer and a classical optimizer in series.
"""

import time
from rastion_hub.auto_problem import AutoProblem
from rastion_hub.auto_optimizer import AutoOptimizer
from rastion_hub.quantum_classical_pipeline import create_quantum_classical_pipeline

def benchmark_problem(problem_repo, problem_name, classical_repo, vqa_repo, exhaustive_repo):
    org = "Rastion"
    print("="*60)
    print(f"Benchmarking {problem_name} problem")
    print("="*60)
    
    # 1. Load the problem instance (assumed to be a QUBO problem with a get_qubo() method)
    problem = AutoProblem.from_repo(f"{org}/{problem_repo}", revision="main")
    
    # 2. Load the VQA pipeline components.
    quantum_optimizer = AutoOptimizer.from_repo(
        f"{org}/{vqa_repo}",
        revision="main",
        override_params={
            "num_layers": 6,        
            "max_iters": 200,
            "nbitstrings": 5,
        }
    )
    
    # Load a classical optimizer to refine the quantum output.
    classical_optimizer = AutoOptimizer.from_repo(
        f"{org}/{classical_repo}",
        revision="main",
        override_params={
            "time_limit": 15
        }
    )
    
    # Compose the pipeline by combining the quantum and classical routines.
    quantum_classical_pipeline = create_quantum_classical_pipeline(quantum_routine=quantum_optimizer, classical_optimizer=classical_optimizer)
    
    # Run the VQA pipeline and time it.
    print("Running VQA pipeline ...")
    start = time.time()
    vqa_solution, vqa_cost = quantum_classical_pipeline.optimize(problem)
    end = time.time()
    vqa_time = end - start
    print(f"VQA Pipeline Solution: {vqa_solution}")
    print(f"VQA Pipeline Cost: {vqa_cost}")
    print(f"Time taken: {vqa_time:.2f} seconds")
    
    # 3. Load the Exhaustive Search solver to obtain the optimal solution.
    exhaustive_solver = AutoOptimizer.from_repo(
        f"{org}/{exhaustive_repo}",
        revision="main"
    )
    print("Running Exhaustive Search ...")
    start = time.time()
    opt_solution, opt_cost = exhaustive_solver.optimize(problem)
    end = time.time()
    ex_time = end - start
    print(f"Exhaustive Search Optimal Solution: {opt_solution}")
    print(f"Exhaustive Search Optimal Cost: {opt_cost}")
    print(f"Time taken: {ex_time:.2f} seconds")
    
    # 4. Compare the results.
    error = abs(vqa_cost - opt_cost)
    rel_error = error / (abs(opt_cost) if opt_cost != 0 else 1)
    print(f"Absolute Error: {error}")
    print(f"Relative Error: {rel_error*100:.2f}%")
    print("\n")

def main():
    # Benchmark the VQA pipeline for three problems:
    # For each, we assume the repositories are named as follows:
    #   - MaxCut problem repo: "max-cut"
    #   - Graph Coloring problem repo: "graph-coloring"
    #   - Gate assignment problem repo: "gate-assignment"
    # And the solvers:
    #   - Quantum optimizer for VQA: "vqa-optimizer"
    #   - Classical optimizer for refining 
    #   - Exhaustive search: "exhaustive-search"
    
    benchmark_problem(
        problem_repo="max-cut",
        problem_name="MaxCut",
        classical_repo="rl-optimizer",
        vqa_repo="vqa-qubit-eff",
        exhaustive_repo="exhaustive-search"
    )
    
    benchmark_problem(
        problem_repo="graph-coloring",
        problem_name="Graph Coloring",
        classical_repo="rl-optimizer",
        vqa_repo="vqa-qubit-eff",
        exhaustive_repo="exhaustive-search"
    )
    
    benchmark_problem(
        problem_repo="gate-assignment",
        problem_name="Gate-assignment",
        classical_repo="rl-optimizer",
        vqa_repo="vqa-qubit-eff",
        exhaustive_repo="exhaustive-search"
    )

    
    
if __name__ == "__main__":
    main()
