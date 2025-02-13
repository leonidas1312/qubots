"""
This script benchmarks the performance of the VQA pipeline (a hybrid quantum-classical solver)
against the optimal solution obtained by an exhaustive search solver for small QUBO problems.
It runs benchmarks for three problems: MaxCut, Graph Coloring, and Knapsack.
The VQA pipeline is built by composing a quantum optimizer and a classical optimizer in series.
In this test, we compare the performance of three classical optimizers:
    - rl-optimizer
    - particle-swarm optimizer
    - tabu-search optimizer
with their respective parameters.
"""

from qubots.auto_problem import AutoProblem
from qubots.auto_optimizer import AutoOptimizer
from qubots.quantum_classical_pipeline import create_quantum_classical_pipeline

def benchmark_problem(problem_repo, problem_name, classical_optimizers, vqa_repo, exhaustive_solver):
    org = "Rastion"
    print("=" * 60)
    print(f"Benchmarking {problem_name} Problem")
    print("=" * 60)
    
    # 1. Load the problem instance (assumed to be a QUBO problem with a get_qubo() method).
    problem = AutoProblem.from_repo(f"{org}/{problem_repo}", revision="main")
    
    # 2. Load the common quantum optimizer for the VQA pipeline.
    quantum_optimizer = AutoOptimizer.from_repo(
        f"{org}/{vqa_repo}",
        revision="main",
        override_params={
            "num_layers": 1,        
            "max_iters": 100,
            "verbose": True
        }
    )
    
    # 3. For each classical optimizer configuration, run the pipeline and compare the results.
    for optimizer_name, config in classical_optimizers.items():
        print("-" * 60)
        print(f"Using Classical Optimizer: {optimizer_name}")
        print("-" * 60)
        
        classical_optimizer = AutoOptimizer.from_repo(
            f"{org}/{config['repo']}",
            revision="main",
            override_params=config["override_params"]
        )
        
        # Compose the quantum-classical pipeline.
        pipeline = create_quantum_classical_pipeline(
            quantum_routine=quantum_optimizer,
            classical_optimizer=classical_optimizer
        )
        
        # Run the VQA pipeline.
        print("Running VQA pipeline ...")
        vqa_solution, vqa_cost = pipeline.optimize(problem)
        print(f"VQA Pipeline Solution: {vqa_solution}")
        print(f"VQA Pipeline Cost: {vqa_cost}")
        
        
        print("Running Exhaustive Search ...")
        opt_solution, opt_cost = exhaustive_solver.optimize(problem)
        print(f"Exhaustive Search Optimal Solution: {opt_solution}")
        print(f"Exhaustive Search Optimal Cost: {opt_cost}")
        
        # 5. Compare the results.
        error = abs(vqa_cost - opt_cost)
        rel_error = error / (abs(opt_cost) if opt_cost != 0 else 1)
        print(f"Absolute Error: {error}")
        print(f"Relative Error: {rel_error * 100:.2f}%")
        print("\n")

def main():
    # Define the classical optimizers with their repositories and override parameters.
    classical_optimizers = {
        "rl-optimizer": {
            "repo": "rl-optimizer",
            "override_params": {
                "time_limit": 1  # seconds
            }
        },
        "particle-swarm": {
            "repo": "particle-swarm",
            "override_params": {
                "swarm_size": 30,
                "max_iters": 100,
                "inertia": 0.5,
                "cognitive": 1.0,
                "social": 1.0,
                "verbose": True
            }
        },
        "tabu-search": {
            "repo": "tabu-search",
            "override_params": {
                "max_iters": 100,
                "tabu_tenure": 5,
                "verbose": True
            }
        }
    }

    
    
    # The repository for the quantum optimizer and the exhaustive search solver.
    vqa_repo = "quantum-qaoa"
    exhaustive_repo = "exhaustive-search"
    org = "Rastion"
    # Load the Exhaustive Search solver to obtain the optimal solution.
    exhaustive_solver = AutoOptimizer.from_repo(
        f"{org}/{exhaustive_repo}",
        revision="main"
    )
    
    # Benchmark the VQA pipeline for three problems.
    benchmark_problem(
        problem_repo="max-cut",
        problem_name="MaxCut",
        classical_optimizers=classical_optimizers,
        vqa_repo=vqa_repo,
        exhaustive_solver=exhaustive_solver
    )
    
    benchmark_problem(
        problem_repo="graph-coloring",
        problem_name="Graph Coloring",
        classical_optimizers=classical_optimizers,
        vqa_repo=vqa_repo,
        exhaustive_solver=exhaustive_solver
    )
    
    benchmark_problem(
        problem_repo="knapsack",
        problem_name="Knapsack",
        classical_optimizers=classical_optimizers,
        vqa_repo=vqa_repo,
        exhaustive_solver=exhaustive_solver
    )
    
if __name__ == "__main__":
    main()
