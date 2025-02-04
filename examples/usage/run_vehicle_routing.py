from rastion_hub.auto_problem import AutoProblem
from rastion_hub.auto_optimizer import AutoOptimizer

def run_vehicle_routing():
    org = "Rastion"
    
    # Load the Vehicle Routing Problem.
    # Ensure that the "vehicle-routing" repository contains a file named "vehicle_routing.py"
    # and a valid "problem_config.json".
    problem = AutoProblem.from_repo(f"{org}/vehicle-routing", revision="main")
    
    # Load the Ant Colony Optimizer.
    # This repository should contain a file named "ant_colony_optimizer.py" (hyphens replaced with underscores)
    # and a valid "solver_config.json".
    optimizer = AutoOptimizer.from_repo(f"{org}/ant-colony", revision="main")
    
    best_solution, best_cost = optimizer.optimize(problem)
    print("Vehicle Routing Problem with ACO")
    print("Best Route:", best_solution)
    print("Best Distance:", best_cost)

if __name__ == "__main__":
    run_vehicle_routing()
