from rastion_core.base_problem import BaseProblem
import random

class EnergyManagementProblem(BaseProblem):
    """
    Energy Management Problem:
    Optimize energy generation/distribution in a smart grid.
    A solution is a schedule of energy production values.
    """
    def __init__(self, demand, production_cost, production_bounds):
        self.demand = demand
        self.production_cost = production_cost
        self.production_bounds = production_bounds  # list of [min, max] for each time period.
        self.periods = len(demand)
    
    def evaluate_solution(self, production_schedule) -> float:
        total_cost = 0
        penalty = 0
        for t in range(self.periods):
            production = production_schedule[t]
            if production < self.production_bounds[t][0] or production > self.production_bounds[t][1]:
                penalty += 1e6
            total_cost += production * self.production_cost[t]
            if production < self.demand[t]:
                penalty += (self.demand[t] - production) * 1000
        return total_cost + penalty
    
    def random_solution(self):
        return [random.uniform(b[0], b[1]) for b in self.production_bounds]
