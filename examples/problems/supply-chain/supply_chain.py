from rastion_core.base_problem import BaseProblem
import random

class SupplyChainProblem(BaseProblem):
    """
    Supply Chain Optimization Problem:
    Optimize decisions along a supply chain (e.g., production, transportation)
    to minimize overall costs.
    This is a simplified formulation.
    """
    def __init__(self, production_costs, transportation_costs, holding_cost, demand):
        self.production_costs = production_costs
        self.transportation_costs = transportation_costs  # matrix: factory to market costs.
        self.holding_cost = holding_cost
        self.demand = demand
        self.num_factories = len(production_costs)
        self.num_markets = len(demand)
    
    def evaluate_solution(self, decisions) -> float:
        # 'decisions' is a dict with keys "production" and "transportation".
        production = decisions.get("production", [0] * self.num_factories)
        transportation = decisions.get("transportation", [[0] * self.num_markets for _ in range(self.num_factories)])
        cost = sum(p * c for p, c in zip(production, self.production_costs))
        for i in range(self.num_factories):
            for j in range(self.num_markets):
                cost += transportation[i][j] * self.transportation_costs[i][j]
        cost += sum(production) * self.holding_cost  # simplified holding cost.
        total_supply = sum(production)
        penalty = max(0, sum(self.demand) - total_supply) * 1e5
        return cost + penalty
    
    def random_solution(self):
        production = [random.uniform(0, d) for d in self.demand]
        transportation = [[random.uniform(0, production[i] / self.num_markets) for _ in range(self.num_markets)]
                          for i in range(self.num_factories)]
        return {"production": production, "transportation": transportation}
