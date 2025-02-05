from rastion_hub.base_problem import BaseProblem
import random

class InventoryManagementProblem(BaseProblem):
    """
    Inventory Management Problem:
    Decide order quantities over a planning horizon to minimize holding and shortage costs.
    A solution is a list of order quantities.
    """
    def __init__(self, demand, holding_cost, shortage_cost, initial_inventory=0):
        self.demand = demand
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.initial_inventory = initial_inventory
        self.periods = len(demand)
    
    def evaluate_solution(self, orders) -> float:
        inventory = self.initial_inventory
        total_cost = 0
        for t in range(self.periods):
            inventory += orders[t] - self.demand[t]
            if inventory >= 0:
                total_cost += inventory * self.holding_cost
            else:
                total_cost += abs(inventory) * self.shortage_cost
        return total_cost
    
    def random_solution(self):
        return [random.randint(0, max(self.demand) + 10) for _ in range(self.periods)]
