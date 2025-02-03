from rastion_core.base_problem import BaseProblem
import random

class NetworkFlowProblem(BaseProblem):
    """
    Network Flow Optimization Problem:
    Given a directed graph with capacities and costs on edges,
    find the flow on each edge that minimizes the total cost while meeting node supply/demand.
    """
    def __init__(self, edges, capacities, costs, supplies):
        self.edges = edges                # List of [u, v] pairs.
        self.capacities = capacities      # List of capacities.
        self.costs = costs                # List of costs per unit flow.
        self.supplies = supplies          # Dict mapping node -> supply (positive for supply, negative for demand).
        self.num_edges = len(edges)
    
    def evaluate_solution(self, flows) -> float:
        penalty = 0
        for i, flow in enumerate(flows):
            if flow < 0 or flow > self.capacities[i]:
                penalty += 1e6
        node_balance = {}
        for (u, v), flow in zip(self.edges, flows):
            node_balance[u] = node_balance.get(u, 0) - flow
            node_balance[v] = node_balance.get(v, 0) + flow
        for node, supply in self.supplies.items():
            penalty += abs(node_balance.get(node, 0) - supply) * 1e5
        cost = sum(flow * cost for flow, cost in zip(flows, self.costs))
        return cost + penalty
    
    def random_solution(self):
        return [random.uniform(0, cap) for cap in self.capacities]
