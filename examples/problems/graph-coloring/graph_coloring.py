from rastion_core.base_problem import BaseProblem
import random

class GraphColoringProblem(BaseProblem):
    """
    Graph Coloring Problem:
    Assign colors to nodes such that no two adjacent nodes share the same color.
    The objective is to minimize the number of conflicts.
    """
    def __init__(self, adjacency_list, num_colors):
        self.adjacency_list = adjacency_list  # Dict: node -> list of adjacent nodes.
        self.num_nodes = len(adjacency_list)
        self.num_colors = num_colors
    
    def evaluate_solution(self, coloring) -> float:
        conflicts = 0
        for node, neighbors in self.adjacency_list.items():
            for neighbor in neighbors:
                if coloring[node] == coloring[neighbor]:
                    conflicts += 1
        return conflicts / 2  # each conflict is counted twice.
    
    def random_solution(self):
        return [random.randint(0, self.num_colors - 1) for _ in range(self.num_nodes)]
