from rastion_hub.base_problem import BaseProblem
import random

class ResourceAllocationProblem(BaseProblem):
    """
    Resource Allocation Problem:
    Allocate limited resources among projects to maximize benefit (or minimize cost).
    A solution is a list of allocations (fractions) that sum to 1.
    """
    def __init__(self, benefits):
        self.benefits = benefits  # List of benefit factors.
        self.num_projects = len(benefits)
    
    def evaluate_solution(self, allocation) -> float:
        allocation = [max(0, a) for a in allocation]
        total = sum(allocation)
        if total == 0:
            return 1e6
        allocation = [a / total for a in allocation]
        cost = -sum(a * b for a, b in zip(allocation, self.benefits))
        penalty = abs(sum(allocation) - 1) * 1e6
        return cost + penalty
    
    def random_solution(self):
        alloc = [random.random() for _ in range(self.num_projects)]
        total = sum(alloc)
        return [a / total for a in alloc]
