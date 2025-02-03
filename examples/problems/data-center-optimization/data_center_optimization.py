from rastion_core.base_problem import BaseProblem
import random

class DataCenterOptimizationProblem(BaseProblem):
    """
    Data Center Resource Optimization:
    Allocate computing resources to tasks to meet requirements while minimizing allocation error.
    """
    def __init__(self, task_requirements, total_resources):
        self.task_requirements = task_requirements  # List of required resources per task.
        self.total_resources = total_resources
        self.num_tasks = len(task_requirements)
    
    def evaluate_solution(self, allocation) -> float:
        penalty = 0
        if sum(allocation) > self.total_resources:
            penalty += (sum(allocation) - self.total_resources) * 1e6
        error = sum(abs(a - r) for a, r in zip(allocation, self.task_requirements))
        return error + penalty
    
    def random_solution(self):
        return [random.uniform(0, self.total_resources / self.num_tasks) for _ in range(self.num_tasks)]
