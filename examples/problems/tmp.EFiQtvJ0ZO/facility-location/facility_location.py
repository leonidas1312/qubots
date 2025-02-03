from rastion_core.base_problem import BaseProblem
import random

class FacilityLocationProblem(BaseProblem):
    """
    A simple Facility Location Problem.
    Decide which facilities to open (binary decision for each facility) to serve customers.
    The objective is to minimize the sum of facility opening costs and customer assignment costs.
    """
    def __init__(self, facility_costs, assignment_costs):
        self.facility_costs = facility_costs
        self.assignment_costs = assignment_costs
        self.num_facilities = len(facility_costs)
        self.num_customers = len(assignment_costs[0])
    
    def evaluate_solution(self, solution) -> float:
        # 'solution' is a binary list: 1 means the facility is open.
        open_cost = sum(self.facility_costs[i] for i, flag in enumerate(solution) if flag)
        assignment_cost = 0
        for j in range(self.num_customers):
            costs = [self.assignment_costs[i][j] for i, flag in enumerate(solution) if flag]
            if costs:
                assignment_cost += min(costs)
            else:
                assignment_cost += 1e6  # heavy penalty if no facility is open
        return open_cost + assignment_cost
    
    def random_solution(self):
        return [random.randint(0, 1) for _ in range(self.num_facilities)]
