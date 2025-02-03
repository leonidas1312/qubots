from rastion_core.base_problem import BaseProblem
import random

class SensorPlacementProblem(BaseProblem):
    """
    Sensor Placement Problem:
    Place a fixed number of sensors in a region to maximize coverage.
    A solution is a list of sensor positions.
    """
    def __init__(self, region_bounds, num_sensors):
        self.region_bounds = region_bounds  # e.g., [[x_min, x_max], [y_min, y_max]]
        self.num_sensors = num_sensors
    
    def evaluate_solution(self, positions) -> float:
        # Use the sum of pairwise Euclidean distances as a proxy for coverage.
        import math
        total_distance = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                total_distance += math.sqrt(dx * dx + dy * dy)
        # Since higher distance means better spread, return negative to minimize.
        return -total_distance
    
    def random_solution(self):
        return [[random.uniform(self.region_bounds[0][0], self.region_bounds[0][1]),
                 random.uniform(self.region_bounds[1][0], self.region_bounds[1][1])]
                for _ in range(self.num_sensors)]
