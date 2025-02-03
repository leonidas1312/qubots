from rastion_core.base_problem import BaseProblem
import random

class VehicleRoutingProblem(BaseProblem):
    """
    A simple formulation of the Vehicle Routing Problem (VRP):
    - Given a depot and several customer locations with demands,
      assign routes for a fleet of vehicles with limited capacity.
    """
    def __init__(self, distance_matrix, demands, vehicle_capacity, num_vehicles, depot=0):
        self.distance_matrix = distance_matrix
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        self.depot = depot
        self.num_customers = len(demands)
    
    def evaluate_solution(self, solution) -> float:
        """
        Evaluate a solution represented as a list of routes.
        Each route is a list of customer indices (with the depot at start and end).
        Computes the total distance with penalties for capacity violations.
        """
        total_distance = 0
        penalty = 0
        for route in solution:
            # Each route should start and end at the depot.
            if not route or route[0] != self.depot or route[-1] != self.depot:
                penalty += 1e6
                continue
            load = 0
            route_distance = 0
            for i in range(len(route)-1):
                frm = route[i]
                to = route[i+1]
                route_distance += self.distance_matrix[frm][to]
                if to != self.depot:
                    load += self.demands[to]
            if load > self.vehicle_capacity:
                penalty += (load - self.vehicle_capacity) * 1000
            total_distance += route_distance
        return total_distance + penalty
    
    def random_solution(self):
        """
        Generate a random solution:
        Randomly partition the customer indices (excluding the depot) into routes.
        """
        customers = list(range(1, self.num_customers))
        random.shuffle(customers)
        routes = []
        size = max(1, len(customers) // self.num_vehicles)
        for i in range(self.num_vehicles):
            route_customers = customers[i*size:(i+1)*size]
            route = [self.depot] + route_customers + [self.depot]
            routes.append(route)
        return routes
