"""
Vehicle Routing Problem (VRP) Implementation for Qubots Framework

This module implements a Vehicle Routing Problem as a qubots CombinatorialProblem.
The VRP involves finding optimal routes for a fleet of vehicles to serve customers
while minimizing total travel distance and respecting vehicle capacity constraints.

Compatible with Rastion platform playground for interactive optimization.

Author: Qubots Community
Version: 1.0.0
"""

import numpy as np
import random
import json
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from qubots import (
        BaseProblem, ProblemMetadata, ProblemType,
        ObjectiveType, DifficultyLevel
    )


@dataclass
class Customer:
    """Represents a customer location with demand and coordinates."""
    id: int
    x: float
    y: float
    demand: int
    service_time: int = 15  # minutes
    time_window_start: int = 0
    time_window_end: int = 1440  # 24 hours in minutes


@dataclass
class Vehicle:
    """Represents a vehicle with capacity and cost information."""
    id: int
    capacity: int
    start_depot: int = 0
    end_depot: int = 0
    cost_per_km: float = 1.0


class VehicleRoutingProblem(BaseProblem):
    """
    Vehicle Routing Problem implementation using qubots framework.

    This problem finds optimal routes for vehicles to serve customers while:
    - Minimizing total travel distance/cost
    - Respecting vehicle capacity constraints
    - Ensuring all customers are served exactly once
    - Starting and ending routes at depot locations

    Solution Format:
        List of routes, where each route is a list of customer IDs.
        Example: [[1, 3, 5], [2, 4], [6, 7, 8]] represents 3 vehicle routes.
    """

    def __init__(self,
                 customers: Optional[List[Customer]] = None,
                 vehicles: Optional[List[Vehicle]] = None,
                 depot_location: Tuple[float, float] = (0.0, 0.0),
                 **kwargs):
        """
        Initialize the Vehicle Routing Problem.

        Args:
            customers: List of customer locations with demands
            vehicles: List of available vehicles with capacities
            depot_location: Coordinates of the depot (x, y)
            **kwargs: Additional parameters for problem configuration
        """
        # Set default data if not provided
        if customers is None:
            customers = self._generate_default_customers()
        if vehicles is None:
            vehicles = self._generate_default_vehicles()

        self.customers = customers
        self.vehicles = vehicles
        self.depot_location = depot_location
        self.n_customers = len(customers)
        self.n_vehicles = len(vehicles)

        # Create distance matrix
        self._distance_matrix = self._calculate_distance_matrix()

        # Problem parameters from kwargs
        self.max_route_duration = kwargs.get('max_route_duration', 480)  # 8 hours
        self.penalty_unserved = kwargs.get('penalty_unserved', 1000.0)
        self.penalty_capacity = kwargs.get('penalty_capacity', 500.0)

        # Initialize with metadata
        metadata = self._get_default_metadata()
        super().__init__(metadata)

    def _get_default_metadata(self) -> 'ProblemMetadata':
        """Return default metadata for VRP."""

        return ProblemMetadata(
            name="Vehicle Routing Problem",
            description="Multi-vehicle delivery optimization with capacity constraints",
            problem_type=ProblemType.COMBINATORIAL,
            objective_type=ObjectiveType.MINIMIZE,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            domain="routing",
            tags={"routing", "logistics", "vrp", "optimization", "delivery"},
            author="Qubots Community",
            version="1.0.0",
            dimension=getattr(self, 'n_customers', 10),  # Default to 10 if not set yet
            constraints_count=2  # capacity and coverage constraints
        )

    def _generate_default_customers(self) -> List[Customer]:
        """Generate default customer data for demonstration."""
        customers = []
        random.seed(42)  # For reproducible results

        for i in range(1, 11):  # 10 customers
            customers.append(Customer(
                id=i,
                x=random.uniform(-10, 10),
                y=random.uniform(-10, 10),
                demand=random.randint(5, 25),
                service_time=random.randint(10, 30)
            ))

        return customers

    def _generate_default_vehicles(self) -> List[Vehicle]:
        """Generate default vehicle fleet."""
        return [
            Vehicle(id=1, capacity=50, cost_per_km=1.0),
            Vehicle(id=2, capacity=60, cost_per_km=1.2),
            Vehicle(id=3, capacity=40, cost_per_km=0.8)
        ]

    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate distance matrix between all locations (depot + customers)."""
        n_locations = self.n_customers + 1  # +1 for depot
        distance_matrix = np.zeros((n_locations, n_locations))

        # All locations: depot (index 0) + customers (indices 1 to n_customers)
        all_locations = [self.depot_location] + [(c.x, c.y) for c in self.customers]

        for i in range(n_locations):
            for j in range(n_locations):
                if i != j:
                    x1, y1 = all_locations[i]
                    x2, y2 = all_locations[j]
                    distance_matrix[i][j] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        return distance_matrix

    def evaluate_solution(self, solution: List[List[int]], verbose: bool = False) -> float:
        """
        Evaluate a VRP solution and return total cost.

        Args:
            solution: List of routes, each route is a list of customer IDs
            verbose: If True, print real-time evaluation details for playground streaming

        Returns:
            Total cost (distance + penalties)
        """
        if not self.validate_solution_format(solution):
            if verbose:
                print("❌ Invalid solution format")
            return float('inf')

        if verbose:
            print(f"🔍 Evaluating solution with {len(solution)} routes...")

        total_cost = 0.0
        served_customers = set()

        for vehicle_idx, route in enumerate(solution):
            if vehicle_idx >= len(self.vehicles):
                continue  # Skip if more routes than vehicles

            vehicle = self.vehicles[vehicle_idx]
            route_cost, route_demand = self._evaluate_route(route, vehicle)
            total_cost += route_cost

            if verbose and route:
                utilization = (route_demand / vehicle.capacity * 100) if vehicle.capacity > 0 else 0
                print(f"   Route {vehicle_idx + 1}: {len(route)} customers, cost: {route_cost:.2f}, demand: {route_demand}/{vehicle.capacity} ({utilization:.1f}%)")

            # Track served customers
            served_customers.update(route)

            # Capacity constraint penalty
            if route_demand > vehicle.capacity:
                penalty = self.penalty_capacity * (route_demand - vehicle.capacity)
                total_cost += penalty
                if verbose:
                    print(f"   ⚠️ Capacity violation: +{penalty:.2f} penalty")

        # Penalty for unserved customers
        all_customer_ids = {c.id for c in self.customers}
        unserved_customers = all_customer_ids - served_customers
        unserved_penalty = len(unserved_customers) * self.penalty_unserved
        total_cost += unserved_penalty

        if verbose:
            if unserved_customers:
                print(f"   ⚠️ Unserved customers: {len(unserved_customers)}, penalty: +{unserved_penalty:.2f}")
            print(f"✅ Total cost: {total_cost:.2f}")

        return total_cost

    def _evaluate_route(self, route: List[int], vehicle: Vehicle) -> Tuple[float, int]:
        """
        Evaluate a single route for a vehicle.

        Returns:
            Tuple of (route_cost, total_demand)
        """
        if not route:
            return 0.0, 0

        total_distance = 0.0
        total_demand = 0

        # Distance from depot to first customer
        if route:
            first_customer_idx = self._get_customer_index(route[0])
            if first_customer_idx is not None:
                total_distance += self._distance_matrix[0][first_customer_idx + 1]

        # Distance between consecutive customers
        for i in range(len(route) - 1):
            from_idx = self._get_customer_index(route[i])
            to_idx = self._get_customer_index(route[i + 1])

            if from_idx is not None and to_idx is not None:
                total_distance += self._distance_matrix[from_idx + 1][to_idx + 1]

        # Distance from last customer back to depot
        if route:
            last_customer_idx = self._get_customer_index(route[-1])
            if last_customer_idx is not None:
                total_distance += self._distance_matrix[last_customer_idx + 1][0]

        # Calculate total demand
        for customer_id in route:
            customer = self._get_customer_by_id(customer_id)
            if customer:
                total_demand += customer.demand

        route_cost = total_distance * vehicle.cost_per_km
        return route_cost, total_demand

    def _get_customer_index(self, customer_id: int) -> Optional[int]:
        """Get customer index by ID."""
        for i, customer in enumerate(self.customers):
            if customer.id == customer_id:
                return i
        return None

    def _get_customer_by_id(self, customer_id: int) -> Optional[Customer]:
        """Get customer object by ID."""
        for customer in self.customers:
            if customer.id == customer_id:
                return customer
        return None

    def validate_solution_format(self, solution: Any) -> bool:
        """Validate that solution has correct format."""
        if not isinstance(solution, list):
            return False

        for route in solution:
            if not isinstance(route, list):
                return False
            for customer_id in route:
                if not isinstance(customer_id, int):
                    return False

        return True

    def get_random_solution(self) -> List[List[int]]:
        """Generate a random valid solution."""
        customer_ids = [c.id for c in self.customers]
        random.shuffle(customer_ids)

        # Distribute customers among vehicles
        routes = [[] for _ in range(self.n_vehicles)]

        for i, customer_id in enumerate(customer_ids):
            vehicle_idx = i % self.n_vehicles
            routes[vehicle_idx].append(customer_id)

        return routes

    def get_solution_summary(self, solution: List[List[int]]) -> Dict[str, Any]:
        """Get detailed summary of a solution."""
        if not self.validate_solution_format(solution):
            return {"error": "Invalid solution format"}

        total_cost = self.evaluate_solution(solution)
        served_customers = set()
        route_details = []

        for vehicle_idx, route in enumerate(solution):
            if vehicle_idx >= len(self.vehicles):
                continue

            vehicle = self.vehicles[vehicle_idx]
            route_cost, route_demand = self._evaluate_route(route, vehicle)

            route_details.append({
                "vehicle_id": vehicle.id,
                "route": route,
                "customers_served": len(route),
                "total_demand": route_demand,
                "vehicle_capacity": vehicle.capacity,
                "capacity_utilization": route_demand / vehicle.capacity if vehicle.capacity > 0 else 0,
                "route_cost": route_cost,
                "feasible": route_demand <= vehicle.capacity
            })

            served_customers.update(route)

        all_customer_ids = {c.id for c in self.customers}
        unserved_customers = all_customer_ids - served_customers

        return {
            "total_cost": total_cost,
            "total_customers": len(all_customer_ids),
            "served_customers": len(served_customers),
            "unserved_customers": len(unserved_customers),
            "unserved_customer_ids": list(unserved_customers),
            "vehicles_used": len([r for r in route_details if r["customers_served"] > 0]),
            "total_vehicles": len(self.vehicles),
            "route_details": route_details,
            "feasible": len(unserved_customers) == 0 and all(r["feasible"] for r in route_details)
        }
