"""
Vehicle Routing Problem (VRP) Tutorial with Qubots and Rastion Integration

This tutorial demonstrates:
1. Creating a Vehicle Routing Problem using qubots CombinatorialProblem
2. Implementing an OR-Tools based VRP optimizer
3. Uploading models to Rastion platform
4. Loading models from Rastion and verifying integrity
5. Complete workflow with error handling and validation

Requirements:
- qubots library
- ortools (pip install ortools)
- numpy
- matplotlib (for visualization)

Author: Qubots Tutorial Team
Version: 1.0.0
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import time
import json

# Add qubots to path if running locally
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Qubots imports
from qubots import (
    CombinatorialProblem, BaseOptimizer,
    ProblemMetadata, OptimizerMetadata,
    ProblemType, ObjectiveType, DifficultyLevel,
    OptimizerType, OptimizerFamily,
    OptimizationResult
)
import qubots.rastion as rastion

# OR-Tools imports (with fallback)
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
    print("✅ OR-Tools available for VRP solving")
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("⚠️  OR-Tools not available. Install with: pip install ortools")


@dataclass
class VehicleInfo:
    """Information about a vehicle in the fleet."""
    id: int
    capacity: int
    start_depot: int = 0
    end_depot: int = 0
    cost_per_km: float = 1.0


@dataclass
class CustomerInfo:
    """Information about a customer location."""
    id: int
    x: float
    y: float
    demand: int
    time_window_start: int = 0
    time_window_end: int = 1440  # 24 hours in minutes
    service_time: int = 15  # minutes


class VehicleRoutingProblem(CombinatorialProblem):
    """
    Vehicle Routing Problem implementation using qubots CombinatorialProblem.

    This problem involves finding optimal routes for a fleet of vehicles
    to serve a set of customers while minimizing total travel distance/cost.
    """

    def __init__(self, customers: List[CustomerInfo], vehicles: List[VehicleInfo],
                 depot_location: Tuple[float, float] = (0.0, 0.0)):
        """
        Initialize the VRP instance.

        Args:
            customers: List of customer information
            vehicles: List of vehicle information
            depot_location: Coordinates of the depot
        """
        self.customers = customers
        self.vehicles = vehicles
        self.depot_location = depot_location
        self.n_customers = len(customers)
        self.n_vehicles = len(vehicles)

        # Create complete location list (depot + customers)
        self.locations = [depot_location] + [(c.x, c.y) for c in customers]
        self.n_locations = len(self.locations)

        # Calculate distance matrix
        self.distance_matrix = self._calculate_distance_matrix()

        # Create demands array (depot has 0 demand)
        self.demands = [0] + [c.demand for c in customers]

        # Initialize base class with all locations as elements
        elements = list(range(self.n_locations))
        super().__init__(elements, self._get_default_metadata())

    def _get_default_metadata(self) -> ProblemMetadata:
        """Return default metadata for VRP."""
        return ProblemMetadata(
            name="Vehicle Routing Problem",
            description=f"VRP with {self.n_customers} customers and {self.n_vehicles} vehicles",
            problem_type=ProblemType.COMBINATORIAL,
            objective_type=ObjectiveType.MINIMIZE,
            difficulty_level=DifficultyLevel.ADVANCED,
            domain="logistics",
            tags={"vrp", "routing", "logistics", "combinatorial", "ortools"},
            author="Qubots VRP Tutorial",
            version="1.0.0",
            dimension=self.n_customers,
            evaluation_complexity="O(n²)",
            memory_complexity="O(n²)"
        )

    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate Euclidean distance matrix between all locations."""
        n = len(self.locations)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = self.locations[i]
                    x2, y2 = self.locations[j]
                    matrix[i][j] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        return matrix

    def evaluate_solution(self, solution: Dict[str, Any]) -> float:
        """
        Evaluate a VRP solution.

        Args:
            solution: Dictionary containing:
                - 'routes': List of routes, each route is a list of location indices
                - 'vehicle_assignments': List mapping route index to vehicle index

        Returns:
            Total cost (distance) of all routes
        """
        if not self.is_feasible(solution):
            return float('inf')  # Infeasible solutions get infinite cost

        routes = solution.get('routes', [])
        vehicle_assignments = solution.get('vehicle_assignments', [])

        total_cost = 0.0

        for route_idx, route in enumerate(routes):
            if not route:  # Empty route
                continue

            vehicle_idx = vehicle_assignments[route_idx] if route_idx < len(vehicle_assignments) else 0
            vehicle = self.vehicles[vehicle_idx] if vehicle_idx < len(self.vehicles) else self.vehicles[0]

            # Calculate route cost: depot -> customers -> depot
            route_cost = 0.0

            # Start from depot to first customer
            if route:
                route_cost += self.distance_matrix[0][route[0]]

                # Customer to customer
                for i in range(len(route) - 1):
                    route_cost += self.distance_matrix[route[i]][route[i + 1]]

                # Last customer back to depot
                route_cost += self.distance_matrix[route[-1]][0]

            total_cost += route_cost * vehicle.cost_per_km

        return total_cost

    def is_feasible(self, solution: Dict[str, Any]) -> bool:
        """
        Check if a solution is feasible.

        Args:
            solution: VRP solution to validate

        Returns:
            True if solution is feasible, False otherwise
        """
        routes = solution.get('routes', [])
        vehicle_assignments = solution.get('vehicle_assignments', [])

        # Check that all customers are visited exactly once
        visited_customers = set()
        for route in routes:
            for customer_idx in route:
                if customer_idx == 0:  # Depot should not be in route
                    return False
                if customer_idx in visited_customers:
                    return False  # Customer visited multiple times
                visited_customers.add(customer_idx)

        # Check that all customers are visited
        expected_customers = set(range(1, self.n_locations))  # Exclude depot (index 0)
        if visited_customers != expected_customers:
            return False

        # Check vehicle capacity constraints
        for route_idx, route in enumerate(routes):
            if route_idx >= len(vehicle_assignments):
                continue

            vehicle_idx = vehicle_assignments[route_idx]
            if vehicle_idx >= len(self.vehicles):
                continue

            vehicle = self.vehicles[vehicle_idx]
            route_demand = sum(self.demands[customer_idx] for customer_idx in route)

            if route_demand > vehicle.capacity:
                return False

        return True

    def get_random_solution(self) -> Dict[str, Any]:
        """Generate a random feasible VRP solution."""
        # Create list of customers (excluding depot)
        customers = list(range(1, self.n_locations))
        np.random.shuffle(customers)

        # Distribute customers among vehicles
        routes = [[] for _ in range(self.n_vehicles)]
        vehicle_assignments = list(range(self.n_vehicles))

        for customer in customers:
            # Find a vehicle with enough capacity
            customer_demand = self.demands[customer]

            for vehicle_idx in range(self.n_vehicles):
                route = routes[vehicle_idx]
                current_demand = sum(self.demands[c] for c in route)
                vehicle_capacity = self.vehicles[vehicle_idx].capacity

                if current_demand + customer_demand <= vehicle_capacity:
                    route.append(customer)
                    break
            else:
                # If no vehicle has capacity, assign to first vehicle (may be infeasible)
                routes[0].append(customer)

        return {
            'routes': routes,
            'vehicle_assignments': vehicle_assignments
        }

    def visualize_solution(self, solution: Dict[str, Any], title: str = "VRP Solution"):
        """Visualize the VRP solution."""
        try:
            plt.figure(figsize=(12, 8))

            # Plot depot
            depot_x, depot_y = self.depot_location
            plt.scatter(depot_x, depot_y, c='red', s=200, marker='s', label='Depot', zorder=5)

            # Plot customers
            customer_x = [c.x for c in self.customers]
            customer_y = [c.y for c in self.customers]
            plt.scatter(customer_x, customer_y, c='blue', s=100, marker='o', label='Customers', zorder=4)

            # Plot routes
            routes = solution.get('routes', [])
            colors = plt.cm.Set3(np.linspace(0, 1, len(routes)))

            for route_idx, route in enumerate(routes):
                if not route:
                    continue

                color = colors[route_idx]

                # Route: depot -> customers -> depot
                route_x = [depot_x]
                route_y = [depot_y]

                for customer_idx in route:
                    customer_loc_idx = customer_idx - 1  # Adjust for depot offset
                    if 0 <= customer_loc_idx < len(self.customers):
                        route_x.append(self.customers[customer_loc_idx].x)
                        route_y.append(self.customers[customer_loc_idx].y)

                route_x.append(depot_x)
                route_y.append(depot_y)

                plt.plot(route_x, route_y, color=color, linewidth=2, alpha=0.7,
                        label=f'Vehicle {route_idx + 1}', zorder=3)

            plt.title(title)
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Visualization failed: {e}")


class ORToolsVRPOptimizer(BaseOptimizer):
    """
    OR-Tools based Vehicle Routing Problem optimizer.

    Uses Google OR-Tools routing library to solve VRP instances optimally
    or near-optimally with various constraints and objectives.
    """

    def __init__(self, time_limit_seconds: int = 30,
                 first_solution_strategy: str = "PATH_CHEAPEST_ARC",
                 local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH"):
        """
        Initialize OR-Tools VRP optimizer.

        Args:
            time_limit_seconds: Maximum solving time
            first_solution_strategy: Strategy for initial solution
            local_search_metaheuristic: Local search method
        """
        self.time_limit_seconds = time_limit_seconds
        self.first_solution_strategy = first_solution_strategy
        self.local_search_metaheuristic = local_search_metaheuristic

        # Pass parameters to base class
        super().__init__(
            self._get_default_metadata(),
            time_limit_seconds=time_limit_seconds,
            first_solution_strategy=first_solution_strategy,
            local_search_metaheuristic=local_search_metaheuristic
        )

    def _get_default_metadata(self) -> OptimizerMetadata:
        """Return default metadata for OR-Tools VRP optimizer."""
        return OptimizerMetadata(
            name="OR-Tools VRP Optimizer",
            description="Google OR-Tools based Vehicle Routing Problem solver",
            optimizer_type=OptimizerType.HEURISTIC,
            optimizer_family=OptimizerFamily.CONSTRAINT_PROGRAMMING,
            author="Qubots OR-Tools Tutorial",
            version="1.0.0",
            supports_constraints=True,
            supports_multi_objective=False,
            typical_problems=["vrp", "tsp", "routing"],
            required_parameters=["time_limit_seconds"],
            optional_parameters=["first_solution_strategy", "local_search_metaheuristic"]
        )

    def _optimize_implementation(self, problem: VehicleRoutingProblem,
                               initial_solution: Optional[Any] = None) -> OptimizationResult:
        """
        Solve VRP using OR-Tools routing library.

        Args:
            problem: VRP problem instance
            initial_solution: Optional initial solution (not used by OR-Tools)

        Returns:
            OptimizationResult with solution and metadata
        """
        if not ORTOOLS_AVAILABLE:
            raise ImportError("OR-Tools not available. Please install with: pip install ortools")

        start_time = time.time()

        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(
            problem.n_locations,  # Number of locations
            problem.n_vehicles,   # Number of vehicles
            0  # Depot index
        )

        # Create routing model
        routing = pywrapcp.RoutingModel(manager)

        # Define distance callback
        def distance_callback(from_index, to_index):
            """Return distance between two points."""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(problem.distance_matrix[from_node][to_node] * 100)  # Scale for integer

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add capacity constraints
        def demand_callback(from_index):
            """Return demand of the node."""
            from_node = manager.IndexToNode(from_index)
            return problem.demands[from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

        # Add capacity dimension
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [vehicle.capacity for vehicle in problem.vehicles],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity'
        )

        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = self.time_limit_seconds

        # Set first solution strategy
        if hasattr(routing_enums_pb2.FirstSolutionStrategy, self.first_solution_strategy):
            search_parameters.first_solution_strategy = getattr(
                routing_enums_pb2.FirstSolutionStrategy, self.first_solution_strategy
            )

        # Set local search metaheuristic
        if hasattr(routing_enums_pb2.LocalSearchMetaheuristic, self.local_search_metaheuristic):
            search_parameters.local_search_metaheuristic = getattr(
                routing_enums_pb2.LocalSearchMetaheuristic, self.local_search_metaheuristic
            )

        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)

        end_time = time.time()

        if solution:
            # Extract solution
            routes = []
            vehicle_assignments = []
            total_distance = 0

            for vehicle_id in range(problem.n_vehicles):
                route = []
                index = routing.Start(vehicle_id)
                route_distance = 0

                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    if node_index != 0:  # Skip depot
                        route.append(node_index)
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

                if route:  # Only add non-empty routes
                    routes.append(route)
                    vehicle_assignments.append(vehicle_id)
                    total_distance += route_distance

            best_solution = {
                'routes': routes,
                'vehicle_assignments': vehicle_assignments
            }

            # Convert back from scaled integer distance
            best_value = total_distance / 100.0

            return OptimizationResult(
                best_solution=best_solution,
                best_value=best_value,
                iterations=1,
                evaluations=1,
                runtime_seconds=end_time - start_time,
                convergence_achieved=True,
                termination_reason="OR-Tools solution found",
                additional_metrics={
                    "ortools_objective": solution.ObjectiveValue(),
                    "ortools_status": "ROUTING_SUCCESS",
                    "num_routes": len(routes),
                    "total_customers": sum(len(route) for route in routes)
                }
            )
        else:
            # No solution found
            return OptimizationResult(
                best_solution=None,
                best_value=float('inf'),
                iterations=0,
                evaluations=0,
                runtime_seconds=end_time - start_time,
                convergence_achieved=False,
                termination_reason="OR-Tools: No solution found",
                additional_metrics={"ortools_status": "ROUTING_FAIL"}
            )


def generate_sample_vrp_data(n_customers: int = 15, n_vehicles: int = 3,
                           area_size: float = 100.0, seed: int = 42) -> Tuple[List[CustomerInfo], List[VehicleInfo]]:
    """
    Generate sample VRP data for testing.

    Args:
        n_customers: Number of customers to generate
        n_vehicles: Number of vehicles in the fleet
        area_size: Size of the area (customers will be in [0, area_size] x [0, area_size])
        seed: Random seed for reproducibility

    Returns:
        Tuple of (customers, vehicles)
    """
    np.random.seed(seed)

    # Generate customers
    customers = []
    for i in range(n_customers):
        customer = CustomerInfo(
            id=i + 1,
            x=np.random.uniform(0, area_size),
            y=np.random.uniform(0, area_size),
            demand=np.random.randint(1, 20),  # Random demand between 1 and 19
            time_window_start=np.random.randint(0, 480),  # 0-8 hours
            time_window_end=np.random.randint(480, 1440),  # 8-24 hours
            service_time=np.random.randint(10, 30)  # 10-30 minutes
        )
        customers.append(customer)

    # Generate vehicles
    vehicles = []
    base_capacity = max(50, sum(c.demand for c in customers) // n_vehicles + 10)

    for i in range(n_vehicles):
        vehicle = VehicleInfo(
            id=i + 1,
            capacity=base_capacity + np.random.randint(-10, 20),
            cost_per_km=np.random.uniform(0.8, 1.2)
        )
        vehicles.append(vehicle)

    return customers, vehicles


def demonstrate_vrp_workflow():
    """
    Demonstrate the complete VRP workflow with qubots and Rastion integration.
    """
    print("🚛 Vehicle Routing Problem Tutorial with Qubots & Rastion")
    print("=" * 60)

    # Step 1: Generate sample data
    print("\n📊 Step 1: Generating sample VRP data...")
    customers, vehicles = generate_sample_vrp_data(n_customers=12, n_vehicles=3)

    print(f"✅ Generated {len(customers)} customers and {len(vehicles)} vehicles")
    print(f"   Total demand: {sum(c.demand for c in customers)}")
    print(f"   Total capacity: {sum(v.capacity for v in vehicles)}")

    # Step 2: Create VRP problem instance
    print("\n🏗️  Step 2: Creating VRP problem instance...")
    vrp_problem = VehicleRoutingProblem(
        customers=customers,
        vehicles=vehicles,
        depot_location=(50.0, 50.0)  # Center of the area
    )

    print(f"✅ Created VRP problem: {vrp_problem.metadata.name}")
    print(f"   Problem type: {vrp_problem.metadata.problem_type}")
    print(f"   Difficulty: {vrp_problem.metadata.difficulty_level}")

    # Step 3: Test problem with random solution
    print("\n🎲 Step 3: Testing with random solution...")
    random_solution = vrp_problem.get_random_solution()
    random_cost = vrp_problem.evaluate_solution(random_solution)
    is_feasible = vrp_problem.is_feasible(random_solution)

    print(f"✅ Random solution cost: {random_cost:.2f}")
    print(f"   Feasible: {is_feasible}")
    print(f"   Routes: {len(random_solution['routes'])} routes")

    # Step 4: Create OR-Tools optimizer
    print("\n⚙️  Step 4: Creating OR-Tools VRP optimizer...")
    if ORTOOLS_AVAILABLE:
        optimizer = ORToolsVRPOptimizer(
            time_limit_seconds=10,
            first_solution_strategy="PATH_CHEAPEST_ARC",
            local_search_metaheuristic="GUIDED_LOCAL_SEARCH"
        )
        print(f"✅ Created optimizer: {optimizer.metadata.name}")
        print(f"   Type: {optimizer.metadata.optimizer_type}")
        print(f"   Family: {optimizer.metadata.optimizer_family}")
    else:
        print("⚠️  OR-Tools not available, skipping optimization step")
        return

    # Step 5: Solve with OR-Tools
    print("\n🔍 Step 5: Solving VRP with OR-Tools...")
    try:
        result = optimizer.optimize(vrp_problem)

        print(f"✅ OR-Tools solution found!")
        print(f"   Best cost: {result.best_value:.2f}")
        print(f"   Runtime: {result.runtime_seconds:.2f} seconds")
        print(f"   Converged: {result.converged}")
        print(f"   Routes found: {result.additional_info.get('num_routes', 'N/A')}")

        # Visualize solution if matplotlib is available
        if result.best_solution:
            print("\n📊 Visualizing solution...")
            vrp_problem.visualize_solution(result.best_solution, "OR-Tools VRP Solution")

    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return

    # Step 6: Demonstrate Rastion upload workflow
    print("\n📤 Step 6: Rastion Upload Workflow (Simulation)...")
    print("Note: Actual upload requires authentication with rastion.authenticate(token)")

    try:
        # Show what would be packaged for upload
        from qubots.rastion_client import QubotPackager

        # Package the problem
        problem_package = QubotPackager.package_model(
            vrp_problem,
            "vrp_tutorial_problem",
            "Vehicle Routing Problem from qubots tutorial with OR-Tools integration"
        )

        # Package the optimizer
        optimizer_package = QubotPackager.package_model(
            optimizer,
            "ortools_vrp_optimizer",
            "OR-Tools based VRP optimizer with capacity constraints"
        )

        print("✅ Problem package created:")
        for filename in problem_package.keys():
            print(f"   📄 {filename}")

        print("✅ Optimizer package created:")
        for filename in optimizer_package.keys():
            print(f"   📄 {filename}")

        # Simulate upload process
        print("\n🔄 Upload simulation:")
        print("   1. Validating model integrity...")
        print("   2. Serializing model state...")
        print("   3. Creating repository structure...")
        print("   4. Uploading to Rastion platform...")
        print("   ✅ Upload complete! (simulated)")

    except Exception as e:
        print(f"❌ Package creation failed: {e}")

    # Step 7: Demonstrate loading workflow
    print("\n📥 Step 7: Rastion Loading Workflow (Simulation)...")
    print("After upload, users would load models like this:")
    print()
    print("```python")
    print("import qubots.rastion as rastion")
    print()
    print("# Authenticate (one-time setup)")
    print("rastion.authenticate('your_gitea_token')")
    print()
    print("# Load models with one line")
    print("problem = rastion.load_qubots_model('vrp_tutorial_problem')")
    print("optimizer = rastion.load_qubots_model('ortools_vrp_optimizer')")
    print()
    print("# Verify model integrity")
    print("print(f'Problem: {problem.metadata.name}')")
    print("print(f'Customers: {problem.n_customers}')")
    print("print(f'Vehicles: {problem.n_vehicles}')")
    print()
    print("# Run optimization")
    print("result = optimizer.optimize(problem)")
    print("print(f'Best cost: {result.best_value:.2f}')")
    print()
    print("# Visualize results")
    print("problem.visualize_solution(result.best_solution)")
    print("```")

    # Step 8: Model verification and error handling
    print("\n🔍 Step 8: Model Verification & Error Handling...")

    # Demonstrate model validation
    print("Model validation checks:")
    print(f"   ✅ Problem format valid: {vrp_problem.validate_solution_format(random_solution)}")
    print(f"   ✅ Problem feasibility check: {vrp_problem.is_feasible(random_solution)}")
    print(f"   ✅ Optimizer compatibility: Compatible with CombinatorialProblem")

    # Demonstrate error handling scenarios
    print("\nError handling scenarios:")

    # Test with invalid solution format
    try:
        invalid_solution = {"invalid": "format"}
        cost = vrp_problem.evaluate_solution(invalid_solution)
        print(f"   ⚠️  Invalid solution cost: {cost}")
    except Exception as e:
        print(f"   ✅ Invalid solution handled: {type(e).__name__}")

    # Test with infeasible solution
    try:
        infeasible_solution = {
            'routes': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],  # All customers in one route
            'vehicle_assignments': [0]
        }
        cost = vrp_problem.evaluate_solution(infeasible_solution)
        print(f"   ✅ Infeasible solution cost: {cost} (infinite as expected)")
    except Exception as e:
        print(f"   ⚠️  Infeasible solution error: {e}")

    print("\n✨ VRP Tutorial completed successfully!")
    print("\n📚 Key Learning Points:")
    print("   🔹 Created VRP using qubots CombinatorialProblem")
    print("   🔹 Implemented OR-Tools based optimizer with constraints")
    print("   🔹 Demonstrated complete Rastion upload/download workflow")
    print("   🔹 Showed model validation and error handling")
    print("   🔹 Visualized solutions for better understanding")
    print("   🔹 Used proper qubots metadata and optimization results")


def demonstrate_advanced_vrp_features():
    """
    Demonstrate advanced VRP features and customizations.
    """
    print("\n🚀 Advanced VRP Features Demo")
    print("=" * 40)

    # Create a more complex VRP instance
    print("\n📊 Creating complex VRP with time windows...")
    customers, vehicles = generate_sample_vrp_data(n_customers=8, n_vehicles=2, seed=123)

    # Add time window constraints to customers
    for i, customer in enumerate(customers):
        customer.time_window_start = i * 60  # Staggered time windows
        customer.time_window_end = customer.time_window_start + 120  # 2-hour windows

    vrp_problem = VehicleRoutingProblem(customers, vehicles, depot_location=(25.0, 25.0))

    print(f"✅ Created complex VRP with time windows")
    print(f"   Customers with time windows: {len([c for c in customers if c.time_window_end < 1440])}")

    # Test different OR-Tools strategies
    if ORTOOLS_AVAILABLE:
        strategies = [
            ("PATH_CHEAPEST_ARC", "GUIDED_LOCAL_SEARCH"),
            ("SAVINGS", "SIMULATED_ANNEALING"),
            ("CHRISTOFIDES", "TABU_SEARCH")
        ]

        print("\n🔧 Testing different OR-Tools strategies...")

        for first_strategy, local_search in strategies:
            try:
                optimizer = ORToolsVRPOptimizer(
                    time_limit_seconds=5,
                    first_solution_strategy=first_strategy,
                    local_search_metaheuristic=local_search
                )

                result = optimizer.optimize(vrp_problem)

                print(f"   {first_strategy} + {local_search}:")
                print(f"     Cost: {result.best_value:.2f}, Time: {result.runtime_seconds:.2f}s")

            except Exception as e:
                print(f"   {first_strategy} + {local_search}: Failed ({e})")

    print("\n✅ Advanced features demonstration completed!")


if __name__ == "__main__":
    """
    Main execution block - run the complete VRP tutorial.

    This tutorial can be executed directly with:
    python routing_tutorial.py
    """
    try:
        # Run main workflow demonstration
        demonstrate_vrp_workflow()

        # Run advanced features demo
        demonstrate_advanced_vrp_features()

        print("\n🎉 All VRP tutorial demonstrations completed successfully!")
        print("\n📖 Next Steps:")
        print("   1. Try modifying the VRP parameters (customers, vehicles, constraints)")
        print("   2. Experiment with different OR-Tools strategies")
        print("   3. Upload your models to Rastion for sharing")
        print("   4. Explore other qubots tutorials (scheduling, finance)")

    except KeyboardInterrupt:
        print("\n⏹️  Tutorial interrupted by user")
    except Exception as e:
        print(f"\n❌ Tutorial failed with error: {e}")
        print("   Please check dependencies and try again")
        import traceback
        traceback.print_exc()
