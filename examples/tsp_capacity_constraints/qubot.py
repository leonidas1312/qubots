"""
Traveling Salesman Problem with Capacity Constraints (TSP-CC)

This module implements a TSP variant where the vehicle has limited capacity
and each city has a demand/pickup requirement. The vehicle must return to
the depot when capacity is reached, creating a multi-trip TSP variant.

Author: Qubots Community
Version: 1.0.0
"""

import random
import os
from typing import List, Any, Dict, Tuple, Optional
from qubots import CombinatorialProblem, ProblemMetadata, ProblemType, ObjectiveType, DifficultyLevel

# TSP parsing functions (copied from base TSP implementation to avoid import issues)
import math

def parse_explicit_matrix(tokens, nb_nodes, weight_format="FULL_MATRIX"):
    # Remove any non-numeric tokens such as "EOF"
    tokens = [token for token in tokens if token.upper() != "EOF"]
    weight_format = weight_format.upper().replace("_", " ").strip()
    matrix = [[0] * nb_nodes for _ in range(nb_nodes)]

    if weight_format == "FUNCTION":
        raise NotImplementedError("FUNCTION weight type is not implemented")

    elif weight_format in ["FULL MATRIX"]:
        it = iter(tokens)
        for i in range(nb_nodes):
            for j in range(nb_nodes):
                matrix[i][j] = int(next(it))

    elif weight_format == "UPPER ROW":
        # Upper triangular (row-wise) without diagonal: tokens for i < j.
        expected_tokens = (nb_nodes * (nb_nodes - 1)) // 2
        if len(tokens) < expected_tokens:
            raise ValueError(f"Not enough tokens for UPPER ROW: expected {expected_tokens}, got {len(tokens)}")
        it = iter(tokens)
        for i in range(nb_nodes - 1):
            for j in range(i + 1, nb_nodes):
                value = int(next(it))
                matrix[i][j] = value
                matrix[j][i] = value

    elif weight_format == "LOWER ROW":
        # Lower triangular (row-wise) without diagonal: tokens for i > j.
        expected_tokens = (nb_nodes * (nb_nodes - 1)) // 2
        if len(tokens) < expected_tokens:
            raise ValueError(f"Not enough tokens for LOWER ROW: expected {expected_tokens}, got {len(tokens)}")
        it = iter(tokens)
        for i in range(1, nb_nodes):
            for j in range(i):
                value = int(next(it))
                matrix[i][j] = value
                matrix[j][i] = value

    elif weight_format == "UPPER DIAG ROW":
        # Upper triangular (row-wise) including diagonal: tokens for i <= j.
        expected_tokens = (nb_nodes * (nb_nodes + 1)) // 2
        if len(tokens) < expected_tokens:
            raise ValueError(f"Not enough tokens for UPPER DIAG ROW: expected {expected_tokens}, got {len(tokens)}")
        it = iter(tokens)
        for i in range(nb_nodes):
            for j in range(i, nb_nodes):
                value = int(next(it))
                matrix[i][j] = value
                if i != j:
                    matrix[j][i] = value

    elif weight_format == "LOWER DIAG ROW":
        # Lower triangular (row-wise) including diagonal: tokens for i >= j.
        expected_tokens = (nb_nodes * (nb_nodes + 1)) // 2
        if len(tokens) < expected_tokens:
            raise ValueError(f"Not enough tokens for LOWER DIAG ROW: expected {expected_tokens}, got {len(tokens)}")
        it = iter(tokens)
        for i in range(nb_nodes):
            for j in range(i + 1):  # j = 0 to i
                value = int(next(it))
                matrix[i][j] = value
                if i != j:
                    matrix[j][i] = value

    elif weight_format == "UPPER COL":
        # Upper triangular (column-wise) without diagonal:
        expected_tokens = (nb_nodes * (nb_nodes - 1)) // 2
        if len(tokens) < expected_tokens:
            raise ValueError(f"Not enough tokens for UPPER COL: expected {expected_tokens}, got {len(tokens)}")
        it = iter(tokens)
        # For each column j from 1 to n-1, rows i=0..(j-1)
        for j in range(1, nb_nodes):
            for i in range(j):
                value = int(next(it))
                matrix[i][j] = value
                matrix[j][i] = value

    elif weight_format == "LOWER COL":
        # Lower triangular (column-wise) without diagonal:
        expected_tokens = (nb_nodes * (nb_nodes - 1)) // 2
        if len(tokens) < expected_tokens:
            raise ValueError(f"Not enough tokens for LOWER COL: expected {expected_tokens}, got {len(tokens)}")
        it = iter(tokens)
        # For each column j from 0 to n-2, rows i=j+1..(n-1)
        for j in range(nb_nodes - 1):
            for i in range(j + 1, nb_nodes):
                value = int(next(it))
                matrix[i][j] = value
                matrix[j][i] = value

    elif weight_format == "UPPER DIAG COL":
        # Upper triangular (column-wise) including diagonal:
        expected_tokens = (nb_nodes * (nb_nodes + 1)) // 2
        if len(tokens) < expected_tokens:
            raise ValueError(f"Not enough tokens for UPPER DIAG COL: expected {expected_tokens}, got {len(tokens)}")
        it = iter(tokens)
        # For each column j from 0 to n-1, rows i=0..j
        for j in range(nb_nodes):
            for i in range(j + 1):
                value = int(next(it))
                matrix[i][j] = value
                if i != j:
                    matrix[j][i] = value

    elif weight_format == "LOWER DIAG COL":
        # Lower triangular (column-wise) including diagonal:
        expected_tokens = (nb_nodes * (nb_nodes + 1)) // 2
        if len(tokens) < expected_tokens:
            raise ValueError(f"Not enough tokens for LOWER DIAG COL: expected {expected_tokens}, got {len(tokens)}")
        it = iter(tokens)
        # For each column j from 0 to n-1, rows i from j to n-1
        for j in range(nb_nodes):
            for i in range(j, nb_nodes):
                value = int(next(it))
                matrix[i][j] = value
                if i != j:
                    matrix[j][i] = value
    else:
        raise ValueError("Unsupported EDGE_WEIGHT_FORMAT: " + weight_format)

    return matrix


def parse_coordinates(tokens, nb_cities):
    """
    Parses coordinates from a NODE_COORD_SECTION.
    Each line is expected to have: <node_index> <x> <y>
    """
    coords = []
    it = iter(tokens)
    for _ in range(nb_cities):
        next(it)  # Skip the node index
        x = float(next(it))
        y = float(next(it))
        coords.append((x, y))
    return coords

def compute_distance_matrix(coords, edge_weight_type, node_coord_type):
    """
    Builds the full distance matrix by applying the correct distance function
    for each pair of nodes.
    """
    nb = len(coords)
    matrix = [[0] * nb for _ in range(nb)]
    for i in range(nb):
        for j in range(nb):
            if i == j:
                matrix[i][j] = 0
            else:
                matrix[i][j] = calc_distance(coords[i], coords[j], edge_weight_type, node_coord_type)
    return matrix

def calc_distance(c1, c2, edge_weight_type, node_coord_type):
    """
    Computes the distance between two nodes c1 and c2 based on the EDGE_WEIGHT_TYPE.
    The node_coord_type can be used to decide between 2D and 3D calculations.
    """
    etype = edge_weight_type.upper()

    if etype in ["EUC_2D", "EUC 2D"]:
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        return int(round(math.sqrt(dx * dx + dy * dy)))

    elif etype in ["EUC_3D", "EUC 3D"]:
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dz = c1[2] - c2[2] if len(c1) > 2 and len(c2) > 2 else 0
        return int(round(math.sqrt(dx * dx + dy * dy + dz * dz)))

    elif etype in ["MAN_2D", "MAN 2D"]:
        dx = abs(c1[0] - c2[0])
        dy = abs(c1[1] - c2[1])
        return int(round(dx + dy))

    elif etype in ["MAN_3D", "MAN 3D"]:
        dx = abs(c1[0] - c2[0])
        dy = abs(c1[1] - c2[1])
        dz = abs(c1[2] - c2[2]) if len(c1) > 2 and len(c2) > 2 else 0
        return int(round(dx + dy + dz))

    elif etype in ["MAX_2D", "MAX 2D"]:
        dx = abs(c1[0] - c2[0])
        dy = abs(c1[1] - c2[1])
        return max(int(round(dx)), int(round(dy)))

    elif etype in ["MAX_3D", "MAX 3D"]:
        dx = abs(c1[0] - c2[0])
        dy = abs(c1[1] - c2[1])
        dz = abs(c1[2] - c2[2]) if len(c1) > 2 and len(c2) > 2 else 0
        return max(int(round(dx)), int(round(dy)), int(round(dz)))

    elif etype == "GEO":
        # For GEO, the coordinates are in DDD.MM format.
        def to_radians(coord):
            deg = int(coord)
            minutes = coord - deg
            return math.pi * (deg + 5.0 * minutes / 3.0) / 180.0

        lat1 = to_radians(c1[0])
        lon1 = to_radians(c1[1])
        lat2 = to_radians(c2[0])
        lon2 = to_radians(c2[1])
        RRR = 6378.388
        q1 = math.cos(lon1 - lon2)
        q2 = math.cos(lat1 - lat2)
        q3 = math.cos(lat1 + lat2)
        return int(RRR * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1)

    elif etype == "ATT":
        # Pseudo-Euclidean (ATT) distance.
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        rij = math.sqrt((dx * dx + dy * dy) / 10.0)
        tij = int(round(rij))
        return tij + 1 if tij < rij else tij

    elif etype in ["CEIL_2D", "CEIL 2D"]:
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        return int(math.ceil(math.sqrt(dx * dx + dy * dy)))

    elif etype in ["XRAY1", "XRAY2"]:
        # Special distance functions for crystallography problems.
        # A proper implementation would mimic the original subroutine.
        # Here we provide a placeholder (e.g. scaling Euclidean distance).
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        d = math.sqrt(dx * dx + dy * dy)
        return int(round(d * 100))

    else:
        raise ValueError("Unsupported EDGE_WEIGHT_TYPE: " + edge_weight_type)


def read_header_and_data(instance_file):
    """
    Reads the TSPLIB file line-by-line, extracting header key/value pairs until
    a known data section is encountered (e.g. NODE_COORD_SECTION or EDGE_WEIGHT_SECTION).
    Returns a tuple (header, section, data_lines) where:
    - header is a dict mapping header keys (uppercase) to values,
    - section is the name of the data section encountered, and
    - data_lines is a list of strings containing the rest of the file.
    """
    header = {}
    data_lines = []
    section = None

    # Resolve relative path with respect to this module's directory.
    if not os.path.isabs(instance_file):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        instance_file = os.path.join(base_dir, instance_file)

    with open(instance_file) as f:
        for line in f:
            stripped = line.strip()
            # Skip empty lines.
            if not stripped:
                continue

            upper_line = stripped.upper()
            # Check if we have reached a data section.
            if upper_line in ["NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION",
                            "DISPLAY_DATA_SECTION", "DEPOT_SECTION"]:
                section = upper_line
                # The rest of the file belongs to the data section.
                # We add the current line's content if needed or simply break.
                # In many TSPLIB files, the section header itself is not part of the data.
                break

            # Process header line: expect "KEY : VALUE" format.
            if ':' in stripped:
                key, value = stripped.split(":", 1)
                header[key.strip().upper()] = value.strip()

        # Read remaining lines (data part)
        if section:
            for line in f:
                if line.strip():  # ignore empty lines
                    data_lines.append(line.strip())

    return header, section, data_lines


class TSPCapacityProblem(CombinatorialProblem):
    """
    Traveling Salesman Problem with Capacity Constraints (TSP-CC)
    
    Extends the classic TSP by adding vehicle capacity constraints.
    Each city has a demand/pickup requirement, and the vehicle has
    limited capacity. The vehicle must return to depot when capacity
    is reached, creating multiple trips if necessary.
    
    Features:
    - Vehicle capacity constraints
    - Customer demand requirements
    - Multi-trip capability
    - Penalty for capacity violations
    """
    
    def __init__(self, instance_file: str = "../tsp/instances/att48.tsp",
                 vehicle_capacity: float = 100.0,
                 demands: Optional[List[float]] = None,
                 capacity_penalty: float = 1000.0,
                 allow_multi_trip: bool = True):
        """
        Initialize TSP-CC problem.
        
        Args:
            instance_file: Path to TSPLIB format instance file
            vehicle_capacity: Maximum vehicle capacity
            demands: Demand/pickup requirement for each city
            capacity_penalty: Penalty for violating capacity constraints
            allow_multi_trip: Whether to allow multiple trips to depot
        """
        # Parse the TSP instance file
        header, section, data_lines = read_header_and_data(instance_file)
        
        # Set header fields with defaults if missing
        self.instance_name = header.get("NAME", "Unknown")
        self.comment = header.get("COMMENT", "")
        self.problem_type = header.get("TYPE", "TSP")
        self.nb_cities = int(header.get("DIMENSION"))
        self.edge_weight_type = header.get("EDGE_WEIGHT_TYPE", "EXPLICIT")
        self.edge_weight_format = header.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX")
        self.node_coord_type = header.get("NODE_COORD_TYPE", "NO_COORDS")
        
        # Tokenize the data section lines
        tokens = []
        for line in data_lines:
            tokens.extend(line.split())
        
        # Parse distance matrix or coordinates
        if section == "EDGE_WEIGHT_SECTION":
            self.dist_matrix = parse_explicit_matrix(tokens, self.nb_cities, self.edge_weight_format)
            self.coords = None
        elif section == "NODE_COORD_SECTION":
            self.coords = parse_coordinates(tokens, self.nb_cities)
            self.dist_matrix = compute_distance_matrix(self.coords, self.edge_weight_type, self.node_coord_type)
        else:
            raise ValueError("Unsupported or missing data section in instance file.")
        
        # Capacity parameters
        self.vehicle_capacity = vehicle_capacity
        self.capacity_penalty = capacity_penalty
        self.allow_multi_trip = allow_multi_trip
        
        # Generate default demands if not provided
        if demands is None:
            self.demands = self._generate_default_demands()
        else:
            if len(demands) != self.nb_cities:
                raise ValueError(f"Demands must be provided for all {self.nb_cities} cities")
            self.demands = demands
        
        # Depot is typically city 0 with zero demand
        self.depot = 0
        self.demands[self.depot] = 0.0
        
        # Create list of cities for CombinatorialProblem
        cities = list(range(self.nb_cities))
        
        # Initialize parent class
        super().__init__(elements=cities)

    def _generate_default_demands(self) -> List[float]:
        """
        Generate reasonable default demands based on vehicle capacity.
        
        Returns:
            List of demand values for each city
        """
        demands = []
        
        # Generate demands that create interesting capacity constraints
        avg_demand = self.vehicle_capacity / (self.nb_cities * 0.6)  # Allow some flexibility
        
        for i in range(self.nb_cities):
            if i == 0:  # Depot has zero demand
                demand = 0.0
            else:
                # Random demand around average with some variation
                demand = random.uniform(avg_demand * 0.5, avg_demand * 1.5)
                demand = round(demand, 1)
            
            demands.append(demand)
        
        return demands

    def _get_default_metadata(self) -> ProblemMetadata:
        """Return default metadata for TSP-CC."""
        return ProblemMetadata(
            name=f"TSP with Capacity Constraints - {self.instance_name}",
            description=f"Find the shortest tour visiting all {self.nb_cities} cities while respecting vehicle capacity constraints. Instance: {self.instance_name}",
            problem_type=ProblemType.COMBINATORIAL,
            objective_type=ObjectiveType.MINIMIZE,
            difficulty_level=DifficultyLevel.ADVANCED,
            domain="routing",
            tags={"tsp_cc", "tsp", "capacity", "routing", "optimization", "combinatorial", "vrp"},
            author="Qubots Community",
            version="1.0.0",
            dimension=self.nb_cities,
            constraints_count=1,  # Capacity constraint
            evaluation_complexity="O(n)",
            memory_complexity="O(n²)",
            benchmark_instances=[f"{self.instance_name}_CC"] if hasattr(self, 'instance_name') else []
        )

    def calculate_route_load(self, route: List[int]) -> float:
        """
        Calculate total load for a route segment.
        
        Args:
            route: List of city indices
            
        Returns:
            Total demand for the route
        """
        return sum(self.demands[city] for city in route)

    def split_into_trips(self, tour: List[int]) -> List[List[int]]:
        """
        Split a tour into multiple trips based on capacity constraints.
        
        Args:
            tour: Complete tour as list of city indices
            
        Returns:
            List of trips, each trip is a list of cities
        """
        if not self.allow_multi_trip:
            return [tour]
        
        trips = []
        current_trip = [self.depot]  # Start at depot
        current_load = 0.0
        
        for city in tour:
            if city == self.depot:
                continue  # Skip depot in the middle of tour
            
            city_demand = self.demands[city]
            
            # Check if adding this city would exceed capacity
            if current_load + city_demand > self.vehicle_capacity and len(current_trip) > 1:
                # Complete current trip and start new one
                current_trip.append(self.depot)  # Return to depot
                trips.append(current_trip)
                current_trip = [self.depot, city]  # Start new trip
                current_load = city_demand
            else:
                # Add city to current trip
                current_trip.append(city)
                current_load += city_demand
        
        # Complete the last trip
        if len(current_trip) > 1:
            current_trip.append(self.depot)
            trips.append(current_trip)
        
        return trips

    def evaluate_solution(self, candidate: List[int]) -> float:
        """
        Evaluate a TSP-CC tour considering distance and capacity violations.
        
        Args:
            candidate: List of city indices representing the tour
            
        Returns:
            Total cost (distance + capacity penalties)
        """
        if not self.is_feasible_format(candidate):
            return float('inf')
        
        if self.allow_multi_trip:
            # Split into capacity-feasible trips
            trips = self.split_into_trips(candidate)
            total_distance = 0
            
            for trip in trips:
                # Calculate distance for each trip
                for i in range(len(trip) - 1):
                    total_distance += self.dist_matrix[trip[i]][trip[i + 1]]
            
            return total_distance
        else:
            # Single trip with capacity penalties
            total_distance = 0
            total_penalty = 0
            current_load = 0.0
            
            n = len(candidate)
            
            for i in range(n):
                current_city = candidate[i]
                next_city = candidate[(i + 1) % n]
                
                # Add distance
                total_distance += self.dist_matrix[current_city][next_city]
                
                # Add demand and check capacity
                current_load += self.demands[current_city]
                
                if current_load > self.vehicle_capacity:
                    # Penalty for capacity violation
                    violation = current_load - self.vehicle_capacity
                    total_penalty += violation * self.capacity_penalty
            
            return total_distance + total_penalty

    def is_feasible_format(self, solution: List[int]) -> bool:
        """
        Check if the solution has valid format.
        
        Args:
            solution: List of city indices
            
        Returns:
            True if valid format, False otherwise
        """
        if not isinstance(solution, list):
            return False
        if len(solution) != self.nb_cities:
            return False
        if sorted(solution) != list(range(self.nb_cities)):
            return False
        return True

    def is_capacity_feasible(self, solution: List[int]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if the solution satisfies capacity constraints.
        
        Args:
            solution: List of city indices
            
        Returns:
            Tuple of (feasible, details_dict)
        """
        if not self.is_feasible_format(solution):
            return False, {"error": "Invalid tour format"}
        
        if self.allow_multi_trip:
            trips = self.split_into_trips(solution)
            violations = []
            
            for trip_idx, trip in enumerate(trips):
                trip_load = self.calculate_route_load(trip)
                if trip_load > self.vehicle_capacity:
                    violations.append({
                        "trip": trip_idx,
                        "cities": trip,
                        "load": trip_load,
                        "capacity": self.vehicle_capacity,
                        "violation": trip_load - self.vehicle_capacity
                    })
            
            is_feasible = len(violations) == 0
            
            return is_feasible, {
                "feasible": is_feasible,
                "trips": trips,
                "num_trips": len(trips),
                "violations": violations,
                "trip_loads": [self.calculate_route_load(trip) for trip in trips]
            }
        else:
            # Single trip analysis
            total_load = self.calculate_route_load(solution)
            is_feasible = total_load <= self.vehicle_capacity
            
            return is_feasible, {
                "feasible": is_feasible,
                "total_load": total_load,
                "capacity": self.vehicle_capacity,
                "violation": max(0, total_load - self.vehicle_capacity)
            }

    def is_feasible(self, solution: List[int]) -> bool:
        """
        Check if the solution is feasible (valid format only for base compatibility).

        Args:
            solution: List of city indices

        Returns:
            True if valid format, False otherwise
        """
        return self.is_feasible_format(solution)

    def random_solution(self) -> List[int]:
        """
        Generate a random valid TSP-CC tour.

        Returns:
            Random permutation of cities
        """
        tour = list(range(self.nb_cities))
        random.shuffle(tour)
        return tour

    def get_neighbor_solution(self, solution: List[int], step_size: float = 1.0) -> List[int]:
        """
        Generate a neighboring solution using 2-opt local search move.

        Args:
            solution: Current tour
            step_size: Not used for TSP-CC (discrete moves)

        Returns:
            Neighboring tour with one 2-opt move applied
        """
        if not self.is_feasible_format(solution):
            return self.random_solution()

        neighbor = solution.copy()
        n = len(neighbor)

        # Perform a random 2-opt move
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)

        # Reverse the segment between i and j
        neighbor[i:j+1] = reversed(neighbor[i:j+1])

        return neighbor

    def get_solution_info(self, solution: List[int]) -> Dict[str, Any]:
        """
        Get detailed information about a TSP-CC solution.

        Args:
            solution: TSP-CC tour

        Returns:
            Dictionary with solution details including capacity analysis
        """
        if not self.is_feasible_format(solution):
            return {
                "feasible": False,
                "total_cost": float('inf'),
                "tour_length": len(solution),
                "error": "Invalid tour format"
            }

        # Calculate basic tour metrics
        total_cost = self.evaluate_solution(solution)

        # Get capacity feasibility analysis
        capacity_feasible, capacity_details = self.is_capacity_feasible(solution)

        # Calculate pure distance (without penalties)
        if self.allow_multi_trip and capacity_feasible:
            # Use the actual trip distances
            total_distance = total_cost  # No penalties when feasible
        else:
            # Calculate base distance
            total_distance = 0
            n = len(solution)
            for i in range(n):
                current_city = solution[i]
                next_city = solution[(i + 1) % n]
                total_distance += self.dist_matrix[current_city][next_city]

        result = {
            "feasible": self.is_feasible_format(solution),
            "capacity_feasible": capacity_feasible,
            "total_cost": total_cost,
            "total_distance": total_distance,
            "capacity_penalty": total_cost - total_distance if not self.allow_multi_trip else 0,
            "tour_length": len(solution),
            "cities_visited": set(solution),
            "vehicle_capacity": self.vehicle_capacity,
            "allow_multi_trip": self.allow_multi_trip
        }

        # Add capacity-specific details
        result.update(capacity_details)

        return result

    def get_demands(self) -> List[float]:
        """
        Get the demand values for all cities.

        Returns:
            List of demand values
        """
        return self.demands

    def get_vehicle_capacity(self) -> float:
        """
        Get the vehicle capacity.

        Returns:
            Vehicle capacity
        """
        return self.vehicle_capacity

    def get_instance_info(self) -> Dict[str, Any]:
        """
        Get information about the TSP-CC instance.

        Returns:
            Dictionary with instance details
        """
        total_demand = sum(self.demands)
        min_trips_needed = total_demand / self.vehicle_capacity if self.vehicle_capacity > 0 else float('inf')

        return {
            "name": self.instance_name,
            "comment": self.comment,
            "type": "TSP-CC",
            "base_instance": self.problem_type,
            "dimension": self.nb_cities,
            "edge_weight_type": self.edge_weight_type,
            "edge_weight_format": self.edge_weight_format,
            "node_coord_type": self.node_coord_type,
            "has_coordinates": self.coords is not None,
            "vehicle_capacity": self.vehicle_capacity,
            "capacity_penalty": self.capacity_penalty,
            "allow_multi_trip": self.allow_multi_trip,
            "depot": self.depot,
            "demands": self.demands,
            "total_demand": total_demand,
            "min_trips_needed": min_trips_needed,
            "avg_demand": total_demand / self.nb_cities,
            "max_demand": max(self.demands),
            "min_demand": min(self.demands)
        }
