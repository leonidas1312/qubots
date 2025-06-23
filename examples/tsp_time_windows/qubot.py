"""
Traveling Salesman Problem with Time Windows (TSPTW)

This module implements a TSP variant where each city has time windows during which
it can be visited. The objective is to find the shortest tour that visits all cities
within their respective time windows.

Author: Qubots Community
Version: 1.0.0
"""

import math
import random
import os
from typing import List, Any, Dict, Tuple, Optional
from qubots import CombinatorialProblem, ProblemMetadata, ProblemType, ObjectiveType, DifficultyLevel

# TSP parsing functions (copied from base TSP implementation to avoid import issues)
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


class TSPTimeWindowsProblem(CombinatorialProblem):
    """
    Traveling Salesman Problem with Time Windows (TSPTW)
    
    Extends the classic TSP by adding time window constraints for each city.
    Each city must be visited within its specified time window, and travel
    time between cities affects the feasibility of the tour.
    
    Features:
    - Time windows for each city (earliest and latest visit times)
    - Travel time between cities
    - Service time at each city
    - Penalty for violating time windows
    """
    
    def __init__(self, instance_file: str = "instances/att48.tsp", 
                 time_windows: Optional[List[Tuple[float, float]]] = None,
                 service_times: Optional[List[float]] = None,
                 travel_speed: float = 1.0,
                 time_penalty: float = 1000.0):
        """
        Initialize TSPTW problem.
        
        Args:
            instance_file: Path to TSPLIB format instance file
            time_windows: List of (earliest, latest) time windows for each city
            service_times: Service time required at each city
            travel_speed: Speed factor for travel time calculation
            time_penalty: Penalty for violating time windows
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
        
        # Time window parameters
        self.travel_speed = travel_speed
        self.time_penalty = time_penalty
        
        # Generate default time windows if not provided
        if time_windows is None:
            self.time_windows = self._generate_default_time_windows()
        else:
            if len(time_windows) != self.nb_cities:
                raise ValueError(f"Time windows must be provided for all {self.nb_cities} cities")
            self.time_windows = time_windows
        
        # Generate default service times if not provided
        if service_times is None:
            self.service_times = [10.0] * self.nb_cities  # Default 10 time units service
        else:
            if len(service_times) != self.nb_cities:
                raise ValueError(f"Service times must be provided for all {self.nb_cities} cities")
            self.service_times = service_times
        
        # Create list of cities for CombinatorialProblem
        cities = list(range(self.nb_cities))
        
        # Initialize parent class
        super().__init__(elements=cities)

    def _generate_default_time_windows(self) -> List[Tuple[float, float]]:
        """
        Generate reasonable default time windows based on problem size.
        
        Returns:
            List of (earliest, latest) time windows
        """
        # Estimate total tour time based on average distances
        avg_distance = sum(sum(row) for row in self.dist_matrix) / (self.nb_cities * self.nb_cities)
        estimated_tour_time = avg_distance * self.nb_cities / self.travel_speed
        
        # Create time windows that allow for reasonable scheduling
        time_windows = []
        window_size = estimated_tour_time / self.nb_cities * 2  # Allow some flexibility
        
        for i in range(self.nb_cities):
            if i == 0:  # Depot has full availability
                earliest = 0.0
                latest = estimated_tour_time * 2
            else:
                # Stagger time windows to create interesting constraints
                center_time = (i / self.nb_cities) * estimated_tour_time
                earliest = max(0, center_time - window_size / 2)
                latest = center_time + window_size / 2
            
            time_windows.append((earliest, latest))
        
        return time_windows

    def _get_default_metadata(self) -> ProblemMetadata:
        """Return default metadata for TSPTW."""
        return ProblemMetadata(
            name=f"TSP with Time Windows - {self.instance_name}",
            description=f"Find the shortest tour visiting all {self.nb_cities} cities within their time windows. Instance: {self.instance_name}",
            problem_type=ProblemType.COMBINATORIAL,
            objective_type=ObjectiveType.MINIMIZE,
            difficulty_level=DifficultyLevel.ADVANCED,
            domain="routing",
            tags={"tsptw", "tsp", "time_windows", "routing", "optimization", "combinatorial", "scheduling"},
            author="Qubots Community",
            version="1.0.0",
            dimension=self.nb_cities,
            constraints_count=self.nb_cities,  # One time window constraint per city
            evaluation_complexity="O(n)",
            memory_complexity="O(n²)",
            benchmark_instances=[f"{self.instance_name}_TW"] if hasattr(self, 'instance_name') else []
        )

    def calculate_travel_time(self, from_city: int, to_city: int) -> float:
        """
        Calculate travel time between two cities.
        
        Args:
            from_city: Starting city index
            to_city: Destination city index
            
        Returns:
            Travel time between cities
        """
        distance = self.dist_matrix[from_city][to_city]
        return distance / self.travel_speed

    def evaluate_solution(self, candidate: List[int]) -> float:
        """
        Evaluate a TSPTW tour considering both distance and time window violations.
        
        Args:
            candidate: List of city indices representing the tour
            
        Returns:
            Total cost (distance + time window penalties)
        """
        if not self.is_feasible_format(candidate):
            return float('inf')
        
        total_distance = 0
        total_penalty = 0
        current_time = 0.0
        
        n = len(candidate)
        
        for i in range(n):
            current_city = candidate[i]
            next_city = candidate[(i + 1) % n]
            
            # Add distance cost
            distance = self.dist_matrix[current_city][next_city]
            total_distance += distance
            
            # Calculate arrival time at current city
            if i > 0:  # Skip for first city (depot)
                prev_city = candidate[i - 1]
                travel_time = self.calculate_travel_time(prev_city, current_city)
                current_time += travel_time
            
            # Check time window constraint
            earliest, latest = self.time_windows[current_city]
            
            if current_time < earliest:
                # Wait until window opens
                current_time = earliest
            elif current_time > latest:
                # Penalty for late arrival
                total_penalty += (current_time - latest) * self.time_penalty
            
            # Add service time
            current_time += self.service_times[current_city]
        
        return total_distance + total_penalty

    def is_feasible_format(self, solution: List[int]) -> bool:
        """
        Check if the solution has valid format (same as TSP).
        
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

    def is_time_feasible(self, solution: List[int]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if the solution satisfies time window constraints.
        
        Args:
            solution: List of city indices
            
        Returns:
            Tuple of (feasible, details_dict)
        """
        if not self.is_feasible_format(solution):
            return False, {"error": "Invalid tour format"}
        
        current_time = 0.0
        violations = []
        arrival_times = []
        
        n = len(solution)
        
        for i in range(n):
            current_city = solution[i]
            
            # Calculate arrival time
            if i > 0:
                prev_city = solution[i - 1]
                travel_time = self.calculate_travel_time(prev_city, current_city)
                current_time += travel_time
            
            arrival_times.append(current_time)
            
            # Check time window
            earliest, latest = self.time_windows[current_city]
            
            if current_time < earliest:
                current_time = earliest  # Wait
            elif current_time > latest:
                violations.append({
                    "city": current_city,
                    "arrival_time": current_time,
                    "latest_time": latest,
                    "violation": current_time - latest
                })
            
            # Add service time
            current_time += self.service_times[current_city]
        
        is_feasible = len(violations) == 0
        
        return is_feasible, {
            "feasible": is_feasible,
            "violations": violations,
            "arrival_times": arrival_times,
            "total_time": current_time
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
        Generate a random valid TSPTW tour.

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
            step_size: Not used for TSPTW (discrete moves)

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
        Get detailed information about a TSPTW solution.

        Args:
            solution: TSPTW tour

        Returns:
            Dictionary with solution details including time analysis
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

        # Calculate pure distance (without penalties)
        total_distance = 0
        n = len(solution)
        for i in range(n):
            current_city = solution[i]
            next_city = solution[(i + 1) % n]
            total_distance += self.dist_matrix[current_city][next_city]

        # Get time feasibility analysis
        time_feasible, time_details = self.is_time_feasible(solution)

        return {
            "feasible": self.is_feasible_format(solution),
            "time_feasible": time_feasible,
            "total_cost": total_cost,
            "total_distance": total_distance,
            "time_penalty": total_cost - total_distance,
            "tour_length": len(solution),
            "cities_visited": set(solution),
            "time_violations": len(time_details.get("violations", [])),
            "total_tour_time": time_details.get("total_time", 0),
            "arrival_times": time_details.get("arrival_times", []),
            "violations_detail": time_details.get("violations", [])
        }

    def get_distance_matrix(self) -> List[List[float]]:
        """
        Get the distance matrix for this TSP instance.

        Returns:
            Distance matrix as list of lists
        """
        return self.dist_matrix

    def get_time_windows(self) -> List[Tuple[float, float]]:
        """
        Get the time windows for all cities.

        Returns:
            List of (earliest, latest) time windows
        """
        return self.time_windows

    def get_service_times(self) -> List[float]:
        """
        Get the service times for all cities.

        Returns:
            List of service times
        """
        return self.service_times

    def get_instance_info(self) -> Dict[str, Any]:
        """
        Get information about the TSPTW instance.

        Returns:
            Dictionary with instance details
        """
        return {
            "name": self.instance_name,
            "comment": self.comment,
            "type": "TSPTW",
            "base_instance": self.problem_type,
            "dimension": self.nb_cities,
            "edge_weight_type": self.edge_weight_type,
            "edge_weight_format": self.edge_weight_format,
            "node_coord_type": self.node_coord_type,
            "has_coordinates": self.coords is not None,
            "travel_speed": self.travel_speed,
            "time_penalty": self.time_penalty,
            "time_windows": self.time_windows,
            "service_times": self.service_times
        }
