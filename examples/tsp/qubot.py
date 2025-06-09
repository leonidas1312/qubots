import math
import random
import os
from typing import List, Any, Dict, Tuple, Optional
from qubots import CombinatorialProblem, ProblemMetadata, ProblemType, ObjectiveType, DifficultyLevel

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

    # Resolve relative path with respect to this module’s directory.
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

class TSPProblem(CombinatorialProblem):
    """
    Traveling Salesman Problem (TSP)

    This class now supports both TSPLib explicit distance matrices (via EDGE_WEIGHT_SECTION)
    and coordinate-based instances (via NODE_COORD_SECTION). In the latter case, distances
    are computed using various distance formulas including ATT (pseudo-Euclidean).

    The TSP seeks to find the shortest possible route that visits each city exactly once
    and returns to the starting city.
    """
    def __init__(self, instance_file: str = "instances/att48.tsp"):
        # Parse the TSP instance file
        header, section, data_lines = read_header_and_data(instance_file)

        # Set header fields with defaults if missing.
        self.instance_name = header.get("NAME", "Unknown")
        self.comment = header.get("COMMENT", "")
        self.problem_type = header.get("TYPE", "TSP")
        self.nb_cities = int(header.get("DIMENSION"))
        self.edge_weight_type = header.get("EDGE_WEIGHT_TYPE", "EXPLICIT")
        self.edge_weight_format = header.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX")
        self.node_coord_type = header.get("NODE_COORD_TYPE", "NO_COORDS")

        # Tokenize the data section lines.
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

        # Create list of cities for CombinatorialProblem
        cities = list(range(self.nb_cities))

        # Initialize parent class
        super().__init__(elements=cities)

    def _get_default_metadata(self) -> ProblemMetadata:
        """Return default metadata for TSP."""
        return ProblemMetadata(
            name=f"Traveling Salesman Problem - {self.instance_name}",
            description=f"Find the shortest tour visiting all {self.nb_cities} cities exactly once and returning to start. Instance: {self.instance_name}",
            problem_type=ProblemType.COMBINATORIAL,
            objective_type=ObjectiveType.MINIMIZE,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            domain="routing",
            tags={"tsp", "traveling_salesman", "routing", "optimization", "combinatorial", "permutation"},
            author="TSPLIB/Qubots Community",
            version="1.0.0",
            dimension=self.nb_cities,
            constraints_count=1,  # All cities must be visited exactly once
            evaluation_complexity="O(n)",
            memory_complexity="O(n²)",
            benchmark_instances=[self.instance_name] if hasattr(self, 'instance_name') else []
        )

    def evaluate_solution(self, candidate: List[int]) -> float:
        """
        Evaluate a TSP tour by calculating total distance.

        Args:
            candidate: List of city indices representing the tour

        Returns:
            Total tour distance, or infinity if invalid tour
        """
        # Validate tour format
        if not self.is_feasible(candidate):
            return float('inf')

        total_distance = 0
        n = len(candidate)

        # Calculate distance for each edge in the tour
        for i in range(n):
            current_city = candidate[i]
            next_city = candidate[(i + 1) % n]  # Wrap around to start
            total_distance += self.dist_matrix[current_city][next_city]

        return total_distance

    def is_feasible(self, solution: List[int]) -> bool:
        """
        Check if the solution is a valid TSP tour.

        Args:
            solution: List of city indices

        Returns:
            True if valid tour, False otherwise
        """
        if not isinstance(solution, list):
            return False
        if len(solution) != self.nb_cities:
            return False
        if sorted(solution) != list(range(self.nb_cities)):
            return False
        return True

    def random_solution(self) -> List[int]:
        """
        Generate a random valid TSP tour.

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
            step_size: Not used for TSP (discrete moves)

        Returns:
            Neighboring tour with one 2-opt move applied
        """
        if not self.is_feasible(solution):
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
        Get detailed information about a solution.

        Args:
            solution: TSP tour

        Returns:
            Dictionary with solution details
        """
        if not self.is_feasible(solution):
            return {
                "feasible": False,
                "total_distance": float('inf'),
                "tour_length": len(solution),
                "error": "Invalid tour"
            }

        total_distance = self.evaluate_solution(solution)

        # Calculate edge distances
        edge_distances = []
        n = len(solution)
        for i in range(n):
            current_city = solution[i]
            next_city = solution[(i + 1) % n]
            edge_distances.append(self.dist_matrix[current_city][next_city])

        return {
            "feasible": True,
            "total_distance": total_distance,
            "tour_length": len(solution),
            "cities_visited": set(solution),
            "edge_distances": edge_distances,
            "min_edge_distance": min(edge_distances),
            "max_edge_distance": max(edge_distances),
            "avg_edge_distance": sum(edge_distances) / len(edge_distances)
        }

    def get_distance_matrix(self) -> List[List[float]]:
        """
        Get the distance matrix for this TSP instance.

        Returns:
            Distance matrix as list of lists
        """
        return self.dist_matrix

    def get_coordinates(self) -> Optional[List[Tuple[float, float]]]:
        """
        Get city coordinates if available.

        Returns:
            List of (x, y) coordinates or None if not available
        """
        return self.coords

    def get_instance_info(self) -> Dict[str, Any]:
        """
        Get information about the TSP instance.

        Returns:
            Dictionary with instance details
        """
        return {
            "name": self.instance_name,
            "comment": self.comment,
            "type": self.problem_type,
            "dimension": self.nb_cities,
            "edge_weight_type": self.edge_weight_type,
            "edge_weight_format": self.edge_weight_format,
            "node_coord_type": self.node_coord_type,
            "has_coordinates": self.coords is not None
        }