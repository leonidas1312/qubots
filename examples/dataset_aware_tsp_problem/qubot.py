"""
Simplified TSP Problem for Qubots Framework

This module implements a Traveling Salesman Problem (TSP) that takes a dataset
as direct input, following the modular architecture: datasets -> problems -> optimizers -> results.
Supports TSPLIB format and provides clean TSP functionality.

Compatible with Rastion platform workflow automation and local development.
"""

import math
import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from qubots import (
    BaseProblem, ProblemMetadata, ProblemType, ObjectiveType,
    DifficultyLevel, EvaluationResult, Dataset
)


class TSPProblem(BaseProblem):
    """
    Simplified TSP Problem for modular qubots architecture.

    Takes a dataset as direct input and formulates the TSP problem.
    Supports TSPLIB format and provides clean TSP functionality.

    Features:
    - Direct dataset input (string or Dataset object)
    - TSPLIB format parsing
    - Coordinate-based and explicit distance matrix support
    - Clean modular design: datasets -> problems -> optimizers -> results
    """

    def __init__(self,
                 dataset: Union[str, Dataset, None] = None,
                 n_cities: Optional[int] = None,
                 seed: int = 42):
        """
        Initialize TSP problem with dataset input.

        Args:
            dataset: Dataset content (string, Dataset object, or None for random instance)
            n_cities: Number of cities (for generated instances when dataset is None)
            seed: Random seed for generated instances
        """
        # Create metadata
        metadata = ProblemMetadata(
            name="TSP Problem",
            description="Traveling Salesman Problem with dataset input",
            problem_type=ProblemType.COMBINATORIAL,
            objective_type=ObjectiveType.MINIMIZE,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            domain="routing",
            tags={"tsp", "routing", "combinatorial"},
            author="Qubots Framework",
            version="2.0.0"
        )

        # Initialize base problem
        super().__init__(metadata=metadata)

        # Store dataset and parameters
        self.dataset = dataset
        self.n_cities = n_cities or 10
        self.seed = seed

        # TSP-specific attributes
        self.coordinates = []
        self.distance_matrix = None
        self.edge_weight_type = "EUC_2D"
        self.edge_weight_format = "FUNCTION"

        # Initialize problem data
        self._initialize_problem_data()
    
    def _get_default_metadata(self) -> ProblemMetadata:
        """Return default metadata for TSP problem."""
        return ProblemMetadata(
            name="TSP Problem",
            description="Traveling Salesman Problem with dataset input",
            problem_type=ProblemType.COMBINATORIAL,
            objective_type=ObjectiveType.MINIMIZE,
            domain="routing",
            tags={"tsp", "routing", "combinatorial"}
        )

    def _initialize_problem_data(self):
        """Initialize problem data from dataset or generate random instance."""
        try:
            if self.dataset is not None:
                # Parse dataset content
                dataset_content = str(self.dataset)  # Convert Dataset object to string if needed
                self._parse_dataset(dataset_content)
            else:
                # Generate random instance
                self._generate_random_instance()

        except Exception as e:
            print(f"Warning: Failed to initialize problem data: {e}")
            # Fallback to small random instance
            self._generate_random_instance()
    
    def _parse_dataset(self, content: str):
        """Parse TSPLIB format dataset."""
        if not content:
            raise ValueError("No dataset content available")

        lines = content.strip().split('\n')
        
        # Parse header information
        dimension = 0
        coord_section = False
        edge_weight_section = False
        data_lines = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('DIMENSION:') or line.startswith('DIMENSION '):
                dimension = int(line.split(':')[-1].strip())
                self.n_cities = dimension
            elif line.startswith('EDGE_WEIGHT_TYPE:'):
                self.edge_weight_type = line.split(':')[-1].strip()
            elif line.startswith('EDGE_WEIGHT_FORMAT:'):
                self.edge_weight_format = line.split(':')[-1].strip()
            elif line == 'NODE_COORD_SECTION':
                coord_section = True
                continue
            elif line == 'EDGE_WEIGHT_SECTION':
                edge_weight_section = True
                continue
            elif line == 'EOF':
                break
            elif coord_section or edge_weight_section:
                if line:
                    data_lines.append(line)
        
        # Parse coordinates or distance matrix
        if coord_section:
            self._parse_coordinates(data_lines)
        elif edge_weight_section:
            self._parse_distance_matrix(data_lines)
        else:
            raise ValueError("No coordinate or distance matrix section found")
    
    def _parse_coordinates(self, data_lines: List[str]):
        """Parse city coordinates from TSPLIB format."""
        self.coordinates = []
        
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 3:
                # Skip city_id (parts[0]) as we only need coordinates
                x = float(parts[1])
                y = float(parts[2])
                self.coordinates.append((x, y))
        
        if len(self.coordinates) != self.n_cities:
            raise ValueError(f"Expected {self.n_cities} coordinates, got {len(self.coordinates)}")
        
        # Calculate distance matrix from coordinates
        self._calculate_distance_matrix_from_coords()
    
    def _parse_distance_matrix(self, data_lines: List[str]):
        """Parse explicit distance matrix from TSPLIB format."""
        # Flatten all numeric tokens
        tokens = []
        for line in data_lines:
            tokens.extend(line.split())
        
        # Remove non-numeric tokens
        numeric_tokens = []
        for token in tokens:
            try:
                numeric_tokens.append(float(token))
            except ValueError:
                continue
        
        # Build distance matrix based on format
        self.distance_matrix = [[0.0] * self.n_cities for _ in range(self.n_cities)]
        
        if self.edge_weight_format == "FULL_MATRIX":
            idx = 0
            for i in range(self.n_cities):
                for j in range(self.n_cities):
                    if idx < len(numeric_tokens):
                        self.distance_matrix[i][j] = numeric_tokens[idx]
                        idx += 1
        elif self.edge_weight_format == "UPPER_ROW":
            idx = 0
            for i in range(self.n_cities):
                for j in range(i + 1, self.n_cities):
                    if idx < len(numeric_tokens):
                        dist = numeric_tokens[idx]
                        self.distance_matrix[i][j] = dist
                        self.distance_matrix[j][i] = dist
                        idx += 1
        elif self.edge_weight_format == "LOWER_DIAG_ROW":
            idx = 0
            for i in range(self.n_cities):
                for j in range(i + 1):
                    if idx < len(numeric_tokens):
                        dist = numeric_tokens[idx]
                        self.distance_matrix[i][j] = dist
                        self.distance_matrix[j][i] = dist
                        idx += 1
    
    def _calculate_distance_matrix_from_coords(self):
        """Calculate distance matrix from coordinates using specified distance type."""
        n = len(self.coordinates)
        self.distance_matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = self._calculate_distance(self.coordinates[i], self.coordinates[j])
                    self.distance_matrix[i][j] = dist
    
    def _calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates based on edge weight type."""
        x1, y1 = coord1
        x2, y2 = coord2
        
        if self.edge_weight_type == "EUC_2D":
            # Euclidean distance
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        elif self.edge_weight_type == "MAN_2D":
            # Manhattan distance
            return abs(x1 - x2) + abs(y1 - y2)
        elif self.edge_weight_type == "MAX_2D":
            # Maximum distance
            return max(abs(x1 - x2), abs(y1 - y2))
        elif self.edge_weight_type == "ATT":
            # Pseudo-Euclidean distance (ATT)
            rij = math.sqrt(((x1 - x2) ** 2 + (y1 - y2) ** 2) / 10.0)
            tij = round(rij)
            return tij + 1 if tij < rij else tij
        elif self.edge_weight_type == "CEIL_2D":
            # Ceiling of Euclidean distance
            return math.ceil(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        else:
            # Default to Euclidean
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    def _generate_random_instance(self):
        """Generate random TSP instance when no dataset is provided."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Generate random coordinates
        self.coordinates = []
        for i in range(self.n_cities):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            self.coordinates.append((x, y))
        
        # Calculate distance matrix
        self._calculate_distance_matrix_from_coords()
    
    def evaluate_solution(self, solution: List[int]) -> Union[float, EvaluationResult]:
        """
        Evaluate a TSP tour solution.
        
        Args:
            solution: List of city indices representing the tour
            
        Returns:
            Tour distance or detailed evaluation result
        """
        if not self.is_valid_solution(solution):
            return EvaluationResult(
                objective_value=float('inf'),
                is_feasible=False,
                constraint_violations=["Invalid tour: not a valid permutation"]
            )
        
        # Calculate tour distance
        total_distance = 0.0
        n = len(solution)
        
        for i in range(n):
            current_city = solution[i]
            next_city = solution[(i + 1) % n]
            total_distance += self.distance_matrix[current_city][next_city]
        
        return EvaluationResult(
            objective_value=total_distance,
            is_feasible=True,
            additional_metrics={
                "tour_length": n,
                "avg_edge_length": total_distance / n
            }
        )
    
    def is_valid_solution(self, solution: List[int]) -> bool:
        """Check if solution is a valid TSP tour."""
        if not solution or len(solution) != self.n_cities:
            return False
        
        # Check if it's a valid permutation
        return sorted(solution) == list(range(self.n_cities))
    
    def random_solution(self) -> List[int]:
        """Generate a random valid TSP tour."""
        tour = list(range(self.n_cities))
        random.shuffle(tour)
        return tour
    
    def get_neighbor_solution(self, solution: List[int]) -> List[int]:
        """Generate neighbor solution using 2-opt move."""
        if len(solution) < 4:
            return solution.copy()
        
        neighbor = solution.copy()
        
        # Random 2-opt move
        i = random.randint(0, len(solution) - 1)
        j = random.randint(0, len(solution) - 1)
        
        if i > j:
            i, j = j, i
        
        if j - i > 1:
            # Reverse the segment between i and j
            neighbor[i:j+1] = reversed(neighbor[i:j+1])
        
        return neighbor
    
    def get_problem_info(self) -> Dict[str, Any]:
        """Get comprehensive problem information."""
        info = {
            "n_cities": self.n_cities,
            "edge_weight_type": self.edge_weight_type,
            "edge_weight_format": self.edge_weight_format,
            "has_coordinates": bool(self.coordinates),
            "has_distance_matrix": bool(self.distance_matrix),
            "has_dataset": self.dataset is not None
        }

        return info

