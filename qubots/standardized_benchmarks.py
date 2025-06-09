"""
Standardized Benchmark Problems for Qubots Leaderboard

Provides a collection of standardized benchmark problems with known optimal solutions
and consistent evaluation criteria for fair comparison across solvers.
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .base_problem import BaseProblem, ProblemMetadata, ProblemType, ObjectiveType, DifficultyLevel


@dataclass
class BenchmarkProblemSpec:
    """Specification for a standardized benchmark problem."""
    name: str
    problem_type: str
    difficulty_level: str
    description: str
    problem_config: Dict[str, Any]
    evaluation_config: Dict[str, Any]
    reference_solution: Optional[str] = None
    reference_value: Optional[float] = None
    time_limit_seconds: int = 300
    memory_limit_mb: int = 1024


class StandardizedTSPProblem(BaseProblem):
    """Standardized Traveling Salesman Problem for benchmarking."""
    
    def __init__(self, instance_name: str = "berlin52", **kwargs):
        """
        Initialize TSP problem with standardized instances.
        
        Args:
            instance_name: Name of the TSP instance (e.g., 'berlin52', 'eil76', 'pr76')
        """
        self.instance_name = instance_name
        self.cities, self.optimal_value = self._load_tsp_instance(instance_name)
        self.n_cities = len(self.cities)
        
        # Calculate distance matrix
        self.distance_matrix = self._calculate_distance_matrix()
        
        super().__init__(
            metadata=ProblemMetadata(
                name=f"TSP-{instance_name}",
                description=f"Traveling Salesman Problem - {instance_name} instance",
                problem_type=ProblemType.COMBINATORIAL,
                objective_type=ObjectiveType.MINIMIZATION,
                difficulty_level=DifficultyLevel.MEDIUM,
                dimension=self.n_cities,
                variable_bounds=[(0, self.n_cities-1)] * self.n_cities,
                known_optimum=self.optimal_value
            )
        )
    
    def evaluate_solution(self, solution: List[int], **kwargs) -> float:
        """Evaluate TSP tour length."""
        if len(solution) != self.n_cities:
            return float('inf')
        
        # Check if solution is a valid permutation
        if set(solution) != set(range(self.n_cities)):
            return float('inf')
        
        total_distance = 0.0
        for i in range(self.n_cities):
            from_city = solution[i]
            to_city = solution[(i + 1) % self.n_cities]
            total_distance += self.distance_matrix[from_city][to_city]
        
        return total_distance
    
    def get_random_solution(self) -> List[int]:
        """Generate a random valid TSP tour."""
        solution = list(range(self.n_cities))
        np.random.shuffle(solution)
        return solution
    
    def is_solution_valid(self, solution: List[int]) -> bool:
        """Check if solution is a valid TSP tour."""
        return (len(solution) == self.n_cities and 
                set(solution) == set(range(self.n_cities)))
    
    def _load_tsp_instance(self, instance_name: str) -> Tuple[List[Tuple[float, float]], float]:
        """Load a standard TSP instance."""
        # Standard TSP instances with known optimal values
        instances = {
            'berlin52': {
                'optimal': 7542,
                'cities': self._generate_berlin52_cities()
            },
            'eil76': {
                'optimal': 538,
                'cities': self._generate_eil76_cities()
            },
            'pr76': {
                'optimal': 108159,
                'cities': self._generate_pr76_cities()
            }
        }
        
        if instance_name not in instances:
            raise ValueError(f"Unknown TSP instance: {instance_name}")
        
        instance = instances[instance_name]
        return instance['cities'], instance['optimal']
    
    def _calculate_distance_matrix(self) -> List[List[float]]:
        """Calculate Euclidean distance matrix between cities."""
        n = len(self.cities)
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = self.cities[i]
                    x2, y2 = self.cities[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    matrix[i][j] = distance
        
        return matrix
    
    def _generate_berlin52_cities(self) -> List[Tuple[float, float]]:
        """Generate Berlin52 TSP instance coordinates."""
        # Simplified version - in practice, would load from TSPLIB
        return [(565.0, 575.0), (25.0, 185.0), (345.0, 750.0), (945.0, 685.0),
                (845.0, 655.0), (880.0, 660.0), (25.0, 230.0), (525.0, 1000.0),
                (580.0, 1175.0), (650.0, 1130.0), (1605.0, 620.0), (1220.0, 580.0),
                (1465.0, 200.0), (1530.0, 5.0), (845.0, 680.0), (725.0, 370.0),
                (145.0, 665.0), (415.0, 635.0), (510.0, 875.0), (560.0, 365.0),
                (300.0, 465.0), (520.0, 585.0), (480.0, 415.0), (835.0, 625.0),
                (975.0, 580.0), (1215.0, 245.0), (1320.0, 315.0), (1250.0, 400.0),
                (660.0, 180.0), (410.0, 250.0), (420.0, 555.0), (575.0, 665.0),
                (1150.0, 1160.0), (700.0, 580.0), (685.0, 595.0), (685.0, 610.0),
                (770.0, 610.0), (795.0, 645.0), (720.0, 635.0), (760.0, 650.0),
                (475.0, 960.0), (95.0, 260.0), (875.0, 920.0), (700.0, 500.0),
                (555.0, 815.0), (830.0, 485.0), (1170.0, 65.0), (830.0, 610.0),
                (605.0, 625.0), (595.0, 360.0), (1340.0, 725.0), (1740.0, 245.0)]
    
    def _generate_eil76_cities(self) -> List[Tuple[float, float]]:
        """Generate EIL76 TSP instance coordinates."""
        # Simplified version - would load from TSPLIB in practice
        return [(22, 22), (36, 26), (21, 45), (45, 35), (55, 20), (33, 34),
                (50, 50), (55, 45), (26, 59), (40, 66), (55, 65), (35, 51),
                (62, 35), (62, 57), (62, 24), (21, 36), (33, 44), (9, 56),
                (62, 48), (66, 14), (44, 13), (26, 13), (11, 28), (7, 43),
                (17, 64), (41, 46), (55, 34), (35, 16), (52, 26), (43, 26),
                (31, 76), (22, 53), (26, 29), (50, 40), (55, 50), (54, 10),
                (60, 15), (47, 66), (30, 60), (30, 50), (12, 17), (15, 14),
                (16, 19), (21, 48), (50, 30), (51, 42), (50, 15), (48, 21),
                (12, 38), (15, 56), (29, 39), (54, 38), (55, 57), (67, 41),
                (10, 70), (6, 25), (65, 27), (40, 60), (70, 64), (64, 4),
                (36, 6), (30, 20), (20, 30), (15, 5), (50, 70), (57, 72),
                (45, 42), (38, 33), (50, 4), (66, 8), (59, 5), (35, 60),
                (27, 24), (40, 20), (40, 37), (40, 40)]
    
    def _generate_pr76_cities(self) -> List[Tuple[float, float]]:
        """Generate PR76 TSP instance coordinates."""
        # Simplified version - would load from TSPLIB in practice
        return [(3600, 2300), (3100, 3300), (4700, 5750), (5400, 5750),
                (5608, 7103), (4493, 7102), (3600, 6950), (3100, 7250),
                (4700, 8450), (5400, 8450), (5610, 10053), (4492, 10052),
                (3600, 10800), (3100, 10950), (4700, 11650), (5400, 11650),
                (6650, 10800), (7300, 10950), (8350, 11650), (8990, 11650),
                (10040, 11650), (10680, 11650), (11730, 11650), (12370, 11650),
                (13420, 11650), (14060, 11650), (15110, 11650), (15750, 11650),
                (16800, 11650), (17440, 11650), (18490, 11650), (19130, 11650),
                (20180, 11650), (20820, 11650), (21870, 11650), (22510, 11650),
                (23560, 11650), (24200, 11650), (25250, 11650), (25890, 11650),
                (26940, 11650), (27580, 11650), (28630, 11650), (29270, 11650),
                (30320, 11650), (30960, 11650), (32010, 11650), (32650, 11650),
                (33700, 11650), (34340, 11650), (35390, 11650), (36030, 11650),
                (37080, 11650), (37720, 11650), (38770, 11650), (39410, 11650),
                (40460, 11650), (41100, 11650), (42150, 11650), (42790, 11650),
                (43840, 11650), (44480, 11650), (45530, 11650), (46170, 11650),
                (47220, 11650), (47860, 11650), (48910, 11650), (49550, 11650),
                (50600, 11650), (51240, 11650), (52290, 11650), (52930, 11650),
                (53980, 11650), (54620, 11650), (55670, 11650), (56310, 11650)]


class StandardizedMaxCutProblem(BaseProblem):
    """Standardized Maximum Cut Problem for benchmarking."""
    
    def __init__(self, graph_type: str = "random", n_vertices: int = 20, density: float = 0.5, seed: int = 42, **kwargs):
        """
        Initialize MaxCut problem with standardized graph instances.
        
        Args:
            graph_type: Type of graph ('random', 'complete', 'cycle')
            n_vertices: Number of vertices
            density: Edge density for random graphs
            seed: Random seed for reproducibility
        """
        self.graph_type = graph_type
        self.n_vertices = n_vertices
        self.density = density
        self.seed = seed
        
        np.random.seed(seed)
        self.adjacency_matrix, self.optimal_value = self._generate_graph()
        
        super().__init__(
            metadata=ProblemMetadata(
                name=f"MaxCut-{graph_type}-{n_vertices}",
                description=f"Maximum Cut Problem - {graph_type} graph with {n_vertices} vertices",
                problem_type=ProblemType.COMBINATORIAL,
                objective_type=ObjectiveType.MAXIMIZATION,
                difficulty_level=DifficultyLevel.MEDIUM,
                dimension=n_vertices,
                variable_bounds=[(0, 1)] * n_vertices,
                known_optimum=self.optimal_value
            )
        )
    
    def evaluate_solution(self, solution: List[int], **kwargs) -> float:
        """Evaluate cut value for a given partition."""
        if len(solution) != self.n_vertices:
            return float('-inf')
        
        # Check if solution is binary
        if not all(x in [0, 1] for x in solution):
            return float('-inf')
        
        cut_value = 0.0
        for i in range(self.n_vertices):
            for j in range(i + 1, self.n_vertices):
                if solution[i] != solution[j]:  # Vertices in different partitions
                    cut_value += self.adjacency_matrix[i][j]
        
        return cut_value
    
    def get_random_solution(self) -> List[int]:
        """Generate a random binary partition."""
        return [np.random.randint(0, 2) for _ in range(self.n_vertices)]
    
    def is_solution_valid(self, solution: List[int]) -> bool:
        """Check if solution is a valid binary partition."""
        return (len(solution) == self.n_vertices and 
                all(x in [0, 1] for x in solution))
    
    def _generate_graph(self) -> Tuple[List[List[float]], Optional[float]]:
        """Generate graph based on type and calculate optimal value if known."""
        if self.graph_type == "random":
            return self._generate_random_graph()
        elif self.graph_type == "complete":
            return self._generate_complete_graph()
        elif self.graph_type == "cycle":
            return self._generate_cycle_graph()
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
    
    def _generate_random_graph(self) -> Tuple[List[List[float]], Optional[float]]:
        """Generate random graph with given density."""
        matrix = [[0.0] * self.n_vertices for _ in range(self.n_vertices)]
        
        for i in range(self.n_vertices):
            for j in range(i + 1, self.n_vertices):
                if np.random.random() < self.density:
                    weight = np.random.uniform(0.1, 1.0)
                    matrix[i][j] = weight
                    matrix[j][i] = weight
        
        # For random graphs, optimal value is unknown
        return matrix, None
    
    def _generate_complete_graph(self) -> Tuple[List[List[float]], Optional[float]]:
        """Generate complete graph with unit weights."""
        matrix = [[1.0 if i != j else 0.0 for j in range(self.n_vertices)] 
                 for i in range(self.n_vertices)]
        
        # For complete graph, optimal cut is n*(n-1)/4 for even n
        if self.n_vertices % 2 == 0:
            optimal = self.n_vertices * (self.n_vertices - 1) // 4
        else:
            optimal = (self.n_vertices - 1) * (self.n_vertices - 1) // 4
        
        return matrix, float(optimal)
    
    def _generate_cycle_graph(self) -> Tuple[List[List[float]], Optional[float]]:
        """Generate cycle graph."""
        matrix = [[0.0] * self.n_vertices for _ in range(self.n_vertices)]
        
        for i in range(self.n_vertices):
            next_vertex = (i + 1) % self.n_vertices
            matrix[i][next_vertex] = 1.0
            matrix[next_vertex][i] = 1.0
        
        # For cycle graph, optimal cut is 2
        return matrix, 2.0


class StandardizedBenchmarkRegistry:
    """Registry for standardized benchmark problems."""
    
    @staticmethod
    def get_benchmark_specs() -> List[BenchmarkProblemSpec]:
        """Get all available benchmark problem specifications."""
        return [
            # TSP Problems
            BenchmarkProblemSpec(
                name="TSP-Berlin52",
                problem_type="tsp",
                difficulty_level="medium",
                description="Berlin52 TSP instance with 52 cities",
                problem_config={"instance_name": "berlin52"},
                evaluation_config={"time_limit": 300, "memory_limit": 1024},
                reference_value=7542.0,
                time_limit_seconds=300
            ),
            BenchmarkProblemSpec(
                name="TSP-EIL76",
                problem_type="tsp", 
                difficulty_level="hard",
                description="EIL76 TSP instance with 76 cities",
                problem_config={"instance_name": "eil76"},
                evaluation_config={"time_limit": 600, "memory_limit": 1024},
                reference_value=538.0,
                time_limit_seconds=600
            ),
            
            # MaxCut Problems
            BenchmarkProblemSpec(
                name="MaxCut-Random-20",
                problem_type="maxcut",
                difficulty_level="easy",
                description="Random MaxCut with 20 vertices, density 0.5",
                problem_config={"graph_type": "random", "n_vertices": 20, "density": 0.5, "seed": 42},
                evaluation_config={"time_limit": 120, "memory_limit": 512},
                time_limit_seconds=120
            ),
            BenchmarkProblemSpec(
                name="MaxCut-Complete-16",
                problem_type="maxcut",
                difficulty_level="medium",
                description="Complete MaxCut with 16 vertices",
                problem_config={"graph_type": "complete", "n_vertices": 16},
                evaluation_config={"time_limit": 180, "memory_limit": 512},
                reference_value=60.0,
                time_limit_seconds=180
            ),
            BenchmarkProblemSpec(
                name="MaxCut-Random-50",
                problem_type="maxcut",
                difficulty_level="hard",
                description="Random MaxCut with 50 vertices, density 0.3",
                problem_config={"graph_type": "random", "n_vertices": 50, "density": 0.3, "seed": 123},
                evaluation_config={"time_limit": 600, "memory_limit": 1024},
                time_limit_seconds=600
            )
        ]
    
    @staticmethod
    def create_problem(spec: BenchmarkProblemSpec) -> BaseProblem:
        """Create a problem instance from specification."""
        if spec.problem_type == "tsp":
            return StandardizedTSPProblem(**spec.problem_config)
        elif spec.problem_type == "maxcut":
            return StandardizedMaxCutProblem(**spec.problem_config)
        else:
            raise ValueError(f"Unknown problem type: {spec.problem_type}")
    
    @staticmethod
    def get_problem_by_name(name: str) -> BaseProblem:
        """Get a problem instance by name."""
        specs = StandardizedBenchmarkRegistry.get_benchmark_specs()
        spec = next((s for s in specs if s.name == name), None)
        
        if not spec:
            raise ValueError(f"Unknown benchmark problem: {name}")
        
        return StandardizedBenchmarkRegistry.create_problem(spec)
