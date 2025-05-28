"""
Maximum Cut Problem (MaxCut) Implementation for Qubots Framework

This module implements the Maximum Cut Problem as a qubots CombinatorialProblem.
The MaxCut problem involves partitioning the vertices of a graph into two sets
such that the number of edges between the two sets is maximized.

Compatible with Rastion platform playground for interactive optimization.

Author: Qubots Community
Version: 1.0.0
"""

import numpy as np
import random
import json
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from qubots import (
    CombinatorialProblem, ProblemMetadata, ProblemType,
    ObjectiveType, DifficultyLevel
)


@dataclass
class Edge:
    """Represents an edge in the graph with weight."""
    u: int  # Source vertex
    v: int  # Target vertex
    weight: float = 1.0  # Edge weight


class MaxCutProblem(CombinatorialProblem):
    """
    Maximum Cut Problem implementation using qubots framework.

    The MaxCut problem finds a partition of graph vertices into two sets S and T
    such that the total weight of edges crossing between S and T is maximized.

    Solution Format:
        Binary list where solution[i] = 0 means vertex i is in set S,
        and solution[i] = 1 means vertex i is in set T.
        Example: [0, 1, 0, 1] means vertices 0,2 in S and vertices 1,3 in T.
    """

    def __init__(self,
                 n_vertices: Optional[int] = None,
                 edges: Optional[List[Edge]] = None,
                 adjacency_matrix: Optional[np.ndarray] = None,
                 graph_type: str = "random",
                 density: float = 0.5,
                 weight_range: Tuple[float, float] = (1.0, 10.0),
                 **kwargs):
        """
        Initialize the Maximum Cut Problem.

        Args:
            n_vertices: Number of vertices in the graph
            edges: List of edges with weights
            adjacency_matrix: Adjacency matrix representation
            graph_type: Type of graph to generate ("random", "complete", "cycle", "grid")
            density: Edge density for random graphs (0.0 to 1.0)
            weight_range: Range for random edge weights (min, max)
            **kwargs: Additional parameters for problem configuration
        """
        # Set default parameters
        if n_vertices is None:
            n_vertices = 10
        
        self.n_vertices = n_vertices
        self.graph_type = graph_type
        self.density = density
        self.weight_range = weight_range

        # Initialize graph representation
        if adjacency_matrix is not None:
            self.adjacency_matrix = adjacency_matrix
            self.n_vertices = adjacency_matrix.shape[0]
            self.edges = self._adjacency_to_edges(adjacency_matrix)
        elif edges is not None:
            self.edges = edges
            self.adjacency_matrix = self._edges_to_adjacency(edges, n_vertices)
        else:
            # Generate default graph
            self.edges, self.adjacency_matrix = self._generate_graph(
                n_vertices, graph_type, density, weight_range
            )

        # Problem parameters from kwargs
        self.penalty_invalid = kwargs.get('penalty_invalid', 0.0)  # No penalty needed for MaxCut
        
        # Initialize with elements (vertices)
        vertices = list(range(self.n_vertices))
        super().__init__(elements=vertices)

    def _get_default_metadata(self) -> 'ProblemMetadata':
        """Return default metadata for MaxCut."""
        return ProblemMetadata(
            name="Maximum Cut Problem",
            description="Graph partitioning optimization to maximize cut weight",
            problem_type=ProblemType.COMBINATORIAL,
            objective_type=ObjectiveType.MAXIMIZE,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            domain="graph_theory",
            tags={"maxcut", "graph", "partitioning", "optimization", "combinatorial"},
            author="Qubots Community",
            version="1.0.0",
            dimension=self.n_vertices,
            constraints_count=0  # MaxCut has no hard constraints
        )

    def _generate_graph(self, n_vertices: int, graph_type: str, density: float, 
                       weight_range: Tuple[float, float]) -> Tuple[List[Edge], np.ndarray]:
        """Generate a graph based on specified parameters."""
        edges = []
        adjacency_matrix = np.zeros((n_vertices, n_vertices))
        
        random.seed(42)  # For reproducible results
        
        if graph_type == "complete":
            # Complete graph - all vertices connected
            for i in range(n_vertices):
                for j in range(i + 1, n_vertices):
                    weight = random.uniform(*weight_range)
                    edges.append(Edge(i, j, weight))
                    adjacency_matrix[i][j] = weight
                    adjacency_matrix[j][i] = weight
                    
        elif graph_type == "cycle":
            # Cycle graph - vertices connected in a ring
            for i in range(n_vertices):
                j = (i + 1) % n_vertices
                weight = random.uniform(*weight_range)
                edges.append(Edge(i, j, weight))
                adjacency_matrix[i][j] = weight
                adjacency_matrix[j][i] = weight
                
        elif graph_type == "grid":
            # Grid graph (approximate square grid)
            side_length = int(np.sqrt(n_vertices))
            for i in range(n_vertices):
                row, col = i // side_length, i % side_length
                # Connect to right neighbor
                if col < side_length - 1:
                    j = i + 1
                    if j < n_vertices:
                        weight = random.uniform(*weight_range)
                        edges.append(Edge(i, j, weight))
                        adjacency_matrix[i][j] = weight
                        adjacency_matrix[j][i] = weight
                # Connect to bottom neighbor
                if row < side_length - 1:
                    j = i + side_length
                    if j < n_vertices:
                        weight = random.uniform(*weight_range)
                        edges.append(Edge(i, j, weight))
                        adjacency_matrix[i][j] = weight
                        adjacency_matrix[j][i] = weight
                        
        else:  # "random" or default
            # Random graph with specified density
            for i in range(n_vertices):
                for j in range(i + 1, n_vertices):
                    if random.random() < density:
                        weight = random.uniform(*weight_range)
                        edges.append(Edge(i, j, weight))
                        adjacency_matrix[i][j] = weight
                        adjacency_matrix[j][i] = weight
        
        return edges, adjacency_matrix

    def _edges_to_adjacency(self, edges: List[Edge], n_vertices: int) -> np.ndarray:
        """Convert edge list to adjacency matrix."""
        adjacency_matrix = np.zeros((n_vertices, n_vertices))
        for edge in edges:
            adjacency_matrix[edge.u][edge.v] = edge.weight
            adjacency_matrix[edge.v][edge.u] = edge.weight
        return adjacency_matrix

    def _adjacency_to_edges(self, adjacency_matrix: np.ndarray) -> List[Edge]:
        """Convert adjacency matrix to edge list."""
        edges = []
        n = adjacency_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency_matrix[i][j] != 0:
                    edges.append(Edge(i, j, adjacency_matrix[i][j]))
        return edges

    def evaluate_solution(self, solution: List[int], verbose: bool = False) -> float:
        """
        Evaluate a MaxCut solution and return the cut weight.

        Args:
            solution: Binary list indicating partition assignment
            verbose: If True, print real-time evaluation details for playground streaming

        Returns:
            Total weight of edges crossing the cut (higher is better)
        """
        if not self.validate_solution_format(solution):
            if verbose:
                print("❌ Invalid solution format")
            return 0.0

        if verbose:
            print(f"🔍 Evaluating MaxCut solution...")
            set_s = [i for i, val in enumerate(solution) if val == 0]
            set_t = [i for i, val in enumerate(solution) if val == 1]
            print(f"   Set S: {set_s} ({len(set_s)} vertices)")
            print(f"   Set T: {set_t} ({len(set_t)} vertices)")

        cut_weight = 0.0
        crossing_edges = 0

        for edge in self.edges:
            # Edge crosses the cut if endpoints are in different sets
            if solution[edge.u] != solution[edge.v]:
                cut_weight += edge.weight
                crossing_edges += 1

        if verbose:
            print(f"   Crossing edges: {crossing_edges}/{len(self.edges)}")
            print(f"✅ Cut weight: {cut_weight:.6f}")

        return cut_weight

    def validate_solution_format(self, solution: Any) -> bool:
        """Validate that solution has correct format."""
        if not isinstance(solution, (list, tuple, np.ndarray)):
            return False
        if len(solution) != self.n_vertices:
            return False
        return all(val in [0, 1] for val in solution)

    def get_random_solution(self) -> List[int]:
        """Generate a random valid solution."""
        return [random.randint(0, 1) for _ in range(self.n_vertices)]

    def get_solution_summary(self, solution: List[int]) -> Dict[str, Any]:
        """Get detailed summary of a solution."""
        if not self.validate_solution_format(solution):
            return {"error": "Invalid solution format"}

        cut_weight = self.evaluate_solution(solution)
        set_s = [i for i, val in enumerate(solution) if val == 0]
        set_t = [i for i, val in enumerate(solution) if val == 1]
        
        # Calculate additional metrics
        crossing_edges = []
        total_weight = 0.0
        
        for edge in self.edges:
            total_weight += edge.weight
            if solution[edge.u] != solution[edge.v]:
                crossing_edges.append((edge.u, edge.v, edge.weight))

        cut_ratio = cut_weight / total_weight if total_weight > 0 else 0.0

        return {
            "cut_weight": cut_weight,
            "total_graph_weight": total_weight,
            "cut_ratio": cut_ratio,
            "crossing_edges_count": len(crossing_edges),
            "total_edges_count": len(self.edges),
            "set_s": set_s,
            "set_t": set_t,
            "set_s_size": len(set_s),
            "set_t_size": len(set_t),
            "balance": abs(len(set_s) - len(set_t)) / self.n_vertices,
            "crossing_edges": crossing_edges[:10],  # Show first 10 for brevity
            "graph_info": {
                "vertices": self.n_vertices,
                "edges": len(self.edges),
                "graph_type": self.graph_type,
                "density": len(self.edges) / (self.n_vertices * (self.n_vertices - 1) / 2) if self.n_vertices > 1 else 0
            }
        }

    def get_complement_solution(self, solution: List[int]) -> List[int]:
        """Get the complement partition (flip all assignments)."""
        return [1 - val for val in solution]

    def get_neighbor_solution(self, solution: List[int], num_flips: int = 1) -> List[int]:
        """Generate a neighbor by flipping a small number of vertex assignments."""
        neighbor = solution.copy()
        vertices_to_flip = random.sample(range(self.n_vertices), 
                                       min(num_flips, self.n_vertices))
        for vertex in vertices_to_flip:
            neighbor[vertex] = 1 - neighbor[vertex]
        return neighbor
