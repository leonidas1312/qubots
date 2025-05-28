"""
OR-Tools MaxCut Optimizer for Qubots Framework

This module implements a Google OR-Tools CP-SAT based solver for the Maximum Cut Problem.
It formulates MaxCut as a constraint satisfaction problem with binary variables and
uses linearization techniques to handle the quadratic objective function.

Compatible with Rastion platform playground for interactive optimization.
No license required - OR-Tools is open source with Apache License 2.0.

Author: Qubots Community
Version: 1.0.0
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from qubots import (
    BaseOptimizer, OptimizerMetadata, OptimizerType,
    OptimizerFamily, OptimizationResult, BaseProblem
)

# OR-Tools imports with error handling
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    cp_model = None


class ORToolsMaxCutOptimizer(BaseOptimizer):
    """
    Google OR-Tools CP-SAT based optimizer for Maximum Cut Problems.

    This optimizer formulates the MaxCut problem using binary variables and linearizes
    the quadratic objective function using auxiliary variables. The CP-SAT solver
    then finds optimal or near-optimal solutions using constraint programming techniques.

    Mathematical Formulation:
        Variables: x_i ∈ {0,1} for each vertex i (partition assignment)
                  y_ij ∈ {0,1} for each edge (i,j) (cut indicator)
        
        Constraints: y_ij >= x_i - x_j  (linearization)
                    y_ij >= x_j - x_i  (linearization)
                    y_ij <= x_i + x_j  (linearization)
                    y_ij <= 2 - x_i - x_j  (linearization)
        
        Objective: maximize Σ w_ij * y_ij for all edges (i,j)

    The y_ij variables equal 1 when vertices i and j are in different partitions
    (cut edge) and 0 when they're in the same partition.
    """

    def __init__(self,
                 time_limit: float = 300.0,
                 num_search_workers: int = 0,
                 max_time_in_seconds: float = 300.0,
                 log_search_progress: bool = False,
                 enumerate_all_solutions: bool = False,
                 use_symmetry: bool = True,
                 use_sat_preprocessing: bool = True,
                 **kwargs):
        """
        Initialize the OR-Tools MaxCut Optimizer.

        Args:
            time_limit: Maximum solving time in seconds
            num_search_workers: Number of parallel workers (0 = automatic)
            max_time_in_seconds: Alternative time limit parameter
            log_search_progress: Enable detailed search progress logging
            enumerate_all_solutions: Find all optimal solutions (if multiple exist)
            use_symmetry: Enable symmetry breaking techniques
            use_sat_preprocessing: Enable SAT preprocessing
            **kwargs: Additional optimizer parameters
        """
        # Check OR-Tools availability
        if not ORTOOLS_AVAILABLE:
            raise ImportError(
                "OR-Tools is not available. Please install OR-Tools Python API. "
                "Run: pip install ortools"
            )

        # Set parameters BEFORE calling super().__init__
        self.time_limit = time_limit
        self.num_search_workers = num_search_workers
        self.max_time_in_seconds = max_time_in_seconds
        self.log_search_progress = log_search_progress
        self.enumerate_all_solutions = enumerate_all_solutions
        self.use_symmetry = use_symmetry
        self.use_sat_preprocessing = use_sat_preprocessing

        # Initialize parent class with all parameters
        super().__init__(
            time_limit=time_limit,
            num_search_workers=num_search_workers,
            max_time_in_seconds=max_time_in_seconds,
            log_search_progress=log_search_progress,
            enumerate_all_solutions=enumerate_all_solutions,
            use_symmetry=use_symmetry,
            use_sat_preprocessing=use_sat_preprocessing,
            **kwargs
        )

        # Runtime statistics
        self.solve_time = 0.0
        self.num_branches = 0
        self.num_conflicts = 0
        self.num_binary_propagations = 0

    def _get_default_metadata(self) -> 'OptimizerMetadata':
        """Return default metadata for OR-Tools MaxCut Optimizer."""
        return OptimizerMetadata(
            name="OR-Tools MaxCut Optimizer",
            description="Constraint programming solver for Maximum Cut using Google OR-Tools CP-SAT with linearization",
            optimizer_type=OptimizerType.EXACT,
            optimizer_family=OptimizerFamily.CONSTRAINT_PROGRAMMING,
            author="Qubots Community",
            version="1.0.0",
            supports_constraints=True,
            supports_continuous=False,
            supports_discrete=True,
            time_complexity="Exponential (NP-Hard)",
            convergence_guaranteed=True,
            parallel_capable=True,
            required_parameters=["time_limit"],
            optional_parameters=["num_search_workers", "log_search_progress", "use_symmetry", "use_sat_preprocessing"],
            parameter_ranges={
                "time_limit": (1.0, 86400.0),  # 1 second to 24 hours
                "num_search_workers": (0, 32),  # 0 = auto, up to 32 workers
                "max_time_in_seconds": (1.0, 86400.0),
            },
            typical_problems=["maxcut", "graph_partitioning", "binary_optimization"],
            benchmark_results={
                "small_maxcut": 1.0,    # Optimal for small instances
                "medium_maxcut": 0.95,  # Near-optimal for medium instances
                "large_maxcut": 0.80    # Good solutions for large instances
            }
        )

    def _optimize_implementation(self, problem: 'BaseProblem', initial_solution=None) -> 'OptimizationResult':
        """
        Main optimization implementation using OR-Tools CP-SAT.

        Args:
            problem: MaxCut problem instance to optimize
            initial_solution: Optional initial solution (used as hint)

        Returns:
            OptimizationResult with optimal or best solution found
        """
        start_time = time.time()

        # Validate problem type
        if not hasattr(problem, 'adjacency_matrix') or not hasattr(problem, 'n_vertices'):
            raise ValueError("Problem must be a MaxCut problem with adjacency_matrix and n_vertices attributes")

        n = problem.n_vertices
        adj_matrix = problem.adjacency_matrix
        edges = problem.edges

        self.log_message('info', f"Setting up OR-Tools CP-SAT model for {n}-vertex MaxCut problem...")
        self.log_message('debug', f"Graph has {len(edges)} edges")

        try:
            # Create CP-SAT model
            model = cp_model.CpModel()

            # Add binary variables for each vertex (partition assignment)
            self.log_message('debug', "Adding vertex variables...")
            x = {}
            for i in range(n):
                x[i] = model.NewBoolVar(f"x_{i}")

            # Add binary variables for each edge (cut indicators)
            self.log_message('debug', "Adding edge cut indicator variables...")
            y = {}
            edge_weights = {}
            for edge in edges:
                i, j = edge.u, edge.v
                if i < j:  # Avoid duplicate edges
                    y[(i, j)] = model.NewBoolVar(f"y_{i}_{j}")
                    edge_weights[(i, j)] = edge.weight

            # Add linearization constraints
            self.log_message('debug', "Adding linearization constraints...")
            self._add_linearization_constraints(model, x, y, edges)

            # Add symmetry breaking constraint (optional)
            if self.use_symmetry and n > 1:
                self.log_message('debug', "Adding symmetry breaking constraint...")
                model.Add(x[0] == 0)  # Fix first vertex to partition 0

            # Build linear objective
            self.log_message('debug', "Building linear objective...")
            objective_terms = []
            for (i, j), weight in edge_weights.items():
                objective_terms.append(int(weight * 1000) * y[(i, j)])  # Scale weights to integers

            if objective_terms:
                model.Maximize(sum(objective_terms))

            # Add initial solution hint if provided
            if initial_solution is not None and self.validate_initial_solution(initial_solution, n):
                self.log_message('debug', "Adding solution hint...")
                for i in range(n):
                    model.AddHint(x[i], initial_solution[i])

            # Configure solver parameters
            solver = cp_model.CpSolver()
            self._configure_solver_parameters(solver)

            # Solve the model
            self.log_message('info', "Starting OR-Tools CP-SAT optimization...")
            self.log_message('info', f"Time limit: {self.time_limit}s, Workers: {self.num_search_workers}")

            status = solver.Solve(model)

            # Extract results
            end_time = time.time()
            self.solve_time = end_time - start_time

            return self._extract_solution(solver, status, x, n, problem, objective_terms)

        except Exception as e:
            self.log_message('error', f"OR-Tools error: {str(e)}")
            # Return best known solution or random solution
            if initial_solution is not None:
                solution = initial_solution
            else:
                solution = problem.get_random_solution()

            return OptimizationResult(
                best_solution=solution,
                best_value=problem.evaluate_solution(solution),
                iterations=0,
                runtime_seconds=time.time() - start_time,
                termination_reason=f"ortools_error: {str(e)}"
            )

    def _add_linearization_constraints(self, model, x, y, edges):
        """Add linearization constraints for the quadratic MaxCut objective."""
        for edge in edges:
            i, j = edge.u, edge.v
            if i < j:  # Avoid duplicate edges
                # y_ij = 1 iff x_i != x_j (vertices in different partitions)
                # Linearization: y_ij >= |x_i - x_j|
                # This is achieved with four constraints:
                model.Add(y[(i, j)] >= x[i] - x[j])      # y_ij >= x_i - x_j
                model.Add(y[(i, j)] >= x[j] - x[i])      # y_ij >= x_j - x_i
                model.Add(y[(i, j)] <= x[i] + x[j])      # y_ij <= x_i + x_j
                model.Add(y[(i, j)] <= 2 - x[i] - x[j])  # y_ij <= 2 - x_i - x_j

    def _configure_solver_parameters(self, solver):
        """Configure OR-Tools CP-SAT solver parameters."""
        # Set time limit
        solver.parameters.max_time_in_seconds = self.time_limit

        # Set number of search workers
        if self.num_search_workers > 0:
            solver.parameters.num_search_workers = self.num_search_workers

        # Enable/disable search progress logging
        solver.parameters.log_search_progress = self.log_search_progress

        # Enable/disable SAT preprocessing
        solver.parameters.use_sat_preprocessing = self.use_sat_preprocessing

        # Additional performance parameters
        solver.parameters.enumerate_all_solutions = self.enumerate_all_solutions

    def _extract_solution(self, solver, status, x, n: int, problem, objective_terms) -> 'OptimizationResult':
        """Extract solution from solved OR-Tools model."""
        status_string = self._get_status_string(status)
        self.log_message('info', f"OR-Tools status: {status_string}")

        # Get solution values
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            solution_values = [solver.Value(x[i]) for i in range(n)]
            binary_solution = [int(val) for val in solution_values]
            
            # Calculate objective value (scaled back)
            if objective_terms:
                objective_value = sum(solver.Value(term) for term in objective_terms) / 1000.0
            else:
                objective_value = 0.0

            # Verify solution with problem evaluation
            verified_value = problem.evaluate_solution(binary_solution)

            self.log_message('info', f"OR-Tools objective: {objective_value:.6f}")
            self.log_message('info', f"Verified value: {verified_value:.6f}")

            # Get additional statistics
            try:
                self.num_branches = solver.NumBranches()
                self.num_conflicts = solver.NumConflicts()
                self.num_binary_propagations = solver.NumBinaryPropagations()
            except:
                pass  # Some statistics might not be available

            termination_reason = "optimal" if status == cp_model.OPTIMAL else "feasible"

        else:
            # No feasible solution found, return random solution
            self.log_message('warning', "No feasible solution found, returning random solution")
            binary_solution = problem.get_random_solution()
            verified_value = problem.evaluate_solution(binary_solution)
            termination_reason = "no_solution"

        # Log final statistics
        self.log_message('info', f"Solve time: {self.solve_time:.3f} seconds")
        if hasattr(self, 'num_branches') and self.num_branches > 0:
            self.log_message('debug', f"Branches: {self.num_branches}")
        if hasattr(self, 'num_conflicts'):
            self.log_message('debug', f"Conflicts: {self.num_conflicts}")

        result = OptimizationResult(
            best_solution=binary_solution,
            best_value=verified_value,
            iterations=getattr(self, 'num_branches', 0),
            runtime_seconds=self.solve_time,
            termination_reason=termination_reason
        )

        # Add OR-Tools specific metrics
        result.additional_metrics = {
            "solver": "OR-Tools CP-SAT",
            "ortools_status": status_string,
            "num_branches": getattr(self, 'num_branches', 0),
            "num_conflicts": getattr(self, 'num_conflicts', 0),
            "num_binary_propagations": getattr(self, 'num_binary_propagations', 0)
        }

        return result

    def _get_status_string(self, status: int) -> str:
        """Convert OR-Tools status code to string."""
        status_map = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.UNKNOWN: "UNKNOWN"
        }
        return status_map.get(status, f"UNKNOWN_STATUS_{status}")

    def validate_initial_solution(self, solution: List[int], n: int) -> bool:
        """Validate initial solution format."""
        if not isinstance(solution, (list, tuple)):
            return False
        if len(solution) != n:
            return False
        return all(val in [0, 1] for val in solution)
