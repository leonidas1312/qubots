"""
HiGHS-based TSP Solver for Qubots Framework

This module implements a Traveling Salesman Problem (TSP) solver using the HiGHS
linear/integer programming solver. It formulates the TSP as an Integer Linear Program
with Miller-Tucker-Zemlin (MTZ) subtour elimination constraints.

Compatible with Rastion platform playground for interactive optimization.
"""

import time
import numpy as np
from typing import List, Any, Dict, Optional, Tuple

try:
    import highspy
except ImportError:
    raise ImportError("HiGHS solver not found. Please install with: pip install highspy")

from qubots import (
    BaseOptimizer, OptimizerMetadata, OptimizerType,
    OptimizerFamily, OptimizationResult, BaseProblem
)


class HiGHSTSPSolver(BaseOptimizer):
    """
    HiGHS-based TSP solver using Integer Linear Programming.
    
    This solver formulates the TSP as an ILP with:
    - Binary variables x_ij indicating if edge (i,j) is in the tour
    - Degree constraints ensuring each city is visited exactly once
    - Miller-Tucker-Zemlin (MTZ) subtour elimination constraints
    """
    
    def __init__(self, 
                 time_limit: float = 60.0,
                 mip_gap: float = 0.01,
                 presolve: bool = True,
                 parallel: bool = True,
                 log_level: int = 1,
                 **kwargs):
        """
        Initialize HiGHS TSP solver.
        
        Args:
            time_limit: Maximum solving time in seconds
            mip_gap: MIP optimality gap tolerance (0.01 = 1%)
            presolve: Enable presolving
            parallel: Enable parallel processing
            log_level: Logging level (0=none, 1=basic, 2=verbose)
            **kwargs: Additional parameters
        """
        super().__init__(
            time_limit=time_limit,
            mip_gap=mip_gap,
            presolve=presolve,
            parallel=parallel,
            log_level=log_level,
            **kwargs
        )
        
        self._time_limit = time_limit
        self._mip_gap = mip_gap
        self._presolve = presolve
        self._parallel = parallel
        self._log_level = log_level
    
    def _get_default_metadata(self) -> OptimizerMetadata:
        """Return default metadata for HiGHS TSP solver."""
        return OptimizerMetadata(
            name="HiGHS TSP Solver",
            description="Integer Linear Programming solver for TSP using HiGHS optimizer with MTZ subtour elimination",
            optimizer_type=OptimizerType.EXACT,
            optimizer_family=OptimizerFamily.LINEAR_PROGRAMMING,
            is_deterministic=True,
            supports_constraints=True,
            supports_multi_objective=False,
            supports_continuous=False,
            supports_discrete=True,
            supports_mixed_integer=True,
            time_complexity="Exponential (exact algorithm)",
            space_complexity="O(n²)",
            convergence_guaranteed=True,
            parallel_capable=True,
            required_parameters=[],
            optional_parameters=["time_limit", "mip_gap", "presolve", "parallel", "log_level"],
            parameter_ranges={
                "time_limit": (1.0, 86400.0),  # 1 second to 24 hours
                "mip_gap": (0.0, 1.0),         # 0% to 100%
                "log_level": (0, 2)            # 0 to 2
            },
            typical_problems=["tsp", "traveling_salesman", "routing"],
            reference_papers=["Miller-Tucker-Zemlin TSP formulation"]
        )
    
    def _optimize_implementation(self, problem: BaseProblem, initial_solution: Optional[Any] = None) -> OptimizationResult:
        _ = initial_solution  # Not used for exact solver
        """
        Core optimization implementation using HiGHS ILP solver.
        
        Args:
            problem: TSP problem instance
            initial_solution: Optional initial solution (not used for exact solver)
            
        Returns:
            OptimizationResult with solution and statistics
        """
        start_time = time.time()
        
        # Validate problem type
        if not hasattr(problem, 'get_distance_matrix'):
            raise ValueError("Problem must be a TSP instance with distance matrix")
        
        # Get problem data
        distance_matrix = problem.get_distance_matrix()
        n_cities = len(distance_matrix)
        
        if n_cities < 3:
            raise ValueError("TSP requires at least 3 cities")
        
        # Log optimization start
        if self._log_callback:
            self._log_callback('info', f'Starting HiGHS TSP optimization for {n_cities} cities', 'highs_solver')
            self._log_callback('info', f'Time limit: {self._time_limit}s, MIP gap: {self._mip_gap}', 'highs_solver')
        
        try:
            # Create HiGHS model
            model = highspy.Highs()
            
            # Configure solver settings
            model.setOptionValue("time_limit", self._time_limit)
            model.setOptionValue("mip_rel_gap", self._mip_gap)
            model.setOptionValue("presolve", "on" if self._presolve else "off")
            model.setOptionValue("parallel", "on" if self._parallel else "off")
            
            # Set log level
            if self._log_level == 0:
                model.setOptionValue("output_flag", False)
            elif self._log_level == 1:
                model.setOptionValue("log_to_console", True)
                model.setOptionValue("log_dev_level", 0)
            else:
                model.setOptionValue("log_to_console", True)
                model.setOptionValue("log_dev_level", 1)
            
            # Build ILP formulation
            solution, objective_value, solve_status = self._build_and_solve_ilp(
                model, distance_matrix, n_cities
            )
            
            runtime = time.time() - start_time
            
            # Determine termination reason and feasibility
            if solve_status == highspy.HighsModelStatus.kOptimal:
                termination_reason = "optimal"
                is_feasible = True
            elif solve_status == highspy.HighsModelStatus.kTimeLimit:
                termination_reason = "time_limit"
                is_feasible = solution is not None
            elif solve_status == highspy.HighsModelStatus.kSolutionLimit:
                termination_reason = "solution_limit"
                is_feasible = solution is not None
            elif solve_status == highspy.HighsModelStatus.kIterationLimit:
                termination_reason = "iteration_limit"
                is_feasible = solution is not None
            elif solve_status == highspy.HighsModelStatus.kInfeasible:
                termination_reason = "infeasible"
                is_feasible = False
                solution = None
                objective_value = float('inf')
            elif solve_status == highspy.HighsModelStatus.kUnbounded:
                termination_reason = "unbounded"
                is_feasible = False
                solution = None
                objective_value = float('inf')
            else:
                termination_reason = f"solver_status_{solve_status}"
                is_feasible = False
                solution = None
                objective_value = float('inf')
            
            # Log results
            if self._log_callback:
                self._log_callback('info', f'HiGHS optimization completed: {termination_reason}', 'highs_solver')
                if is_feasible:
                    self._log_callback('info', f'Best solution value: {objective_value}', 'highs_solver')
                self._log_callback('info', f'Runtime: {runtime:.3f} seconds', 'highs_solver')
            
            return OptimizationResult(
                best_solution=solution,
                best_value=objective_value,
                is_feasible=is_feasible,
                runtime_seconds=runtime,
                iterations=1,  # HiGHS doesn't expose iteration count for MIP
                termination_reason=termination_reason,
                additional_metrics={
                    "solver_status": str(solve_status),
                    "mip_gap_achieved": model.getInfo().mip_gap if hasattr(model.getInfo(), 'mip_gap') else 0.0,
                    "nodes_processed": model.getInfo().mip_node_count if hasattr(model.getInfo(), 'mip_node_count') else 0.0
                }
            )
            
        except Exception as e:
            runtime = time.time() - start_time
            if self._log_callback:
                self._log_callback('error', f'HiGHS optimization failed: {str(e)}', 'highs_solver')
            
            return OptimizationResult(
                best_solution=None,
                best_value=float('inf'),
                is_feasible=False,
                runtime_seconds=runtime,
                iterations=0,
                termination_reason=f"error: {str(e)}",
                additional_metrics={"error": str(e)}
            )
    
    def _build_and_solve_ilp(self, model: highspy.Highs, distance_matrix: List[List[float]], 
                           n_cities: int) -> Tuple[Optional[List[int]], float, int]:
        """
        Build and solve the TSP ILP formulation.
        
        Args:
            model: HiGHS model instance
            distance_matrix: Distance matrix for TSP
            n_cities: Number of cities
            
        Returns:
            Tuple of (solution_tour, objective_value, solve_status)
        """
        # Decision variables: x[i][j] = 1 if edge (i,j) is in tour, 0 otherwise
        # We'll use a flattened representation: x[i*n + j] for edge (i,j)
        
        n_vars = n_cities * n_cities + n_cities  # x_ij variables + u_i variables for MTZ
        
        # Variable bounds: binary for x_ij, continuous for u_i
        col_lower = np.array([0.0] * n_vars)
        col_upper = np.array([1.0] * (n_cities * n_cities) + [float(n_cities)] * n_cities)

        # Objective coefficients
        obj_coeffs = []
        for i in range(n_cities):
            for j in range(n_cities):
                if i == j:
                    obj_coeffs.append(1e6)  # Large penalty for self-loops instead of inf
                else:
                    obj_coeffs.append(distance_matrix[i][j])
        obj_coeffs.extend([0.0] * n_cities)  # u_i variables don't contribute to objective
        obj_coeffs = np.array(obj_coeffs)

        # Add variables to model (HiGHS expects numpy arrays)
        model.addVars(n_vars, col_lower, col_upper)

        # Set objective
        model.changeColsCost(n_vars, list(range(n_vars)), obj_coeffs)

        # Set variable types (binary for x_ij, continuous for u_i)
        for i in range(n_cities * n_cities):
            model.changeColIntegrality(i, highspy.HighsVarType.kInteger)
            model.changeColBounds(i, 0.0, 1.0)  # Binary variables
        
        # Build constraints
        self._add_degree_constraints(model, n_cities)
        self._add_mtz_constraints(model, n_cities)
        
        # Solve the model
        model.run()
        
        # Extract solution
        solution = model.getSolution()
        status = model.getModelStatus()
        
        if solution:
            tour = self._extract_tour_from_solution(solution.col_value, n_cities)
            objective = model.getInfo().objective_function_value
            return tour, objective, status
        else:
            return None, float('inf'), status

    def _add_degree_constraints(self, model: highspy.Highs, n_cities: int):
        """Add degree constraints: each city has exactly one incoming and one outgoing edge."""

        # Out-degree constraints: sum_j x_ij = 1 for all i
        for i in range(n_cities):
            row_indices = []
            row_values = []
            for j in range(n_cities):
                if i != j:  # No self-loops
                    row_indices.append(i * n_cities + j)
                    row_values.append(1.0)

            model.addRow(1.0, 1.0, len(row_indices), np.array(row_indices), np.array(row_values))

        # In-degree constraints: sum_i x_ij = 1 for all j
        for j in range(n_cities):
            row_indices = []
            row_values = []
            for i in range(n_cities):
                if i != j:  # No self-loops
                    row_indices.append(i * n_cities + j)
                    row_values.append(1.0)

            model.addRow(1.0, 1.0, len(row_indices), np.array(row_indices), np.array(row_values))

    def _add_mtz_constraints(self, model: highspy.Highs, n_cities: int):
        """Add Miller-Tucker-Zemlin subtour elimination constraints."""

        # MTZ constraints: u_i - u_j + n * x_ij <= n - 1 for all i,j != 0, i != j
        # This prevents subtours by ensuring a consistent ordering

        x_offset = n_cities * n_cities  # Offset for u variables

        for i in range(1, n_cities):  # Start from 1 (city 0 is depot)
            for j in range(1, n_cities):
                if i != j:
                    row_indices = np.array([x_offset + i, x_offset + j, i * n_cities + j])
                    row_values = np.array([1.0, -1.0, float(n_cities)])

                    model.addRow(-1e20, float(n_cities - 1),
                               len(row_indices), row_indices, row_values)

        # Bounds for u variables: 1 <= u_i <= n for i != 0
        # This is handled in variable bounds, but we add explicit constraints for clarity
        for i in range(1, n_cities):
            # u_i >= 1
            model.addRow(1.0, 1e20, 1, np.array([x_offset + i]), np.array([1.0]))
            # u_i <= n
            model.addRow(-1e20, float(n_cities), 1, np.array([x_offset + i]), np.array([1.0]))

    def _extract_tour_from_solution(self, solution_values: List[float], n_cities: int) -> List[int]:
        """Extract TSP tour from HiGHS solution."""

        # Build adjacency list from x_ij variables
        edges = []
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j and solution_values[i * n_cities + j] > 0.5:  # Binary variable is 1
                    edges.append((i, j))

        if len(edges) != n_cities:
            raise ValueError(f"Invalid solution: expected {n_cities} edges, got {len(edges)}")

        # Build adjacency dictionary
        next_city = {}
        for i, j in edges:
            next_city[i] = j

        # Construct tour starting from city 0
        tour = []
        current = 0
        for _ in range(n_cities):
            tour.append(current)
            if current not in next_city:
                raise ValueError(f"Invalid solution: no outgoing edge from city {current}")
            current = next_city[current]

        # Verify we returned to start
        if current != 0:
            raise ValueError("Invalid solution: tour does not return to start")

        return tour

    def supports_problem(self, problem: BaseProblem) -> bool:
        """Check if this optimizer supports the given problem type."""
        from qubots import ProblemType
        return (hasattr(problem, 'get_distance_matrix') and
                hasattr(problem, 'evaluate_solution') and
                getattr(problem.metadata, 'problem_type', None) == ProblemType.COMBINATORIAL)

    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about the algorithm."""
        return {
            "algorithm_name": "Integer Linear Programming",
            "solver": "HiGHS",
            "formulation": "Miller-Tucker-Zemlin (MTZ)",
            "complexity": "Exponential (exact algorithm)",
            "guarantees": "Global optimum (if solved to optimality)",
            "problem_size_limit": "Practical limit ~100-200 cities",
            "subtour_elimination": "MTZ constraints",
            "variable_count": "O(n²)",
            "constraint_count": "O(n²)"
        }
