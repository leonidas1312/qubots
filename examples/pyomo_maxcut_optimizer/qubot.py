"""
Pyomo MaxCut Optimizer for Qubots Framework

This module implements a Pyomo-based solver for the Maximum Cut Problem.
It formulates MaxCut as a quadratic binary optimization problem using Pyomo
modeling framework and supports multiple solvers (Gurobi, CPLEX, CBC, etc.).

Compatible with Rastion platform playground for interactive optimization.

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

# Pyomo imports with error handling
try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverStatus, TerminationCondition
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    pyo = None
    SolverStatus = None
    TerminationCondition = None


class PyomoMaxCutOptimizer(BaseOptimizer):
    """
    Pyomo-based optimizer for Maximum Cut Problems.

    This optimizer formulates the MaxCut problem as a quadratic binary optimization
    problem using Pyomo modeling framework. It supports multiple solvers including
    Gurobi, CPLEX, CBC, and other MILP solvers.

    Mathematical Formulation:
        maximize: Σ w_ij * (x_i + x_j - 2*x_i*x_j) for all edges (i,j)
        subject to: x_i ∈ {0,1} for all vertices i

    The quadratic term (x_i + x_j - 2*x_i*x_j) equals 1 when vertices i and j
    are in different sets (cut edge) and 0 when they're in the same set.
    """

    def __init__(self,
                 solver_name: str = "gurobi",
                 time_limit: float = 300.0,
                 mip_gap: float = 0.01,
                 threads: int = 0,
                 solver_options: Optional[Dict[str, Any]] = None,
                 tee: bool = False,
                 **kwargs):
        """
        Initialize the Pyomo MaxCut Optimizer.

        Args:
            solver_name: Name of the solver to use ("gurobi", "cplex", "cbc", "glpk", etc.)
            time_limit: Maximum solving time in seconds
            mip_gap: Relative optimality gap tolerance (0.01 = 1%)
            threads: Number of threads (0 = automatic, solver-dependent)
            solver_options: Additional solver-specific options
            tee: Enable solver output streaming
            **kwargs: Additional optimizer parameters
        """
        # Check Pyomo availability
        if not PYOMO_AVAILABLE:
            raise ImportError(
                "Pyomo is not available. Please install Pyomo: pip install pyomo"
            )

        # Set parameters BEFORE calling super().__init__
        self.solver_name = solver_name
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.threads = threads
        self.solver_options = solver_options or {}
        self.tee = tee

        # Initialize parent class with all parameters
        super().__init__(
            solver_name=solver_name,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            solver_options=solver_options,
            tee=tee,
            **kwargs
        )

        # Runtime statistics
        self.solve_time = 0.0
        self.solver_status = None
        self.termination_condition = None
        self.best_bound = None
        self.gap_closed = 0.0

    def _get_default_metadata(self) -> 'OptimizerMetadata':
        """Return default metadata for Pyomo MaxCut Optimizer."""
        return OptimizerMetadata(
            name="Pyomo MaxCut Optimizer",
            description="Flexible solver for Maximum Cut using Pyomo modeling framework with multiple solver backends",
            optimizer_type=OptimizerType.EXACT,
            optimizer_family=OptimizerFamily.MATHEMATICAL_PROGRAMMING,
            author="Qubots Community",
            version="1.0.0",
            supports_constraints=True,
            supports_continuous=False,
            supports_discrete=True,
            time_complexity="Exponential (NP-Hard)",
            convergence_guaranteed=True,
            parallel_capable=True,
            required_parameters=["solver_name", "time_limit"],
            optional_parameters=["mip_gap", "threads", "solver_options", "tee"],
            parameter_ranges={
                "time_limit": (1.0, 86400.0),  # 1 second to 24 hours
                "mip_gap": (0.0001, 0.5),      # 0.01% to 50%
                "threads": (0, 64),            # 0 = auto, up to 64 threads
            },
            typical_problems=["maxcut", "graph_partitioning", "quadratic_binary"],
            benchmark_results={
                "small_maxcut": 1.0,    # Optimal for small instances
                "medium_maxcut": 0.98,  # Near-optimal for medium instances
                "large_maxcut": 0.85    # Good solutions for large instances
            }
        )

    def _optimize_implementation(self, problem: 'BaseProblem', initial_solution=None) -> 'OptimizationResult':
        """
        Main optimization implementation using Pyomo.

        Args:
            problem: MaxCut problem instance to optimize
            initial_solution: Optional initial solution (used as warm start)

        Returns:
            OptimizationResult with optimal or best solution found
        """
        start_time = time.time()

        # Validate problem type
        if not hasattr(problem, 'adjacency_matrix') or not hasattr(problem, 'n_vertices'):
            raise ValueError("Problem must be a MaxCut problem with adjacency_matrix and n_vertices attributes")

        n = problem.n_vertices
        adj_matrix = problem.adjacency_matrix

        self.log_message('info', f"Setting up Pyomo model for {n}-vertex MaxCut problem...")
        self.log_message('debug', f"Graph has {len(problem.edges)} edges")
        self.log_message('info', f"Using solver: {self.solver_name}")

        try:
            # Create Pyomo model
            model = self._build_pyomo_model(adj_matrix, n)
            
            # Add initial solution if provided
            if initial_solution is not None and self.validate_initial_solution(initial_solution, n):
                self.log_message('debug', "Setting warm start solution...")
                for i in range(n):
                    model.x[i].set_value(initial_solution[i])

            # Create solver
            solver = self._create_solver()

            # Solve the model
            self.log_message('info', "Starting Pyomo optimization...")
            self.log_message('info', f"Time limit: {self.time_limit}s, MIP gap: {self.mip_gap*100:.2f}%")
            
            results = solver.solve(model, tee=self.tee)

            # Extract results
            end_time = time.time()
            self.solve_time = end_time - start_time

            return self._extract_solution(model, results, n, problem)

        except Exception as e:
            self.log_message('error', f"Pyomo error: {str(e)}")
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
                termination_reason=f"pyomo_error: {str(e)}"
            )

    def _build_pyomo_model(self, adj_matrix: np.ndarray, n: int):
        """Build the Pyomo model for MaxCut."""
        model = pyo.ConcreteModel()
        
        # Create binary variables for each vertex
        model.x = pyo.Var(range(n), domain=pyo.Binary)
        
        # Build objective function
        objective_expr = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i][j] != 0:
                    weight = adj_matrix[i][j]
                    # Add linear terms: w_ij * (x_i + x_j)
                    objective_expr += weight * (model.x[i] + model.x[j])
                    # Add quadratic term: -2 * w_ij * x_i * x_j
                    objective_expr -= 2 * weight * model.x[i] * model.x[j]
        
        # Set objective to maximize
        model.objective = pyo.Objective(expr=objective_expr, sense=pyo.maximize)
        
        return model

    def _create_solver(self):
        """Create and configure the Pyomo solver."""
        try:
            solver = pyo.SolverFactory(self.solver_name)
            
            if solver is None or not solver.available():
                # Try fallback solvers
                fallback_solvers = ["cbc", "glpk", "ipopt"]
                for fallback in fallback_solvers:
                    if fallback != self.solver_name:
                        self.log_message('warning', f"Solver {self.solver_name} not available, trying {fallback}")
                        solver = pyo.SolverFactory(fallback)
                        if solver is not None and solver.available():
                            self.solver_name = fallback
                            break
                
                if solver is None or not solver.available():
                    raise RuntimeError(f"No suitable solver found. Tried: {self.solver_name}, {fallback_solvers}")
            
            # Set common solver options
            solver_options = self.solver_options.copy()
            
            # Set time limit (solver-specific)
            if self.solver_name.lower() in ["gurobi"]:
                solver_options["TimeLimit"] = self.time_limit
                solver_options["MIPGap"] = self.mip_gap
                if self.threads > 0:
                    solver_options["Threads"] = self.threads
            elif self.solver_name.lower() in ["cplex"]:
                solver_options["timelimit"] = self.time_limit
                solver_options["mip.tolerances.mipgap"] = self.mip_gap
                if self.threads > 0:
                    solver_options["threads"] = self.threads
            elif self.solver_name.lower() in ["cbc"]:
                solver_options["seconds"] = self.time_limit
                solver_options["ratio"] = self.mip_gap
                if self.threads > 0:
                    solver_options["threads"] = self.threads
            
            # Apply solver options
            for option, value in solver_options.items():
                solver.options[option] = value
            
            return solver
            
        except Exception as e:
            raise RuntimeError(f"Failed to create solver {self.solver_name}: {str(e)}")

    def _extract_solution(self, model, results, n: int, problem) -> 'OptimizationResult':
        """Extract solution from solved Pyomo model."""
        self.solver_status = results.solver.status
        self.termination_condition = results.solver.termination_condition
        
        status_string = str(self.solver_status)
        termination_string = str(self.termination_condition)
        
        self.log_message('info', f"Solver status: {status_string}")
        self.log_message('info', f"Termination condition: {termination_string}")
        
        # Get solution values
        if (self.termination_condition == TerminationCondition.optimal or 
            self.termination_condition == TerminationCondition.feasible):
            
            solution_values = [pyo.value(model.x[i]) for i in range(n)]
            # Round to binary values
            binary_solution = [int(round(val)) for val in solution_values]
            
            try:
                objective_value = pyo.value(model.objective)
            except:
                objective_value = None
            
            # Verify solution with problem evaluation
            verified_value = problem.evaluate_solution(binary_solution)
            
            if objective_value is not None:
                self.log_message('info', f"Pyomo objective: {objective_value:.6f}")
            self.log_message('info', f"Verified value: {verified_value:.6f}")
            
            # Get additional statistics if available
            try:
                if hasattr(results.problem, 'upper_bound'):
                    self.best_bound = results.problem.upper_bound
                if hasattr(results.problem, 'lower_bound') and self.best_bound is not None:
                    lower_bound = results.problem.lower_bound
                    if lower_bound is not None and self.best_bound is not None:
                        self.gap_closed = abs(self.best_bound - lower_bound) / max(abs(self.best_bound), 1e-10)
            except:
                pass  # Some statistics might not be available
            
            termination_reason = "optimal" if self.termination_condition == TerminationCondition.optimal else "feasible"
            
        else:
            # No feasible solution found, return random solution
            self.log_message('warning', "No feasible solution found, returning random solution")
            binary_solution = problem.get_random_solution()
            verified_value = problem.evaluate_solution(binary_solution)
            termination_reason = "no_solution"
        
        # Log final statistics
        self.log_message('info', f"Solve time: {self.solve_time:.3f} seconds")
        if hasattr(self, 'gap_closed') and self.gap_closed is not None:
            self.log_message('debug', f"Optimality gap: {self.gap_closed*100:.4f}%")

        result = OptimizationResult(
            best_solution=binary_solution,
            best_value=verified_value,
            iterations=0,  # Pyomo doesn't always provide iteration count
            runtime_seconds=self.solve_time,
            termination_reason=termination_reason
        )

        # Add Pyomo-specific metrics
        result.additional_metrics = {
            "solver": self.solver_name,
            "solver_status": status_string,
            "termination_condition": termination_string,
            "best_bound": getattr(self, 'best_bound', None),
            "optimality_gap": getattr(self, 'gap_closed', None)
        }

        return result

    def validate_initial_solution(self, solution: List[int], n: int) -> bool:
        """Validate initial solution format."""
        if not isinstance(solution, (list, tuple)):
            return False
        if len(solution) != n:
            return False
        return all(val in [0, 1] for val in solution)
