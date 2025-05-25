"""
Fantasy Football OR-Tools Optimizer for Qubots Framework.

This module implements a specialized OR-Tools CP-SAT optimizer designed specifically
for fantasy football lineup optimization problems. It uses constraint programming
to find optimal lineups that maximize projected points while satisfying all
DraftKings constraints.

The optimizer can be uploaded to and loaded from the Rastion platform for
seamless integration with the qubots ecosystem.

Author: Qubots Community
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from qubots import (
    BaseOptimizer,
    OptimizerMetadata,
    OptimizerType,
    OptimizerFamily,
    OptimizationResult,
    BaseProblem
)

# OR-Tools integration
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False


@dataclass
class FantasyFootballORToolsConfig:
    """Configuration for Fantasy Football OR-Tools optimizer."""
    time_limit_seconds: int = 60
    num_search_workers: int = 4
    log_search_progress: bool = False
    use_hint: bool = True
    randomize_search: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'time_limit_seconds': self.time_limit_seconds,
            'num_search_workers': self.num_search_workers,
            'log_search_progress': self.log_search_progress,
            'use_hint': self.use_hint,
            'randomize_search': self.randomize_search
        }


class FantasyFootballORToolsOptimizer(BaseOptimizer):
    """
    OR-Tools CP-SAT optimizer for fantasy football lineup optimization.

    This optimizer uses Google's OR-Tools Constraint Programming SAT solver
    to find optimal fantasy football lineups. It's particularly effective for
    this type of combinatorial optimization problem with multiple constraints.

    Key Features:
    - Uses CP-SAT solver for exact optimization
    - Handles all DraftKings position constraints
    - Respects salary cap limitations
    - Supports time limits and parallel search
    - Provides detailed solution statistics

    Example Usage:
        ```python
        from fantasy_football import FantasyFootballProblem

        # Create problem and optimizer
        problem = FantasyFootballProblem(csv_file='player_data.csv')
        optimizer = FantasyFootballORToolsOptimizer(time_limit_seconds=30)

        # Run optimization
        result = optimizer.optimize(problem)
        print(f'Best lineup points: {result.best_value:.2f}')
        print(f'Optimal: {result.convergence_achieved}')

        # Show best lineup
        best_lineup = problem.get_lineup_summary(result.best_solution)
        print(best_lineup)
        ```

    Rastion Integration:
        ```python
        import qubots.rastion as rastion

        # Load problem and optimizer from Rastion
        problem = rastion.load_qubots_model('fantasy_football_problem')
        optimizer = rastion.load_qubots_model('fantasy_football_ortools_optimizer')

        # Run optimization
        result = optimizer.optimize(problem)
        ```
    """

    def __init__(self,
                 time_limit_seconds: int = 60,
                 num_search_workers: int = 4,
                 log_search_progress: bool = False,
                 use_hint: bool = True,
                 randomize_search: bool = True,
                 **kwargs):
        """
        Initialize the Fantasy Football OR-Tools optimizer.

        Args:
            time_limit_seconds: Maximum time to spend searching (default: 60)
            num_search_workers: Number of parallel search workers (default: 4)
            log_search_progress: Whether to log search progress (default: False)
            use_hint: Whether to use initial solution hint (default: True)
            randomize_search: Whether to randomize search strategy (default: True)
            **kwargs: Additional parameters passed to base class
        """
        # Check OR-Tools availability
        if not ORTOOLS_AVAILABLE:
            raise ImportError(
                "OR-Tools not available. Please install with: pip install ortools"
            )

        # Create configuration
        self.config = FantasyFootballORToolsConfig(
            time_limit_seconds=time_limit_seconds,
            num_search_workers=num_search_workers,
            log_search_progress=log_search_progress,
            use_hint=use_hint,
            randomize_search=randomize_search
        )

        # Initialize base optimizer
        super().__init__(**kwargs)

        # Solver statistics
        self.last_solve_time = 0.0
        self.last_num_branches = 0
        self.last_num_conflicts = 0
        self.last_status = None

    def _get_default_metadata(self) -> OptimizerMetadata:
        """Return default metadata for this optimizer."""
        return OptimizerMetadata(
            name="Fantasy Football OR-Tools Optimizer",
            description="CP-SAT optimizer for fantasy football lineup optimization using Google OR-Tools",
            optimizer_type=OptimizerType.EXACT,
            optimizer_family=OptimizerFamily.CONSTRAINT_PROGRAMMING,
            domain="sports_analytics",
            tags={"fantasy_football", "ortools", "constraint_programming", "exact", "sports"},
            author="Qubots Community",
            version="1.0.0",
            parameters={
                "time_limit_seconds": "Maximum solving time in seconds",
                "num_search_workers": "Number of parallel search workers",
                "log_search_progress": "Enable search progress logging",
                "use_hint": "Use initial solution as search hint",
                "randomize_search": "Randomize search strategy"
            },
            capabilities={
                "handles_constraints": True,
                "parallel_execution": True,
                "time_limited": True,
                "exact_solver": True,
                "supports_hints": True
            }
        )

    def _optimize_implementation(self, problem: BaseProblem, initial_solution: Optional[Any] = None) -> OptimizationResult:
        """
        Core optimization implementation using OR-Tools CP-SAT.

        Args:
            problem: Fantasy football problem instance
            initial_solution: Optional initial solution to use as hint

        Returns:
            OptimizationResult with comprehensive optimization information
        """
        start_time = time.time()

        # Validate problem compatibility
        if not hasattr(problem, 'df') or not hasattr(problem, 'n_players'):
            raise ValueError("Problem must be a FantasyFootballProblem with df and n_players attributes")

        # Create CP model
        model = cp_model.CpModel()

        # Decision variables: binary variable for each player
        player_vars = []
        for i in range(problem.n_players):
            player_vars.append(model.NewBoolVar(f'player_{i}'))

        # Objective: maximize total projected points
        points = [float(problem.df.iloc[i]['DK.points']) for i in range(problem.n_players)]
        model.Maximize(sum(points[i] * player_vars[i] for i in range(problem.n_players)))

        # Constraint 1: Exactly 9 players in lineup
        model.Add(sum(player_vars) == 9)

        # Constraint 2: Salary cap
        salaries = [int(problem.df.iloc[i]['DK.salary']) for i in range(problem.n_players)]
        model.Add(sum(salaries[i] * player_vars[i] for i in range(problem.n_players)) <= problem.max_salary)
        model.Add(sum(salaries[i] * player_vars[i] for i in range(problem.n_players)) >= problem.min_salary)

        # Position constraints
        self._add_position_constraints(model, player_vars, problem)

        # Create solver
        solver = cp_model.CpSolver()

        # Configure solver parameters
        solver.parameters.max_time_in_seconds = self.config.time_limit_seconds
        solver.parameters.num_search_workers = self.config.num_search_workers
        solver.parameters.log_search_progress = self.config.log_search_progress
        solver.parameters.randomize_search = self.config.randomize_search

        # Add initial solution hint if provided and enabled
        if self.config.use_hint and initial_solution is not None:
            self._add_solution_hint(model, player_vars, initial_solution, problem)

        # Solve the model
        status = solver.Solve(model)
        solve_time = time.time() - start_time

        # Store solver statistics
        self.last_solve_time = solve_time
        self.last_num_branches = solver.NumBranches()
        self.last_num_conflicts = solver.NumConflicts()
        self.last_status = status

        # Process results
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Extract solution
            solution = [solver.Value(var) for var in player_vars]
            objective_value = float(solver.ObjectiveValue())

            # Determine if optimal
            is_optimal = (status == cp_model.OPTIMAL)

            return OptimizationResult(
                best_solution=solution,
                best_value=objective_value,
                is_feasible=True,
                iterations=solver.NumBranches(),
                evaluations=solver.NumBranches(),
                runtime_seconds=solve_time,
                convergence_achieved=is_optimal,
                termination_reason="optimal" if is_optimal else "feasible_solution_found",
                additional_info={
                    "solver_status": self._status_to_string(status),
                    "num_branches": solver.NumBranches(),
                    "num_conflicts": solver.NumConflicts(),
                    "num_booleans": solver.NumBooleans(),
                    "wall_time": solver.WallTime(),
                    "user_time": solver.UserTime(),
                    "deterministic_time": solver.DeterministicTime()
                }
            )
        else:
            # No feasible solution found
            return OptimizationResult(
                best_solution=None,
                best_value=float('-inf'),
                is_feasible=False,
                iterations=solver.NumBranches(),
                evaluations=solver.NumBranches(),
                runtime_seconds=solve_time,
                convergence_achieved=False,
                termination_reason=self._status_to_string(status),
                additional_info={
                    "solver_status": self._status_to_string(status),
                    "num_branches": solver.NumBranches(),
                    "num_conflicts": solver.NumConflicts()
                }
            )

    def _add_position_constraints(self, model: cp_model.CpModel, player_vars: List, problem: BaseProblem):
        """Add position-specific constraints to the model."""
        df = problem.df

        # QB constraint: exactly 1
        if 'Pos_QB' in df.columns:
            qb_vars = [player_vars[i] for i in range(problem.n_players) if df.iloc[i]['Pos_QB'] == 1]
            if qb_vars:
                model.Add(sum(qb_vars) == 1)

        # RB constraint: 2-3 players
        if 'Pos_RB' in df.columns:
            rb_vars = [player_vars[i] for i in range(problem.n_players) if df.iloc[i]['Pos_RB'] == 1]
            if rb_vars:
                model.Add(sum(rb_vars) >= 2)
                model.Add(sum(rb_vars) <= 3)

        # WR constraint: 3-4 players
        if 'Pos_WR' in df.columns:
            wr_vars = [player_vars[i] for i in range(problem.n_players) if df.iloc[i]['Pos_WR'] == 1]
            if wr_vars:
                model.Add(sum(wr_vars) >= 3)
                model.Add(sum(wr_vars) <= 4)

        # TE constraint: 1-2 players
        if 'Pos_TE' in df.columns:
            te_vars = [player_vars[i] for i in range(problem.n_players) if df.iloc[i]['Pos_TE'] == 1]
            if te_vars:
                model.Add(sum(te_vars) >= 1)
                model.Add(sum(te_vars) <= 2)

        # DEF constraint: exactly 1
        if 'Pos_Def' in df.columns:
            def_vars = [player_vars[i] for i in range(problem.n_players) if df.iloc[i]['Pos_Def'] == 1]
            if def_vars:
                model.Add(sum(def_vars) == 1)

        # Flex constraint: total RB+WR+TE should be exactly 7
        if 'PosFlex' in df.columns:
            flex_vars = [player_vars[i] for i in range(problem.n_players) if df.iloc[i]['PosFlex'] == 1]
            if flex_vars:
                model.Add(sum(flex_vars) == 7)

    def _add_solution_hint(self, model: cp_model.CpModel, player_vars: List,
                          initial_solution: Any, problem: BaseProblem):
        """Add initial solution as hint to guide the search."""
        try:
            if len(initial_solution) == problem.n_players:
                for i, value in enumerate(initial_solution):
                    if value in [0, 1]:
                        model.AddHint(player_vars[i], int(value))
        except (TypeError, ValueError, IndexError):
            # If hint is invalid, just skip it
            pass

    def _status_to_string(self, status: int) -> str:
        """Convert OR-Tools status to string."""
        status_map = {
            cp_model.OPTIMAL: "optimal",
            cp_model.FEASIBLE: "feasible",
            cp_model.INFEASIBLE: "infeasible",
            cp_model.MODEL_INVALID: "model_invalid",
            cp_model.UNKNOWN: "unknown"
        }
        return status_map.get(status, f"unknown_status_{status}")

    def get_solver_statistics(self) -> Dict[str, Any]:
        """Get detailed solver statistics from the last run."""
        return {
            "solve_time_seconds": self.last_solve_time,
            "num_branches": self.last_num_branches,
            "num_conflicts": self.last_num_conflicts,
            "solver_status": self._status_to_string(self.last_status) if self.last_status is not None else "not_run",
            "configuration": self.config.to_dict()
        }

    def update_config(self, **kwargs):
        """Update optimizer configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")