"""
OR-Tools-based Calendar Rearrangement Optimizer for Qubots Framework

This module implements an integer programming optimizer using Google OR-Tools for calendar
rearrangement problems. It provides optimal solutions with comprehensive visualization
and analysis of meeting schedules.

Compatible with Rastion platform workflow automation and local development.
"""

from ortools.linear_solver import pywraplp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import time
from qubots import (
    BaseOptimizer, OptimizerMetadata, OptimizerType, OptimizerFamily,
    OptimizationResult, BaseProblem
)


class ORToolsCalendarOptimizer(BaseOptimizer):
    """
    OR-Tools-based Integer Programming optimizer for calendar rearrangement problems.

    This optimizer uses integer programming to find optimal solutions for calendar
    rearrangement problems. It formulates the problem as a mixed-integer linear
    program (MILP) and solves it using OR-Tools' solvers.

    Features:
    - Optimal solution guarantee for calendar rearrangement
    - Multiple solver support (SCIP, CBC, Gurobi, CPLEX)
    - Comprehensive visualization of meeting schedules
    - Before/after calendar comparison
    - Meeting distribution analysis
    """

    def __init__(self, solver_name: str = "SCIP", time_limit: float = 300.0, **kwargs):
        """
        Initialize OR-Tools Calendar Optimizer.

        Args:
            solver_name: Solver to use ("SCIP", "CBC", "GUROBI", "CPLEX")
            time_limit: Maximum solving time in seconds
            **kwargs: Additional parameters passed to BaseOptimizer
        """
        metadata = OptimizerMetadata(
            name="OR-Tools Calendar Optimizer",
            description="Integer programming optimizer for calendar rearrangement using Google OR-Tools",
            optimizer_type=OptimizerType.EXACT,
            optimizer_family=OptimizerFamily.LINEAR_PROGRAMMING,
            author="Qubots Framework",
            version="1.0.0",
            is_deterministic=True,
            supports_constraints=True,
            supports_continuous=False,
            supports_discrete=True,
            supports_mixed_integer=True,
            time_complexity="Exponential for MILP",
            space_complexity="O(meetings × days)",
            convergence_guaranteed=True,
            parallel_capable=True
        )

        super().__init__(metadata, **kwargs)
        
        self.solver_name = solver_name
        self.time_limit = time_limit
        self._setup_solver()

    def _get_default_metadata(self) -> OptimizerMetadata:
        """Return default metadata for the optimizer."""
        return self.metadata

    def _setup_solver(self):
        """Setup the OR-Tools solver."""
        self.solver_type = self.solver_name.upper()

    def _optimize_implementation(self, problem: BaseProblem, 
                               initial_solution: Optional[Any] = None) -> OptimizationResult:
        """
        Core optimization implementation using OR-Tools integer programming.

        Args:
            problem: Calendar rearrangement problem instance
            initial_solution: Optional initial solution (used as warm start)

        Returns:
            OptimizationResult with optimal solution and comprehensive analysis
        """
        start_time = time.time()
        
        try:
            # Create the solver
            solver = pywraplp.Solver.CreateSolver(self.solver_type)
            if not solver:
                raise Exception(f"Could not create solver for {self.solver_name}")
            
            # Set time limit
            solver.SetTimeLimit(int(self.time_limit * 1000))  # Convert to milliseconds
            
            # Extract problem data
            moveable_meetings = problem.moveable_meetings
            available_days = problem.available_days
            n_meetings = len(moveable_meetings)
            n_days = len(available_days)
            
            if n_meetings == 0:
                # No meetings to move
                return self._create_empty_result(problem, start_time)
            
            # Decision variables: x[i][j] = 1 if meeting i is assigned to day j
            x = {}
            for i in range(n_meetings):
                for j in range(n_days):
                    x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')
            
            # Constraint 1: Each meeting must be assigned to exactly one day
            for i in range(n_meetings):
                solver.Add(sum(x[i, j] for j in range(n_days)) == 1)
            
            # Constraint 2: Capacity constraints for each day
            for j in range(n_days):
                day = available_days[j]
                
                # Calculate existing hours on this day
                existing_hours = 0.0
                for meeting in problem.meetings:
                    if (meeting.current_day == day and 
                        meeting.current_day != problem.target_day_off):
                        existing_hours += meeting.duration_hours
                
                # Add constraint for moved meetings
                moved_hours = sum(x[i, j] * moveable_meetings[i].duration_hours 
                                for i in range(n_meetings))
                solver.Add(existing_hours + moved_hours <= problem.max_hours_per_day)
            
            # Objective: Minimize total rescheduling cost
            objective_terms = []
            for i in range(n_meetings):
                meeting = moveable_meetings[i]
                
                # Base rescheduling cost
                base_cost = problem.rescheduling_cost_weight * (1 + 0.1 * meeting.participants)
                
                # Priority cost
                priority_cost = problem.priority_weight * meeting.priority
                
                total_meeting_cost = base_cost + priority_cost
                
                # Add cost for each possible assignment
                for j in range(n_days):
                    objective_terms.append(total_meeting_cost * x[i, j])
            
            solver.Minimize(sum(objective_terms))
            
            # Solve the problem
            status = solver.Solve()
            
            # Process results
            if status == pywraplp.Solver.OPTIMAL:
                # Extract solution
                assignments = []
                for i in range(n_meetings):
                    for j in range(n_days):
                        if x[i, j].solution_value() > 0.5:
                            assignments.append(j)
                            break
                
                solution = {'assignments': assignments}
                objective_value = solver.Objective().Value()
                
                # Create comprehensive result
                result = OptimizationResult(
                    best_solution=solution,
                    best_value=objective_value,
                    is_feasible=True,
                    runtime_seconds=time.time() - start_time,
                    iterations=solver.iterations(),
                    termination_reason=f"OR-Tools {self.solver_name} - OPTIMAL",
                    additional_metrics={
                        'status': 'OPTIMAL',
                        'solver_status': status,
                        'gap': 0.0,
                        'bound': objective_value,
                        'meetings_moved': n_meetings,
                        'available_days': n_days,
                        'solver_time': solver.WallTime() / 1000.0,
                        'nodes_explored': solver.nodes() if hasattr(solver, 'nodes') else 0
                    }
                )
                
                # Add visualization
                self._add_visualization(result, problem, solution)
                
                return result
                
            elif status == pywraplp.Solver.FEASIBLE:
                # Feasible but not optimal (time limit reached)
                assignments = []
                for i in range(n_meetings):
                    for j in range(n_days):
                        if x[i, j].solution_value() > 0.5:
                            assignments.append(j)
                            break
                
                solution = {'assignments': assignments}
                objective_value = solver.Objective().Value()
                
                result = OptimizationResult(
                    best_solution=solution,
                    best_value=objective_value,
                    is_feasible=True,
                    runtime_seconds=time.time() - start_time,
                    iterations=solver.iterations(),
                    termination_reason=f"OR-Tools {self.solver_name} - FEASIBLE",
                    additional_metrics={
                        'status': 'FEASIBLE',
                        'solver_status': status,
                        'gap': 'unknown',
                        'bound': solver.Objective().BestBound(),
                        'meetings_moved': n_meetings,
                        'available_days': n_days,
                        'solver_time': solver.WallTime() / 1000.0,
                        'time_limit_reached': True
                    }
                )
                
                self._add_visualization(result, problem, solution)
                return result
                
            else:
                # Infeasible or error
                return OptimizationResult(
                    best_solution=None,
                    best_value=float('inf'),
                    is_feasible=False,
                    runtime_seconds=time.time() - start_time,
                    iterations=0,
                    termination_reason=f"OR-Tools {self.solver_name} - INFEASIBLE",
                    additional_metrics={
                        'status': 'INFEASIBLE',
                        'solver_status': status,
                        'error': 'Problem is infeasible or solver error',
                        'meetings_moved': n_meetings,
                        'available_days': n_days
                    }
                )
                
        except Exception as e:
            return OptimizationResult(
                best_solution=None,
                best_value=float('inf'),
                is_feasible=False,
                runtime_seconds=time.time() - start_time,
                iterations=0,
                termination_reason=f"OR-Tools {self.solver_name} - ERROR",
                additional_metrics={
                    'status': 'ERROR',
                    'error': str(e)
                }
            )

    def _create_empty_result(self, problem: BaseProblem, start_time: float) -> OptimizationResult:
        """Create result when no meetings need to be moved."""
        return OptimizationResult(
            best_solution={'assignments': []},
            best_value=0.0,
            is_feasible=True,
            runtime_seconds=time.time() - start_time,
            iterations=0,
            termination_reason=f"OR-Tools {self.solver_name} - OPTIMAL",
            additional_metrics={
                'status': 'OPTIMAL',
                'message': f'No meetings to move from {problem.target_day_off}',
                'meetings_moved': 0,
                'available_days': len(problem.available_days)
            }
        )

    def _add_visualization(self, result: OptimizationResult, problem: BaseProblem, solution: Dict[str, Any]):
        """Add visualization plots to the optimization result."""
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 16))

            # Create a grid layout: 3 rows, 2 columns, with the first row spanning both columns
            gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)

            # Main schedule visualization (top row, spanning both columns)
            ax_schedule = fig.add_subplot(gs[0, :])
            self._plot_schedule_gantt(ax_schedule, problem, solution)

            # Secondary plots (bottom two rows)
            ax1 = fig.add_subplot(gs[1, 0])
            ax2 = fig.add_subplot(gs[1, 1])
            ax3 = fig.add_subplot(gs[2, 0])
            ax4 = fig.add_subplot(gs[2, 1])

            fig.suptitle('Calendar Rearrangement Optimization Results', fontsize=18, fontweight='bold')

            # Plot 1: Before/After Meeting Distribution
            self._plot_meeting_distribution(ax1, problem, solution)

            # Plot 2: Daily Hours Comparison
            self._plot_daily_hours(ax2, problem, solution)

            # Plot 3: Meeting Priority Distribution
            self._plot_priority_distribution(ax3, problem, solution)

            # Plot 4: Cost Breakdown
            self._plot_cost_breakdown(ax4, problem, solution)

            plt.tight_layout()
            result.plots = {'calendar_analysis': fig}

            # Display the plot
            plt.show()

        except Exception as e:
            print(f"Warning: Could not generate visualization: {e}")

    def _plot_schedule_gantt(self, ax, problem, solution):
        """Plot a Gantt chart showing the before/after schedule."""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        # Create before and after schedules
        before_schedule = {day: [] for day in days}
        after_schedule = {day: [] for day in days}

        # Populate before schedule
        for meeting in problem.meetings:
            if meeting.current_day in before_schedule:
                before_schedule[meeting.current_day].append({
                    'name': meeting.meeting_name,
                    'duration': meeting.duration_hours,
                    'priority': meeting.priority,
                    'flexible': meeting.flexible
                })

        # Populate after schedule (start with before, then apply changes)
        for day in days:
            after_schedule[day] = before_schedule[day].copy()

        # Remove moved meetings from target day and add to new days
        assignments = solution['assignments']
        moved_meetings = []

        # Remove moveable meetings from target day
        for meeting in problem.moveable_meetings:
            for i, m in enumerate(after_schedule[problem.target_day_off]):
                if m['name'] == meeting.meeting_name:
                    moved_meetings.append(after_schedule[problem.target_day_off].pop(i))
                    break

        # Add moved meetings to new days
        for i, day_idx in enumerate(assignments):
            new_day = problem.available_days[day_idx]
            if i < len(moved_meetings):
                after_schedule[new_day].append(moved_meetings[i])

        # Plot the schedules
        y_positions = np.arange(len(days))
        bar_height = 0.35

        # Calculate cumulative hours for stacked bars
        for i, day in enumerate(days):
            # Before schedule
            cumulative_before = 0
            for j, meeting in enumerate(before_schedule[day]):
                color = plt.cm.Reds(0.3 + 0.1 * meeting['priority'])
                alpha = 0.8 if meeting['flexible'] else 0.5
                ax.barh(i - bar_height/2, meeting['duration'], left=cumulative_before,
                       height=bar_height, color=color, alpha=alpha,
                       label=f"Before - {meeting['name'][:15]}..." if len(meeting['name']) > 15 else f"Before - {meeting['name']}")
                cumulative_before += meeting['duration']

            # After schedule
            cumulative_after = 0
            for j, meeting in enumerate(after_schedule[day]):
                color = plt.cm.Blues(0.3 + 0.1 * meeting['priority'])
                alpha = 0.8 if meeting['flexible'] else 0.5
                ax.barh(i + bar_height/2, meeting['duration'], left=cumulative_after,
                       height=bar_height, color=color, alpha=alpha,
                       label=f"After - {meeting['name'][:15]}..." if len(meeting['name']) > 15 else f"After - {meeting['name']}")
                cumulative_after += meeting['duration']

            # Add total hours text
            ax.text(max(cumulative_before, cumulative_after) + 0.1, i - bar_height/2,
                   f'{cumulative_before:.1f}h', va='center', fontsize=8, color='red')
            ax.text(max(cumulative_before, cumulative_after) + 0.1, i + bar_height/2,
                   f'{cumulative_after:.1f}h', va='center', fontsize=8, color='blue')

        # Highlight the target day off
        target_day_idx = days.index(problem.target_day_off) if problem.target_day_off in days else -1
        if target_day_idx >= 0:
            ax.axhspan(target_day_idx - 0.4, target_day_idx + 0.4, alpha=0.2, color='green',
                      label=f'{problem.target_day_off} - Target Day Off')

        # Add capacity line
        ax.axvline(x=problem.max_hours_per_day, color='red', linestyle='--',
                  label=f'Max Capacity ({problem.max_hours_per_day}h)')

        ax.set_yticks(y_positions)
        ax.set_yticklabels(days)
        ax.set_xlabel('Hours')
        ax.set_title('Weekly Schedule: Before (Red) vs After (Blue) Optimization')
        ax.grid(True, alpha=0.3, axis='x')

        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.6, label='Before Optimization'),
            Patch(facecolor='blue', alpha=0.6, label='After Optimization'),
            Patch(facecolor='green', alpha=0.2, label=f'{problem.target_day_off} - Target Day Off'),
            plt.Line2D([0], [0], color='red', linestyle='--', label=f'Max Capacity ({problem.max_hours_per_day}h)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    def _plot_meeting_distribution(self, ax, problem, solution):
        """Plot before/after meeting distribution across days."""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        # Count meetings before rearrangement
        before_counts = {day: 0 for day in days}
        for meeting in problem.meetings:
            if meeting.current_day in before_counts:
                before_counts[meeting.current_day] += 1
        
        # Count meetings after rearrangement
        after_counts = before_counts.copy()
        assignments = solution['assignments']
        
        # Remove moved meetings from target day
        for meeting in problem.moveable_meetings:
            after_counts[problem.target_day_off] -= 1
        
        # Add moved meetings to new days
        for i, day_idx in enumerate(assignments):
            new_day = problem.available_days[day_idx]
            after_counts[new_day] += 1
        
        x = np.arange(len(days))
        width = 0.35
        
        before_values = [before_counts[day] for day in days]
        after_values = [after_counts[day] for day in days]
        
        ax.bar(x - width/2, before_values, width, label='Before', alpha=0.8, color='lightcoral')
        ax.bar(x + width/2, after_values, width, label='After', alpha=0.8, color='lightblue')
        
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Number of Meetings')
        ax.set_title('Meeting Distribution: Before vs After')
        ax.set_xticks(x)
        ax.set_xticklabels(days, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_daily_hours(self, ax, problem, solution):
        """Plot daily meeting hours before and after rearrangement."""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        # Calculate hours before
        before_hours = {day: 0.0 for day in days}
        for meeting in problem.meetings:
            if meeting.current_day in before_hours:
                before_hours[meeting.current_day] += meeting.duration_hours
        
        # Calculate hours after
        after_hours = before_hours.copy()
        assignments = solution['assignments']
        
        # Remove moved meetings from target day
        for meeting in problem.moveable_meetings:
            after_hours[problem.target_day_off] -= meeting.duration_hours
        
        # Add moved meetings to new days
        for i, day_idx in enumerate(assignments):
            new_day = problem.available_days[day_idx]
            meeting = problem.moveable_meetings[i]
            after_hours[new_day] += meeting.duration_hours
        
        x = np.arange(len(days))
        width = 0.35
        
        before_values = [before_hours[day] for day in days]
        after_values = [after_hours[day] for day in days]
        
        bars1 = ax.bar(x - width/2, before_values, width, label='Before', alpha=0.8, color='orange')
        bars2 = ax.bar(x + width/2, after_values, width, label='After', alpha=0.8, color='green')
        
        # Add capacity line
        ax.axhline(y=problem.max_hours_per_day, color='red', linestyle='--', 
                  label=f'Max Capacity ({problem.max_hours_per_day}h)')
        
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Meeting Hours')
        ax.set_title('Daily Meeting Hours: Before vs After')
        ax.set_xticks(x)
        ax.set_xticklabels(days, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_priority_distribution(self, ax, problem, solution):
        """Plot priority distribution of moved meetings."""
        if not problem.moveable_meetings:
            ax.text(0.5, 0.5, 'No meetings to move', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Priority Distribution of Moved Meetings')
            return
        
        priorities = [meeting.priority for meeting in problem.moveable_meetings]
        assignments = solution['assignments']
        
        # Create data for plotting
        priority_day_data = []
        for i, priority in enumerate(priorities):
            day_idx = assignments[i]
            day = problem.available_days[day_idx]
            priority_day_data.append({'Priority': priority, 'New Day': day})
        
        df = pd.DataFrame(priority_day_data)
        
        # Create stacked bar chart
        priority_counts = df.groupby(['Priority', 'New Day']).size().unstack(fill_value=0)
        priority_counts.plot(kind='bar', stacked=True, ax=ax, alpha=0.8)
        
        ax.set_xlabel('Meeting Priority')
        ax.set_ylabel('Number of Meetings')
        ax.set_title('Priority Distribution of Moved Meetings')
        ax.legend(title='Moved to Day', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    def _plot_cost_breakdown(self, ax, problem, solution):
        """Plot cost breakdown of the solution."""
        if not problem.moveable_meetings:
            ax.text(0.5, 0.5, 'No rescheduling costs', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Rescheduling Cost Breakdown')
            return
        
        assignments = solution['assignments']
        base_costs = []
        priority_costs = []
        meeting_names = []
        
        for i, meeting in enumerate(problem.moveable_meetings):
            base_cost = problem.rescheduling_cost_weight * (1 + 0.1 * meeting.participants)
            priority_cost = problem.priority_weight * meeting.priority
            
            base_costs.append(base_cost)
            priority_costs.append(priority_cost)
            meeting_names.append(meeting.meeting_name[:15] + '...' if len(meeting.meeting_name) > 15 
                               else meeting.meeting_name)
        
        x = np.arange(len(meeting_names))
        
        ax.bar(x, base_costs, label='Base Cost', alpha=0.8, color='skyblue')
        ax.bar(x, priority_costs, bottom=base_costs, label='Priority Cost', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Meetings')
        ax.set_ylabel('Rescheduling Cost')
        ax.set_title('Rescheduling Cost Breakdown by Meeting')
        ax.set_xticks(x)
        ax.set_xticklabels(meeting_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

