"""
School Resource Allocation Optimizer for Qubots Framework

This optimizer uses OR-Tools constraint programming to solve school scheduling problems
by finding optimal teacher-subject-classroom-timeslot assignments.

Author: Qubots Framework
Version: 1.0.0
"""

import time
import random
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

# Import qubots framework components
try:
    from qubots.base_optimizer import BaseOptimizer, OptimizationResult, OptimizerMetadata, OptimizerType, OptimizerFamily
    from qubots.base_problem import BaseProblem
except ImportError:
    # Fallback for local development
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from qubots.base_optimizer import BaseOptimizer, OptimizationResult, OptimizerMetadata, OptimizerType, OptimizerFamily
    from qubots.base_problem import BaseProblem

# Import OR-Tools
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("Warning: OR-Tools not available. Install with: pip install ortools")

# Import school problem components
try:
    from examples.school_resource_allocation_problem.qubot import Assignment, SchoolResourceAllocationProblem
except ImportError:
    # Define minimal Assignment class for fallback
    @dataclass
    class Assignment:
        teacher_id: str
        subject_id: str
        room_id: str
        time_slot_id: int
        student_count: int = 20


class SchoolResourceAllocationOptimizer(BaseOptimizer):
    """
    OR-Tools based optimizer for school resource allocation problems.
    
    This optimizer uses constraint programming to find optimal schedules that:
    - Eliminate conflicts (teacher/room double-booking)
    - Satisfy subject hour requirements
    - Respect teacher qualifications and room requirements
    - Minimize costs while maximizing quality
    
    Parameters:
    - max_solve_time_seconds: Maximum time to spend solving (default: 300)
    - num_search_workers: Number of parallel search workers (default: 4)
    - emphasis: Search emphasis - 'feasibility', 'optimality', or 'balanced' (default: 'balanced')
    - enable_logging: Whether to enable OR-Tools logging (default: False)
    """
    
    def __init__(self,
                 max_solve_time_seconds: float = 300.0,
                 num_search_workers: int = 4,
                 emphasis: str = 'balanced',
                 enable_logging: bool = False,
                 **kwargs):
        """
        Initialize school resource allocation optimizer.

        Args:
            max_solve_time_seconds: Maximum time to spend solving (default: 300)
            num_search_workers: Number of parallel search workers (default: 4)
            emphasis: Search emphasis - 'feasibility', 'optimality', or 'balanced' (default: 'balanced')
            enable_logging: Whether to enable OR-Tools logging (default: False)
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)

        # Set parameters as instance variables
        self.max_solve_time_seconds = max_solve_time_seconds
        self.num_search_workers = num_search_workers
        self.emphasis = emphasis
        self.enable_logging = enable_logging

        if not ORTOOLS_AVAILABLE:
            raise ImportError("OR-Tools is required for this optimizer. Install with: pip install ortools")
    
    def _get_default_metadata(self) -> OptimizerMetadata:
        """Return default metadata for school resource allocation optimizer."""
        return OptimizerMetadata(
            name="School Resource Allocation Optimizer",
            description="OR-Tools constraint programming optimizer for school scheduling problems with conflict resolution and cost optimization",
            optimizer_type=OptimizerType.EXACT,
            optimizer_family=OptimizerFamily.CONSTRAINT_PROGRAMMING,
            author="Qubots Framework",
            version="1.0.0",
            is_deterministic=True,
            supports_constraints=True,
            supports_multi_objective=True,
            supports_continuous=False,
            supports_discrete=True,
            supports_mixed_integer=True,
            time_complexity="O(exponential in worst case)",
            space_complexity="O(n³)",
            convergence_guaranteed=True,
            parallel_capable=True,
            required_parameters=[],
            optional_parameters=["max_solve_time_seconds", "num_search_workers", "emphasis", "enable_logging"],
            parameter_ranges={
                "max_solve_time_seconds": (1.0, 3600.0),
                "num_search_workers": (1, 16)
            }
        )
    
    def _optimize_implementation(self, problem: BaseProblem, initial_solution: Optional[Any] = None) -> OptimizationResult:
        """
        Core optimization implementation using OR-Tools CP-SAT solver.
        """
        start_time = time.time()
        
        # Validate problem type
        if not hasattr(problem, 'teachers') or not hasattr(problem, 'subjects'):
            raise ValueError("Problem must be a SchoolResourceAllocationProblem instance")
        
        # Extract parameters
        max_solve_time = self.max_solve_time_seconds
        num_workers = self.num_search_workers
        emphasis = self.emphasis
        enable_logging = self.enable_logging
        
        self.log_message('info', f"Starting OR-Tools optimization with {len(problem.teachers)} teachers, {len(problem.subjects)} subjects")
        
        # Create CP model
        model = cp_model.CpModel()
        
        # Decision variables: assignment[t, s, r, slot] = 1 if teacher t teaches subject s in room r at time slot
        assignments = {}
        teacher_ids = list(problem.teachers.keys())
        subject_ids = list(problem.subjects.keys())
        room_ids = list(problem.classrooms.keys())
        time_slots = list(problem.time_slots.keys())
        
        # Create binary variables for each possible assignment
        for t_idx, teacher_id in enumerate(teacher_ids):
            for s_idx, subject_id in enumerate(subject_ids):
                for r_idx, room_id in enumerate(room_ids):
                    for slot in time_slots:
                        var_name = f"assign_{teacher_id}_{subject_id}_{room_id}_{slot}"
                        assignments[(t_idx, s_idx, r_idx, slot)] = model.NewBoolVar(var_name)
        
        # Constraint 1: No teacher double-booking
        for t_idx in range(len(teacher_ids)):
            for slot in time_slots:
                model.Add(
                    sum(assignments[(t_idx, s_idx, r_idx, slot)] 
                        for s_idx in range(len(subject_ids))
                        for r_idx in range(len(room_ids))) <= 1
                )
        
        # Constraint 2: No room double-booking
        for r_idx in range(len(room_ids)):
            for slot in time_slots:
                model.Add(
                    sum(assignments[(t_idx, s_idx, r_idx, slot)]
                        for t_idx in range(len(teacher_ids))
                        for s_idx in range(len(subject_ids))) <= 1
                )
        
        # Constraint 3: Subject hour requirements
        for s_idx, subject_id in enumerate(subject_ids):
            subject = problem.subjects[subject_id]
            model.Add(
                sum(assignments[(t_idx, s_idx, r_idx, slot)]
                    for t_idx in range(len(teacher_ids))
                    for r_idx in range(len(room_ids))
                    for slot in time_slots) >= subject.required_hours_per_week
            )
        
        # Constraint 4: Teacher qualifications
        for t_idx, teacher_id in enumerate(teacher_ids):
            teacher = problem.teachers[teacher_id]
            for s_idx, subject_id in enumerate(subject_ids):
                subject = problem.subjects[subject_id]
                # If teacher is not qualified for subject, prevent assignment
                if subject_id not in teacher.subjects and subject.name not in teacher.subjects:
                    for r_idx in range(len(room_ids)):
                        for slot in time_slots:
                            model.Add(assignments[(t_idx, s_idx, r_idx, slot)] == 0)
        
        # Constraint 5: Room type requirements
        for s_idx, subject_id in enumerate(subject_ids):
            subject = problem.subjects[subject_id]
            for r_idx, room_id in enumerate(room_ids):
                room = problem.classrooms[room_id]
                # If room type doesn't match subject requirements, prevent assignment
                if subject.required_room_type != "standard" and room.room_type != subject.required_room_type:
                    for t_idx in range(len(teacher_ids)):
                        for slot in time_slots:
                            model.Add(assignments[(t_idx, s_idx, r_idx, slot)] == 0)
        
        # Constraint 6: Teacher daily hour limits
        for t_idx, teacher_id in enumerate(teacher_ids):
            teacher = problem.teachers[teacher_id]
            # Group time slots by day (assuming 8 slots per day)
            slots_per_day = 8
            for day in range(len(time_slots) // slots_per_day):
                day_slots = range(day * slots_per_day, (day + 1) * slots_per_day)
                model.Add(
                    sum(assignments[(t_idx, s_idx, r_idx, slot)]
                        for s_idx in range(len(subject_ids))
                        for r_idx in range(len(room_ids))
                        for slot in day_slots) <= teacher.max_hours_per_day
                )
        
        # Objective: Minimize cost while maximizing quality
        cost_terms = []
        quality_terms = []
        
        for t_idx, teacher_id in enumerate(teacher_ids):
            teacher = problem.teachers[teacher_id]
            for s_idx, subject_id in enumerate(subject_ids):
                for r_idx, room_id in enumerate(room_ids):
                    room = problem.classrooms[room_id]
                    for slot in time_slots:
                        var = assignments[(t_idx, s_idx, r_idx, slot)]
                        
                        # Cost component (teacher + room cost)
                        total_cost = int((teacher.hourly_cost + room.hourly_cost) * 10)  # Scale for integer
                        cost_terms.append(total_cost * var)
                        
                        # Quality component (teacher experience)
                        quality_bonus = teacher.experience_level * 10
                        quality_terms.append(quality_bonus * var)
        
        # Multi-objective: minimize cost, maximize quality
        total_cost = sum(cost_terms)
        total_quality = sum(quality_terms)
        
        # Weighted objective (cost minimization with quality bonus)
        model.Minimize(total_cost - total_quality)
        
        # Configure solver
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max_solve_time
        solver.parameters.num_search_workers = num_workers
        solver.parameters.log_search_progress = enable_logging
        
        # Set search emphasis
        if emphasis == 'feasibility':
            solver.parameters.cp_model_presolve = False
        elif emphasis == 'optimality':
            solver.parameters.optimize_with_core = True
        # 'balanced' uses default settings
        
        self.log_message('info', f"Solving with {len(assignments)} variables and OR-Tools CP-SAT")
        
        # Solve the model
        status = solver.Solve(model)
        
        solve_time = time.time() - start_time
        
        # Process results
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Extract solution
            solution_assignments = []
            total_cost_value = 0
            total_quality_value = 0
            
            for t_idx, teacher_id in enumerate(teacher_ids):
                teacher = problem.teachers[teacher_id]
                for s_idx, subject_id in enumerate(subject_ids):
                    for r_idx, room_id in enumerate(room_ids):
                        room = problem.classrooms[room_id]
                        for slot in time_slots:
                            var = assignments[(t_idx, s_idx, r_idx, slot)]
                            if solver.Value(var) == 1:
                                # Create assignment
                                assignment = Assignment(
                                    teacher_id=teacher_id,
                                    subject_id=subject_id,
                                    room_id=room_id,
                                    time_slot_id=slot,
                                    student_count=min(problem.subjects[subject_id].max_class_size, room.capacity)
                                )
                                solution_assignments.append(assignment)
                                
                                # Track costs and quality
                                total_cost_value += teacher.hourly_cost + room.hourly_cost
                                total_quality_value += teacher.experience_level
            
            # Evaluate solution using problem's evaluation method
            evaluation_result = problem.evaluate_solution_detailed(solution_assignments)
            
            self.log_message('info', f"Found {'optimal' if status == cp_model.OPTIMAL else 'feasible'} solution with {len(solution_assignments)} assignments")
            
            return OptimizationResult(
                best_solution=solution_assignments,
                best_value=evaluation_result.objective_value,
                is_feasible=evaluation_result.is_feasible,
                iterations=1,
                evaluations=1,
                runtime_seconds=solve_time,
                convergence_achieved=(status == cp_model.OPTIMAL),
                termination_reason="optimal" if status == cp_model.OPTIMAL else "feasible",
                optimization_history=[{
                    "iteration": 1,
                    "objective_value": evaluation_result.objective_value,
                    "is_feasible": evaluation_result.is_feasible,
                    "assignments_count": len(solution_assignments)
                }],
                additional_metrics={
                    "solver_status": solver.StatusName(status),
                    "assignments_count": len(solution_assignments),
                    "total_cost": total_cost_value,
                    "total_quality": total_quality_value,
                    "constraint_violations": len(evaluation_result.constraint_violations),
                    "solver_time": solve_time,
                    "num_variables": len(assignments)
                }
            )
        
        else:
            # No solution found
            self.log_message('warning', f"No solution found. Solver status: {solver.StatusName(status)}")
            
            # Return empty solution
            return OptimizationResult(
                best_solution=[],
                best_value=float('inf'),
                is_feasible=False,
                iterations=1,
                evaluations=0,
                runtime_seconds=solve_time,
                convergence_achieved=False,
                termination_reason=f"no_solution_{solver.StatusName(status).lower()}",
                optimization_history=[],
                additional_metrics={
                    "solver_status": solver.StatusName(status),
                    "solver_time": solve_time,
                    "num_variables": len(assignments)
                }
            )

    def plot_optimization_results(self, result: OptimizationResult, problem: BaseProblem, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of the optimization results.
        """
        if not result.best_solution:
            print("No solution to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('School Resource Allocation Optimization Results', fontsize=16, fontweight='bold')

        # Plot 1: Teacher Workload Distribution
        ax1 = axes[0, 0]
        teacher_workload = {}
        for assignment in result.best_solution:
            if assignment.teacher_id not in teacher_workload:
                teacher_workload[assignment.teacher_id] = 0
            teacher_workload[assignment.teacher_id] += 1

        teachers = list(teacher_workload.keys())
        workloads = list(teacher_workload.values())

        bars1 = ax1.bar(range(len(teachers)), workloads, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Teachers')
        ax1.set_ylabel('Hours Assigned')
        ax1.set_title('Teacher Workload Distribution')
        ax1.set_xticks(range(len(teachers)))
        ax1.set_xticklabels([t[:6] for t in teachers], rotation=45)

        # Add value labels on bars
        for bar, workload in zip(bars1, workloads):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(workload), ha='center', va='bottom')

        # Plot 2: Room Utilization
        ax2 = axes[0, 1]
        room_utilization = {}
        for assignment in result.best_solution:
            if assignment.room_id not in room_utilization:
                room_utilization[assignment.room_id] = 0
            room_utilization[assignment.room_id] += 1

        rooms = list(room_utilization.keys())
        utilizations = list(room_utilization.values())

        bars2 = ax2.bar(range(len(rooms)), utilizations, color='lightgreen', alpha=0.7)
        ax2.set_xlabel('Classrooms')
        ax2.set_ylabel('Hours Used')
        ax2.set_title('Classroom Utilization')
        ax2.set_xticks(range(len(rooms)))
        ax2.set_xticklabels([r[:6] for r in rooms], rotation=45)

        # Add value labels on bars
        for bar, util in zip(bars2, utilizations):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(util), ha='center', va='bottom')

        # Plot 3: Subject Coverage
        ax3 = axes[1, 0]
        subject_hours = {}
        for assignment in result.best_solution:
            if assignment.subject_id not in subject_hours:
                subject_hours[assignment.subject_id] = 0
            subject_hours[assignment.subject_id] += 1

        # Compare with required hours
        subjects = list(problem.subjects.keys())
        assigned_hours = [subject_hours.get(s, 0) for s in subjects]
        required_hours = [problem.subjects[s].required_hours_per_week for s in subjects]

        x = np.arange(len(subjects))
        width = 0.35

        bars3a = ax3.bar(x - width/2, assigned_hours, width, label='Assigned', color='orange', alpha=0.7)
        bars3b = ax3.bar(x + width/2, required_hours, width, label='Required', color='red', alpha=0.7)

        ax3.set_xlabel('Subjects')
        ax3.set_ylabel('Hours per Week')
        ax3.set_title('Subject Hour Coverage')
        ax3.set_xticks(x)
        ax3.set_xticklabels([s[:6] for s in subjects], rotation=45)
        ax3.legend()

        # Plot 4: Schedule Heatmap (simplified)
        ax4 = axes[1, 1]

        # Create a simplified schedule matrix (teachers vs time slots)
        time_slots = list(problem.time_slots.keys())
        teachers = list(problem.teachers.keys())

        # Create matrix: rows = teachers, cols = time slots
        schedule_matrix = np.zeros((len(teachers), min(len(time_slots), 40)))  # Limit to 40 slots for visibility

        for assignment in result.best_solution:
            if assignment.time_slot_id < schedule_matrix.shape[1]:
                teacher_idx = teachers.index(assignment.teacher_id)
                schedule_matrix[teacher_idx, assignment.time_slot_id] = 1

        im = ax4.imshow(schedule_matrix, cmap='Blues', aspect='auto')
        ax4.set_xlabel('Time Slots')
        ax4.set_ylabel('Teachers')
        ax4.set_title('Schedule Overview (Blue = Assigned)')
        ax4.set_yticks(range(len(teachers)))
        ax4.set_yticklabels([t[:8] for t in teachers])

        # Add colorbar
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

        # Print summary statistics
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Total assignments: {len(result.best_solution)}")
        print(f"Objective value: {result.best_value:.2f}")
        print(f"Feasible solution: {result.is_feasible}")
        print(f"Solve time: {result.runtime_seconds:.2f} seconds")
        print(f"Solver status: {result.additional_metrics.get('solver_status', 'Unknown')}")

        if result.additional_metrics:
            print(f"Total cost: ${result.additional_metrics.get('total_cost', 0):.2f}")
            print(f"Constraint violations: {result.additional_metrics.get('constraint_violations', 0)}")

        print("\nTeacher Workload:")
        for teacher_id, hours in teacher_workload.items():
            teacher_name = problem.teachers[teacher_id].name if hasattr(problem, 'teachers') else teacher_id
            print(f"  {teacher_name}: {hours} hours")

        print("\nSubject Coverage:")
        for subject_id in subjects:
            assigned = subject_hours.get(subject_id, 0)
            required = problem.subjects[subject_id].required_hours_per_week
            status = "✓" if assigned >= required else "✗"
            subject_name = problem.subjects[subject_id].name if hasattr(problem, 'subjects') else subject_id
            print(f"  {status} {subject_name}: {assigned}/{required} hours")

        print("="*60)
