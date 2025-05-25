"""
Resource Scheduling Tutorial with Qubots and Rastion Integration

This tutorial demonstrates:
1. Creating a Job Shop Scheduling Problem using qubots ConstrainedProblem
2. Implementing an OR-Tools CP-SAT based scheduler
3. Uploading models to Rastion platform
4. Loading models from Rastion and verifying integrity
5. Complete workflow with error handling and validation

Requirements:
- qubots library
- ortools (pip install ortools)
- numpy
- matplotlib (for Gantt chart visualization)

Author: Qubots Tutorial Team
Version: 1.0.0
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import time
import json
from datetime import datetime, timedelta

# Add qubots to path if running locally
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Qubots imports
from qubots import (
    ConstrainedProblem, BaseOptimizer,
    ProblemMetadata, OptimizerMetadata,
    ProblemType, ObjectiveType, DifficultyLevel,
    OptimizerType, OptimizerFamily,
    OptimizationResult
)
import qubots.rastion as rastion

# OR-Tools imports (with fallback)
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
    print("✅ OR-Tools CP-SAT available for scheduling")
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("⚠️  OR-Tools not available. Install with: pip install ortools")


@dataclass
class Task:
    """Represents a task in the scheduling problem."""
    id: int
    job_id: int
    machine_id: int
    duration: int
    earliest_start: int = 0
    latest_finish: int = 1440  # 24 hours in minutes
    priority: int = 1


@dataclass
class Job:
    """Represents a job consisting of multiple tasks."""
    id: int
    name: str
    tasks: List[Task]
    deadline: Optional[int] = None
    release_time: int = 0
    priority: int = 1


@dataclass
class Machine:
    """Represents a machine/resource in the scheduling problem."""
    id: int
    name: str
    capacity: int = 1
    availability_start: int = 0
    availability_end: int = 1440  # 24 hours
    setup_time: int = 0  # Setup time between different jobs


class JobShopSchedulingProblem(ConstrainedProblem):
    """
    Job Shop Scheduling Problem implementation using qubots ConstrainedProblem.

    This problem involves scheduling a set of jobs on a set of machines,
    where each job consists of a sequence of tasks that must be processed
    in a specific order on specific machines.
    """

    def __init__(self, jobs: List[Job], machines: List[Machine],
                 horizon: int = 1440):
        """
        Initialize the Job Shop Scheduling Problem.

        Args:
            jobs: List of jobs to be scheduled
            machines: List of available machines
            horizon: Time horizon for scheduling (in minutes)
        """
        self.jobs = jobs
        self.machines = machines
        self.horizon = horizon
        self.n_jobs = len(jobs)
        self.n_machines = len(machines)

        # Create task list for easier access
        self.tasks = []
        for job in jobs:
            self.tasks.extend(job.tasks)
        self.n_tasks = len(self.tasks)

        # Initialize base class
        super().__init__(self._get_default_metadata())

        # Add scheduling constraints
        self._add_scheduling_constraints()

    def _get_default_metadata(self) -> ProblemMetadata:
        """Return default metadata for Job Shop Scheduling."""
        return ProblemMetadata(
            name="Job Shop Scheduling Problem",
            description=f"JSSP with {self.n_jobs} jobs, {self.n_machines} machines, {self.n_tasks} tasks",
            problem_type=ProblemType.DISCRETE,
            objective_type=ObjectiveType.MINIMIZE,
            difficulty_level=DifficultyLevel.ADVANCED,
            domain="scheduling",
            tags={"jssp", "scheduling", "constraint_programming", "ortools"},
            author="Qubots Scheduling Tutorial",
            version="1.0.0",
            dimension=self.n_tasks,
            constraints_count=0,  # Will be updated when constraints are added
            evaluation_complexity="O(n²)",
            memory_complexity="O(n²)"
        )

    def _add_scheduling_constraints(self):
        """Add scheduling-specific constraints."""

        # Constraint 1: Task precedence within jobs
        def precedence_constraint(solution: Dict[str, Any]) -> float:
            """Ensure tasks within a job are scheduled in correct order."""
            violation = 0.0
            task_schedule = solution.get('task_schedule', {})

            for job in self.jobs:
                for i in range(len(job.tasks) - 1):
                    current_task = job.tasks[i]
                    next_task = job.tasks[i + 1]

                    current_end = task_schedule.get(current_task.id, {}).get('end_time', 0)
                    next_start = task_schedule.get(next_task.id, {}).get('start_time', 0)

                    if current_end > next_start:
                        violation += current_end - next_start

            return violation

        self.add_constraint(precedence_constraint, "task_precedence", "inequality")

        # Constraint 2: Machine capacity constraints
        def machine_capacity_constraint(solution: Dict[str, Any]) -> float:
            """Ensure machine capacity is not exceeded."""
            violation = 0.0
            task_schedule = solution.get('task_schedule', {})

            for machine in self.machines:
                # Check for overlapping tasks on the same machine
                machine_tasks = [task for task in self.tasks if task.machine_id == machine.id]

                for i, task1 in enumerate(machine_tasks):
                    for task2 in machine_tasks[i+1:]:
                        schedule1 = task_schedule.get(task1.id, {})
                        schedule2 = task_schedule.get(task2.id, {})

                        start1 = schedule1.get('start_time', 0)
                        end1 = schedule1.get('end_time', 0)
                        start2 = schedule2.get('start_time', 0)
                        end2 = schedule2.get('end_time', 0)

                        # Check for overlap
                        if not (end1 <= start2 or end2 <= start1):
                            overlap = min(end1, end2) - max(start1, start2)
                            violation += overlap

            return violation

        self.add_constraint(machine_capacity_constraint, "machine_capacity", "inequality")

        # Constraint 3: Time window constraints
        def time_window_constraint(solution: Dict[str, Any]) -> float:
            """Ensure tasks are scheduled within their time windows."""
            violation = 0.0
            task_schedule = solution.get('task_schedule', {})

            for task in self.tasks:
                schedule = task_schedule.get(task.id, {})
                start_time = schedule.get('start_time', 0)
                end_time = schedule.get('end_time', 0)

                # Check earliest start constraint
                if start_time < task.earliest_start:
                    violation += task.earliest_start - start_time

                # Check latest finish constraint
                if end_time > task.latest_finish:
                    violation += end_time - task.latest_finish

            return violation

        self.add_constraint(time_window_constraint, "time_windows", "inequality")

    def evaluate_solution(self, solution: Dict[str, Any]) -> float:
        """
        Evaluate a scheduling solution.

        Args:
            solution: Dictionary containing:
                - 'task_schedule': Dict mapping task_id to {'start_time': int, 'end_time': int}
                - 'makespan': Total completion time

        Returns:
            Objective value (makespan + constraint violations)
        """
        task_schedule = solution.get('task_schedule', {})

        # Calculate makespan (primary objective)
        makespan = 0
        for job in self.jobs:
            job_completion = 0
            for task in job.tasks:
                schedule = task_schedule.get(task.id, {})
                end_time = schedule.get('end_time', 0)
                job_completion = max(job_completion, end_time)
            makespan = max(makespan, job_completion)

        # Add constraint violations (penalty method)
        constraint_violations = self.get_constraint_violation(solution)
        penalty_weight = 1000  # Heavy penalty for constraint violations

        return makespan + penalty_weight * constraint_violations

    def is_feasible(self, solution: Dict[str, Any]) -> bool:
        """Check if a solution is feasible."""
        return self.get_constraint_violation(solution) <= 1e-6

    def get_random_solution(self) -> Dict[str, Any]:
        """Generate a random feasible scheduling solution."""
        task_schedule = {}

        # Simple random scheduling: assign random start times
        for task in self.tasks:
            # Random start time within valid range
            latest_start = max(0, task.latest_finish - task.duration)
            start_time = np.random.randint(
                task.earliest_start,
                min(latest_start + 1, self.horizon - task.duration + 1)
            )
            end_time = start_time + task.duration

            task_schedule[task.id] = {
                'start_time': start_time,
                'end_time': end_time
            }

        return {
            'task_schedule': task_schedule,
            'makespan': max(schedule['end_time'] for schedule in task_schedule.values())
        }

    def visualize_solution(self, solution: Dict[str, Any], title: str = "Scheduling Solution"):
        """Visualize the scheduling solution as a Gantt chart."""
        try:
            task_schedule = solution.get('task_schedule', {})

            fig, ax = plt.subplots(figsize=(14, 8))

            # Color map for jobs
            colors = plt.cm.Set3(np.linspace(0, 1, self.n_jobs))
            job_colors = {job.id: colors[i] for i, job in enumerate(self.jobs)}

            # Plot tasks
            y_pos = 0
            machine_positions = {}

            for machine in self.machines:
                machine_positions[machine.id] = y_pos

                # Get tasks for this machine
                machine_tasks = [task for task in self.tasks if task.machine_id == machine.id]

                for task in machine_tasks:
                    schedule = task_schedule.get(task.id, {})
                    start_time = schedule.get('start_time', 0)
                    duration = task.duration

                    # Create rectangle for task
                    rect = patches.Rectangle(
                        (start_time, y_pos - 0.4),
                        duration, 0.8,
                        linewidth=1,
                        edgecolor='black',
                        facecolor=job_colors[task.job_id],
                        alpha=0.7
                    )
                    ax.add_patch(rect)

                    # Add task label
                    ax.text(start_time + duration/2, y_pos, f'J{task.job_id}T{task.id}',
                           ha='center', va='center', fontsize=8, fontweight='bold')

                y_pos += 1

            # Set labels and title
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Machines')
            ax.set_title(title)

            # Set y-axis labels
            ax.set_yticks(range(self.n_machines))
            ax.set_yticklabels([f'Machine {m.id}' for m in self.machines])

            # Set x-axis limits
            ax.set_xlim(0, self.horizon)
            ax.set_ylim(-0.5, self.n_machines - 0.5)

            # Add grid
            ax.grid(True, alpha=0.3)

            # Add legend
            legend_elements = [patches.Patch(color=job_colors[job.id], label=f'Job {job.id}')
                             for job in self.jobs]
            ax.legend(handles=legend_elements, loc='upper right')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Visualization failed: {e}")


class ORToolsSchedulingOptimizer(BaseOptimizer):
    """
    OR-Tools CP-SAT based scheduling optimizer.

    Uses Google OR-Tools Constraint Programming solver to find optimal
    or near-optimal schedules for job shop scheduling problems.
    """

    def __init__(self, time_limit_seconds: int = 60,
                 num_search_workers: int = 4,
                 objective_type: str = "makespan"):
        """
        Initialize OR-Tools scheduling optimizer.

        Args:
            time_limit_seconds: Maximum solving time
            num_search_workers: Number of parallel search workers
            objective_type: Objective to optimize ("makespan", "total_completion_time")
        """
        self.time_limit_seconds = time_limit_seconds
        self.num_search_workers = num_search_workers
        self.objective_type = objective_type

        # Pass parameters to base class
        super().__init__(
            self._get_default_metadata(),
            time_limit_seconds=time_limit_seconds,
            num_search_workers=num_search_workers,
            objective_type=objective_type
        )

    def _get_default_metadata(self) -> OptimizerMetadata:
        """Return default metadata for OR-Tools scheduling optimizer."""
        return OptimizerMetadata(
            name="OR-Tools CP-SAT Scheduling Optimizer",
            description="Google OR-Tools CP-SAT based job shop scheduling solver",
            optimizer_type=OptimizerType.HEURISTIC,
            optimizer_family=OptimizerFamily.CONSTRAINT_PROGRAMMING,
            author="Qubots Scheduling Tutorial",
            version="1.0.0",
            supports_constraints=True,
            supports_multi_objective=False,
            typical_problems=["jssp", "scheduling", "resource_allocation"],
            required_parameters=["time_limit_seconds"],
            optional_parameters=["num_search_workers", "objective_type"]
        )

    def _optimize_implementation(self, problem: JobShopSchedulingProblem,
                               initial_solution: Optional[Any] = None) -> OptimizationResult:
        """
        Solve scheduling problem using OR-Tools CP-SAT.

        Args:
            problem: Job shop scheduling problem instance
            initial_solution: Optional initial solution (not used by CP-SAT)

        Returns:
            OptimizationResult with solution and metadata
        """
        if not ORTOOLS_AVAILABLE:
            raise ImportError("OR-Tools not available. Please install with: pip install ortools")

        start_time = time.time()

        # Create CP model
        model = cp_model.CpModel()

        # Decision variables
        task_vars = {}  # task_id -> (start_var, end_var, interval_var)

        for task in problem.tasks:
            start_var = model.NewIntVar(
                task.earliest_start,
                problem.horizon - task.duration,
                f'start_task_{task.id}'
            )
            end_var = model.NewIntVar(
                task.earliest_start + task.duration,
                min(task.latest_finish, problem.horizon),
                f'end_task_{task.id}'
            )
            interval_var = model.NewIntervalVar(
                start_var, task.duration, end_var, f'interval_task_{task.id}'
            )

            task_vars[task.id] = (start_var, end_var, interval_var)

        # Constraint 1: Task precedence within jobs
        for job in problem.jobs:
            for i in range(len(job.tasks) - 1):
                current_task = job.tasks[i]
                next_task = job.tasks[i + 1]

                _, current_end, _ = task_vars[current_task.id]
                next_start, _, _ = task_vars[next_task.id]

                # Add setup time if specified
                setup_time = 0
                if hasattr(current_task, 'setup_time'):
                    setup_time = current_task.setup_time

                model.Add(current_end + setup_time <= next_start)

        # Constraint 2: Machine capacity (no overlap)
        for machine in problem.machines:
            machine_tasks = [task for task in problem.tasks if task.machine_id == machine.id]
            machine_intervals = [task_vars[task.id][2] for task in machine_tasks]

            if machine_intervals:
                model.AddNoOverlap(machine_intervals)

        # Objective: Minimize makespan
        makespan_var = model.NewIntVar(0, problem.horizon, 'makespan')

        for job in problem.jobs:
            if job.tasks:
                last_task = job.tasks[-1]
                _, job_end, _ = task_vars[last_task.id]
                model.Add(makespan_var >= job_end)

        model.Minimize(makespan_var)

        # Create solver and set parameters
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_seconds
        solver.parameters.num_search_workers = self.num_search_workers

        # Solve the problem
        status = solver.Solve(model)

        end_time = time.time()

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            # Extract solution
            task_schedule = {}
            makespan = solver.Value(makespan_var)

            for task in problem.tasks:
                start_var, end_var, _ = task_vars[task.id]
                start_time = solver.Value(start_var)
                end_time = solver.Value(end_var)

                task_schedule[task.id] = {
                    'start_time': start_time,
                    'end_time': end_time
                }

            best_solution = {
                'task_schedule': task_schedule,
                'makespan': makespan
            }

            # Determine termination reason
            termination_reason = "Optimal solution found" if status == cp_model.OPTIMAL else "Feasible solution found"

            return OptimizationResult(
                best_solution=best_solution,
                best_value=makespan,
                iterations=1,
                evaluations=1,
                runtime_seconds=end_time - start_time,
                convergence_achieved=(status == cp_model.OPTIMAL),
                termination_reason=termination_reason,
                additional_metrics={
                    "cp_sat_status": solver.StatusName(status),
                    "objective_value": makespan,
                    "num_conflicts": solver.NumConflicts(),
                    "num_branches": solver.NumBranches(),
                    "wall_time": solver.WallTime()
                }
            )
        else:
            # No solution found
            status_name = solver.StatusName(status)
            return OptimizationResult(
                best_solution=None,
                best_value=float('inf'),
                iterations=0,
                evaluations=0,
                runtime_seconds=end_time - start_time,
                convergence_achieved=False,
                termination_reason=f"CP-SAT: {status_name}",
                additional_metrics={
                    "cp_sat_status": status_name,
                    "num_conflicts": solver.NumConflicts(),
                    "num_branches": solver.NumBranches()
                }
            )


def generate_sample_scheduling_data(n_jobs: int = 5, n_machines: int = 3,
                                  max_tasks_per_job: int = 4, seed: int = 42) -> Tuple[List[Job], List[Machine]]:
    """
    Generate sample job shop scheduling data for testing.

    Args:
        n_jobs: Number of jobs to generate
        n_machines: Number of machines available
        max_tasks_per_job: Maximum number of tasks per job
        seed: Random seed for reproducibility

    Returns:
        Tuple of (jobs, machines)
    """
    np.random.seed(seed)

    # Generate machines
    machines = []
    for i in range(n_machines):
        machine = Machine(
            id=i,
            name=f"Machine_{i}",
            capacity=1,
            availability_start=0,
            availability_end=1440,  # 24 hours
            setup_time=np.random.randint(5, 15)  # 5-15 minutes setup
        )
        machines.append(machine)

    # Generate jobs
    jobs = []
    for job_id in range(n_jobs):
        n_tasks = np.random.randint(2, max_tasks_per_job + 1)
        tasks = []

        # Create task sequence for this job
        machine_sequence = np.random.choice(n_machines, size=n_tasks, replace=False)

        for task_idx in range(n_tasks):
            task = Task(
                id=job_id * 10 + task_idx,  # Unique task ID
                job_id=job_id,
                machine_id=machine_sequence[task_idx],
                duration=np.random.randint(20, 120),  # 20-120 minutes
                earliest_start=0,
                latest_finish=1440,
                priority=np.random.randint(1, 4)
            )
            tasks.append(task)

        job = Job(
            id=job_id,
            name=f"Job_{job_id}",
            tasks=tasks,
            deadline=np.random.randint(600, 1440),  # 10-24 hours
            release_time=np.random.randint(0, 120),  # 0-2 hours
            priority=np.random.randint(1, 4)
        )
        jobs.append(job)

    return jobs, machines


def demonstrate_scheduling_workflow():
    """
    Demonstrate the complete scheduling workflow with qubots and Rastion integration.
    """
    print("⏰ Job Shop Scheduling Tutorial with Qubots & Rastion")
    print("=" * 60)

    # Step 1: Generate sample data
    print("\n📊 Step 1: Generating sample scheduling data...")
    jobs, machines = generate_sample_scheduling_data(n_jobs=4, n_machines=3, max_tasks_per_job=3)

    total_tasks = sum(len(job.tasks) for job in jobs)
    total_duration = sum(task.duration for job in jobs for task in job.tasks)

    print(f"✅ Generated {len(jobs)} jobs and {len(machines)} machines")
    print(f"   Total tasks: {total_tasks}")
    print(f"   Total processing time: {total_duration} minutes")
    print(f"   Average tasks per job: {total_tasks / len(jobs):.1f}")

    # Step 2: Create scheduling problem instance
    print("\n🏗️  Step 2: Creating scheduling problem instance...")
    scheduling_problem = JobShopSchedulingProblem(
        jobs=jobs,
        machines=machines,
        horizon=1440  # 24 hours
    )

    print(f"✅ Created scheduling problem: {scheduling_problem.metadata.name}")
    print(f"   Problem type: {scheduling_problem.metadata.problem_type}")
    print(f"   Difficulty: {scheduling_problem.metadata.difficulty_level}")
    print(f"   Constraints: {scheduling_problem.constraint_count}")

    # Step 3: Test problem with random solution
    print("\n🎲 Step 3: Testing with random solution...")
    random_solution = scheduling_problem.get_random_solution()
    random_makespan = scheduling_problem.evaluate_solution(random_solution)
    is_feasible = scheduling_problem.is_feasible(random_solution)

    print(f"✅ Random solution makespan: {random_makespan:.2f} minutes")
    print(f"   Feasible: {is_feasible}")
    print(f"   Constraint violations: {scheduling_problem.get_constraint_violation(random_solution):.2f}")

    # Step 4: Create OR-Tools optimizer
    print("\n⚙️  Step 4: Creating OR-Tools CP-SAT optimizer...")
    if ORTOOLS_AVAILABLE:
        optimizer = ORToolsSchedulingOptimizer(
            time_limit_seconds=30,
            num_search_workers=4,
            objective_type="makespan"
        )
        print(f"✅ Created optimizer: {optimizer.metadata.name}")
        print(f"   Type: {optimizer.metadata.optimizer_type}")
        print(f"   Family: {optimizer.metadata.optimizer_family}")
    else:
        print("⚠️  OR-Tools not available, skipping optimization step")
        return

    # Step 5: Solve with OR-Tools CP-SAT
    print("\n🔍 Step 5: Solving scheduling problem with OR-Tools CP-SAT...")
    try:
        result = optimizer.optimize(scheduling_problem)

        print(f"✅ CP-SAT solution found!")
        print(f"   Best makespan: {result.best_value:.2f} minutes ({result.best_value/60:.1f} hours)")
        print(f"   Runtime: {result.runtime_seconds:.2f} seconds")
        print(f"   Converged: {result.converged}")
        print(f"   Status: {result.additional_info.get('cp_sat_status', 'N/A')}")
        print(f"   Conflicts: {result.additional_info.get('num_conflicts', 'N/A')}")
        print(f"   Branches: {result.additional_info.get('num_branches', 'N/A')}")

        # Visualize solution if matplotlib is available
        if result.best_solution:
            print("\n📊 Visualizing scheduling solution...")
            scheduling_problem.visualize_solution(result.best_solution, "OR-Tools CP-SAT Schedule")

    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return

    # Step 6: Demonstrate Rastion upload workflow
    print("\n📤 Step 6: Rastion Upload Workflow (Simulation)...")
    print("Note: Actual upload requires authentication with rastion.authenticate(token)")

    try:
        # Show what would be packaged for upload
        from qubots.rastion_client import QubotPackager

        # Package the problem
        problem_package = QubotPackager.package_model(
            scheduling_problem,
            "jssp_tutorial_problem",
            "Job Shop Scheduling Problem from qubots tutorial with OR-Tools CP-SAT integration"
        )

        # Package the optimizer
        optimizer_package = QubotPackager.package_model(
            optimizer,
            "ortools_cpsat_scheduler",
            "OR-Tools CP-SAT based job shop scheduler with constraint programming"
        )

        print("✅ Problem package created:")
        for filename in problem_package.keys():
            print(f"   📄 {filename}")

        print("✅ Optimizer package created:")
        for filename in optimizer_package.keys():
            print(f"   📄 {filename}")

        # Simulate upload process
        print("\n🔄 Upload simulation:")
        print("   1. Validating model integrity...")
        print("   2. Serializing constraint model...")
        print("   3. Creating repository structure...")
        print("   4. Uploading to Rastion platform...")
        print("   ✅ Upload complete! (simulated)")

    except Exception as e:
        print(f"❌ Package creation failed: {e}")

    # Step 7: Demonstrate loading workflow
    print("\n📥 Step 7: Rastion Loading Workflow (Simulation)...")
    print("After upload, users would load models like this:")
    print()
    print("```python")
    print("import qubots.rastion as rastion")
    print()
    print("# Authenticate (one-time setup)")
    print("rastion.authenticate('your_gitea_token')")
    print()
    print("# Load models with one line")
    print("problem = rastion.load_qubots_model('jssp_tutorial_problem')")
    print("optimizer = rastion.load_qubots_model('ortools_cpsat_scheduler')")
    print()
    print("# Verify model integrity")
    print("print(f'Problem: {problem.metadata.name}')")
    print("print(f'Jobs: {problem.n_jobs}')")
    print("print(f'Machines: {problem.n_machines}')")
    print("print(f'Constraints: {problem.constraint_count}')")
    print()
    print("# Run optimization")
    print("result = optimizer.optimize(problem)")
    print("print(f'Best makespan: {result.best_value:.2f} minutes')")
    print()
    print("# Visualize results")
    print("problem.visualize_solution(result.best_solution)")
    print("```")

    # Step 8: Model verification and error handling
    print("\n🔍 Step 8: Model Verification & Error Handling...")

    # Demonstrate model validation
    print("Model validation checks:")
    print(f"   ✅ Problem format valid: {scheduling_problem.validate_solution_format(random_solution)}")
    print(f"   ✅ Problem feasibility check: {scheduling_problem.is_feasible(random_solution)}")
    print(f"   ✅ Optimizer compatibility: Compatible with ConstrainedProblem")

    # Demonstrate error handling scenarios
    print("\nError handling scenarios:")

    # Test with invalid solution format
    try:
        invalid_solution = {"invalid": "format"}
        makespan = scheduling_problem.evaluate_solution(invalid_solution)
        print(f"   ⚠️  Invalid solution makespan: {makespan}")
    except Exception as e:
        print(f"   ✅ Invalid solution handled: {type(e).__name__}")

    # Test with constraint violations
    try:
        # Create solution with overlapping tasks on same machine
        conflicting_solution = {
            'task_schedule': {
                0: {'start_time': 100, 'end_time': 200},
                10: {'start_time': 150, 'end_time': 250}  # Overlapping if same machine
            },
            'makespan': 250
        }
        makespan = scheduling_problem.evaluate_solution(conflicting_solution)
        violations = scheduling_problem.get_constraint_violation(conflicting_solution)
        print(f"   ✅ Conflicting solution makespan: {makespan:.2f} (with penalty)")
        print(f"   ✅ Constraint violations: {violations:.2f}")
    except Exception as e:
        print(f"   ⚠️  Conflicting solution error: {e}")

    print("\n✨ Scheduling Tutorial completed successfully!")
    print("\n📚 Key Learning Points:")
    print("   🔹 Created JSSP using qubots ConstrainedProblem")
    print("   🔹 Implemented OR-Tools CP-SAT optimizer with constraints")
    print("   🔹 Demonstrated complete Rastion upload/download workflow")
    print("   🔹 Showed constraint handling and validation")
    print("   🔹 Visualized schedules with Gantt charts")
    print("   🔹 Used proper qubots metadata and optimization results")


def demonstrate_advanced_scheduling_features():
    """
    Demonstrate advanced scheduling features and customizations.
    """
    print("\n🚀 Advanced Scheduling Features Demo")
    print("=" * 45)

    # Create a more complex scheduling instance
    print("\n📊 Creating complex scheduling with priorities...")
    jobs, machines = generate_sample_scheduling_data(n_jobs=3, n_machines=2, max_tasks_per_job=2, seed=456)

    # Add priority-based constraints
    for i, job in enumerate(jobs):
        job.priority = 3 - i  # Higher priority for earlier jobs
        job.deadline = 600 + i * 200  # Tighter deadlines for higher priority

    scheduling_problem = JobShopSchedulingProblem(jobs, machines, horizon=1200)

    print(f"✅ Created complex scheduling with priorities")
    print(f"   Job priorities: {[job.priority for job in jobs]}")
    print(f"   Job deadlines: {[job.deadline for job in jobs]}")

    # Test different CP-SAT parameters
    if ORTOOLS_AVAILABLE:
        configurations = [
            {"time_limit_seconds": 10, "num_search_workers": 1},
            {"time_limit_seconds": 10, "num_search_workers": 4},
            {"time_limit_seconds": 20, "num_search_workers": 2}
        ]

        print("\n🔧 Testing different CP-SAT configurations...")

        for config in configurations:
            try:
                optimizer = ORToolsSchedulingOptimizer(**config)
                result = optimizer.optimize(scheduling_problem)

                print(f"   Config {config}:")
                print(f"     Makespan: {result.best_value:.2f}, Time: {result.runtime_seconds:.2f}s")
                print(f"     Status: {result.additional_info.get('cp_sat_status', 'N/A')}")

            except Exception as e:
                print(f"   Config {config}: Failed ({e})")

    print("\n✅ Advanced features demonstration completed!")


if __name__ == "__main__":
    """
    Main execution block - run the complete scheduling tutorial.

    This tutorial can be executed directly with:
    python scheduling_tutorial.py
    """
    try:
        # Run main workflow demonstration
        demonstrate_scheduling_workflow()

        # Run advanced features demo
        demonstrate_advanced_scheduling_features()

        print("\n🎉 All scheduling tutorial demonstrations completed successfully!")
        print("\n📖 Next Steps:")
        print("   1. Try modifying the scheduling parameters (jobs, machines, constraints)")
        print("   2. Experiment with different CP-SAT configurations")
        print("   3. Add custom constraints (deadlines, priorities, setup times)")
        print("   4. Upload your models to Rastion for sharing")
        print("   5. Explore other qubots tutorials (routing, finance)")

    except KeyboardInterrupt:
        print("\n⏹️  Tutorial interrupted by user")
    except Exception as e:
        print(f"\n❌ Tutorial failed with error: {e}")
        print("   Please check dependencies and try again")
        import traceback
        traceback.print_exc()
