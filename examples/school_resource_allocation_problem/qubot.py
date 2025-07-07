"""
School Resource Allocation Problem for Qubots Framework

This problem optimizes the allocation of teachers, classrooms, and time slots
to minimize scheduling conflicts while maximizing resource utilization and
teacher-subject compatibility.

Author: Qubots Framework
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import random
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import io

# Import qubots framework components
try:
    from qubots.base_problem import BaseProblem, ProblemMetadata, ProblemType, ObjectiveType, DifficultyLevel, EvaluationResult
except ImportError:
    # Fallback for local development
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from qubots.base_problem import BaseProblem, ProblemMetadata, ProblemType, ObjectiveType, DifficultyLevel, EvaluationResult


@dataclass
class Teacher:
    """Represents a teacher with qualifications and constraints."""
    teacher_id: str
    name: str
    subjects: List[str]  # Subjects the teacher can teach
    max_hours_per_day: int = 8
    max_hours_per_week: int = 40
    preferred_time_slots: List[int] = field(default_factory=list)  # 0-based time slot indices
    unavailable_time_slots: List[int] = field(default_factory=list)
    experience_level: int = 1  # 1-5 scale
    hourly_cost: float = 50.0


@dataclass
class Classroom:
    """Represents a classroom with capacity and equipment."""
    room_id: str
    name: str
    capacity: int
    room_type: str  # "standard", "lab", "computer", "gym", "auditorium"
    equipment: List[str] = field(default_factory=list)
    hourly_cost: float = 10.0


@dataclass
class Subject:
    """Represents a subject with requirements."""
    subject_id: str
    name: str
    required_hours_per_week: int
    min_class_size: int = 1
    max_class_size: int = 30
    required_room_type: str = "standard"
    required_equipment: List[str] = field(default_factory=list)
    priority: int = 1  # 1-5 scale, higher is more important


@dataclass
class TimeSlot:
    """Represents a time slot in the schedule."""
    slot_id: int
    day: str  # "Monday", "Tuesday", etc.
    start_time: str  # "08:00"
    end_time: str  # "09:00"
    duration_hours: float = 1.0


@dataclass
class Assignment:
    """Represents a teacher-subject-classroom-timeslot assignment."""
    teacher_id: str
    subject_id: str
    room_id: str
    time_slot_id: int
    student_count: int = 20


class SchoolResourceAllocationProblem(BaseProblem):
    """
    School Resource Allocation Optimization Problem.
    
    This problem optimizes the assignment of teachers to subjects, classrooms, and time slots
    to minimize conflicts and costs while maximizing educational quality and resource utilization.
    
    Objectives:
    1. Minimize scheduling conflicts (hard constraints)
    2. Minimize total cost (teacher + classroom costs)
    3. Maximize teacher-subject compatibility
    4. Maximize resource utilization
    5. Respect teacher preferences and availability
    
    Solution Format:
        List of Assignment objects representing the complete schedule
    """
    
    def __init__(self,
                 csv_data: Optional[str] = None,
                 dataset_content: Optional[str] = None,
                 teachers_data: Optional[str] = None,
                 classrooms_data: Optional[str] = None,
                 subjects_data: Optional[str] = None,
                 time_slots_data: Optional[str] = None,
                 conflict_penalty: float = 1000.0,
                 cost_weight: float = 0.3,
                 compatibility_weight: float = 0.4,
                 utilization_weight: float = 0.3,
                 **kwargs):
        """
        Initialize school resource allocation problem.

        Args:
            csv_data: CSV data containing comprehensive school information
            dataset_content: Pre-loaded dataset content from Rastion platform
            teachers_data: CSV data containing teacher information
            classrooms_data: CSV data containing classroom information
            subjects_data: CSV data containing subject information
            time_slots_data: CSV data containing time slot information
            conflict_penalty: Penalty cost for each scheduling conflict
            cost_weight: Weight for cost minimization objective
            compatibility_weight: Weight for teacher-subject compatibility maximization
            utilization_weight: Weight for resource utilization maximization
            **kwargs: Additional parameters
        """
        # Store parameters
        self.conflict_penalty = conflict_penalty
        self.cost_weight = cost_weight
        self.compatibility_weight = compatibility_weight
        self.utilization_weight = utilization_weight

        # Initialize data structures BEFORE calling super().__init__
        self.teachers: Dict[str, Teacher] = {}
        self.classrooms: Dict[str, Classroom] = {}
        self.subjects: Dict[str, Subject] = {}
        self.time_slots: Dict[int, TimeSlot] = {}

        # Load data BEFORE calling super().__init__
        self._load_data(csv_data, dataset_content, teachers_data, classrooms_data, subjects_data, time_slots_data)

        # Initialize base class after data is loaded
        super().__init__(**kwargs)
    
    def _load_data(self, csv_data: Optional[str], dataset_content: Optional[str], 
                   teachers_data: Optional[str], classrooms_data: Optional[str],
                   subjects_data: Optional[str], time_slots_data: Optional[str]):
        """Load school data from various sources."""
        
        if dataset_content:
            # Load from pre-loaded dataset
            self._load_from_dataset(dataset_content)
        elif csv_data:
            # Load from single CSV
            self._load_from_csv(csv_data)
        elif any([teachers_data, classrooms_data, subjects_data, time_slots_data]):
            # Load from separate CSV files
            self._load_from_separate_csvs(teachers_data, classrooms_data, subjects_data, time_slots_data)
        else:
            # Load default data
            self._load_default_data()
    
    def _load_default_data(self):
        """Load default school data for demonstration."""
        
        # Default teachers
        teachers_data = [
            {"teacher_id": "T001", "name": "Dr. Smith", "subjects": "Mathematics,Physics", "max_hours_per_day": 6, "experience_level": 5, "hourly_cost": 60},
            {"teacher_id": "T002", "name": "Ms. Johnson", "subjects": "English,Literature", "max_hours_per_day": 7, "experience_level": 4, "hourly_cost": 55},
            {"teacher_id": "T003", "name": "Mr. Brown", "subjects": "Chemistry,Biology", "max_hours_per_day": 6, "experience_level": 3, "hourly_cost": 50},
            {"teacher_id": "T004", "name": "Mrs. Davis", "subjects": "History,Geography", "max_hours_per_day": 8, "experience_level": 4, "hourly_cost": 52},
            {"teacher_id": "T005", "name": "Mr. Wilson", "subjects": "Computer Science,Mathematics", "max_hours_per_day": 7, "experience_level": 3, "hourly_cost": 58},
        ]
        
        for t_data in teachers_data:
            teacher = Teacher(
                teacher_id=t_data["teacher_id"],
                name=t_data["name"],
                subjects=t_data["subjects"].split(","),
                max_hours_per_day=t_data["max_hours_per_day"],
                experience_level=t_data["experience_level"],
                hourly_cost=t_data["hourly_cost"]
            )
            self.teachers[teacher.teacher_id] = teacher
        
        # Default classrooms
        classrooms_data = [
            {"room_id": "R101", "name": "Math Room 1", "capacity": 30, "room_type": "standard", "hourly_cost": 8},
            {"room_id": "R102", "name": "Science Lab", "capacity": 25, "room_type": "lab", "hourly_cost": 15},
            {"room_id": "R103", "name": "Computer Lab", "capacity": 28, "room_type": "computer", "hourly_cost": 20},
            {"room_id": "R104", "name": "English Room", "capacity": 32, "room_type": "standard", "hourly_cost": 8},
            {"room_id": "R105", "name": "History Room", "capacity": 35, "room_type": "standard", "hourly_cost": 8},
        ]
        
        for c_data in classrooms_data:
            classroom = Classroom(
                room_id=c_data["room_id"],
                name=c_data["name"],
                capacity=c_data["capacity"],
                room_type=c_data["room_type"],
                hourly_cost=c_data["hourly_cost"]
            )
            self.classrooms[classroom.room_id] = classroom
        
        # Default subjects
        subjects_data = [
            {"subject_id": "S001", "name": "Mathematics", "required_hours_per_week": 5, "max_class_size": 30, "priority": 5},
            {"subject_id": "S002", "name": "English", "required_hours_per_week": 4, "max_class_size": 32, "priority": 5},
            {"subject_id": "S003", "name": "Physics", "required_hours_per_week": 3, "max_class_size": 25, "priority": 4},
            {"subject_id": "S004", "name": "Chemistry", "required_hours_per_week": 3, "max_class_size": 25, "required_room_type": "lab", "priority": 4},
            {"subject_id": "S005", "name": "Computer Science", "required_hours_per_week": 2, "max_class_size": 28, "required_room_type": "computer", "priority": 3},
            {"subject_id": "S006", "name": "History", "required_hours_per_week": 3, "max_class_size": 35, "priority": 3},
        ]
        
        for s_data in subjects_data:
            subject = Subject(
                subject_id=s_data["subject_id"],
                name=s_data["name"],
                required_hours_per_week=s_data["required_hours_per_week"],
                max_class_size=s_data["max_class_size"],
                required_room_type=s_data.get("required_room_type", "standard"),
                priority=s_data["priority"]
            )
            self.subjects[subject.subject_id] = subject
        
        # Default time slots (5 days, 8 hours per day)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        start_times = ["08:00", "09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00"]
        
        slot_id = 0
        for day in days:
            for i, start_time in enumerate(start_times):
                end_hour = int(start_time.split(":")[0]) + 1
                end_time = f"{end_hour:02d}:00"
                
                time_slot = TimeSlot(
                    slot_id=slot_id,
                    day=day,
                    start_time=start_time,
                    end_time=end_time
                )
                self.time_slots[slot_id] = time_slot
                slot_id += 1

    def _load_from_csv(self, csv_data: str):
        """Load data from a single CSV containing all information."""
        # Implementation for loading from single CSV
        # This would parse a comprehensive CSV with all school data
        pass

    def _load_from_separate_csvs(self, teachers_data: Optional[str], classrooms_data: Optional[str],
                                subjects_data: Optional[str], time_slots_data: Optional[str]):
        """Load data from separate CSV files."""
        import io
        import csv

        # Load teachers data
        if teachers_data:
            reader = csv.DictReader(io.StringIO(teachers_data))
            for row in reader:
                teacher = Teacher(
                    teacher_id=row["teacher_id"],
                    name=row["name"],
                    subjects=row["subjects"].split(",") if row["subjects"] else [],
                    max_hours_per_day=int(row.get("max_hours_per_day", 8)),
                    experience_level=int(row.get("experience_level", 3)),
                    hourly_cost=float(row.get("hourly_cost", 50))
                )
                self.teachers[teacher.teacher_id] = teacher

        # Load classrooms data
        if classrooms_data:
            reader = csv.DictReader(io.StringIO(classrooms_data))
            for row in reader:
                classroom = Classroom(
                    room_id=row["room_id"],
                    name=row["name"],
                    capacity=int(row["capacity"]),
                    room_type=row.get("room_type", "standard"),
                    hourly_cost=float(row.get("hourly_cost", 10))
                )
                self.classrooms[classroom.room_id] = classroom

        # Load subjects data
        if subjects_data:
            reader = csv.DictReader(io.StringIO(subjects_data))
            for row in reader:
                subject = Subject(
                    subject_id=row["subject_id"],
                    name=row["name"],
                    required_hours_per_week=int(row["required_hours_per_week"]),
                    max_class_size=int(row.get("max_class_size", 30)),
                    required_room_type=row.get("required_room_type", "standard"),
                    priority=int(row.get("priority", 3))
                )
                self.subjects[subject.subject_id] = subject

        # Load time slots data or use default
        if time_slots_data:
            reader = csv.DictReader(io.StringIO(time_slots_data))
            for row in reader:
                time_slot = TimeSlot(
                    slot_id=int(row["slot_id"]),
                    day=row["day"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    duration_hours=float(row.get("duration_hours", 1.0))
                )
                self.time_slots[time_slot.slot_id] = time_slot
        else:
            # Use default time slots if not provided
            self._load_default_time_slots()

    def _load_default_time_slots(self):
        """Load default time slots (5 days, 8 hours per day)."""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        start_times = ["08:00", "09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00"]

        slot_id = 0
        for day in days:
            for i, start_time in enumerate(start_times):
                end_hour = int(start_time.split(":")[0]) + 1
                end_time = f"{end_hour:02d}:00"

                time_slot = TimeSlot(
                    slot_id=slot_id,
                    day=day,
                    start_time=start_time,
                    end_time=end_time
                )
                self.time_slots[slot_id] = time_slot
                slot_id += 1

    def _load_from_dataset(self, dataset_content: str):
        """Load data from pre-loaded dataset."""
        # Implementation for loading from dataset
        pass

    def _get_default_metadata(self) -> ProblemMetadata:
        """Return default metadata for School Resource Allocation."""
        return ProblemMetadata(
            name="School Resource Allocation Problem",
            description="Optimize teacher-subject-classroom-timeslot assignments to minimize conflicts and costs while maximizing educational quality",
            problem_type=ProblemType.COMBINATORIAL,
            objective_type=ObjectiveType.MINIMIZE,
            difficulty_level=DifficultyLevel.ADVANCED,
            domain="education",
            tags={"scheduling", "resource_allocation", "education", "optimization", "timetabling"},
            author="Qubots Framework",
            version="1.0.0",
            dimension=len(self.teachers) * len(self.subjects) * len(self.time_slots),
            constraints_count=5,  # Multiple constraint types
            evaluation_complexity="O(n³)",
            memory_complexity="O(n²)"
        )

    def evaluate_solution(self, solution: List[Assignment]) -> Union[float, EvaluationResult]:
        """
        Evaluate a school resource allocation solution.

        Args:
            solution: List of Assignment objects representing the schedule

        Returns:
            EvaluationResult with objective value and constraint violations
        """
        start_time = datetime.now()

        # Initialize metrics
        total_cost = 0.0
        conflict_violations = []
        compatibility_score = 0.0
        utilization_score = 0.0

        # Track assignments for conflict detection
        teacher_schedule = {}  # teacher_id -> {time_slot_id: assignment}
        room_schedule = {}     # room_id -> {time_slot_id: assignment}
        subject_hours = {}     # subject_id -> total_hours_assigned

        # Process each assignment
        for assignment in solution:
            # Cost calculation
            teacher = self.teachers.get(assignment.teacher_id)
            classroom = self.classrooms.get(assignment.room_id)

            if teacher and classroom:
                total_cost += teacher.hourly_cost + classroom.hourly_cost

            # Conflict detection
            # Teacher double-booking
            if assignment.teacher_id not in teacher_schedule:
                teacher_schedule[assignment.teacher_id] = {}

            if assignment.time_slot_id in teacher_schedule[assignment.teacher_id]:
                conflict_violations.append(f"Teacher {assignment.teacher_id} double-booked at time slot {assignment.time_slot_id}")
            else:
                teacher_schedule[assignment.teacher_id][assignment.time_slot_id] = assignment

            # Room double-booking
            if assignment.room_id not in room_schedule:
                room_schedule[assignment.room_id] = {}

            if assignment.time_slot_id in room_schedule[assignment.room_id]:
                conflict_violations.append(f"Room {assignment.room_id} double-booked at time slot {assignment.time_slot_id}")
            else:
                room_schedule[assignment.room_id][assignment.time_slot_id] = assignment

            # Subject hours tracking
            if assignment.subject_id not in subject_hours:
                subject_hours[assignment.subject_id] = 0
            subject_hours[assignment.subject_id] += 1

            # Teacher-subject compatibility
            subject = self.subjects.get(assignment.subject_id)
            if teacher and subject and (assignment.subject_id in teacher.subjects or subject.name in teacher.subjects):
                compatibility_score += teacher.experience_level
            else:
                conflict_violations.append(f"Teacher {assignment.teacher_id} not qualified for subject {assignment.subject_id}")

            # Room-subject compatibility
            subject = self.subjects.get(assignment.subject_id)
            if subject and classroom:
                if subject.required_room_type != "standard" and classroom.room_type != subject.required_room_type:
                    conflict_violations.append(f"Subject {assignment.subject_id} requires {subject.required_room_type} room, assigned to {classroom.room_type}")

                if assignment.student_count > classroom.capacity:
                    conflict_violations.append(f"Class size {assignment.student_count} exceeds room capacity {classroom.capacity}")

        # Check subject hour requirements
        for subject_id, subject in self.subjects.items():
            assigned_hours = subject_hours.get(subject_id, 0)
            if assigned_hours < subject.required_hours_per_week:
                conflict_violations.append(f"Subject {subject_id} needs {subject.required_hours_per_week} hours, only {assigned_hours} assigned")

        # Calculate utilization scores
        total_possible_assignments = len(self.teachers) * len(self.time_slots)
        actual_assignments = len(solution)
        utilization_score = actual_assignments / max(total_possible_assignments, 1)

        # Normalize compatibility score
        max_possible_compatibility = len(solution) * 5  # Max experience level is 5
        if max_possible_compatibility > 0:
            compatibility_score = compatibility_score / max_possible_compatibility

        # Calculate final objective value
        penalty_cost = len(conflict_violations) * self.conflict_penalty
        normalized_cost = total_cost / max(len(solution), 1)  # Cost per assignment

        objective_value = (
            penalty_cost +
            self.cost_weight * normalized_cost -
            self.compatibility_weight * compatibility_score * 100 -
            self.utilization_weight * utilization_score * 100
        )

        # Calculate evaluation time
        evaluation_time = (datetime.now() - start_time).total_seconds() * 1000

        return EvaluationResult(
            objective_value=objective_value,
            is_feasible=len(conflict_violations) == 0,
            constraint_violations=conflict_violations,
            evaluation_time_ms=evaluation_time,
            additional_metrics={
                "total_cost": total_cost,
                "conflict_count": len(conflict_violations),
                "compatibility_score": compatibility_score,
                "utilization_score": utilization_score,
                "assignments_count": len(solution),
                "penalty_cost": penalty_cost,
                "subject_coverage": len(subject_hours) / max(len(self.subjects), 1)
            }
        )

    def random_solution(self) -> List[Assignment]:
        """Generate a random solution for the school resource allocation problem."""
        assignments = []

        # Create assignments for each subject's required hours
        for subject_id, subject in self.subjects.items():
            # Find qualified teachers
            qualified_teachers = [
                teacher_id for teacher_id, teacher in self.teachers.items()
                if subject_id in teacher.subjects or subject.name in teacher.subjects
            ]

            if not qualified_teachers:
                continue

            # Find suitable classrooms
            suitable_rooms = [
                room_id for room_id, room in self.classrooms.items()
                if room.room_type == subject.required_room_type or subject.required_room_type == "standard"
            ]

            if not suitable_rooms:
                continue

            # Assign required hours for this subject
            for _ in range(subject.required_hours_per_week):
                teacher_id = random.choice(qualified_teachers)
                room_id = random.choice(suitable_rooms)
                time_slot_id = random.choice(list(self.time_slots.keys()))
                student_count = random.randint(subject.min_class_size, min(subject.max_class_size, self.classrooms[room_id].capacity))

                assignment = Assignment(
                    teacher_id=teacher_id,
                    subject_id=subject_id,
                    room_id=room_id,
                    time_slot_id=time_slot_id,
                    student_count=student_count
                )
                assignments.append(assignment)

        return assignments

    def is_feasible(self, solution: List[Assignment]) -> bool:
        """Check if a solution is feasible (no conflicts)."""
        result = self.evaluate_solution_detailed(solution)
        return result.is_feasible

    def get_solution_summary(self, solution: List[Assignment]) -> Dict[str, Any]:
        """Get a detailed summary of the solution."""
        result = self.evaluate_solution_detailed(solution)

        # Calculate additional statistics
        teacher_workload = {}
        room_utilization = {}
        subject_coverage = {}

        for assignment in solution:
            # Teacher workload
            if assignment.teacher_id not in teacher_workload:
                teacher_workload[assignment.teacher_id] = 0
            teacher_workload[assignment.teacher_id] += 1

            # Room utilization
            if assignment.room_id not in room_utilization:
                room_utilization[assignment.room_id] = 0
            room_utilization[assignment.room_id] += 1

            # Subject coverage
            if assignment.subject_id not in subject_coverage:
                subject_coverage[assignment.subject_id] = 0
            subject_coverage[assignment.subject_id] += 1

        return {
            "total_assignments": len(solution),
            "is_feasible": result.is_feasible,
            "total_cost": result.additional_metrics["total_cost"],
            "conflict_count": result.additional_metrics["conflict_count"],
            "compatibility_score": result.additional_metrics["compatibility_score"],
            "utilization_score": result.additional_metrics["utilization_score"],
            "subject_coverage": result.additional_metrics["subject_coverage"],
            "teacher_workload": teacher_workload,
            "room_utilization": room_utilization,
            "subject_hours_assigned": subject_coverage,
            "constraint_violations": result.constraint_violations,
            "objective_value": result.objective_value
        }

    def get_neighbor_solution(self, solution: List[Assignment], step_size: float = 1.0) -> List[Assignment]:
        """Generate a neighboring solution by making small modifications."""
        if not solution:
            return self.random_solution()

        new_solution = solution.copy()
        num_changes = max(1, int(len(solution) * step_size * 0.1))  # Change up to 10% of assignments

        for _ in range(num_changes):
            if not new_solution:
                break

            # Select random assignment to modify
            idx = random.randint(0, len(new_solution) - 1)
            assignment = new_solution[idx]

            # Choose what to modify
            modification_type = random.choice(["teacher", "room", "time_slot"])

            if modification_type == "teacher":
                # Find alternative qualified teacher
                subject = self.subjects.get(assignment.subject_id)
                if subject:
                    qualified_teachers = [
                        teacher_id for teacher_id, teacher in self.teachers.items()
                        if assignment.subject_id in teacher.subjects or subject.name in teacher.subjects
                    ]
                    if qualified_teachers:
                        new_teacher = random.choice(qualified_teachers)
                        new_solution[idx] = Assignment(
                            teacher_id=new_teacher,
                            subject_id=assignment.subject_id,
                            room_id=assignment.room_id,
                            time_slot_id=assignment.time_slot_id,
                            student_count=assignment.student_count
                        )

            elif modification_type == "room":
                # Find alternative suitable room
                subject = self.subjects.get(assignment.subject_id)
                if subject:
                    suitable_rooms = [
                        room_id for room_id, room in self.classrooms.items()
                        if room.room_type == subject.required_room_type or subject.required_room_type == "standard"
                    ]
                    if suitable_rooms:
                        new_room = random.choice(suitable_rooms)
                        new_solution[idx] = Assignment(
                            teacher_id=assignment.teacher_id,
                            subject_id=assignment.subject_id,
                            room_id=new_room,
                            time_slot_id=assignment.time_slot_id,
                            student_count=assignment.student_count
                        )

            elif modification_type == "time_slot":
                # Change to different time slot
                new_time_slot = random.choice(list(self.time_slots.keys()))
                new_solution[idx] = Assignment(
                    teacher_id=assignment.teacher_id,
                    subject_id=assignment.subject_id,
                    room_id=assignment.room_id,
                    time_slot_id=new_time_slot,
                    student_count=assignment.student_count
                )

        return new_solution

    def validate_solution_format(self, solution: Any) -> bool:
        """Validate that a solution has the correct format."""
        if not isinstance(solution, list):
            return False

        for item in solution:
            if not isinstance(item, Assignment):
                return False

            # Check if referenced entities exist
            if item.teacher_id not in self.teachers:
                return False
            if item.subject_id not in self.subjects:
                return False
            if item.room_id not in self.classrooms:
                return False
            if item.time_slot_id not in self.time_slots:
                return False

        return True

    def get_problem_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the problem instance."""
        return {
            "teachers_count": len(self.teachers),
            "classrooms_count": len(self.classrooms),
            "subjects_count": len(self.subjects),
            "time_slots_count": len(self.time_slots),
            "total_required_hours": sum(subject.required_hours_per_week for subject in self.subjects.values()),
            "teachers": {tid: {
                "name": teacher.name,
                "subjects": teacher.subjects,
                "max_hours_per_day": teacher.max_hours_per_day,
                "experience_level": teacher.experience_level,
                "hourly_cost": teacher.hourly_cost
            } for tid, teacher in self.teachers.items()},
            "classrooms": {rid: {
                "name": room.name,
                "capacity": room.capacity,
                "room_type": room.room_type,
                "hourly_cost": room.hourly_cost
            } for rid, room in self.classrooms.items()},
            "subjects": {sid: {
                "name": subject.name,
                "required_hours_per_week": subject.required_hours_per_week,
                "max_class_size": subject.max_class_size,
                "required_room_type": subject.required_room_type,
                "priority": subject.priority
            } for sid, subject in self.subjects.items()},
            "time_slots": {slot_id: {
                "day": slot.day,
                "start_time": slot.start_time,
                "end_time": slot.end_time
            } for slot_id, slot in self.time_slots.items()}
        }
