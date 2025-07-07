"""
Calendar Rearrangement Optimization Problem for Qubots Framework

This problem optimizes the rearrangement of calendar meetings to free up a specific day
while respecting constraints like meeting durations, participant availability, and priorities.

The problem accepts CSV data with columns:
- meeting_id: Unique identifier for the meeting
- meeting_name: Name/title of the meeting
- duration_hours: Duration of the meeting in hours
- priority: Priority level (1-5, where 5 is highest priority)
- current_day: Current day of the week (Monday, Tuesday, etc.)
- flexible: Whether the meeting can be moved (True/False)
- participants: Number of participants (affects rescheduling difficulty)

Features:
- CSV data input for realistic meeting information
- Configurable target day to free up
- Constraint-based optimization respecting meeting flexibility
- Priority-weighted objective function
- Support for different rescheduling preferences
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import io

from qubots import BaseProblem, ProblemMetadata, ProblemType, ObjectiveType, DifficultyLevel


@dataclass
class MeetingData:
    """Data structure for individual meeting information."""
    meeting_id: str
    meeting_name: str
    duration_hours: float
    priority: int
    current_day: str
    flexible: bool
    participants: int
    
    def __post_init__(self):
        """Validate meeting data."""
        if self.duration_hours <= 0 or self.duration_hours > 8:
            raise ValueError(f"Duration {self.duration_hours} must be between 0 and 8 hours")
        if self.priority < 1 or self.priority > 5:
            raise ValueError(f"Priority {self.priority} must be between 1 and 5")
        if self.participants < 1:
            raise ValueError(f"Participants {self.participants} must be at least 1")


class CalendarRearrangementProblem(BaseProblem):
    """
    Calendar Rearrangement Problem for optimizing meeting schedules.
    
    Minimizes disruption cost while freeing up a target day by:
    - Moving flexible meetings to other available days
    - Respecting meeting priorities and participant constraints
    - Minimizing total rescheduling cost
    """
    
    def __init__(self, 
                 csv_data: str = None,
                 csv_file_path: str = None,
                 target_day_off: str = "Wednesday",
                 available_days: List[str] = None,
                 max_hours_per_day: float = 8.0,
                 rescheduling_cost_weight: float = 1.0,
                 priority_weight: float = 2.0,
                 **kwargs):
        """
        Initialize calendar rearrangement problem.
        
        Args:
            csv_data: CSV content as string
            csv_file_path: Path to CSV file (alternative to csv_data)
            target_day_off: Day to free up (default: "Wednesday")
            available_days: List of available days for rescheduling
            max_hours_per_day: Maximum meeting hours per day
            rescheduling_cost_weight: Weight for rescheduling cost
            priority_weight: Weight for priority considerations
            **kwargs: Additional parameters
        """
        self.target_day_off = target_day_off
        self.available_days = available_days or ["Monday", "Tuesday", "Thursday", "Friday"]
        self.max_hours_per_day = max_hours_per_day
        self.rescheduling_cost_weight = rescheduling_cost_weight
        self.priority_weight = priority_weight
        
        # Load meeting data from CSV
        self.meetings = self._load_meeting_data(csv_data, csv_file_path)
        self.n_meetings = len(self.meetings)
        
        # Filter meetings that need to be moved (on target day and flexible)
        self.moveable_meetings = [m for m in self.meetings 
                                 if m.current_day == self.target_day_off and m.flexible]
        self.n_moveable = len(self.moveable_meetings)
        
        # Set up problem metadata
        metadata = self._get_default_metadata()
        super().__init__(metadata, **kwargs)
    
    def _get_default_metadata(self) -> ProblemMetadata:
        """Return default metadata for calendar rearrangement problem."""
        return ProblemMetadata(
            name="Calendar Rearrangement Problem",
            description=f"Optimize meeting rearrangement to free up {self.target_day_off}, "
                       f"with {self.n_moveable} moveable meetings",
            problem_type=ProblemType.COMBINATORIAL,
            objective_type=ObjectiveType.MINIMIZE,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            domain="scheduling",
            tags={"calendar", "scheduling", "meetings", "optimization", "rearrangement"},
            dimension=self.n_moveable * len(self.available_days),
            constraints_count=len(self.available_days) + 1,  # capacity constraints + assignment
            evaluation_complexity="O(n*d)",
            memory_complexity="O(n*d)"
        )
    
    def _load_meeting_data(self, csv_data: str = None, csv_file_path: str = None) -> List[MeetingData]:
        """Load meeting data from CSV source."""
        if csv_data:
            # Load from string data
            df = pd.read_csv(io.StringIO(csv_data))
        elif csv_file_path:
            # Load from file path
            df = pd.read_csv(csv_file_path)
        else:
            # Use default sample data
            df = self._create_default_data()
        
        # Validate required columns
        required_cols = ['meeting_id', 'meeting_name', 'duration_hours', 'priority', 
                        'current_day', 'flexible', 'participants']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert to MeetingData objects
        meetings = []
        for _, row in df.iterrows():
            meeting = MeetingData(
                meeting_id=str(row['meeting_id']),
                meeting_name=str(row['meeting_name']),
                duration_hours=float(row['duration_hours']),
                priority=int(row['priority']),
                current_day=str(row['current_day']),
                flexible=bool(row['flexible']),
                participants=int(row['participants'])
            )
            meetings.append(meeting)
        
        return meetings
    
    def _create_default_data(self) -> pd.DataFrame:
        """Create default sample meeting data for demonstration."""
        return pd.DataFrame({
            'meeting_id': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8'],
            'meeting_name': ['Team Standup', 'Project Review', 'Client Call', 'Design Meeting',
                           'Budget Planning', 'Training Session', 'One-on-One', 'Strategy Meeting'],
            'duration_hours': [0.5, 2.0, 1.0, 1.5, 1.0, 2.0, 0.5, 2.0],
            'priority': [3, 5, 4, 3, 4, 2, 3, 5],
            'current_day': ['Wednesday', 'Wednesday', 'Monday', 'Wednesday',
                          'Tuesday', 'Wednesday', 'Wednesday', 'Thursday'],
            'flexible': [True, True, True, True, True, True, True, False],  # Make more meetings flexible
            'participants': [5, 8, 3, 4, 6, 12, 2, 10]
        })
    
    def evaluate_solution(self, solution: Union[List[int], np.ndarray, Dict[str, Any]]) -> float:
        """
        Evaluate calendar rearrangement solution.
        
        Args:
            solution: Assignment of moveable meetings to available days
                     Can be list/array of day indices or dict with 'assignments' key
            
        Returns:
            Total rescheduling cost with penalties for constraint violations
        """
        # Extract assignments from solution
        if isinstance(solution, dict):
            assignments = solution.get('assignments', solution.get('schedule', []))
        else:
            assignments = solution
        
        assignments = np.array(assignments)
        
        if len(assignments) != self.n_moveable:
            return 1e6  # Large penalty for wrong dimension
        
        # Check if assignments are valid day indices
        if np.any(assignments < 0) or np.any(assignments >= len(self.available_days)):
            return 1e6  # Invalid day assignments
        
        total_cost = 0.0
        penalty = 0.0
        
        # Calculate capacity constraints for each day
        day_hours = {day: 0.0 for day in self.available_days}
        
        # Add existing meetings on available days
        for meeting in self.meetings:
            if meeting.current_day in self.available_days and meeting.current_day != self.target_day_off:
                day_hours[meeting.current_day] += meeting.duration_hours
        
        # Add moved meetings
        for i, day_idx in enumerate(assignments):
            day = self.available_days[day_idx]
            meeting = self.moveable_meetings[i]
            day_hours[day] += meeting.duration_hours
        
        # Check capacity constraints
        for day, hours in day_hours.items():
            if hours > self.max_hours_per_day:
                penalty += 1000 * (hours - self.max_hours_per_day) ** 2
        
        # Calculate rescheduling costs
        for i, day_idx in enumerate(assignments):
            meeting = self.moveable_meetings[i]
            
            # Base rescheduling cost (higher for more participants)
            base_cost = self.rescheduling_cost_weight * (1 + 0.1 * meeting.participants)
            
            # Priority adjustment (higher priority = higher cost to move)
            priority_cost = self.priority_weight * meeting.priority
            
            total_cost += base_cost + priority_cost
        
        return total_cost + penalty
    
    def is_feasible(self, solution: Union[List[int], np.ndarray, Dict[str, Any]]) -> bool:
        """Check if solution satisfies all constraints."""
        if isinstance(solution, dict):
            assignments = solution.get('assignments', solution.get('schedule', []))
        else:
            assignments = solution
        
        assignments = np.array(assignments)
        
        if len(assignments) != self.n_moveable:
            return False
        
        # Check valid day indices
        if np.any(assignments < 0) or np.any(assignments >= len(self.available_days)):
            return False
        
        # Check capacity constraints
        day_hours = {day: 0.0 for day in self.available_days}
        
        # Add existing meetings
        for meeting in self.meetings:
            if meeting.current_day in self.available_days and meeting.current_day != self.target_day_off:
                day_hours[meeting.current_day] += meeting.duration_hours
        
        # Add moved meetings
        for i, day_idx in enumerate(assignments):
            day = self.available_days[day_idx]
            meeting = self.moveable_meetings[i]
            day_hours[day] += meeting.duration_hours
        
        # Check if any day exceeds capacity
        for hours in day_hours.values():
            if hours > self.max_hours_per_day:
                return False
        
        return True
    
    def random_solution(self) -> Dict[str, Any]:
        """Generate a random feasible solution."""
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Random assignment of meetings to days
            assignments = np.random.randint(0, len(self.available_days), self.n_moveable)
            
            if self.is_feasible({'assignments': assignments}):
                return {'assignments': assignments.tolist()}
        
        # If no feasible random solution found, use greedy approach
        return self._greedy_solution()
    
    def _greedy_solution(self) -> Dict[str, Any]:
        """Generate a greedy feasible solution."""
        assignments = []
        day_hours = {day: 0.0 for day in self.available_days}
        
        # Add existing meetings
        for meeting in self.meetings:
            if meeting.current_day in self.available_days and meeting.current_day != self.target_day_off:
                day_hours[meeting.current_day] += meeting.duration_hours
        
        # Sort meetings by priority (high priority first) and duration (short first)
        sorted_meetings = sorted(enumerate(self.moveable_meetings), 
                               key=lambda x: (-x[1].priority, x[1].duration_hours))
        
        for orig_idx, meeting in sorted_meetings:
            # Find day with minimum load that can accommodate this meeting
            best_day_idx = None
            min_load = float('inf')
            
            for day_idx, day in enumerate(self.available_days):
                if day_hours[day] + meeting.duration_hours <= self.max_hours_per_day:
                    if day_hours[day] < min_load:
                        min_load = day_hours[day]
                        best_day_idx = day_idx
            
            if best_day_idx is not None:
                assignments.append((orig_idx, best_day_idx))
                day_hours[self.available_days[best_day_idx]] += meeting.duration_hours
            else:
                # Force assignment to least loaded day (may violate capacity)
                best_day_idx = min(range(len(self.available_days)), 
                                 key=lambda i: day_hours[self.available_days[i]])
                assignments.append((orig_idx, best_day_idx))
                day_hours[self.available_days[best_day_idx]] += meeting.duration_hours
        
        # Sort by original index to maintain order
        assignments.sort(key=lambda x: x[0])
        final_assignments = [day_idx for _, day_idx in assignments]
        
        return {'assignments': final_assignments}

    def get_feasibility_info(self) -> Dict[str, Any]:
        """Get detailed information about problem feasibility."""
        info = {
            'total_meetings': self.n_meetings,
            'moveable_meetings': self.n_moveable,
            'target_day_off': self.target_day_off,
            'available_days': self.available_days,
            'max_hours_per_day': self.max_hours_per_day
        }

        # Calculate existing hours on each available day
        existing_hours = {day: 0.0 for day in self.available_days}
        for meeting in self.meetings:
            if meeting.current_day in existing_hours and meeting.current_day != self.target_day_off:
                existing_hours[meeting.current_day] += meeting.duration_hours

        info['existing_hours_per_day'] = existing_hours

        # Calculate total hours that need to be moved
        total_hours_to_move = sum(m.duration_hours for m in self.moveable_meetings)
        info['total_hours_to_move'] = total_hours_to_move

        # Calculate available capacity
        available_capacity = {}
        total_available_capacity = 0
        for day in self.available_days:
            capacity = self.max_hours_per_day - existing_hours[day]
            available_capacity[day] = max(0, capacity)
            total_available_capacity += available_capacity[day]

        info['available_capacity_per_day'] = available_capacity
        info['total_available_capacity'] = total_available_capacity
        info['is_theoretically_feasible'] = total_available_capacity >= total_hours_to_move

        # List meetings to be moved
        info['meetings_to_move'] = [
            {
                'name': m.meeting_name,
                'duration': m.duration_hours,
                'priority': m.priority,
                'participants': m.participants
            }
            for m in self.moveable_meetings
        ]

        return info

