"""
Enhanced Fantasy Football Optimization Problem for the Qubots Framework.

This module implements a comprehensive fantasy football lineup optimization problem
that maximizes projected points while satisfying DraftKings constraints and salary caps.
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from qubots import (
    BaseProblem, ProblemMetadata, ProblemType, ObjectiveType,
    DifficultyLevel, EvaluationResult
)


class FantasyFootballProblem(BaseProblem):
    """
    Fantasy Football Lineup Optimization Problem.

    This problem optimizes fantasy football lineups for DraftKings contests by:
    - Maximizing projected fantasy points
    - Satisfying position requirements (1 QB, 2-3 RB, 3-4 WR, 1-2 TE, 1 DEF)
    - Staying within salary cap constraints
    - Ensuring exactly 9 players in the lineup

    The problem uses binary decision variables where 1 indicates a player is selected
    and 0 indicates they are not selected.
    """

    def __init__(self,
                 dataframe: Optional[pd.DataFrame] = None,
                 csv_file: Optional[str] = None,
                 min_salary: int = 0,
                 max_salary: int = 50000,
                 excluded_players: Optional[List[str]] = None):
        """
        Initialize the Fantasy Football Problem.

        Args:
            dataframe: Pre-loaded player data DataFrame
            csv_file: Path to CSV file with player data
            min_salary: Minimum total salary for lineup
            max_salary: Maximum total salary for lineup (salary cap)
            excluded_players: List of player names to exclude from consideration
        """
        # Initialize metadata first
        metadata = ProblemMetadata(
            name="Fantasy Football Lineup Optimization",
            description="Optimize DraftKings fantasy football lineups to maximize points while satisfying constraints",
            problem_type=ProblemType.COMBINATORIAL,
            objective_type=ObjectiveType.MAXIMIZE,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            domain="sports_analytics",
            tags={"fantasy_football", "sports", "optimization", "draftkings", "lineup"},
            author="Qubots Community",
            version="1.0.0",
            constraints_count=8  # Position constraints + salary constraint + lineup size
        )

        super().__init__(metadata)

        # Load data
        if dataframe is not None:
            self.df = dataframe.copy()
        elif csv_file is not None:
            self.df = pd.read_csv(csv_file)
        else:
            # Try to load default data file
            default_files = ["2016week8sun.csv", "sample_fantasy_data.csv"]
            loaded = False
            for default_file in default_files:
                if os.path.exists(default_file):
                    self.df = pd.read_csv(default_file)
                    loaded = True
                    break
            if not loaded:
                raise ValueError("Must provide either dataframe or csv_file parameter, or have a default data file available")

        # Problem parameters
        self.excluded_players = excluded_players or []
        self.min_salary = min_salary
        self.max_salary = max_salary
        self.n_players = len(self.df)

        # Preprocess data
        self._preprocess_data()

        # Update metadata with actual problem dimensions
        self._metadata.dimension = self.n_players

    def _get_default_metadata(self) -> ProblemMetadata:
        """Return default metadata for fantasy football problems."""
        return ProblemMetadata(
            name="Fantasy Football Lineup Optimization",
            description="Optimize DraftKings fantasy football lineups to maximize points while satisfying constraints",
            problem_type=ProblemType.COMBINATORIAL,
            objective_type=ObjectiveType.MAXIMIZE,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            domain="sports_analytics",
            tags={"fantasy_football", "sports", "optimization", "draftkings", "lineup"},
            author="Qubots Community",
            version="1.0.0",
            constraints_count=8
        )

    def _preprocess_data(self):
        """Preprocess the player data for optimization."""
        # Ensure required columns exist
        required_columns = ['Name', 'Pos', 'DK.points', 'DK.salary']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # One-hot encoding for positions
        pos_dummies = pd.get_dummies(self.df['Pos'], prefix='Pos')
        self.df = pd.concat([self.df, pos_dummies], axis=1)

        # Create flex position indicator (RB, WR, TE can fill flex spots)
        flex_positions = ['Pos_RB', 'Pos_WR', 'Pos_TE']
        available_flex = [pos for pos in flex_positions if pos in self.df.columns]
        if available_flex:
            self.df['PosFlex'] = self.df[available_flex].sum(axis=1).clip(upper=1)
        else:
            self.df['PosFlex'] = 0

        # Handle excluded players by setting their points to 0
        if self.excluded_players:
            mask = self.df['Name'].isin(self.excluded_players)
            self.df.loc[mask, 'DK.points'] = 0

        # Store position columns for constraint checking
        self.position_columns = [col for col in self.df.columns if col.startswith('Pos_')]

    def evaluate_solution(self, solution: Any) -> float:
        """
        Evaluate a fantasy football lineup solution.

        Args:
            solution: Binary list indicating which players are selected

        Returns:
            Total projected fantasy points for the lineup
        """
        solution_array = np.array(solution)

        # Calculate total fantasy points
        total_points = float(np.dot(solution_array, self.df['DK.points']))

        return total_points

    def is_feasible(self, solution: Any) -> bool:
        """
        Check if a solution satisfies all fantasy football constraints.

        Args:
            solution: Binary list indicating which players are selected

        Returns:
            True if solution is feasible, False otherwise
        """
        solution_array = np.array(solution)

        # Basic validation
        if solution_array.shape[0] != self.n_players:
            return False

        # Must be binary
        if not np.all((solution_array == 0) | (solution_array == 1)):
            return False

        # Position constraints
        constraints = self._get_position_constraints()
        for constraint_name, (min_count, max_count, column) in constraints.items():
            if column in self.df.columns:
                count = int(np.dot(solution_array, self.df[column]))
                if not (min_count <= count <= max_count):
                    return False

        # Salary constraint
        total_salary = float(np.dot(solution_array, self.df['DK.salary']))
        if not (self.min_salary <= total_salary <= self.max_salary):
            return False

        # Total lineup size (should be exactly 9 for DraftKings)
        total_players = int(np.sum(solution_array))
        if total_players != 9:
            return False

        return True

    def _get_position_constraints(self) -> Dict[str, Tuple[int, int, str]]:
        """Get position constraints for DraftKings lineup."""
        return {
            'quarterback': (1, 1, 'Pos_QB'),
            'running_back': (2, 3, 'Pos_RB'),
            'wide_receiver': (3, 4, 'Pos_WR'),
            'tight_end': (1, 2, 'Pos_TE'),
            'defense': (1, 1, 'Pos_Def'),
            'flex_total': (7, 7, 'PosFlex')  # Total flex positions (RB+WR+TE)
        }

    def random_solution(self) -> Any:
        """
        Generate a random feasible fantasy football lineup.

        Returns:
            Random feasible solution as binary list
        """
        max_attempts = 1000

        for attempt in range(max_attempts):
            solution = np.zeros(self.n_players, dtype=int)

            try:
                # Select required positions
                constraints = self._get_position_constraints()

                # Select QB (exactly 1)
                if 'Pos_QB' in self.df.columns:
                    qb_indices = self.df[self.df['Pos_QB'] == 1].index.tolist()
                    if qb_indices:
                        selected_qb = np.random.choice(qb_indices)
                        solution[selected_qb] = 1

                # Select DEF (exactly 1)
                if 'Pos_Def' in self.df.columns:
                    def_indices = self.df[self.df['Pos_Def'] == 1].index.tolist()
                    if def_indices:
                        selected_def = np.random.choice(def_indices)
                        solution[selected_def] = 1

                # Select flex positions (RB, WR, TE) to total 7
                flex_indices = self.df[self.df['PosFlex'] == 1].index.tolist()
                available_flex = [i for i in flex_indices if solution[i] == 0]

                if len(available_flex) >= 7:
                    selected_flex = np.random.choice(available_flex, 7, replace=False)
                    solution[selected_flex] = 1

                # Check if solution is feasible
                if self.is_feasible(solution.tolist()):
                    return solution.tolist()

            except (ValueError, IndexError):
                continue

        # If no feasible solution found, return a simple fallback
        solution = np.zeros(self.n_players, dtype=int)
        solution[:min(9, self.n_players)] = 1
        return solution.tolist()

    def get_lineup_summary(self, solution: List[int]) -> pd.DataFrame:
        """
        Get a summary of the selected lineup.

        Args:
            solution: Binary list indicating which players are selected

        Returns:
            DataFrame with selected players and their details
        """
        mask = np.array(solution) == 1
        selected_players = self.df[mask].copy()

        if len(selected_players) > 0:
            # Add summary statistics
            total_salary = selected_players['DK.salary'].sum()
            total_points = selected_players['DK.points'].sum()

            print(f"Lineup Summary:")
            print(f"Total Players: {len(selected_players)}")
            print(f"Total Salary: ${total_salary:,}")
            print(f"Total Projected Points: {total_points:.2f}")
            print(f"Salary Remaining: ${self.max_salary - total_salary:,}")

        return selected_players[['Name', 'Pos', 'Team', 'DK.points', 'DK.salary']] if len(selected_players) > 0 else pd.DataFrame()


# Example usage and testing
if __name__ == "__main__":
    # Create problem instance
    problem = FantasyFootballProblem()

    print(f"Fantasy Football Problem: {problem.metadata.name}")
    print(f"Number of players: {problem.n_players}")
    print(f"Problem type: {problem.metadata.problem_type}")

    # Generate and test a random solution
    solution = problem.random_solution()
    print(f"\nRandom solution feasible: {problem.is_feasible(solution)}")
    print(f"Solution points: {problem.evaluate_solution(solution):.2f}")

    # Show lineup
    lineup = problem.get_lineup_summary(solution)
    print("\nSelected lineup:")
    print(lineup)
