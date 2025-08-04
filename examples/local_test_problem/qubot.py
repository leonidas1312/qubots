"""
Simple test problem for local development.
"""

import random
from qubots import BaseProblem, ProblemMetadata, ProblemType, ObjectiveType, DifficultyLevel


class LocalTestProblem(BaseProblem):
    """A simple test problem that generates random solutions."""
    
    def __init__(self, size=10):
        """Initialize the problem."""
        self.size = size
        super().__init__()
    
    def _get_default_metadata(self):
        """Get problem metadata."""
        return ProblemMetadata(
            name="Local Test Problem",
            description="A simple test problem for local development",
            problem_type=ProblemType.DISCRETE,
            objective_type=ObjectiveType.MAXIMIZE,  # Maximize sum of bits
            difficulty_level=DifficultyLevel.BEGINNER,
            domain="testing",
            tags={"test", "local", "simple", "binary"},
            dimension=self.size,
            constraints_count=0,
            evaluation_complexity="O(n)",
            memory_complexity="O(n)"
        )
    
    def random_solution(self):
        """Generate a random solution."""
        return [random.randint(0, 1) for _ in range(self.size)]
    
    def evaluate_solution(self, solution):
        """Evaluate a solution (sum of bits)."""
        return sum(solution)
    
    def is_feasible(self, solution):
        """Check if solution is feasible."""
        return len(solution) == self.size and all(x in [0, 1] for x in solution)
