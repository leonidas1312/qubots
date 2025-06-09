"""
Qubots Leaderboard Integration

Provides functionality for submitting results to the Rastion platform leaderboard,
retrieving rankings, and managing standardized benchmark problems.
"""

import json
import time
import platform
import psutil
import sys
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

from .base_problem import BaseProblem
from .base_optimizer import BaseOptimizer, OptimizationResult
from .benchmarking import BenchmarkResult, BenchmarkMetrics
from .rastion_client import get_global_client


@dataclass
class LeaderboardSubmission:
    """Represents a submission to the leaderboard."""
    problem_id: int
    solver_name: str
    solver_username: str
    solver_repository: str
    solver_version: Optional[str]
    solver_config: Dict[str, Any]
    best_value: float
    runtime_seconds: float
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    iterations: Optional[int] = None
    evaluations: Optional[int] = None
    success_rate: float = 100.0
    hardware_info: Optional[Dict[str, Any]] = None
    execution_metadata: Optional[Dict[str, Any]] = None


@dataclass
class StandardizedProblem:
    """Represents a standardized benchmark problem."""
    id: int
    name: str
    problem_type: str
    description: str
    difficulty_level: str
    problem_config: Dict[str, Any]
    evaluation_config: Dict[str, Any]
    reference_solution: Optional[str] = None
    reference_value: Optional[float] = None
    time_limit_seconds: int = 300
    memory_limit_mb: int = 1024


class LeaderboardClient:
    """Client for interacting with the Rastion platform leaderboard."""
    
    def __init__(self, client=None):
        """Initialize leaderboard client."""
        self.client = client or get_global_client()
        self.base_url = f"{self.client.base_url}/api/leaderboard"
    
    def get_standardized_problems(self, 
                                problem_type: Optional[str] = None,
                                difficulty_level: Optional[str] = None) -> List[StandardizedProblem]:
        """Get list of standardized benchmark problems."""
        params = {}
        if problem_type:
            params['problem_type'] = problem_type
        if difficulty_level:
            params['difficulty_level'] = difficulty_level
        
        response = self.client.get(f"{self.base_url}/problems", params=params)
        
        if response.status_code == 200:
            data = response.json()
            return [StandardizedProblem(**problem) for problem in data['problems']]
        else:
            raise Exception(f"Failed to fetch problems: {response.text}")
    
    def get_leaderboard(self, 
                       problem_id: int,
                       limit: int = 50,
                       sort_by: str = 'rank_overall',
                       validated_only: bool = False) -> List[Dict[str, Any]]:
        """Get leaderboard for a specific problem."""
        params = {
            'limit': limit,
            'sort_by': sort_by,
            'validated_only': str(validated_only).lower()
        }
        
        response = self.client.get(f"{self.base_url}/problems/{problem_id}/leaderboard", params=params)
        
        if response.status_code == 200:
            data = response.json()
            return data['leaderboard']
        else:
            raise Exception(f"Failed to fetch leaderboard: {response.text}")
    
    def submit_result(self, submission: LeaderboardSubmission) -> Dict[str, Any]:
        """Submit a result to the leaderboard."""
        submission_data = asdict(submission)
        
        response = self.client.post(
            f"{self.base_url}/problems/{submission.problem_id}/submit",
            json=submission_data
        )
        
        if response.status_code == 201:
            return response.json()['submission']
        elif response.status_code == 409:
            raise Exception("Duplicate submission detected")
        else:
            raise Exception(f"Failed to submit result: {response.text}")
    
    def get_solver_profile(self, solver_repository: str) -> Dict[str, Any]:
        """Get solver profile and statistics."""
        encoded_repo = solver_repository.replace('/', '%2F')
        response = self.client.get(f"{self.base_url}/solvers/{encoded_repo}")
        
        if response.status_code == 200:
            return response.json()['profile']
        elif response.status_code == 404:
            return None
        else:
            raise Exception(f"Failed to fetch solver profile: {response.text}")
    
    def get_leaderboard_stats(self) -> Dict[str, Any]:
        """Get overall leaderboard statistics."""
        response = self.client.get(f"{self.base_url}/stats")
        
        if response.status_code == 200:
            return response.json()['stats']
        else:
            raise Exception(f"Failed to fetch stats: {response.text}")


class LeaderboardIntegration:
    """Integration layer for automatic leaderboard submissions."""
    
    def __init__(self, client: Optional[LeaderboardClient] = None):
        """Initialize leaderboard integration."""
        self.client = client or LeaderboardClient()
    
    def submit_benchmark_result(self,
                              benchmark_result: BenchmarkResult,
                              problem_id: int,
                              solver_repository: str,
                              solver_version: Optional[str] = None) -> Dict[str, Any]:
        """Submit a benchmark result to the leaderboard."""
        
        # Extract hardware information
        hardware_info = self._get_hardware_info()
        
        # Create submission
        submission = LeaderboardSubmission(
            problem_id=problem_id,
            solver_name=benchmark_result.optimizer_name,
            solver_username="", # Will be filled by backend from auth
            solver_repository=solver_repository,
            solver_version=solver_version,
            solver_config=benchmark_result.optimizer_config,
            best_value=benchmark_result.metrics.best_value,
            mean_value=benchmark_result.metrics.mean_value,
            std_value=benchmark_result.metrics.std_value,
            runtime_seconds=benchmark_result.metrics.mean_runtime_seconds,
            iterations=int(benchmark_result.metrics.mean_iterations_to_convergence) if benchmark_result.metrics.mean_iterations_to_convergence > 0 else None,
            evaluations=None, # TODO: Extract from benchmark result if available
            success_rate=benchmark_result.metrics.success_rate,
            hardware_info=hardware_info,
            execution_metadata={
                'benchmark_id': benchmark_result.benchmark_id,
                'benchmark_type': benchmark_result.benchmark_type.value,
                'num_runs': benchmark_result.num_runs,
                'timestamp': benchmark_result.timestamp.isoformat()
            }
        )
        
        return self.client.submit_result(submission)
    
    def submit_optimization_result(self,
                                 result: OptimizationResult,
                                 problem_id: int,
                                 solver_name: str,
                                 solver_repository: str,
                                 solver_config: Dict[str, Any],
                                 solver_version: Optional[str] = None) -> Dict[str, Any]:
        """Submit a single optimization result to the leaderboard."""
        
        # Extract hardware information
        hardware_info = self._get_hardware_info()
        
        # Create submission
        submission = LeaderboardSubmission(
            problem_id=problem_id,
            solver_name=solver_name,
            solver_username="", # Will be filled by backend from auth
            solver_repository=solver_repository,
            solver_version=solver_version,
            solver_config=solver_config,
            best_value=result.best_value,
            runtime_seconds=result.runtime_seconds,
            iterations=getattr(result, 'iterations', None),
            evaluations=getattr(result, 'evaluations', None),
            success_rate=100.0 if result.is_feasible else 0.0,
            hardware_info=hardware_info,
            execution_metadata={
                'termination_reason': getattr(result, 'termination_reason', None),
                'timestamp': datetime.now().isoformat()
            }
        )
        
        return self.client.submit_result(submission)
    
    def run_standardized_benchmark(self,
                                 problem_id: int,
                                 optimizer: BaseOptimizer,
                                 solver_repository: str,
                                 num_runs: int = 10,
                                 solver_version: Optional[str] = None) -> Dict[str, Any]:
        """Run a standardized benchmark and submit results."""
        
        # Get the standardized problem
        problems = self.client.get_standardized_problems()
        problem_spec = next((p for p in problems if p.id == problem_id), None)
        
        if not problem_spec:
            raise ValueError(f"Standardized problem {problem_id} not found")
        
        # Load the problem (this would need to be implemented based on problem type)
        problem = self._load_standardized_problem(problem_spec)
        
        # Run benchmark
        from .benchmarking import BenchmarkSuite, BenchmarkType
        
        suite = BenchmarkSuite(f"Leaderboard_{problem_spec.name}")
        suite.add_problem("standard", problem)
        suite.add_optimizer("solver", optimizer)
        
        benchmark_result = suite.run_benchmark(
            "standard", 
            "solver", 
            num_runs=num_runs,
            benchmark_type=BenchmarkType.PERFORMANCE
        )
        
        # Submit to leaderboard
        return self.submit_benchmark_result(
            benchmark_result,
            problem_id,
            solver_repository,
            solver_version
        )
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Collect hardware information for normalization."""
        try:
            cpu_info = platform.processor()
            cpu_count = psutil.cpu_count(logical=False)
            memory_gb = round(psutil.virtual_memory().total / (1024**3))
            
            # Try to get CPU frequency
            try:
                cpu_freq = psutil.cpu_freq()
                cpu_frequency = cpu_freq.max / 1000 if cpu_freq else None  # Convert to GHz
            except:
                cpu_frequency = None
            
            return {
                'cpu_model': cpu_info,
                'cpu_cores': cpu_count,
                'cpu_frequency_ghz': cpu_frequency,
                'memory_gb': memory_gb,
                'os_type': platform.system(),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            }
        except Exception as e:
            print(f"Warning: Could not collect hardware info: {e}")
            return {}
    
    def _load_standardized_problem(self, problem_spec: StandardizedProblem) -> BaseProblem:
        """Load a standardized problem instance."""
        # This would need to be implemented based on the problem type
        # For now, we'll raise an error indicating this needs implementation
        raise NotImplementedError(
            f"Loading standardized problems of type '{problem_spec.problem_type}' "
            "is not yet implemented. This would require a registry of problem loaders."
        )


# Convenience functions
def get_leaderboard_client() -> LeaderboardClient:
    """Get a leaderboard client instance."""
    return LeaderboardClient()


def submit_to_leaderboard(result: OptimizationResult,
                        problem_id: int,
                        solver_name: str,
                        solver_repository: str,
                        solver_config: Dict[str, Any],
                        solver_version: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to submit a result to the leaderboard."""
    integration = LeaderboardIntegration()
    return integration.submit_optimization_result(
        result, problem_id, solver_name, solver_repository, 
        solver_config, solver_version
    )


def get_problem_leaderboard(problem_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Convenience function to get leaderboard for a problem."""
    client = LeaderboardClient()
    return client.get_leaderboard(problem_id, limit=limit)


def get_standardized_problems(problem_type: Optional[str] = None) -> List[StandardizedProblem]:
    """Convenience function to get standardized problems."""
    client = LeaderboardClient()
    return client.get_standardized_problems(problem_type=problem_type)
