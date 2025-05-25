"""
Comprehensive benchmarking and evaluation system for qubots.
Provides standardized benchmarking, performance metrics, and comparison tools.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import statistics
import json
from datetime import datetime
import numpy as np
from .base_problem import BaseProblem
from .base_optimizer import BaseOptimizer, OptimizationResult

class BenchmarkType(Enum):
    """Types of benchmarks."""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    ROBUSTNESS = "robustness"
    SCALABILITY = "scalability"
    CONVERGENCE = "convergence"

@dataclass
class BenchmarkMetrics:
    """Comprehensive metrics for benchmark evaluation."""
    # Performance metrics
    best_value: float
    mean_value: float
    std_value: float
    median_value: float
    worst_value: float
    
    # Time metrics
    mean_runtime_seconds: float
    std_runtime_seconds: float
    total_runtime_seconds: float
    
    # Convergence metrics
    convergence_rate: float  # Percentage of runs that converged
    mean_iterations_to_convergence: float
    
    # Robustness metrics
    success_rate: float  # Percentage of successful runs
    coefficient_of_variation: float  # std/mean for stability
    
    # Additional metrics
    evaluations_per_second: float
    memory_usage_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "best_value": self.best_value,
            "mean_value": self.mean_value,
            "std_value": self.std_value,
            "median_value": self.median_value,
            "worst_value": self.worst_value,
            "mean_runtime_seconds": self.mean_runtime_seconds,
            "std_runtime_seconds": self.std_runtime_seconds,
            "total_runtime_seconds": self.total_runtime_seconds,
            "convergence_rate": self.convergence_rate,
            "mean_iterations_to_convergence": self.mean_iterations_to_convergence,
            "success_rate": self.success_rate,
            "coefficient_of_variation": self.coefficient_of_variation,
            "evaluations_per_second": self.evaluations_per_second,
            "memory_usage_mb": self.memory_usage_mb
        }

@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    benchmark_id: str
    problem_name: str
    optimizer_name: str
    metrics: BenchmarkMetrics
    individual_runs: List[OptimizationResult]
    benchmark_type: BenchmarkType
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Configuration
    num_runs: int = 1
    problem_config: Dict[str, Any] = field(default_factory=dict)
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "benchmark_id": self.benchmark_id,
            "problem_name": self.problem_name,
            "optimizer_name": self.optimizer_name,
            "metrics": self.metrics.to_dict(),
            "benchmark_type": self.benchmark_type.value,
            "timestamp": self.timestamp.isoformat(),
            "num_runs": self.num_runs,
            "problem_config": self.problem_config,
            "optimizer_config": self.optimizer_config,
            "individual_runs_count": len(self.individual_runs)
        }

class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for optimization algorithms.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize benchmark suite.
        
        Args:
            name: Name of the benchmark suite
            description: Description of the benchmark suite
        """
        self.name = name
        self.description = description
        self.problems = {}
        self.optimizers = {}
        self.results = []
        self.configurations = {}
    
    def add_problem(self, name: str, problem: BaseProblem, config: Dict[str, Any] = None):
        """Add a problem to the benchmark suite."""
        self.problems[name] = {
            "problem": problem,
            "config": config or {}
        }
    
    def add_optimizer(self, name: str, optimizer: BaseOptimizer, config: Dict[str, Any] = None):
        """Add an optimizer to the benchmark suite."""
        self.optimizers[name] = {
            "optimizer": optimizer,
            "config": config or {}
        }
    
    def run_benchmark(self, problem_name: str, optimizer_name: str, 
                     num_runs: int = 10, benchmark_type: BenchmarkType = BenchmarkType.PERFORMANCE,
                     convergence_threshold: float = 1e-6, max_stagnation: int = 100) -> BenchmarkResult:
        """
        Run benchmark for specific problem-optimizer combination.
        
        Args:
            problem_name: Name of the problem
            optimizer_name: Name of the optimizer
            num_runs: Number of independent runs
            benchmark_type: Type of benchmark
            convergence_threshold: Threshold for convergence detection
            max_stagnation: Maximum iterations without improvement for convergence
            
        Returns:
            BenchmarkResult with comprehensive metrics
        """
        if problem_name not in self.problems:
            raise ValueError(f"Problem '{problem_name}' not found in benchmark suite")
        if optimizer_name not in self.optimizers:
            raise ValueError(f"Optimizer '{optimizer_name}' not found in benchmark suite")
        
        problem = self.problems[problem_name]["problem"]
        optimizer = self.optimizers[optimizer_name]["optimizer"]
        
        # Run multiple independent trials
        results = []
        total_start_time = time.time()
        
        for run_idx in range(num_runs):
            # Reset problem and optimizer statistics
            problem.reset_statistics()
            optimizer.reset_statistics()
            
            try:
                # Run optimization
                result = optimizer.optimize(problem)
                results.append(result)
                
            except Exception as e:
                # Handle failed runs
                failed_result = OptimizationResult(
                    best_solution=None,
                    best_value=float('inf'),
                    is_feasible=False,
                    runtime_seconds=0.0,
                    termination_reason=f"error: {str(e)}"
                )
                results.append(failed_result)
        
        total_end_time = time.time()
        
        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(results, convergence_threshold, max_stagnation)
        
        # Create benchmark result
        benchmark_result = BenchmarkResult(
            benchmark_id=f"{self.name}_{problem_name}_{optimizer_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            problem_name=problem_name,
            optimizer_name=optimizer_name,
            metrics=metrics,
            individual_runs=results,
            benchmark_type=benchmark_type,
            num_runs=num_runs,
            problem_config=self.problems[problem_name]["config"],
            optimizer_config=self.optimizers[optimizer_name]["config"]
        )
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def run_full_benchmark(self, num_runs: int = 10) -> List[BenchmarkResult]:
        """Run benchmark for all problem-optimizer combinations."""
        results = []
        
        for problem_name in self.problems:
            for optimizer_name in self.optimizers:
                result = self.run_benchmark(problem_name, optimizer_name, num_runs)
                results.append(result)
        
        return results
    
    def _calculate_metrics(self, results: List[OptimizationResult], 
                          convergence_threshold: float, max_stagnation: int) -> BenchmarkMetrics:
        """Calculate comprehensive metrics from optimization results."""
        # Filter successful runs
        successful_runs = [r for r in results if r.is_feasible and r.best_value != float('inf')]
        
        if not successful_runs:
            # All runs failed
            return BenchmarkMetrics(
                best_value=float('inf'),
                mean_value=float('inf'),
                std_value=0.0,
                median_value=float('inf'),
                worst_value=float('inf'),
                mean_runtime_seconds=0.0,
                std_runtime_seconds=0.0,
                total_runtime_seconds=sum(r.runtime_seconds for r in results),
                convergence_rate=0.0,
                mean_iterations_to_convergence=0.0,
                success_rate=0.0,
                coefficient_of_variation=0.0,
                evaluations_per_second=0.0
            )
        
        # Extract values and times
        values = [r.best_value for r in successful_runs]
        runtimes = [r.runtime_seconds for r in successful_runs]
        evaluations = [r.evaluations for r in successful_runs if r.evaluations > 0]
        
        # Calculate convergence metrics
        converged_runs = []
        for result in successful_runs:
            if hasattr(result, 'convergence_achieved') and result.convergence_achieved:
                converged_runs.append(result)
        
        convergence_rate = len(converged_runs) / len(results) * 100
        mean_iterations_to_convergence = (
            statistics.mean([r.iterations for r in converged_runs]) 
            if converged_runs else 0.0
        )
        
        # Calculate performance metrics
        best_value = min(values)
        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values) if len(values) > 1 else 0.0
        median_value = statistics.median(values)
        worst_value = max(values)
        
        # Calculate time metrics
        mean_runtime = statistics.mean(runtimes)
        std_runtime = statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0
        total_runtime = sum(r.runtime_seconds for r in results)
        
        # Calculate efficiency metrics
        success_rate = len(successful_runs) / len(results) * 100
        coefficient_of_variation = (std_value / mean_value) if mean_value != 0 else float('inf')
        
        evaluations_per_second = (
            statistics.mean([e / r for e, r in zip(evaluations, runtimes) if r > 0])
            if evaluations and runtimes else 0.0
        )
        
        return BenchmarkMetrics(
            best_value=best_value,
            mean_value=mean_value,
            std_value=std_value,
            median_value=median_value,
            worst_value=worst_value,
            mean_runtime_seconds=mean_runtime,
            std_runtime_seconds=std_runtime,
            total_runtime_seconds=total_runtime,
            convergence_rate=convergence_rate,
            mean_iterations_to_convergence=mean_iterations_to_convergence,
            success_rate=success_rate,
            coefficient_of_variation=coefficient_of_variation,
            evaluations_per_second=evaluations_per_second
        )
    
    def get_leaderboard(self, problem_name: str, metric: str = "best_value") -> List[Dict[str, Any]]:
        """
        Get leaderboard for a specific problem.
        
        Args:
            problem_name: Name of the problem
            metric: Metric to rank by
            
        Returns:
            List of optimizer results ranked by metric
        """
        problem_results = [r for r in self.results if r.problem_name == problem_name]
        
        # Sort by metric (assuming lower is better for most metrics)
        reverse_metrics = ["success_rate", "convergence_rate", "evaluations_per_second"]
        reverse = metric in reverse_metrics
        
        leaderboard = []
        for result in problem_results:
            metric_value = getattr(result.metrics, metric, None)
            if metric_value is not None:
                leaderboard.append({
                    "optimizer_name": result.optimizer_name,
                    "metric_value": metric_value,
                    "benchmark_result": result
                })
        
        leaderboard.sort(key=lambda x: x["metric_value"], reverse=reverse)
        return leaderboard
    
    def export_results(self, filename: str):
        """Export benchmark results to JSON file."""
        export_data = {
            "suite_name": self.name,
            "suite_description": self.description,
            "export_timestamp": datetime.now().isoformat(),
            "results": [result.to_dict() for result in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def compare_optimizers(self, problem_name: str, metric: str = "best_value") -> Dict[str, Any]:
        """
        Compare optimizers on a specific problem.
        
        Args:
            problem_name: Name of the problem
            metric: Metric to compare
            
        Returns:
            Comparison statistics
        """
        problem_results = [r for r in self.results if r.problem_name == problem_name]
        
        comparison = {
            "problem_name": problem_name,
            "metric": metric,
            "optimizers": {}
        }
        
        for result in problem_results:
            metric_value = getattr(result.metrics, metric, None)
            if metric_value is not None:
                comparison["optimizers"][result.optimizer_name] = {
                    "value": metric_value,
                    "rank": 0,  # Will be filled later
                    "relative_performance": 0.0  # Will be filled later
                }
        
        # Calculate ranks and relative performance
        sorted_optimizers = sorted(
            comparison["optimizers"].items(),
            key=lambda x: x[1]["value"]
        )
        
        best_value = sorted_optimizers[0][1]["value"] if sorted_optimizers else 0
        
        for rank, (optimizer_name, data) in enumerate(sorted_optimizers, 1):
            comparison["optimizers"][optimizer_name]["rank"] = rank
            if best_value != 0:
                comparison["optimizers"][optimizer_name]["relative_performance"] = (
                    data["value"] / best_value
                )
        
        return comparison
