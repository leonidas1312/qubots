"""
Execution Validator for Qubots Models

Provides comprehensive validation for model execution, including direct execution,
playground execution, and performance monitoring with real-time output streaming.
"""

import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable

from ..base_problem import BaseProblem
from ..base_optimizer import BaseOptimizer
from ..playground_integration import execute_playground_optimization
import qubots.rastion as rastion


@dataclass
class ExecutionResult:
    """Result of an execution validation test."""
    success: bool
    execution_type: str = ""  # "direct", "playground", "combined"
    execution_time: float = 0.0
    result_data: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    error_details: Dict[str, Any] = field(default_factory=dict)
    real_time_output: List[str] = field(default_factory=list)


class ExecutionError(Exception):
    """Custom exception for execution validation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ExecutionValidator:
    """
    Comprehensive validator for model execution.
    
    This class provides methods to validate that models execute correctly
    in various scenarios including direct execution and playground execution.
    """
    
    def __init__(self, verbose: bool = True, capture_output: bool = True):
        """
        Initialize the execution validator.
        
        Args:
            verbose: Enable verbose output
            capture_output: Capture real-time output during execution
        """
        self.verbose = verbose
        self.capture_output = capture_output
        self.output_buffer: List[str] = []
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            log_msg = f"[{timestamp}] EXEC {level}: {message}"
            print(log_msg)
            
            if self.capture_output:
                self.output_buffer.append(log_msg)
    
    def validate_direct_execution(self, problem_name: str, optimizer_name: str,
                                problem_params: Optional[Dict[str, Any]] = None,
                                optimizer_params: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Validate direct execution of problem and optimizer.
        
        Args:
            problem_name: Name of the problem repository
            optimizer_name: Name of the optimizer repository
            problem_params: Parameters for the problem
            optimizer_params: Parameters for the optimizer
            
        Returns:
            ExecutionResult with validation details
        """
        start_time = time.time()
        self.output_buffer.clear()
        
        try:
            self.log(f"Direct execution test: {problem_name} + {optimizer_name}")
            
            # Load models
            self.log("Loading problem...")
            problem = rastion.load_qubots_model(problem_name)
            
            self.log("Loading optimizer...")
            optimizer = rastion.load_qubots_model(optimizer_name)
            
            # Apply parameters if provided
            if problem_params:
                self._apply_problem_parameters(problem, problem_params)
            
            if optimizer_params:
                self._apply_optimizer_parameters(optimizer, optimizer_params)
            
            # Execute optimization
            self.log("Starting optimization...")
            optimization_start = time.time()
            
            result = optimizer.optimize(problem)
            
            optimization_time = time.time() - optimization_start
            total_time = time.time() - start_time
            
            # Validate result
            result_validation = self._validate_optimization_result(result)
            
            return ExecutionResult(
                success=True,
                execution_type="direct",
                execution_time=total_time,
                result_data={
                    "optimization_result": result,
                    "result_validation": result_validation,
                    "problem_type": type(problem).__name__,
                    "optimizer_type": type(optimizer).__name__
                },
                performance_metrics={
                    "optimization_time": optimization_time,
                    "total_time": total_time,
                    "loading_time": optimization_start - start_time
                },
                real_time_output=self.output_buffer.copy()
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            
            return ExecutionResult(
                success=False,
                execution_type="direct",
                execution_time=total_time,
                error_message=str(e),
                error_details={
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                },
                real_time_output=self.output_buffer.copy()
            )
    
    def validate_playground_execution(self, problem_name: str, optimizer_name: str,
                                    problem_params: Optional[Dict[str, Any]] = None,
                                    optimizer_params: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Validate playground execution of problem and optimizer.
        
        Args:
            problem_name: Name of the problem repository
            optimizer_name: Name of the optimizer repository
            problem_params: Parameters for the problem
            optimizer_params: Parameters for the optimizer
            
        Returns:
            ExecutionResult with validation details
        """
        start_time = time.time()
        self.output_buffer.clear()
        
        try:
            self.log(f"Playground execution test: {problem_name} + {optimizer_name}")
            
            # Set default parameters
            if problem_params is None:
                problem_params = {}
            if optimizer_params is None:
                optimizer_params = {"population_size": 10, "generations": 5}
            
            # Execute in playground
            self.log("Starting playground optimization...")
            playground_start = time.time()
            
            result = execute_playground_optimization(
                problem_name=problem_name,
                optimizer_name=optimizer_name,
                problem_params=problem_params,
                optimizer_params=optimizer_params
            )
            
            playground_time = time.time() - playground_start
            total_time = time.time() - start_time
            
            # Validate playground result
            if isinstance(result, dict) and "error" in result:
                raise ExecutionError(f"Playground execution failed: {result['error']}")
            
            playground_validation = self._validate_playground_result(result)
            
            return ExecutionResult(
                success=True,
                execution_type="playground",
                execution_time=total_time,
                result_data={
                    "playground_result": result,
                    "result_validation": playground_validation,
                    "problem_params": problem_params,
                    "optimizer_params": optimizer_params
                },
                performance_metrics={
                    "playground_time": playground_time,
                    "total_time": total_time
                },
                real_time_output=self.output_buffer.copy()
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            
            return ExecutionResult(
                success=False,
                execution_type="playground",
                execution_time=total_time,
                error_message=str(e),
                error_details={
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                },
                real_time_output=self.output_buffer.copy()
            )
    
    def validate_combined_execution(self, problem_name: str, optimizer_name: str,
                                  test_params: Optional[List[Dict[str, Any]]] = None) -> ExecutionResult:
        """
        Validate both direct and playground execution with multiple parameter sets.
        
        Args:
            problem_name: Name of the problem repository
            optimizer_name: Name of the optimizer repository
            test_params: List of parameter combinations to test
            
        Returns:
            ExecutionResult with combined validation details
        """
        start_time = time.time()
        self.output_buffer.clear()
        
        try:
            self.log(f"Combined execution test: {problem_name} + {optimizer_name}")
            
            if test_params is None:
                test_params = [
                    {"problem_params": {}, "optimizer_params": {"population_size": 5, "generations": 3}},
                    {"problem_params": {}, "optimizer_params": {"population_size": 10, "generations": 5}}
                ]
            
            results = {
                "direct_tests": [],
                "playground_tests": [],
                "comparison_metrics": {}
            }
            
            # Run tests with different parameters
            for i, params in enumerate(test_params):
                self.log(f"Running test configuration {i+1}/{len(test_params)}")
                
                problem_params = params.get("problem_params", {})
                optimizer_params = params.get("optimizer_params", {})
                
                # Direct execution test
                direct_result = self.validate_direct_execution(
                    problem_name, optimizer_name, problem_params, optimizer_params
                )
                results["direct_tests"].append(direct_result)
                
                # Playground execution test
                playground_result = self.validate_playground_execution(
                    problem_name, optimizer_name, problem_params, optimizer_params
                )
                results["playground_tests"].append(playground_result)
            
            # Calculate comparison metrics
            results["comparison_metrics"] = self._calculate_comparison_metrics(
                results["direct_tests"], results["playground_tests"]
            )
            
            total_time = time.time() - start_time
            
            # Determine overall success
            all_direct_success = all(r.success for r in results["direct_tests"])
            all_playground_success = all(r.success for r in results["playground_tests"])
            overall_success = all_direct_success and all_playground_success
            
            return ExecutionResult(
                success=overall_success,
                execution_type="combined",
                execution_time=total_time,
                result_data=results,
                performance_metrics={
                    "total_time": total_time,
                    "test_configurations": len(test_params),
                    "direct_success_rate": sum(r.success for r in results["direct_tests"]) / len(results["direct_tests"]),
                    "playground_success_rate": sum(r.success for r in results["playground_tests"]) / len(results["playground_tests"])
                },
                real_time_output=self.output_buffer.copy()
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            
            return ExecutionResult(
                success=False,
                execution_type="combined",
                execution_time=total_time,
                error_message=str(e),
                error_details={
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                },
                real_time_output=self.output_buffer.copy()
            )
    
    def _apply_problem_parameters(self, problem: BaseProblem, params: Dict[str, Any]):
        """Apply parameters to a problem instance."""
        for key, value in params.items():
            if hasattr(problem, key):
                setattr(problem, key, value)
                self.log(f"Applied problem parameter: {key} = {value}")
    
    def _apply_optimizer_parameters(self, optimizer: BaseOptimizer, params: Dict[str, Any]):
        """Apply parameters to an optimizer instance."""
        for key, value in params.items():
            if hasattr(optimizer, key):
                setattr(optimizer, key, value)
                self.log(f"Applied optimizer parameter: {key} = {value}")
    
    def _validate_optimization_result(self, result: Any) -> Dict[str, Any]:
        """Validate an optimization result."""
        validation = {
            "valid": True,
            "result_type": type(result).__name__,
            "has_best_value": hasattr(result, 'best_value'),
            "has_best_solution": hasattr(result, 'best_solution'),
            "has_runtime": hasattr(result, 'runtime_seconds')
        }
        
        if hasattr(result, 'best_value'):
            validation["best_value"] = result.best_value
            validation["best_value_type"] = type(result.best_value).__name__
        
        if hasattr(result, 'runtime_seconds'):
            validation["runtime_seconds"] = result.runtime_seconds
        
        return validation
    
    def _validate_playground_result(self, result: Any) -> Dict[str, Any]:
        """Validate a playground execution result."""
        validation = {
            "valid": True,
            "result_type": type(result).__name__
        }
        
        if isinstance(result, dict):
            validation["is_dict"] = True
            validation["keys"] = list(result.keys())
            
            if "best_value" in result:
                validation["has_best_value"] = True
                validation["best_value"] = result["best_value"]
            
            if "runtime_seconds" in result:
                validation["has_runtime"] = True
                validation["runtime_seconds"] = result["runtime_seconds"]
        else:
            validation["is_dict"] = False
            
        return validation
    
    def _calculate_comparison_metrics(self, direct_results: List[ExecutionResult], 
                                    playground_results: List[ExecutionResult]) -> Dict[str, Any]:
        """Calculate comparison metrics between direct and playground execution."""
        metrics = {
            "direct_avg_time": 0.0,
            "playground_avg_time": 0.0,
            "time_difference": 0.0,
            "consistency_check": True
        }
        
        if direct_results and playground_results:
            # Calculate average execution times
            direct_times = [r.execution_time for r in direct_results if r.success]
            playground_times = [r.execution_time for r in playground_results if r.success]
            
            if direct_times:
                metrics["direct_avg_time"] = sum(direct_times) / len(direct_times)
            
            if playground_times:
                metrics["playground_avg_time"] = sum(playground_times) / len(playground_times)
            
            if direct_times and playground_times:
                metrics["time_difference"] = abs(metrics["direct_avg_time"] - metrics["playground_avg_time"])
        
        return metrics
