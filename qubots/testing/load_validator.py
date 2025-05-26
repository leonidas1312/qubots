"""
Load Validator for Qubots Models

Provides comprehensive validation for the model loading process from the Rastion platform,
including load testing, model verification, and interface validation.
"""

import time
import traceback
import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable

from ..base_problem import BaseProblem
from ..base_optimizer import BaseOptimizer
import qubots.rastion as rastion


@dataclass
class LoadResult:
    """Result of a load validation test."""
    success: bool
    repository_name: str = ""
    model_type: str = ""
    load_time: float = 0.0
    validation_details: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    error_details: Dict[str, Any] = field(default_factory=dict)


class LoadError(Exception):
    """Custom exception for load validation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class LoadValidator:
    """
    Comprehensive validator for model loading process.
    
    This class provides methods to validate that models can be loaded
    correctly from the Rastion platform and have the expected interfaces.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the load validator.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] LOAD {level}: {message}")
    
    def validate_model_loading(self, repository_name: str, 
                             username: Optional[str] = None) -> LoadResult:
        """
        Validate the complete model loading process.
        
        Args:
            repository_name: Name of the repository to load
            username: Repository owner (auto-detected if None)
            
        Returns:
            LoadResult with validation details
        """
        start_time = time.time()
        
        try:
            self.log(f"Loading model: {repository_name}")
            
            # Step 1: Load the model
            model = rastion.load_qubots_model(repository_name, username)
            
            # Step 2: Validate model type
            model_validation = self._validate_model_type(model)
            
            # Step 3: Validate model interface
            interface_validation = self._validate_model_interface(model)
            
            # Step 4: Validate model metadata
            metadata_validation = self._validate_model_metadata(model)
            
            # Step 5: Test basic functionality
            functionality_validation = self._validate_basic_functionality(model)
            
            load_time = time.time() - start_time
            
            return LoadResult(
                success=True,
                repository_name=repository_name,
                model_type=type(model).__name__,
                load_time=load_time,
                validation_details={
                    "model_validation": model_validation,
                    "interface_validation": interface_validation,
                    "metadata_validation": metadata_validation,
                    "functionality_validation": functionality_validation
                }
            )
            
        except Exception as e:
            load_time = time.time() - start_time
            
            return LoadResult(
                success=False,
                repository_name=repository_name,
                load_time=load_time,
                error_message=str(e),
                error_details={
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            )
    
    def _validate_model_type(self, model: Any) -> Dict[str, Any]:
        """Validate that the loaded model is a valid qubots model."""
        is_problem = isinstance(model, BaseProblem)
        is_optimizer = isinstance(model, BaseOptimizer)
        
        if not (is_problem or is_optimizer):
            raise LoadError(f"Loaded model is not a valid qubots model: {type(model)}")
        
        return {
            "valid": True,
            "is_problem": is_problem,
            "is_optimizer": is_optimizer,
            "model_class": type(model).__name__,
            "model_module": type(model).__module__,
            "base_classes": [cls.__name__ for cls in type(model).__mro__]
        }
    
    def _validate_model_interface(self, model: Any) -> Dict[str, Any]:
        """Validate that the model has the expected interface."""
        validation_results = {
            "valid": True,
            "missing_methods": [],
            "method_signatures": {},
            "properties": {}
        }
        
        if isinstance(model, BaseProblem):
            required_methods = [
                "evaluate_solution",
                "is_valid",
                "get_random_solution"
            ]
            
            # Check for optional methods
            optional_methods = [
                "evaluate",
                "get_solution_summary",
                "validate_solution_format"
            ]
            
        elif isinstance(model, BaseOptimizer):
            required_methods = [
                "optimize",
                "_optimize_implementation"
            ]
            
            optional_methods = [
                "get_default_parameters",
                "validate_parameters"
            ]
        else:
            raise LoadError(f"Unknown model type: {type(model)}")
        
        # Check required methods
        for method_name in required_methods:
            if not hasattr(model, method_name):
                validation_results["missing_methods"].append(method_name)
                validation_results["valid"] = False
            else:
                method = getattr(model, method_name)
                if callable(method):
                    try:
                        sig = inspect.signature(method)
                        validation_results["method_signatures"][method_name] = str(sig)
                    except Exception:
                        validation_results["method_signatures"][method_name] = "signature_unavailable"
        
        # Check optional methods (informational)
        for method_name in optional_methods:
            if hasattr(model, method_name):
                method = getattr(model, method_name)
                if callable(method):
                    try:
                        sig = inspect.signature(method)
                        validation_results["method_signatures"][method_name] = str(sig)
                    except Exception:
                        validation_results["method_signatures"][method_name] = "signature_unavailable"
        
        # Check properties
        if hasattr(model, '_metadata') and model._metadata:
            validation_results["properties"]["has_metadata"] = True
            validation_results["properties"]["metadata_type"] = type(model._metadata).__name__
        else:
            validation_results["properties"]["has_metadata"] = False
        
        return validation_results
    
    def _validate_model_metadata(self, model: Any) -> Dict[str, Any]:
        """Validate the model's metadata."""
        validation_results = {
            "valid": True,
            "has_metadata": False,
            "metadata_fields": {}
        }
        
        if hasattr(model, '_metadata') and model._metadata:
            validation_results["has_metadata"] = True
            metadata = model._metadata
            
            # Check common metadata fields
            expected_fields = ["name", "description", "author", "version"]
            for field in expected_fields:
                if hasattr(metadata, field):
                    value = getattr(metadata, field)
                    validation_results["metadata_fields"][field] = {
                        "present": True,
                        "value": str(value),
                        "type": type(value).__name__
                    }
                else:
                    validation_results["metadata_fields"][field] = {
                        "present": False
                    }
        
        return validation_results
    
    def _validate_basic_functionality(self, model: Any) -> Dict[str, Any]:
        """Test basic functionality of the loaded model."""
        validation_results = {
            "valid": True,
            "tests_performed": [],
            "test_results": {}
        }
        
        try:
            if isinstance(model, BaseProblem):
                # Test problem functionality
                self._test_problem_functionality(model, validation_results)
            elif isinstance(model, BaseOptimizer):
                # Test optimizer functionality
                self._test_optimizer_functionality(model, validation_results)
        
        except Exception as e:
            validation_results["valid"] = False
            validation_results["error"] = str(e)
            validation_results["error_trace"] = traceback.format_exc()
        
        return validation_results
    
    def _test_problem_functionality(self, problem: BaseProblem, results: Dict[str, Any]):
        """Test basic problem functionality."""
        # Test random solution generation
        try:
            solution = problem.get_random_solution()
            results["tests_performed"].append("get_random_solution")
            results["test_results"]["get_random_solution"] = {
                "success": True,
                "solution_type": type(solution).__name__,
                "solution_length": len(solution) if hasattr(solution, '__len__') else "unknown"
            }
        except Exception as e:
            results["test_results"]["get_random_solution"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test solution validation
        if "get_random_solution" in results["test_results"] and \
           results["test_results"]["get_random_solution"]["success"]:
            try:
                is_valid = problem.is_valid(solution)
                results["tests_performed"].append("is_valid")
                results["test_results"]["is_valid"] = {
                    "success": True,
                    "result": is_valid,
                    "result_type": type(is_valid).__name__
                }
            except Exception as e:
                results["test_results"]["is_valid"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Test solution evaluation
        if "get_random_solution" in results["test_results"] and \
           results["test_results"]["get_random_solution"]["success"]:
            try:
                evaluation = problem.evaluate_solution(solution)
                results["tests_performed"].append("evaluate_solution")
                results["test_results"]["evaluate_solution"] = {
                    "success": True,
                    "result": evaluation,
                    "result_type": type(evaluation).__name__
                }
            except Exception as e:
                results["test_results"]["evaluate_solution"] = {
                    "success": False,
                    "error": str(e)
                }
    
    def _test_optimizer_functionality(self, optimizer: BaseOptimizer, results: Dict[str, Any]):
        """Test basic optimizer functionality."""
        # Test parameter access
        try:
            if hasattr(optimizer, 'get_default_parameters'):
                params = optimizer.get_default_parameters()
                results["tests_performed"].append("get_default_parameters")
                results["test_results"]["get_default_parameters"] = {
                    "success": True,
                    "params_type": type(params).__name__,
                    "params_count": len(params) if hasattr(params, '__len__') else "unknown"
                }
        except Exception as e:
            results["test_results"]["get_default_parameters"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test that optimize method exists and is callable
        try:
            optimize_method = getattr(optimizer, 'optimize')
            results["tests_performed"].append("optimize_method_check")
            results["test_results"]["optimize_method_check"] = {
                "success": True,
                "is_callable": callable(optimize_method),
                "signature": str(inspect.signature(optimize_method)) if callable(optimize_method) else "not_callable"
            }
        except Exception as e:
            results["test_results"]["optimize_method_check"] = {
                "success": False,
                "error": str(e)
            }
    
    def validate_multiple_models(self, repository_names: List[str]) -> Dict[str, LoadResult]:
        """
        Validate loading of multiple models.
        
        Args:
            repository_names: List of repository names to validate
            
        Returns:
            Dictionary mapping repository names to LoadResults
        """
        results = {}
        
        for repo_name in repository_names:
            self.log(f"Validating model: {repo_name}")
            result = self.validate_model_loading(repo_name)
            results[repo_name] = result
            
            if result.success:
                self.log(f"✅ {repo_name}: Load validation passed")
            else:
                self.log(f"❌ {repo_name}: Load validation failed - {result.error_message}")
        
        return results
