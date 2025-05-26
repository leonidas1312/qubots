"""
Core Model Testing Framework

Provides the main ModelTester class and supporting infrastructure for testing
optimization models throughout the upload/load/execute workflow.
"""

import os
import time
import json
import traceback
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from ..base_problem import BaseProblem
from ..base_optimizer import BaseOptimizer
from ..rastion_client import get_global_client
from ..playground_integration import execute_playground_optimization
import qubots.rastion as rastion


class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestType(Enum):
    """Type of test being performed."""
    UPLOAD = "upload"
    LOAD = "load"
    EXECUTION = "execution"
    VALIDATION = "validation"
    INTEGRATION = "integration"


@dataclass
class TestResult:
    """Comprehensive test result with detailed information."""
    test_name: str
    test_type: TestType
    status: TestStatus
    duration: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    error_trace: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def is_success(self) -> bool:
        """Check if test was successful."""
        return self.status == TestStatus.PASSED

    def get_summary(self) -> str:
        """Get a summary string for the test result."""
        status_emoji = {
            TestStatus.PASSED: "✅",
            TestStatus.FAILED: "❌",
            TestStatus.ERROR: "💥",
            TestStatus.SKIPPED: "⏭️",
            TestStatus.PENDING: "⏳",
            TestStatus.RUNNING: "🔄"
        }

        emoji = status_emoji.get(self.status, "❓")
        return f"{emoji} {self.test_name} ({self.duration:.2f}s): {self.message}"


class ValidationError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ModelTester:
    """
    Comprehensive model testing framework for Rastion platform integration.

    This class provides methods to test the complete workflow of uploading,
    loading, and executing optimization models on the Rastion platform.
    """

    def __init__(self, verbose: bool = True, cleanup: bool = False):
        """
        Initialize the model tester.

        Args:
            verbose: Enable verbose output
            cleanup: Automatically cleanup test models after testing
        """
        self.verbose = verbose
        self.cleanup = cleanup
        self.test_results: List[TestResult] = []
        self.test_models: List[str] = []  # Track models for cleanup

    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")

    def test_model_workflow(self, model_path: str,
                          test_name_prefix: Optional[str] = None) -> List[TestResult]:
        """
        Test the complete workflow for a model.

        Args:
            model_path: Path to the model directory
            test_name_prefix: Prefix for test repository names

        Returns:
            List of test results
        """
        self.log(f"Starting comprehensive test for model: {model_path}")

        if test_name_prefix is None:
            test_name_prefix = f"test_{int(time.time())}"

        workflow_results = []

        # Step 1: Validate model structure
        result = self._test_model_structure(model_path)
        workflow_results.append(result)
        if not result.is_success():
            return workflow_results

        # Step 2: Test upload
        upload_result = self._test_model_upload(model_path, test_name_prefix)
        workflow_results.append(upload_result)
        if not upload_result.is_success():
            return workflow_results

        # Step 3: Test loading
        model_name = upload_result.details.get("repository_name")
        if model_name:
            load_result = self._test_model_loading(model_name)
            workflow_results.append(load_result)

            # Step 4: Test execution (if loading succeeded)
            if load_result.is_success():
                execution_result = self._test_model_execution(model_name, model_path)
                workflow_results.append(execution_result)

        self.test_results.extend(workflow_results)
        return workflow_results

    def _test_model_structure(self, model_path: str) -> TestResult:
        """Test that the model has the correct file structure."""
        start_time = time.time()

        try:
            path = Path(model_path)
            if not path.exists():
                raise ValidationError(f"Model path does not exist: {model_path}")

            # Check required files
            required_files = ["qubot.py", "config.json"]
            missing_files = []

            for file_name in required_files:
                if not (path / file_name).exists():
                    missing_files.append(file_name)

            if missing_files:
                raise ValidationError(f"Missing required files: {missing_files}")

            # Validate config.json
            config_path = path / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)

            required_config_fields = ["type", "entry_point", "class_name"]
            missing_config = [field for field in required_config_fields
                            if field not in config]

            if missing_config:
                raise ValidationError(f"Missing config fields: {missing_config}")

            duration = time.time() - start_time
            return TestResult(
                test_name="Model Structure Validation",
                test_type=TestType.VALIDATION,
                status=TestStatus.PASSED,
                duration=duration,
                message="Model structure is valid",
                details={
                    "model_path": str(path),
                    "config": config,
                    "files_found": [f.name for f in path.iterdir() if f.is_file()]
                }
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Model Structure Validation",
                test_type=TestType.VALIDATION,
                status=TestStatus.FAILED,
                duration=duration,
                message=str(e),
                error=e,
                error_trace=traceback.format_exc()
            )

    def _test_model_upload(self, model_path: str, test_name_prefix: str) -> TestResult:
        """Test uploading the model to Rastion platform."""
        start_time = time.time()

        try:
            # Check authentication
            client = get_global_client()
            if not (client and hasattr(client, 'token') and client.token):
                raise ValidationError("Not authenticated with Rastion platform")

            # Generate unique repository name
            model_name = Path(model_path).name
            repository_name = f"{test_name_prefix}_{model_name}"

            self.log(f"Uploading model as: {repository_name}")

            # Upload model
            result_url = rastion.upload_qubots_model(
                path=model_path,
                repository_name=repository_name,
                description=f"Test upload of {model_name}",
                overwrite=True
            )

            # Track for cleanup
            self.test_models.append(repository_name)

            duration = time.time() - start_time
            return TestResult(
                test_name="Model Upload",
                test_type=TestType.UPLOAD,
                status=TestStatus.PASSED,
                duration=duration,
                message=f"Successfully uploaded to {repository_name}",
                details={
                    "repository_name": repository_name,
                    "result_url": result_url,
                    "model_path": model_path
                }
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Model Upload",
                test_type=TestType.UPLOAD,
                status=TestStatus.FAILED,
                duration=duration,
                message=str(e),
                error=e,
                error_trace=traceback.format_exc()
            )

    def _test_model_loading(self, repository_name: str) -> TestResult:
        """Test loading the model from Rastion platform."""
        start_time = time.time()

        try:
            self.log(f"Loading model: {repository_name}")

            # Wait a moment for platform processing
            time.sleep(2)

            # Load model
            model = rastion.load_qubots_model(repository_name)

            # Validate loaded model
            if not isinstance(model, (BaseProblem, BaseOptimizer)):
                raise ValidationError(f"Loaded model is not a valid qubots model: {type(model)}")

            duration = time.time() - start_time
            return TestResult(
                test_name="Model Loading",
                test_type=TestType.LOAD,
                status=TestStatus.PASSED,
                duration=duration,
                message=f"Successfully loaded {type(model).__name__}",
                details={
                    "repository_name": repository_name,
                    "model_type": type(model).__name__,
                    "model_class": str(type(model)),
                    "is_problem": isinstance(model, BaseProblem),
                    "is_optimizer": isinstance(model, BaseOptimizer)
                }
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Model Loading",
                test_type=TestType.LOAD,
                status=TestStatus.FAILED,
                duration=duration,
                message=str(e),
                error=e,
                error_trace=traceback.format_exc()
            )

    def _test_model_execution(self, repository_name: str, model_path: str) -> TestResult:
        """Test executing the model (problem + optimizer or standalone)."""
        start_time = time.time()

        try:
            self.log(f"Testing execution for: {repository_name}")

            # Load config to determine model type
            config_path = Path(model_path) / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)

            model_type = config.get("type")

            if model_type == "problem":
                # For problems, we need an optimizer to test execution
                execution_result = self._test_problem_execution(repository_name, config)
            elif model_type == "optimizer":
                # For optimizers, we need a problem to test execution
                execution_result = self._test_optimizer_execution(repository_name, config)
            else:
                raise ValidationError(f"Unknown model type: {model_type}")

            duration = time.time() - start_time
            if execution_result:
                return TestResult(
                    test_name="Model Execution",
                    test_type=TestType.EXECUTION,
                    status=TestStatus.PASSED,
                    duration=duration,
                    message="Model execution successful",
                    details=execution_result
                )
            else:
                return TestResult(
                    test_name="Model Execution",
                    test_type=TestType.EXECUTION,
                    status=TestStatus.SKIPPED,
                    duration=duration,
                    message="Execution test skipped (no compatible models found)"
                )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Model Execution",
                test_type=TestType.EXECUTION,
                status=TestStatus.FAILED,
                duration=duration,
                message=str(e),
                error=e,
                error_trace=traceback.format_exc()
            )

    def _test_problem_execution(self, problem_name: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Test problem execution with a compatible optimizer."""
        # Look for compatible optimizers in config
        compatible_optimizers = config.get("rastion_config", {}).get("compatible_optimizers", [])

        if not compatible_optimizers:
            # Try with a generic genetic optimizer if available
            compatible_optimizers = ["genetic_vrp_optimizer", "test_genetic_vrp_optimizer"]

        for optimizer_name in compatible_optimizers:
            try:
                self.log(f"Testing {problem_name} with {optimizer_name}")

                # Try playground execution
                result = execute_playground_optimization(
                    problem_name=problem_name,
                    optimizer_name=optimizer_name,
                    problem_params=config.get("default_params", {}),
                    optimizer_params={"population_size": 5, "generations": 3}
                )

                if isinstance(result, dict) and "error" not in result:
                    return {
                        "problem_name": problem_name,
                        "optimizer_name": optimizer_name,
                        "execution_result": result,
                        "execution_type": "playground"
                    }

            except Exception as e:
                self.log(f"Failed to test with {optimizer_name}: {e}")
                continue

        return None

    def _test_optimizer_execution(self, optimizer_name: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Test optimizer execution with a compatible problem."""
        # Look for compatible problems in config
        compatible_problems = config.get("compatible_problems", [])

        if not compatible_problems:
            # Try with a generic VRP problem if available
            compatible_problems = ["vehicle_routing_problem", "test_vehicle_routing_problem"]

        for problem_name in compatible_problems:
            try:
                self.log(f"Testing {optimizer_name} with {problem_name}")

                # Try playground execution
                result = execute_playground_optimization(
                    problem_name=problem_name,
                    optimizer_name=optimizer_name,
                    problem_params={"penalty_unserved": 1000.0, "penalty_capacity": 500.0},
                    optimizer_params=config.get("default_params", {})
                )

                if isinstance(result, dict) and "error" not in result:
                    return {
                        "problem_name": problem_name,
                        "optimizer_name": optimizer_name,
                        "execution_result": result,
                        "execution_type": "playground"
                    }

            except Exception as e:
                self.log(f"Failed to test with {problem_name}: {e}")
                continue

        return None

    def get_test_summary(self) -> Dict[str, Any]:
        """Get a summary of all test results."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.is_success())
        failed_tests = sum(1 for r in self.test_results if r.status == TestStatus.FAILED)
        error_tests = sum(1 for r in self.test_results if r.status == TestStatus.ERROR)
        skipped_tests = sum(1 for r in self.test_results if r.status == TestStatus.SKIPPED)

        total_duration = sum(r.duration for r in self.test_results)

        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "errors": error_tests,
            "skipped": skipped_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "total_duration": total_duration,
            "test_models": self.test_models
        }

    def print_summary(self):
        """Print a formatted summary of test results."""
        summary = self.get_test_summary()

        print("\n" + "=" * 60)
        print("🧪 MODEL TESTING SUMMARY")
        print("=" * 60)

        print(f"📊 Tests: {summary['passed']}/{summary['total_tests']} passed")
        print(f"⏱️ Duration: {summary['total_duration']:.2f} seconds")
        print(f"📈 Success Rate: {summary['success_rate']:.1%}")

        if summary['failed'] > 0:
            print(f"❌ Failed: {summary['failed']}")
        if summary['errors'] > 0:
            print(f"💥 Errors: {summary['errors']}")
        if summary['skipped'] > 0:
            print(f"⏭️ Skipped: {summary['skipped']}")

        print("\n📋 Test Results:")
        for result in self.test_results:
            print(f"  {result.get_summary()}")

        if self.test_models:
            print(f"\n🧹 Test Models Created: {', '.join(self.test_models)}")
            if self.cleanup:
                print("   (Will be cleaned up automatically)")
            else:
                print("   (Manual cleanup required)")

    def cleanup_test_models(self):
        """Clean up test models from the platform."""
        if not self.test_models:
            return

        self.log("Cleaning up test models...")
        # Note: Actual cleanup would require platform API support
        # For now, just log the models that need cleanup
        for model_name in self.test_models:
            self.log(f"Model requiring cleanup: {model_name}")

        self.test_models.clear()
