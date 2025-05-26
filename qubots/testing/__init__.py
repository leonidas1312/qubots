"""
Comprehensive Testing and Debugging Framework for Qubots Platform Integration

This module provides a complete testing framework for systematically uploading,
loading, and validating optimization models on the Rastion platform.

Key Features:
- Model upload testing and validation
- Model loading verification
- Execution testing with real-time monitoring
- Comprehensive debugging tools
- VRP implementation testing
- Error reporting and troubleshooting

Usage:
    from qubots.testing import ModelTester, run_comprehensive_test
    
    # Run comprehensive test suite
    results = run_comprehensive_test("examples/vehicle_routing_problem")
    
    # Test specific model
    tester = ModelTester()
    result = tester.test_model_workflow("my_model_path")
"""

from .model_testing import (
    ModelTester,
    TestResult,
    ValidationError,
    TestStatus,
    TestType
)

from .upload_validator import (
    UploadValidator,
    UploadResult,
    UploadError
)

from .load_validator import (
    LoadValidator,
    LoadResult,
    LoadError
)

from .execution_validator import (
    ExecutionValidator,
    ExecutionResult,
    ExecutionError
)

from .debugging_tools import (
    DebuggingTools,
    DebugReport,
    ErrorAnalyzer,
    PerformanceProfiler
)

from .test_runner import (
    run_comprehensive_test,
    run_vrp_integration_test,
    validate_model_upload,
    validate_model_loading,
    validate_model_execution
)

__version__ = "1.0.0"

__all__ = [
    # Core testing classes
    "ModelTester",
    "TestResult",
    "ValidationError",
    "TestStatus",
    "TestType",
    
    # Upload validation
    "UploadValidator",
    "UploadResult",
    "UploadError",
    
    # Load validation
    "LoadValidator",
    "LoadResult",
    "LoadError",
    
    # Execution validation
    "ExecutionValidator",
    "ExecutionResult",
    "ExecutionError",
    
    # Debugging tools
    "DebuggingTools",
    "DebugReport",
    "ErrorAnalyzer",
    "PerformanceProfiler",
    
    # Test runners
    "run_comprehensive_test",
    "run_vrp_integration_test",
    "validate_model_upload",
    "validate_model_loading",
    "validate_model_execution",
]
