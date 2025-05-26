"""
Test Runner for Qubots Platform Integration

Provides high-level functions to run comprehensive tests and validations
for the complete upload/load/execute workflow.
"""

import time
from typing import Dict, List, Any, Optional, Union

from .model_testing import ModelTester, TestResult
from .upload_validator import UploadValidator, UploadResult
from .load_validator import LoadValidator, LoadResult
from .execution_validator import ExecutionValidator, ExecutionResult
from .debugging_tools import DebuggingTools, DebugReport


def run_comprehensive_test(model_path: str, 
                         test_name_prefix: Optional[str] = None,
                         cleanup: bool = False,
                         verbose: bool = True) -> Dict[str, Any]:
    """
    Run a comprehensive test of the complete model workflow.
    
    This function tests the entire process of uploading, loading, and executing
    a model on the Rastion platform.
    
    Args:
        model_path: Path to the model directory
        test_name_prefix: Prefix for test repository names
        cleanup: Whether to cleanup test models after testing
        verbose: Enable verbose output
        
    Returns:
        Comprehensive test results
    """
    print("🚀 Starting Comprehensive Qubots Model Test")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize tester
    tester = ModelTester(verbose=verbose, cleanup=cleanup)
    
    try:
        # Run the complete workflow test
        workflow_results = tester.test_model_workflow(model_path, test_name_prefix)
        
        # Generate summary
        summary = tester.get_test_summary()
        
        # Print results
        tester.print_summary()
        
        total_time = time.time() - start_time
        
        return {
            "success": summary["success_rate"] == 1.0,
            "workflow_results": workflow_results,
            "summary": summary,
            "total_time": total_time,
            "test_models": tester.test_models
        }
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        
        # Generate debug report
        debug_tools = DebuggingTools(verbose=verbose)
        debug_report = debug_tools.generate_debug_report(model_path, e)
        debug_tools.print_debug_report(debug_report)
        
        return {
            "success": False,
            "error": str(e),
            "debug_report": debug_report,
            "total_time": time.time() - start_time
        }


def run_vrp_integration_test(verbose: bool = True) -> Dict[str, Any]:
    """
    Run integration test specifically for VRP models.
    
    This function tests the VRP problem and genetic optimizer examples
    to validate the complete integration workflow.
    
    Args:
        verbose: Enable verbose output
        
    Returns:
        VRP integration test results
    """
    print("🚛 Starting VRP Integration Test")
    print("=" * 50)
    
    start_time = time.time()
    
    # Test both VRP models
    vrp_problem_path = "examples/vehicle_routing_problem"
    genetic_optimizer_path = "examples/genetic_vrp_optimizer"
    
    results = {
        "vrp_problem": None,
        "genetic_optimizer": None,
        "combined_execution": None,
        "success": False
    }
    
    try:
        # Test VRP problem
        print("\n📦 Testing VRP Problem...")
        vrp_result = run_comprehensive_test(
            vrp_problem_path, 
            test_name_prefix="vrp_integration_test",
            verbose=verbose
        )
        results["vrp_problem"] = vrp_result
        
        # Test genetic optimizer
        print("\n🧬 Testing Genetic VRP Optimizer...")
        optimizer_result = run_comprehensive_test(
            genetic_optimizer_path,
            test_name_prefix="vrp_integration_test", 
            verbose=verbose
        )
        results["genetic_optimizer"] = optimizer_result
        
        # Test combined execution if both succeeded
        if vrp_result["success"] and optimizer_result["success"]:
            print("\n🔄 Testing Combined Execution...")
            
            execution_validator = ExecutionValidator(verbose=verbose)
            
            # Get repository names from upload results
            vrp_repo = None
            optimizer_repo = None
            
            for test_result in vrp_result["workflow_results"]:
                if test_result.test_name == "Model Upload" and test_result.is_success():
                    vrp_repo = test_result.details.get("repository_name")
                    break
            
            for test_result in optimizer_result["workflow_results"]:
                if test_result.test_name == "Model Upload" and test_result.is_success():
                    optimizer_repo = test_result.details.get("repository_name")
                    break
            
            if vrp_repo and optimizer_repo:
                combined_result = execution_validator.validate_combined_execution(
                    vrp_repo, optimizer_repo
                )
                results["combined_execution"] = combined_result
                
                results["success"] = (vrp_result["success"] and 
                                    optimizer_result["success"] and 
                                    combined_result.success)
            else:
                print("⚠️ Could not find repository names for combined execution test")
        
        total_time = time.time() - start_time
        results["total_time"] = total_time
        
        # Print final summary
        print("\n" + "=" * 60)
        print("🎯 VRP INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        if results["success"]:
            print("✅ All VRP integration tests passed!")
        else:
            print("❌ Some VRP integration tests failed")
        
        print(f"⏱️ Total time: {total_time:.2f} seconds")
        
        return results
        
    except Exception as e:
        print(f"❌ VRP integration test failed: {e}")
        
        # Generate debug report
        debug_tools = DebuggingTools(verbose=verbose)
        debug_report = debug_tools.generate_debug_report(error=e)
        debug_tools.print_debug_report(debug_report)
        
        return {
            "success": False,
            "error": str(e),
            "debug_report": debug_report,
            "total_time": time.time() - start_time
        }


def validate_model_upload(model_path: str, repository_name: str, 
                        description: str, verbose: bool = True) -> UploadResult:
    """
    Validate model upload process.
    
    Args:
        model_path: Path to the model directory
        repository_name: Name for the repository
        description: Description for the model
        verbose: Enable verbose output
        
    Returns:
        Upload validation result
    """
    validator = UploadValidator(verbose=verbose)
    return validator.validate_upload_process(model_path, repository_name, description)


def validate_model_loading(repository_name: str, 
                         username: Optional[str] = None,
                         verbose: bool = True) -> LoadResult:
    """
    Validate model loading process.
    
    Args:
        repository_name: Name of the repository to load
        username: Repository owner (auto-detected if None)
        verbose: Enable verbose output
        
    Returns:
        Load validation result
    """
    validator = LoadValidator(verbose=verbose)
    return validator.validate_model_loading(repository_name, username)


def validate_model_execution(problem_name: str, optimizer_name: str,
                           execution_type: str = "combined",
                           problem_params: Optional[Dict[str, Any]] = None,
                           optimizer_params: Optional[Dict[str, Any]] = None,
                           verbose: bool = True) -> ExecutionResult:
    """
    Validate model execution process.
    
    Args:
        problem_name: Name of the problem repository
        optimizer_name: Name of the optimizer repository
        execution_type: Type of execution test ("direct", "playground", "combined")
        problem_params: Parameters for the problem
        optimizer_params: Parameters for the optimizer
        verbose: Enable verbose output
        
    Returns:
        Execution validation result
    """
    validator = ExecutionValidator(verbose=verbose)
    
    if execution_type == "direct":
        return validator.validate_direct_execution(
            problem_name, optimizer_name, problem_params, optimizer_params
        )
    elif execution_type == "playground":
        return validator.validate_playground_execution(
            problem_name, optimizer_name, problem_params, optimizer_params
        )
    elif execution_type == "combined":
        return validator.validate_combined_execution(
            problem_name, optimizer_name
        )
    else:
        raise ValueError(f"Unknown execution type: {execution_type}")


def diagnose_model_issues(model_path: str, 
                        error: Optional[Exception] = None,
                        verbose: bool = True) -> DebugReport:
    """
    Diagnose issues with a model.
    
    Args:
        model_path: Path to the model directory
        error: Error that occurred (optional)
        verbose: Enable verbose output
        
    Returns:
        Debug report with analysis and recommendations
    """
    debug_tools = DebuggingTools(verbose=verbose)
    report = debug_tools.generate_debug_report(model_path, error)
    
    if verbose:
        debug_tools.print_debug_report(report)
    
    return report


def test_existing_models(model_names: List[str], 
                       test_execution: bool = True,
                       verbose: bool = True) -> Dict[str, Any]:
    """
    Test existing models on the Rastion platform.
    
    Args:
        model_names: List of model repository names to test
        test_execution: Whether to test execution (requires compatible pairs)
        verbose: Enable verbose output
        
    Returns:
        Test results for all models
    """
    print(f"🧪 Testing {len(model_names)} Existing Models")
    print("=" * 50)
    
    start_time = time.time()
    
    # Test loading for all models
    load_validator = LoadValidator(verbose=verbose)
    load_results = load_validator.validate_multiple_models(model_names)
    
    results = {
        "load_results": load_results,
        "execution_results": {},
        "summary": {
            "total_models": len(model_names),
            "load_success": sum(1 for r in load_results.values() if r.success),
            "execution_success": 0
        }
    }
    
    # Test execution if requested
    if test_execution:
        execution_validator = ExecutionValidator(verbose=verbose)
        
        # Find problem/optimizer pairs
        problems = [name for name, result in load_results.items() 
                   if result.success and "problem" in result.model_type.lower()]
        optimizers = [name for name, result in load_results.items() 
                     if result.success and "optimizer" in result.model_type.lower()]
        
        print(f"\n🔄 Found {len(problems)} problems and {len(optimizers)} optimizers")
        
        # Test compatible pairs
        for problem in problems:
            for optimizer in optimizers:
                if "vrp" in problem.lower() and "vrp" in optimizer.lower():
                    print(f"Testing execution: {problem} + {optimizer}")
                    
                    exec_result = execution_validator.validate_playground_execution(
                        problem, optimizer
                    )
                    
                    results["execution_results"][f"{problem}+{optimizer}"] = exec_result
                    
                    if exec_result.success:
                        results["summary"]["execution_success"] += 1
    
    total_time = time.time() - start_time
    results["total_time"] = total_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 EXISTING MODELS TEST SUMMARY")
    print("=" * 60)
    
    summary = results["summary"]
    print(f"📦 Models tested: {summary['total_models']}")
    print(f"✅ Load success: {summary['load_success']}/{summary['total_models']}")
    
    if test_execution:
        print(f"🔄 Execution success: {summary['execution_success']}")
    
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    
    return results
