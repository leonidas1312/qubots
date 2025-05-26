#!/usr/bin/env python3
"""
Command Line Interface for Qubots Testing Framework

Provides command-line access to the testing and debugging tools.

Usage:
    python -m qubots.testing.cli test <model_path>
    python -m qubots.testing.cli validate-upload <model_path> <repo_name>
    python -m qubots.testing.cli validate-load <repo_name>
    python -m qubots.testing.cli diagnose <model_path>
    python -m qubots.testing.cli vrp-test
"""

import sys
import argparse
from pathlib import Path

from .test_runner import (
    run_comprehensive_test,
    run_vrp_integration_test,
    validate_model_upload,
    validate_model_loading,
    validate_model_execution,
    diagnose_model_issues,
    test_existing_models
)


def cmd_test(args):
    """Run comprehensive test for a model."""
    print(f"🧪 Testing model: {args.model_path}")
    
    results = run_comprehensive_test(
        model_path=args.model_path,
        test_name_prefix=args.prefix,
        cleanup=args.cleanup,
        verbose=args.verbose
    )
    
    if results["success"]:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


def cmd_validate_upload(args):
    """Validate model upload process."""
    print(f"📤 Validating upload: {args.model_path} -> {args.repository_name}")
    
    result = validate_model_upload(
        model_path=args.model_path,
        repository_name=args.repository_name,
        description=args.description or f"Test upload of {args.repository_name}",
        verbose=args.verbose
    )
    
    if result.success:
        print(f"✅ Upload validation successful: {result.repository_url}")
        return 0
    else:
        print(f"❌ Upload validation failed: {result.error_message}")
        return 1


def cmd_validate_load(args):
    """Validate model loading process."""
    print(f"📥 Validating load: {args.repository_name}")
    
    result = validate_model_loading(
        repository_name=args.repository_name,
        username=args.username,
        verbose=args.verbose
    )
    
    if result.success:
        print(f"✅ Load validation successful: {result.model_type}")
        return 0
    else:
        print(f"❌ Load validation failed: {result.error_message}")
        return 1


def cmd_validate_execution(args):
    """Validate model execution."""
    print(f"🔄 Validating execution: {args.problem_name} + {args.optimizer_name}")
    
    result = validate_model_execution(
        problem_name=args.problem_name,
        optimizer_name=args.optimizer_name,
        execution_type=args.execution_type,
        verbose=args.verbose
    )
    
    if result.success:
        print(f"✅ Execution validation successful")
        return 0
    else:
        print(f"❌ Execution validation failed: {result.error_message}")
        return 1


def cmd_diagnose(args):
    """Diagnose model issues."""
    print(f"🔍 Diagnosing model: {args.model_path}")
    
    debug_report = diagnose_model_issues(
        model_path=args.model_path,
        verbose=args.verbose
    )
    
    print(f"\n📋 Diagnosis completed")
    print(f"   Recommendations: {len(debug_report.recommendations)}")
    
    return 0


def cmd_vrp_test(args):
    """Run VRP integration test."""
    print("🚛 Running VRP integration test")
    
    results = run_vrp_integration_test(verbose=args.verbose)
    
    if results["success"]:
        print("\n✅ VRP integration test passed!")
        return 0
    else:
        print("\n❌ VRP integration test failed")
        return 1


def cmd_test_existing(args):
    """Test existing models on platform."""
    model_names = args.model_names.split(',')
    print(f"📦 Testing {len(model_names)} existing models")
    
    results = test_existing_models(
        model_names=model_names,
        test_execution=args.test_execution,
        verbose=args.verbose
    )
    
    summary = results["summary"]
    success_rate = summary["load_success"] / summary["total_models"]
    
    print(f"\n📊 Results: {summary['load_success']}/{summary['total_models']} models loaded successfully")
    
    if success_rate >= 0.8:
        return 0
    else:
        return 1


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Qubots Testing Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s test examples/vehicle_routing_problem
  %(prog)s validate-upload examples/vrp my_vrp_test
  %(prog)s validate-load my_vrp_test
  %(prog)s diagnose examples/vehicle_routing_problem
  %(prog)s vrp-test
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run comprehensive test")
    test_parser.add_argument("model_path", help="Path to model directory")
    test_parser.add_argument("--prefix", default=None, help="Test name prefix")
    test_parser.add_argument("--cleanup", action="store_true", help="Cleanup test models")
    test_parser.set_defaults(func=cmd_test)
    
    # Validate upload command
    upload_parser = subparsers.add_parser("validate-upload", help="Validate upload process")
    upload_parser.add_argument("model_path", help="Path to model directory")
    upload_parser.add_argument("repository_name", help="Repository name for upload")
    upload_parser.add_argument("--description", help="Repository description")
    upload_parser.set_defaults(func=cmd_validate_upload)
    
    # Validate load command
    load_parser = subparsers.add_parser("validate-load", help="Validate load process")
    load_parser.add_argument("repository_name", help="Repository name to load")
    load_parser.add_argument("--username", help="Repository owner username")
    load_parser.set_defaults(func=cmd_validate_load)
    
    # Validate execution command
    exec_parser = subparsers.add_parser("validate-execution", help="Validate execution")
    exec_parser.add_argument("problem_name", help="Problem repository name")
    exec_parser.add_argument("optimizer_name", help="Optimizer repository name")
    exec_parser.add_argument("--execution-type", choices=["direct", "playground", "combined"], 
                           default="playground", help="Type of execution test")
    exec_parser.set_defaults(func=cmd_validate_execution)
    
    # Diagnose command
    diagnose_parser = subparsers.add_parser("diagnose", help="Diagnose model issues")
    diagnose_parser.add_argument("model_path", help="Path to model directory")
    diagnose_parser.set_defaults(func=cmd_diagnose)
    
    # VRP test command
    vrp_parser = subparsers.add_parser("vrp-test", help="Run VRP integration test")
    vrp_parser.set_defaults(func=cmd_vrp_test)
    
    # Test existing models command
    existing_parser = subparsers.add_parser("test-existing", help="Test existing models")
    existing_parser.add_argument("model_names", help="Comma-separated list of model names")
    existing_parser.add_argument("--test-execution", action="store_true", 
                                help="Test execution (requires compatible pairs)")
    existing_parser.set_defaults(func=cmd_test_existing)
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
