#!/usr/bin/env python3
"""
Run Playground Tests

This script demonstrates how to test your qubots repositories for playground
consistency before uploading to Rastion. It runs a series of tests to ensure
your optimization will work the same way in the Rastion playground as locally.

Usage:
    python run_playground_tests.py [options]

Author: Qubots Community
Version: 1.0.0
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_test(script_name: str, problem: str, optimizer: str, extra_args: list = None) -> bool:
    """
    Run a test script and return success status.
    
    Args:
        script_name: Name of the test script to run
        problem: Problem repository name
        optimizer: Optimizer repository name
        extra_args: Additional arguments to pass to the script
        
    Returns:
        True if test passed, False otherwise
    """
    if extra_args is None:
        extra_args = []
    
    cmd = [sys.executable, script_name, problem, optimizer] + extra_args
    
    print(f"\n🚀 Running: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, check=False)
        success = result.returncode == 0
        
        if success:
            print("✅ Test PASSED")
        else:
            print(f"❌ Test FAILED (exit code: {result.returncode})")
        
        return success
        
    except Exception as e:
        print(f"❌ Test ERROR: {e}")
        return False


def main():
    """Main function to run playground tests."""
    parser = argparse.ArgumentParser(
        description="Run playground consistency tests for qubots repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  
  # Test all available problem-optimizer pairs
  python run_playground_tests.py
  
  # Test specific problem-optimizer pair
  python run_playground_tests.py --problem tsp --optimizer highs_tsp_solver
  
  # Run only consistency test
  python run_playground_tests.py --consistency-only
        """
    )
    
    parser.add_argument("--problem", help="Specific problem repository to test")
    parser.add_argument("--optimizer", help="Specific optimizer repository to test")
    parser.add_argument("--consistency-only", action="store_true",
                       help="Run only the consistency test")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of test iterations (default: 3)")
    
    args = parser.parse_args()
    
    print("🎯 Qubots Playground Testing Suite")
    print("="*50)
    
    # Define test cases (problem, optimizer pairs)
    test_cases = []
    
    if args.problem and args.optimizer:
        # Test specific pair
        test_cases = [(args.problem, args.optimizer)]
    else:
        # Test predefined pairs that should work
        test_cases = [
            ("tsp", "highs_tsp_solver"),
            ("maxcut_problem", "ortools_maxcut_optimizer"),
            ("vehicle_routing_problem", "genetic_vrp_optimizer"),
        ]
    
    print(f"📋 Test cases: {len(test_cases)}")
    for problem, optimizer in test_cases:
        print(f"   • {problem} + {optimizer}")
    
    # Track results
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    # Run tests for each case
    for problem, optimizer in test_cases:
        print(f"\n" + "="*70)
        print(f"🧪 Testing: {problem} + {optimizer}")
        print("="*70)
        
        case_passed = True
        
        # 1. Run consistency test (always run this)
        print(f"\n1️⃣ Playground Consistency Test")
        success = run_test("playground_consistency_test.py", problem, optimizer, 
                          ["--iterations", str(args.iterations)])
        total_tests += 1
        if success:
            passed_tests += 1
        else:
            case_passed = False
            failed_tests.append(f"{problem}+{optimizer}: consistency test")
        
        # 2. Run comprehensive playground test (if not consistency-only)
        if not args.consistency_only:
            print(f"\n2️⃣ Comprehensive Playground Test")
            success = run_test("test_playground_execution.py", problem, optimizer,
                              ["--iterations", str(args.iterations)])
            total_tests += 1
            if success:
                passed_tests += 1
            else:
                case_passed = False
                failed_tests.append(f"{problem}+{optimizer}: comprehensive test")
        
        # 3. Run local optimization test for comparison
        if not args.consistency_only:
            print(f"\n3️⃣ Local Optimization Test")
            success = run_test("load_and_test_optimization.py", problem, optimizer,
                              ["--iterations", str(args.iterations)])
            total_tests += 1
            if success:
                passed_tests += 1
            else:
                case_passed = False
                failed_tests.append(f"{problem}+{optimizer}: local test")
        
        # Summary for this case
        if case_passed:
            print(f"\n✅ {problem} + {optimizer}: ALL TESTS PASSED")
        else:
            print(f"\n❌ {problem} + {optimizer}: SOME TESTS FAILED")
    
    # Final summary
    print(f"\n" + "="*70)
    print("📊 FINAL TEST SUMMARY")
    print("="*70)
    print(f"🎯 Total tests run: {total_tests}")
    print(f"✅ Tests passed: {passed_tests}")
    print(f"❌ Tests failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\n❌ Failed tests:")
        for failed in failed_tests:
            print(f"   • {failed}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if len(failed_tests) == 0:
        print("   🎉 All tests passed! Your repositories are ready for Rastion upload.")
        print("   📤 You can safely run: python upload_repo_to_rastion.py <repo_path>")
    elif len(failed_tests) < total_tests // 2:
        print("   ⚠️ Most tests passed, but some issues detected.")
        print("   🔍 Review the failed tests and fix issues before uploading.")
    else:
        print("   ❌ Many tests failed. Significant issues detected.")
        print("   🛠️ Fix the underlying problems before attempting to upload.")
    
    print("="*70)
    
    # Exit with appropriate code
    if len(failed_tests) == 0:
        sys.exit(0)  # All tests passed
    elif len(failed_tests) < total_tests // 2:
        sys.exit(1)  # Some tests failed
    else:
        sys.exit(2)  # Many tests failed


if __name__ == "__main__":
    main()
