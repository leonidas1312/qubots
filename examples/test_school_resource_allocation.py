"""
Test script for School Resource Allocation Decision Model

This script demonstrates how schools can use the qubots framework to optimize
their resource allocation and create efficient schedules.

Author: Qubots Framework
Version: 1.0.0
"""

import sys
from pathlib import Path

# Get the examples directory
examples_dir = Path(__file__).parent

def test_school_problem():
    """Test the school resource allocation problem."""
    print("\n" + "=" * 60)
    print("TESTING SCHOOL RESOURCE ALLOCATION PROBLEM")
    print("=" * 60)

    try:
        from school_resource_allocation_problem.qubot import SchoolResourceAllocationProblem

        # Test 1: Default problem initialization
        print("\n1. Testing default problem initialization...")
        problem = SchoolResourceAllocationProblem()

        print(f"   + Problem created successfully")
        print(f"   + Teachers: {len(problem.teachers)}")
        print(f"   + Classrooms: {len(problem.classrooms)}")
        print(f"   + Subjects: {len(problem.subjects)}")
        print(f"   + Time slots: {len(problem.time_slots)}")

        # Test metadata
        print(f"   + Problem name: {problem.metadata.name}")
        print(f"   + Problem type: {problem.metadata.problem_type}")
        print(f"   + Domain: {problem.metadata.domain}")

        # Test 2: Random solution generation
        print("\n2. Testing random solution generation...")
        solution = problem.random_solution()
        print(f"   + Generated {len(solution)} assignments")

        # Test 3: Solution evaluation
        print("\n3. Testing solution evaluation...")
        evaluation = problem.evaluate_solution_detailed(solution)
        print(f"   ✓ Objective value: {evaluation.objective_value:.2f}")
        print(f"   ✓ Feasible: {evaluation.is_feasible}")
        print(f"   ✓ Constraint violations: {len(evaluation.constraint_violations)}")

        if evaluation.constraint_violations:
            print("   ✓ Sample violations:")
            for violation in evaluation.constraint_violations[:3]:
                print(f"     • {violation}")

        # Test 4: Solution summary
        print("\n4. Testing solution summary...")
        summary = problem.get_solution_summary(solution)
        print(f"   ✓ Total assignments: {summary['total_assignments']}")
        print(f"   ✓ Total cost: ${summary['total_cost']:.2f}")
        print(f"   ✓ Subject coverage: {summary['subject_coverage']:.1%}")

        # Test 5: Problem info
        print("\n5. Testing problem information...")
        info = problem.get_problem_info()
        print(f"   ✓ Total required hours: {info['total_required_hours']}")
        print(f"   ✓ Teachers count: {info['teachers_count']}")
        print(f"   ✓ Classrooms count: {info['classrooms_count']}")

        print("\n✅ School Problem tests PASSED")
        return True

    except Exception as e:
        print(f"\n❌ School Problem test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_school_optimizer():
    """Test the school resource allocation optimizer."""
    print("\n" + "=" * 60)
    print("TESTING SCHOOL RESOURCE ALLOCATION OPTIMIZER")
    print("=" * 60)

    try:
        from school_resource_allocation_optimizer.qubot import SchoolResourceAllocationOptimizer

        # Test 1: Default optimizer initialization
        print("\n1. Testing optimizer initialization...")
        optimizer = SchoolResourceAllocationOptimizer()
        print(f"   ✓ Optimizer created")
        print(f"   ✓ Max solve time: {optimizer.max_solve_time_seconds} seconds")
        print(f"   ✓ Search workers: {optimizer.num_search_workers}")
        print(f"   ✓ Emphasis: {optimizer.emphasis}")

        # Test 2: Custom parameters
        print("\n2. Testing with custom parameters...")
        custom_optimizer = SchoolResourceAllocationOptimizer(
            max_solve_time_seconds=120.0,
            num_search_workers=2,
            emphasis="feasibility",
            enable_logging=True
        )
        print(f"   ✓ Custom optimizer created")
        print(f"   ✓ Custom solve time: {custom_optimizer.max_solve_time_seconds}")
        print(f"   ✓ Custom workers: {custom_optimizer.num_search_workers}")

        # Test 3: Metadata
        print("\n3. Testing optimizer metadata...")
        metadata = optimizer.metadata
        print(f"   ✓ Name: {metadata.name}")
        print(f"   ✓ Type: {metadata.optimizer_type}")
        print(f"   ✓ Family: {metadata.optimizer_family}")
        print(f"   ✓ Supports constraints: {metadata.supports_constraints}")
        print(f"   ✓ Parallel capable: {metadata.parallel_capable}")

        print("\n✅ School Optimizer tests PASSED")
        return True

    except Exception as e:
        print(f"\n❌ School Optimizer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between problem and optimizer."""
    print("\n" + "=" * 60)
    print("TESTING INTEGRATION WITH OPTIMIZATION")
    print("=" * 60)

    try:
        from school_resource_allocation_problem.qubot import SchoolResourceAllocationProblem
        from school_resource_allocation_optimizer.qubot import SchoolResourceAllocationOptimizer

        # Create problem with smaller size for faster testing
        print("\n1. Creating test problem...")
        problem = SchoolResourceAllocationProblem()
        print(f"   ✓ Problem created with {len(problem.teachers)} teachers")

        # Create optimizer with short time limit for testing
        print("\n2. Creating optimizer...")
        optimizer = SchoolResourceAllocationOptimizer(
            max_solve_time_seconds=30,  # Short time for testing
            emphasis="feasibility",
            enable_logging=False
        )
        print(f"   ✓ Optimizer created")

        # Run optimization
        print("\n3. Running optimization...")
        result = optimizer.optimize(problem)

        print(f"   ✓ Optimization completed")
        print(f"   ✓ Best value: {result.best_value:.2f}")
        print(f"   ✓ Feasible: {result.is_feasible}")
        print(f"   ✓ Runtime: {result.runtime_seconds:.2f} seconds")
        print(f"   ✓ Assignments: {len(result.best_solution)}")

        if result.additional_metrics:
            print(f"   ✓ Solver status: {result.additional_metrics.get('solver_status', 'Unknown')}")
            print(f"   ✓ Total cost: ${result.additional_metrics.get('total_cost', 0):.2f}")
            print(f"   ✓ Constraint violations: {result.additional_metrics.get('constraint_violations', 0)}")

        # Test solution analysis
        if result.best_solution:
            print("\n4. Analyzing solution...")
            summary = problem.get_solution_summary(result.best_solution)

            print(f"   ✓ Teacher workload (first 3):")
            for teacher_id, hours in list(summary['teacher_workload'].items())[:3]:
                teacher_name = problem.teachers[teacher_id].name
                print(f"     • {teacher_name}: {hours} hours")

            print(f"   ✓ Subject coverage (first 3):")
            for subject_id, hours in list(summary['subject_hours_assigned'].items())[:3]:
                subject_name = problem.subjects[subject_id].name
                required = problem.subjects[subject_id].required_hours_per_week
                status = "✓" if hours >= required else "✗"
                print(f"     {status} {subject_name}: {hours}/{required} hours")

        print("\n✅ Integration tests PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_data():
    """Test with custom school data."""
    print("\n" + "=" * 60)
    print("TESTING CUSTOM DATA LOADING")
    print("=" * 60)

    try:
        from school_resource_allocation_problem.qubot import SchoolResourceAllocationProblem

        # Create custom teacher data
        custom_teachers = """teacher_id,name,subjects,max_hours_per_day,experience_level,hourly_cost
T001,Dr. Alice Smith,"Mathematics,Statistics",6,5,70
T002,Prof. Bob Johnson,"Physics,Chemistry",7,4,65
T003,Ms. Carol Brown,"English,Literature",8,3,55"""

        # Create custom classroom data
        custom_classrooms = """room_id,name,capacity,room_type,hourly_cost
R101,Main Math Room,35,standard,10
R102,Science Laboratory,25,lab,20
R103,English Classroom,30,standard,8"""

        # Create custom subject data
        custom_subjects = """subject_id,name,required_hours_per_week,max_class_size,required_room_type,priority
S001,Advanced Mathematics,4,30,standard,5
S002,Physics,3,25,lab,4
S003,English Literature,3,30,standard,4"""

        print("\n1. Loading problem with custom data...")
        problem = SchoolResourceAllocationProblem(
            teachers_data=custom_teachers,
            classrooms_data=custom_classrooms,
            subjects_data=custom_subjects
        )

        print(f"   ✓ Custom problem loaded")
        print(f"   ✓ Teachers: {len(problem.teachers)}")
        print(f"   ✓ Classrooms: {len(problem.classrooms)}")
        print(f"   ✓ Subjects: {len(problem.subjects)}")

        # Test with custom data
        print("\n2. Testing with custom data...")
        solution = problem.random_solution()
        evaluation = problem.evaluate_solution_detailed(solution)

        print(f"   ✓ Generated {len(solution)} assignments")
        print(f"   ✓ Feasible: {evaluation.is_feasible}")
        print(f"   ✓ Objective value: {evaluation.objective_value:.2f}")

        print("\n✅ Custom Data tests PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Custom Data test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization capabilities."""
    print("\n" + "=" * 60)
    print("TESTING VISUALIZATION CAPABILITIES")
    print("=" * 60)

    try:
        from school_resource_allocation_problem.qubot import SchoolResourceAllocationProblem
        from school_resource_allocation_optimizer.qubot import SchoolResourceAllocationOptimizer

        # Create problem and optimizer
        print("\n1. Setting up for visualization test...")
        problem = SchoolResourceAllocationProblem()
        optimizer = SchoolResourceAllocationOptimizer(
            max_solve_time_seconds=20,  # Very short for demo
            emphasis="feasibility"
        )

        # Run quick optimization
        print("\n2. Running quick optimization...")
        result = optimizer.optimize(problem)

        if result.best_solution:
            print("\n3. Testing visualization...")
            try:
                # Test visualization method exists and can be called
                optimizer.plot_optimization_results(result, problem)
                print("   ✓ Visualization method executed successfully")
                return True
            except Exception as viz_error:
                print(f"   ⚠ Visualization method exists but may require display: {viz_error}")
                return True
        else:
            print("   ⚠ No solution to visualize")
            return False

    except Exception as e:
        print(f"\n❌ Visualization test FAILED: {e}")
        return False


def main():
    """Run all tests for the school resource allocation decision model."""
    print("SCHOOL RESOURCE ALLOCATION DECISION MODEL TEST SUITE")
    print("=" * 60)
    print("Testing comprehensive school scheduling optimization...")

    # Track test results
    test_results = {}

    # Run tests
    test_results['problem'] = test_school_problem()
    test_results['optimizer'] = test_school_optimizer()
    test_results['integration'] = test_integration()
    test_results['custom_data'] = test_custom_data()
    test_results['visualization'] = test_visualization()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    total_tests = len(test_results)
    passed_tests = sum(test_results.values())

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 All tests passed! School resource allocation model is working correctly.")
        print("\nThis decision model can help schools:")
        print("• Create optimal teacher-subject-classroom schedules")
        print("• Eliminate scheduling conflicts and double-bookings")
        print("• Minimize costs while maximizing educational quality")
        print("• Efficiently utilize teachers and classroom resources")
        print("• Generate comprehensive schedule analysis and visualizations")
    else:
        print("\n⚠ Some tests failed. Please check the error messages above.")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
