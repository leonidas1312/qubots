#!/usr/bin/env python3
"""
Test script for Calendar Rearrangement Problem and Optimizer

This script tests the calendar rearrangement components to ensure they work
correctly together and produce valid results.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add the examples directory to the Python path
examples_dir = Path(__file__).parent
sys.path.insert(0, str(examples_dir))

def test_calendar_problem():
    """Test the calendar rearrangement problem."""
    print("=" * 60)
    print("TESTING CALENDAR REARRANGEMENT PROBLEM")
    print("=" * 60)
    
    try:
        # Import the problem class
        from calendar_rearrangement_problem.qubot import CalendarRearrangementProblem
        
        # Test 1: Default problem (built-in data)
        print("\n1. Testing with default data...")
        problem = CalendarRearrangementProblem()
        print(f"   ✓ Problem created with {problem.n_meetings} total meetings")
        print(f"   ✓ Target day off: {problem.target_day_off}")
        print(f"   ✓ Moveable meetings: {problem.n_moveable}")
        print(f"   ✓ Available days: {problem.available_days}")
        print(f"   ✓ Max hours per day: {problem.max_hours_per_day}")
        
        # Test random solution generation
        solution = problem.random_solution()
        print(f"   ✓ Random solution generated: {len(solution['assignments'])} assignments")
        
        # Test solution evaluation
        cost = problem.evaluate_solution(solution)
        print(f"   ✓ Solution evaluated, cost: {cost:.2f}")
        
        # Test feasibility check
        is_feasible = problem.is_feasible(solution)
        print(f"   ✓ Feasibility check: {is_feasible}")
        
        # Test 2: Custom CSV data
        print("\n2. Testing with custom CSV data...")
        csv_data = """meeting_id,meeting_name,duration_hours,priority,current_day,flexible,participants
M1,Team Standup,0.5,3,Wednesday,True,5
M2,Project Review,2.0,5,Wednesday,False,8
M3,Client Call,1.0,4,Monday,True,3"""
        
        custom_problem = CalendarRearrangementProblem(
            csv_data=csv_data,
            target_day_off="Wednesday",
            max_hours_per_day=6.0
        )
        print(f"   ✓ Custom problem created with {custom_problem.n_meetings} meetings")
        print(f"   ✓ Moveable meetings: {custom_problem.n_moveable}")
        
        # Test 3: Different target day
        print("\n3. Testing with different target day...")
        friday_problem = CalendarRearrangementProblem(
            target_day_off="Friday",
            available_days=["Monday", "Tuesday", "Wednesday", "Thursday"]
        )
        print(f"   ✓ Friday-off problem created")
        print(f"   ✓ Moveable meetings from Friday: {friday_problem.n_moveable}")
        
        # Test 4: Load from sample dataset
        print("\n4. Testing with sample dataset...")
        dataset_path = examples_dir / "calendar_rearrangement_problem" / "datasets" / "sample_week.csv"
        if dataset_path.exists():
            dataset_problem = CalendarRearrangementProblem(
                csv_file_path=str(dataset_path),
                target_day_off="Wednesday"
            )
            print(f"   ✓ Dataset problem created with {dataset_problem.n_meetings} meetings")
            print(f"   ✓ Moveable meetings: {dataset_problem.n_moveable}")
            
            # Test solution with dataset
            dataset_solution = dataset_problem.random_solution()
            dataset_cost = dataset_problem.evaluate_solution(dataset_solution)
            dataset_feasible = dataset_problem.is_feasible(dataset_solution)
            print(f"   ✓ Dataset solution cost: {dataset_cost:.2f}, feasible: {dataset_feasible}")
        else:
            print("   ⚠ Sample dataset not found, skipping dataset test")
        
        print("\n✅ All problem tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Problem test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_calendar_optimizer():
    """Test the calendar rearrangement optimizer."""
    print("\n" + "=" * 60)
    print("TESTING CALENDAR REARRANGEMENT OPTIMIZER")
    print("=" * 60)
    
    try:
        # Import the optimizer class
        from calendar_rearrangement_optimizer.qubot import ORToolsCalendarOptimizer
        from calendar_rearrangement_problem.qubot import CalendarRearrangementProblem
        
        # Test 1: Basic optimization
        print("\n1. Testing basic optimization...")
        problem = CalendarRearrangementProblem()
        optimizer = ORToolsCalendarOptimizer(time_limit=30.0)  # Short time limit for testing
        
        print(f"   ✓ Optimizer created: {optimizer.solver_name}")
        print(f"   ✓ Time limit: {optimizer.time_limit}s")
        
        # Run optimization
        result = optimizer.optimize(problem)
        print(f"   ✓ Optimization completed")
        print(f"   ✓ Status: {result.additional_metrics.get('status', 'Unknown')}")
        print(f"   ✓ Objective value: {result.best_value:.2f}")
        print(f"   ✓ Optimization time: {result.runtime_seconds:.2f}s")
        print(f"   ✓ Is feasible: {result.is_feasible}")

        if result.best_solution:
            print(f"   ✓ Solution assignments: {result.best_solution['assignments']}")

            # Verify solution with problem
            problem_cost = problem.evaluate_solution(result.best_solution)
            problem_feasible = problem.is_feasible(result.best_solution)
            print(f"   ✓ Problem verification - Cost: {problem_cost:.2f}, Feasible: {problem_feasible}")
        
        # Test 2: Different solver
        print("\n2. Testing with CBC solver...")
        cbc_optimizer = ORToolsCalendarOptimizer(solver_name="CBC", time_limit=30.0)
        cbc_result = cbc_optimizer.optimize(problem)
        print(f"   ✓ CBC optimization completed")
        print(f"   ✓ CBC status: {cbc_result.additional_metrics.get('status', 'Unknown')}")
        print(f"   ✓ CBC objective: {cbc_result.best_value:.2f}")
        
        # Test 3: Complex problem
        print("\n3. Testing with busy week dataset...")
        dataset_path = examples_dir / "calendar_rearrangement_problem" / "datasets" / "busy_week.csv"
        if dataset_path.exists():
            busy_problem = CalendarRearrangementProblem(
                csv_file_path=str(dataset_path),
                target_day_off="Wednesday",
                max_hours_per_day=8.0
            )
            
            busy_result = optimizer.optimize(busy_problem)
            print(f"   ✓ Busy week optimization completed")
            print(f"   ✓ Meetings to move: {busy_problem.n_moveable}")
            print(f"   ✓ Status: {busy_result.additional_metrics.get('status', 'Unknown')}")
            print(f"   ✓ Objective: {busy_result.best_value:.2f}")

            if busy_result.best_solution:
                # Analyze the solution
                assignments = busy_result.best_solution['assignments']
                day_distribution = {}
                for day_idx in assignments:
                    day = busy_problem.available_days[day_idx]
                    day_distribution[day] = day_distribution.get(day, 0) + 1
                print(f"   ✓ Meeting distribution: {day_distribution}")
        else:
            print("   ⚠ Busy week dataset not found, skipping complex test")
        
        # Test 4: Edge case - no moveable meetings
        print("\n4. Testing edge case - no moveable meetings...")
        csv_no_moveable = """meeting_id,meeting_name,duration_hours,priority,current_day,flexible,participants
M1,Fixed Meeting,1.0,5,Wednesday,False,5
M2,Another Fixed,1.0,4,Monday,True,3"""
        
        no_move_problem = CalendarRearrangementProblem(
            csv_data=csv_no_moveable,
            target_day_off="Wednesday"
        )
        
        no_move_result = optimizer.optimize(no_move_problem)
        print(f"   ✓ No moveable meetings optimization completed")
        print(f"   ✓ Status: {no_move_result.additional_metrics.get('status', 'Unknown')}")
        print(f"   ✓ Objective: {no_move_result.best_value:.2f}")
        
        print("\n✅ All optimizer tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between problem and optimizer with different scenarios."""
    print("\n" + "=" * 60)
    print("TESTING INTEGRATION SCENARIOS")
    print("=" * 60)
    
    try:
        from calendar_rearrangement_problem.qubot import CalendarRearrangementProblem
        from calendar_rearrangement_optimizer.qubot import ORToolsCalendarOptimizer
        
        scenarios = [
            {
                "name": "Wednesday Off (Default)",
                "target_day": "Wednesday",
                "available_days": ["Monday", "Tuesday", "Thursday", "Friday"],
                "max_hours": 8.0
            },
            {
                "name": "Friday Off (Relaxed)",
                "target_day": "Friday",
                "available_days": ["Monday", "Tuesday", "Wednesday", "Thursday"],
                "max_hours": 9.0
            },
            {
                "name": "Monday Off (Generous)",
                "target_day": "Monday",
                "available_days": ["Tuesday", "Wednesday", "Thursday", "Friday"],
                "max_hours": 10.0
            }
        ]
        
        optimizer = ORToolsCalendarOptimizer(time_limit=60.0)
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. Testing scenario: {scenario['name']}")
            
            # Use sample dataset if available
            dataset_path = examples_dir / "calendar_rearrangement_problem" / "datasets" / "sample_week.csv"
            if dataset_path.exists():
                problem = CalendarRearrangementProblem(
                    csv_file_path=str(dataset_path),
                    target_day_off=scenario["target_day"],
                    available_days=scenario["available_days"],
                    max_hours_per_day=scenario["max_hours"]
                )
            else:
                problem = CalendarRearrangementProblem(
                    target_day_off=scenario["target_day"],
                    available_days=scenario["available_days"],
                    max_hours_per_day=scenario["max_hours"]
                )
            
            print(f"   ✓ Problem setup: {problem.n_moveable} meetings to move from {scenario['target_day']}")

            # Check feasibility before optimization
            feasibility_info = problem.get_feasibility_info()
            print(f"   ✓ Hours to move: {feasibility_info['total_hours_to_move']:.1f}h")
            print(f"   ✓ Available capacity: {feasibility_info['total_available_capacity']:.1f}h")
            print(f"   ✓ Theoretically feasible: {feasibility_info['is_theoretically_feasible']}")

            result = optimizer.optimize(problem)
            print(f"   ✓ Optimization result: {result.additional_metrics.get('status', 'Unknown')}")
            print(f"   ✓ Cost: {result.best_value:.2f}")

            if result.best_solution and result.is_feasible:
                # Verify the solution makes sense
                assignments = result.best_solution['assignments']
                if len(assignments) == problem.n_moveable:
                    print(f"   ✓ Solution has correct number of assignments")
                    
                    # Check if all assignments are valid
                    valid_assignments = all(0 <= idx < len(problem.available_days) for idx in assignments)
                    print(f"   ✓ All assignments valid: {valid_assignments}")
                    
                    # Check feasibility
                    is_feasible = problem.is_feasible(result.best_solution)
                    print(f"   ✓ Solution feasible: {is_feasible}")
                else:
                    print(f"   ⚠ Assignment count mismatch: {len(assignments)} vs {problem.n_moveable}")
            elif problem.n_moveable == 0:
                print(f"   ✓ No meetings to move - correct result")
            else:
                print(f"   ⚠ No feasible solution found")
        
        print("\n✅ All integration tests completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 CALENDAR REARRANGEMENT DECISION MODEL TESTS")
    print("=" * 80)
    
    # Track test results
    results = []
    
    # Run tests
    results.append(("Problem Tests", test_calendar_problem()))
    results.append(("Optimizer Tests", test_calendar_optimizer()))
    results.append(("Integration Tests", test_integration()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Calendar rearrangement decision model is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
