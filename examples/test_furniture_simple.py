"""
Simple test script for Furniture Arrangement Optimization.

This script uses smaller datasets for faster testing and validation.
"""

import sys
import os
from pathlib import Path

# Add examples directory to path for imports
examples_dir = Path(__file__).parent
sys.path.insert(0, str(examples_dir))

def test_simple_furniture_arrangement():
    """Test furniture arrangement with simple datasets."""
    print("🏠 Simple Furniture Arrangement Test")
    print("=" * 50)
    
    try:
        from furniture_arrangement_problem.qubot import FurnitureArrangementProblem
        from furniture_arrangement_optimizer.qubot import FurnitureArrangementOptimizer
        
        print("\n1. Creating simple problem...")
        
        # Use simple datasets
        furniture_file = examples_dir / "furniture_arrangement_problem" / "datasets" / "simple_furniture.csv"
        room_file = examples_dir / "furniture_arrangement_problem" / "datasets" / "simple_rooms.csv"
        
        # Create problem with simple furniture set
        problem = FurnitureArrangementProblem(
            room_config="small_room",
            furniture_selection=['sofa_small', 'coffee_table', 'tv_small', 'tv_stand', 'armchair'],
            furniture_csv_file=str(furniture_file),
            room_csv_file=str(room_file)
        )
        
        print(f"   ✓ Room: {problem.room.name} ({problem.room.width}×{problem.room.depth} cm)")
        print(f"   ✓ Furniture pieces: {len(problem.furniture_pieces)}")
        print(f"   ✓ Furniture list: {list(problem.furniture_pieces.keys())}")
        
        print("\n2. Testing random solution...")
        solution = problem.random_solution()
        cost = problem.evaluate_solution(solution)
        is_valid = problem.is_valid_solution(solution)
        
        print(f"   ✓ Random solution cost: {cost:.2f}")
        print(f"   ✓ Valid arrangement: {is_valid}")
        
        print("\n3. Creating simple optimizer...")
        optimizer = FurnitureArrangementOptimizer(
            initial_temperature=200.0,
            cooling_rate=0.9,
            max_iterations=500,  # Reduced for quick test
            moves_per_temperature=10,
            create_plots=False,  # Disable plots for simple test
            random_seed=42
        )
        
        print(f"   ✓ Optimizer created: {optimizer.metadata.name}")
        
        print("\n4. Running optimization...")
        result = optimizer._optimize_implementation(problem)
        
        print(f"   ✓ Optimization completed!")
        print(f"   ✓ Best cost: {result.best_value:.2f}")
        print(f"   ✓ Iterations: {result.iterations}")
        print(f"   ✓ Runtime: {result.runtime_seconds:.2f} seconds")
        print(f"   ✓ Valid arrangement: {result.additional_metrics['is_valid_arrangement']}")
        print(f"   ✓ Space utilization: {result.additional_metrics['space_utilization']:.1f}%")
        
        print("\n5. Testing solution summary...")
        if result.best_solution:
            summary = problem.get_solution_summary(result.best_solution)
            print("   ✓ Solution summary generated")
            print(f"   First line: {summary.split(chr(10))[0]}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in simple test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_tiny_room():
    """Test with the tiniest room configuration."""
    print("\n🏠 Tiny Room Test")
    print("=" * 30)
    
    try:
        from furniture_arrangement_problem.qubot import FurnitureArrangementProblem
        
        # Use simple datasets
        furniture_file = examples_dir / "furniture_arrangement_problem" / "datasets" / "simple_furniture.csv"
        room_file = examples_dir / "furniture_arrangement_problem" / "datasets" / "simple_rooms.csv"
        
        # Create problem with minimal furniture
        problem = FurnitureArrangementProblem(
            room_config="tiny_room",
            furniture_selection=['sofa_small', 'coffee_table', 'tv_small'],
            furniture_csv_file=str(furniture_file),
            room_csv_file=str(room_file)
        )
        
        print(f"   ✓ Room: {problem.room.name} ({problem.room.width}×{problem.room.depth} cm)")
        print(f"   ✓ Room area: {(problem.room.width * problem.room.depth / 10000):.1f} m²")
        print(f"   ✓ Furniture pieces: {len(problem.furniture_pieces)}")
        
        # Test solution
        solution = problem.random_solution()
        cost = problem.evaluate_solution(solution)
        is_valid = problem.is_valid_solution(solution)
        
        print(f"   ✓ Solution cost: {cost:.2f}")
        print(f"   ✓ Valid: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in tiny room test: {str(e)}")
        return False

def test_with_visualizations():
    """Test with visualizations enabled."""
    print("\n🎨 Visualization Test")
    print("=" * 30)
    
    try:
        from furniture_arrangement_problem.qubot import FurnitureArrangementProblem
        from furniture_arrangement_optimizer.qubot import FurnitureArrangementOptimizer
        
        # Use simple datasets
        furniture_file = examples_dir / "furniture_arrangement_problem" / "datasets" / "simple_furniture.csv"
        room_file = examples_dir / "furniture_arrangement_problem" / "datasets" / "simple_rooms.csv"
        
        # Create problem
        problem = FurnitureArrangementProblem(
            room_config="medium_room",
            furniture_selection=['sofa_small', 'coffee_table', 'tv_small', 'tv_stand', 'armchair', 'side_table'],
            furniture_csv_file=str(furniture_file),
            room_csv_file=str(room_file)
        )
        
        # Create optimizer with visualizations
        optimizer = FurnitureArrangementOptimizer(
            initial_temperature=300.0,
            max_iterations=200,  # Very short for demo
            moves_per_temperature=5,
            create_plots=True,   # Enable visualizations
            save_plots=False,    # Don't save files
            plot_interval=50,    # Frequent updates
            random_seed=42
        )
        
        print(f"   ✓ Problem: {problem.room.name}")
        print(f"   ✓ Optimizer: {optimizer.metadata.name}")
        print("   ✓ Running optimization with visualizations...")
        
        # This should show plots
        result = optimizer._optimize_implementation(problem)
        
        print(f"   ✓ Completed with cost: {result.best_value:.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in visualization test: {str(e)}")
        return False

def main():
    """Run simple furniture arrangement tests."""
    print("🏠 Simple Furniture Arrangement Test Suite")
    print("=" * 60)
    print("Testing with smaller datasets for faster validation...")
    
    tests = [
        ("Simple Furniture Arrangement", test_simple_furniture_arrangement),
        ("Tiny Room Configuration", test_tiny_room),
        ("Visualization Test", test_with_visualizations),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("🏠 SIMPLE TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All simple tests passed!")
        print("\nThe furniture arrangement system is working correctly.")
        print("You can now run the full test suite or use the system directly!")
    else:
        print(f"\n⚠️ {total-passed} test(s) failed. Please check the errors above.")

if __name__ == "__main__":
    main()
