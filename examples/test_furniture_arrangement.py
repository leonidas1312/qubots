"""
Test script for Furniture Arrangement Optimization with Qubots Framework.

This script demonstrates the complete furniture arrangement optimization system
including problem setup, optimization, and comprehensive visualizations.
"""

import sys
import os
from pathlib import Path

# Add examples directory to path for imports
examples_dir = Path(__file__).parent
sys.path.insert(0, str(examples_dir))

def test_furniture_arrangement_basic():
    """Test basic furniture arrangement functionality."""
    print("🏠 Testing Basic Furniture Arrangement")
    print("=" * 50)
    
    try:
        from furniture_arrangement_problem.qubot import FurnitureArrangementProblem
        
        # Test 1: Default medium rectangular room
        print("\n1. Testing default medium rectangular room...")
        problem = FurnitureArrangementProblem()
        
        print(f"   ✓ Room: {problem.room.name}")
        print(f"   ✓ Room dimensions: {problem.room.width}×{problem.room.depth} cm")
        print(f"   ✓ Furniture pieces: {len(problem.furniture_pieces)}")
        print(f"   ✓ Door position: ({problem.room.door_x}, {problem.room.door_y})")
        print(f"   ✓ Windows: {len(problem.room.windows)}")
        
        # Test random solution generation
        solution = problem.random_solution()
        print(f"   ✓ Random solution generated: {len(solution)} furniture pieces")
        
        # Test solution evaluation
        cost = problem.evaluate_solution(solution)
        print(f"   ✓ Solution evaluated, cost: {cost:.2f}")
        
        # Test solution validation
        is_valid = problem.is_valid_solution(solution)
        print(f"   ✓ Solution validity: {is_valid}")
        
        # Test solution summary
        summary = problem.get_solution_summary(solution)
        print(f"   ✓ Solution summary generated")
        print(f"   Summary preview: {summary.split(chr(10))[1]}")  # Second line
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in basic test: {str(e)}")
        return False

def test_different_room_configurations():
    """Test different room configurations."""
    print("\n🏠 Testing Different Room Configurations")
    print("=" * 50)
    
    room_configs = [
        ("small_square", "Small Square Living Room"),
        ("large_rect", "Large Rectangular Living Room"), 
        ("open_plan", "Open Plan Living Area"),
        ("studio_apt", "Studio Apartment"),
        ("cozy_cottage", "Cozy Cottage Living Room")
    ]
    
    try:
        from furniture_arrangement_problem.qubot import FurnitureArrangementProblem
        
        for room_id, room_name in room_configs:
            print(f"\n📐 Testing: {room_name}")
            print("-" * 40)
            
            # Create problem with specific room
            problem = FurnitureArrangementProblem(room_config=room_id)
            
            print(f"   Room ID: {problem.room.id}")
            print(f"   Dimensions: {problem.room.width}×{problem.room.depth} cm")
            print(f"   Area: {(problem.room.width * problem.room.depth / 10000):.1f} m²")
            print(f"   Room type: {problem.room.room_type}")
            
            # Generate and evaluate solution
            solution = problem.random_solution()
            cost = problem.evaluate_solution(solution)
            is_valid = problem.is_valid_solution(solution)
            
            print(f"   Solution cost: {cost:.2f}")
            print(f"   Valid: {is_valid}")
            print(f"   Furniture pieces placed: {len(solution)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing room configurations: {str(e)}")
        return False

def test_custom_furniture_selection():
    """Test custom furniture selection."""
    print("\n🪑 Testing Custom Furniture Selection")
    print("=" * 50)
    
    try:
        from furniture_arrangement_problem.qubot import FurnitureArrangementProblem
        
        # Test 1: Minimal furniture set
        print("\n1. Testing minimal furniture set...")
        minimal_furniture = ['sofa_3seat', 'coffee_table_rect', 'tv_55inch', 'tv_stand_large']
        
        problem = FurnitureArrangementProblem(
            room_config="medium_rect",
            furniture_selection=minimal_furniture
        )
        
        print(f"   ✓ Selected {len(problem.furniture_pieces)} furniture pieces")
        print(f"   ✓ Furniture: {list(problem.furniture_pieces.keys())}")
        
        solution = problem.random_solution()
        cost = problem.evaluate_solution(solution)
        print(f"   ✓ Minimal set cost: {cost:.2f}")
        
        # Test 2: Luxury furniture set
        print("\n2. Testing luxury furniture set...")
        luxury_furniture = [
            'sofa_3seat', 'sofa_2seat', 'armchair_1', 'armchair_2',
            'coffee_table_round', 'side_table_1', 'side_table_2',
            'tv_65inch', 'tv_stand_large', 'bookshelf_tall',
            'floor_lamp_1', 'floor_lamp_2', 'plant_large', 'plant_medium',
            'rug_large', 'ottoman_round', 'bar_cart'
        ]
        
        problem = FurnitureArrangementProblem(
            room_config="large_rect",
            furniture_selection=luxury_furniture
        )
        
        print(f"   ✓ Selected {len(problem.furniture_pieces)} furniture pieces")
        
        solution = problem.random_solution()
        cost = problem.evaluate_solution(solution)
        print(f"   ✓ Luxury set cost: {cost:.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing custom furniture: {str(e)}")
        return False

def test_optimization_with_visualizations():
    """Test the complete optimization process with visualizations."""
    print("\n🎯 Testing Optimization with Visualizations")
    print("=" * 50)
    
    try:
        from furniture_arrangement_problem.qubot import FurnitureArrangementProblem
        from furniture_arrangement_optimizer.qubot import FurnitureArrangementOptimizer
        
        print("\n1. Setting up optimization problem...")
        
        # Create a medium-sized problem for testing
        problem = FurnitureArrangementProblem(
            room_config="medium_rect",
            furniture_selection=[
                'sofa_3seat', 'coffee_table_rect', 'tv_55inch', 'tv_stand_large',
                'armchair_1', 'side_table_1', 'floor_lamp_1', 'plant_large', 'rug_large'
            ],
            space_weight=0.4,
            accessibility_weight=0.3,
            aesthetic_weight=0.3
        )
        
        print(f"   ✓ Problem created: {problem.room.name}")
        print(f"   ✓ Furniture pieces: {len(problem.furniture_pieces)}")
        
        print("\n2. Creating optimizer with visualization...")
        
        # Create optimizer with reduced iterations for testing
        optimizer = FurnitureArrangementOptimizer(
            initial_temperature=500.0,
            cooling_rate=0.9,
            max_iterations=1000,  # Reduced for testing
            moves_per_temperature=20,
            create_plots=True,  # Enable visualizations
            save_plots=False,   # Don't save files during testing
            plot_interval=200,  # More frequent updates for testing
            random_seed=42      # For reproducible results
        )
        
        print(f"   ✓ Optimizer created: {optimizer.metadata.name}")
        print(f"   ✓ Algorithm: {optimizer.metadata.optimizer_family.value}")
        print(f"   ✓ Max iterations: {optimizer.max_iterations}")
        
        print("\n3. Running optimization...")
        print("   (This will show visualization plots during optimization)")
        
        # Run optimization
        result = optimizer.optimize(problem)
        
        print(f"\n   ✓ Optimization completed!")
        print(f"   ✓ Best cost: {result.best_value:.2f}")
        print(f"   ✓ Iterations: {result.iterations}")
        print(f"   ✓ Runtime: {result.runtime_seconds:.2f} seconds")
        print(f"   ✓ Acceptance rate: {result.additional_metrics['acceptance_rate']:.2%}")
        print(f"   ✓ Space utilization: {result.additional_metrics['space_utilization']:.1f}%")
        print(f"   ✓ Valid arrangement: {result.additional_metrics['is_valid_arrangement']}")
        
        # Test solution summary
        if result.best_solution:
            summary = problem.get_solution_summary(result.best_solution)
            print(f"\n   Solution Summary:")
            for line in summary.split('\n')[:6]:  # First 6 lines
                print(f"   {line}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in optimization test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_variations():
    """Test different parameter configurations."""
    print("\n⚙️ Testing Parameter Variations")
    print("=" * 50)
    
    try:
        from furniture_arrangement_problem.qubot import FurnitureArrangementProblem
        
        # Test different objective weights
        weight_configs = [
            (0.7, 0.2, 0.1, "Space-focused"),
            (0.2, 0.7, 0.1, "Accessibility-focused"), 
            (0.2, 0.2, 0.6, "Aesthetic-focused"),
            (0.33, 0.33, 0.34, "Balanced")
        ]
        
        for space_w, access_w, aesthetic_w, description in weight_configs:
            print(f"\n📊 Testing: {description}")
            print("-" * 30)
            
            problem = FurnitureArrangementProblem(
                room_config="medium_rect",
                space_weight=space_w,
                accessibility_weight=access_w,
                aesthetic_weight=aesthetic_w
            )
            
            # Generate multiple solutions and compare
            costs = []
            for _ in range(5):
                solution = problem.random_solution()
                cost = problem.evaluate_solution(solution)
                costs.append(cost)
            
            avg_cost = sum(costs) / len(costs)
            min_cost = min(costs)
            max_cost = max(costs)
            
            print(f"   Weights: Space={space_w}, Access={access_w}, Aesthetic={aesthetic_w}")
            print(f"   Average cost: {avg_cost:.2f}")
            print(f"   Cost range: {min_cost:.2f} - {max_cost:.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing parameters: {str(e)}")
        return False

def main():
    """Run all furniture arrangement tests."""
    print("🏠 Furniture Arrangement Optimization Test Suite")
    print("=" * 60)
    print("Testing the complete qubots furniture arrangement system...")
    
    tests = [
        ("Basic Functionality", test_furniture_arrangement_basic),
        ("Room Configurations", test_different_room_configurations),
        ("Custom Furniture Selection", test_custom_furniture_selection),
        ("Parameter Variations", test_parameter_variations),
        ("Optimization with Visualizations", test_optimization_with_visualizations),
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
    print("🏠 FURNITURE ARRANGEMENT TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! The furniture arrangement system is working correctly.")
        print("\nYou can now use the system to optimize your living room layout!")
        print("\nExample usage:")
        print("```python")
        print("from qubots import AutoProblem, AutoOptimizer")
        print("")
        print("# Load furniture arrangement problem")
        print("problem = AutoProblem.from_repo('examples/furniture_arrangement_problem')")
        print("")
        print("# Load optimizer")
        print("optimizer = AutoOptimizer.from_repo('examples/furniture_arrangement_optimizer')")
        print("")
        print("# Optimize your living room!")
        print("result = optimizer.optimize(problem)")
        print("```")
    else:
        print(f"\n⚠️ {total-passed} test(s) failed. Please check the errors above.")

if __name__ == "__main__":
    main()
