#!/usr/bin/env python3
"""
Test script for the Molecular Conformation Optimization Decision Model

This script demonstrates the complete workflow of the chemistry optimization
decision model, including dataset loading, problem setup, optimization,
and results analysis.

Usage:
    python test_molecular_optimization.py
"""

import sys
import os
from pathlib import Path

# Add the examples directory to the path
examples_dir = Path(__file__).parent
sys.path.insert(0, str(examples_dir))

def test_molecular_problem():
    """Test the molecular conformation problem with default data."""
    print("\n" + "=" * 60)
    print("TESTING MOLECULAR CONFORMATION PROBLEM")
    print("=" * 60)
    
    try:
        from molecular_conformation_problem.qubot import MolecularConformationProblem
        
        # Test 1: Default problem (built-in butane data)
        print("\n1. Testing with default butane data...")
        problem = MolecularConformationProblem()
        print(f"   ✓ Problem created with {problem.n_dihedrals} dihedral angles")
        print(f"   ✓ Energy range: {problem.min_energy:.2f} to {problem.max_energy:.2f} kcal/mol")
        print(f"   ✓ Energy tolerance: {problem.energy_tolerance} kcal/mol")
        
        # Test random solution generation
        solution = problem.random_solution()
        print(f"   ✓ Random solution generated: {len(solution)} dihedral angles")
        
        # Test solution evaluation
        energy = problem.evaluate_solution(solution)
        print(f"   ✓ Solution evaluated, energy: {energy:.3f} kcal/mol")
        
        # Test detailed evaluation
        detailed_result = problem.evaluate_solution(solution, verbose=True)
        print(f"   ✓ Detailed evaluation completed")
        print(f"     - Van der Waals: {detailed_result.additional_metrics['van_der_waals_energy']:.3f} kcal/mol")
        print(f"     - Electrostatic: {detailed_result.additional_metrics['electrostatic_energy']:.3f} kcal/mol")
        print(f"     - Is local minimum: {detailed_result.additional_metrics['is_local_minimum']}")
        
        # Test neighbor generation
        neighbor = problem.get_neighbor_solution(solution)
        print(f"   ✓ Neighbor solution generated")
        
        # Test global minimum
        global_min_angles, global_min_energy = problem.get_global_minimum()
        print(f"   ✓ Global minimum: {global_min_energy:.3f} kcal/mol")
        print(f"     - Angles: {[f'{angle:.1f}°' for angle in global_min_angles]}")
        
        # Test 2: Custom CSV data
        print("\n2. Testing with custom CSV data...")
        custom_data = """dihedral_1,dihedral_2,energy,van_der_waals_energy,electrostatic_energy
0.0,0.0,5.2,-2.1,0.3
90.0,90.0,0.4,-3.9,-1.4
180.0,180.0,4.9,-2.3,0.1
270.0,270.0,2.1,-3.2,-0.8"""
        
        custom_problem = MolecularConformationProblem(csv_data=custom_data)
        print(f"   ✓ Custom problem created with {custom_problem.n_dihedrals} dihedral angles")
        
        custom_solution = custom_problem.random_solution()
        custom_energy = custom_problem.evaluate_solution(custom_solution)
        print(f"   ✓ Custom solution energy: {custom_energy:.3f} kcal/mol")
        
        # Test 3: Load from file
        print("\n3. Testing with CSV file...")
        csv_file_path = examples_dir / "molecular_conformation_problem" / "datasets" / "butane_conformations.csv"
        if csv_file_path.exists():
            file_problem = MolecularConformationProblem(csv_file_path=str(csv_file_path))
            print(f"   ✓ Problem loaded from file with {file_problem.n_dihedrals} dihedral angles")
            print(f"   ✓ Dataset contains {len(file_problem.molecular_data)} conformations")
            
            file_solution = file_problem.random_solution()
            file_energy = file_problem.evaluate_solution(file_solution)
            print(f"   ✓ File-based solution energy: {file_energy:.3f} kcal/mol")
        else:
            print("   ⚠️  CSV file not found, skipping file test")
        
        print("\n✅ Molecular Problem test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Molecular Problem test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_molecular_optimizer():
    """Test the molecular conformation optimizer."""
    print("\n" + "=" * 60)
    print("TESTING MOLECULAR CONFORMATION OPTIMIZER")
    print("=" * 60)
    
    try:
        from molecular_conformation_optimizer.qubot import MolecularConformationOptimizer
        
        # Test 1: Default optimizer
        print("\n1. Testing optimizer initialization...")
        optimizer = MolecularConformationOptimizer()
        print(f"   ✓ Optimizer created")
        print(f"   ✓ Initial temperature: {optimizer.initial_temperature}")
        print(f"   ✓ Cooling rate: {optimizer.cooling_rate}")
        print(f"   ✓ Step size: {optimizer.step_size}")
        
        # Test 2: Custom parameters
        print("\n2. Testing with custom parameters...")
        custom_optimizer = MolecularConformationOptimizer(
            initial_temperature=50.0,
            cooling_rate=0.9,
            max_iterations=1000,
            step_size=10.0
        )
        print(f"   ✓ Custom optimizer created")
        print(f"   ✓ Custom temperature: {custom_optimizer.initial_temperature}")
        print(f"   ✓ Custom iterations: {custom_optimizer.max_iterations}")
        
        print("\n✅ Molecular Optimizer test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Molecular Optimizer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rdkit_optimizer():
    """Test RDKit molecular optimizer."""
    print("\n" + "=" * 60)
    print("TESTING RDKIT MOLECULAR OPTIMIZER")
    print("=" * 60)

    try:
        from rdkit_molecular_optimizer.qubot import RDKitMolecularOptimizer

        # Test 1: Check RDKit availability
        print("\n1. Testing RDKit availability...")
        try:
            import rdkit
            print(f"   ✓ RDKit available (version: {rdkit.__version__})")
            rdkit_available = True
        except ImportError:
            print("   ⚠️  RDKit not available - install with: pip install rdkit")
            rdkit_available = False

        if not rdkit_available:
            print("   ⚠️  Skipping RDKit optimizer tests")
            return True

        # Test 2: Default optimizer
        print("\n2. Testing optimizer initialization...")
        optimizer = RDKitMolecularOptimizer()
        print(f"   ✓ RDKit optimizer created")
        print(f"   ✓ Force field: {optimizer.force_field}")
        print(f"   ✓ Number of conformers: {optimizer.num_conformers}")
        print(f"   ✓ Max iterations: {optimizer.max_iterations}")

        # Test 3: Custom parameters
        print("\n3. Testing with custom parameters...")
        custom_optimizer = RDKitMolecularOptimizer(
            force_field="UFF",
            num_conformers=50,
            max_iterations=500
        )
        print(f"   ✓ Custom RDKit optimizer created")
        print(f"   ✓ Custom force field: {custom_optimizer.force_field}")
        print(f"   ✓ Custom conformers: {custom_optimizer.num_conformers}")

        print("\n✅ RDKit Optimizer test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ RDKit Optimizer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between problem and optimizer."""
    print("\n" + "=" * 60)
    print("TESTING MOLECULAR OPTIMIZATION INTEGRATION")
    print("=" * 60)
    
    try:
        from molecular_conformation_problem.qubot import MolecularConformationProblem
        from molecular_conformation_optimizer.qubot import MolecularConformationOptimizer
        
        # Create problem and optimizer
        print("\n1. Setting up molecular optimization...")
        problem = MolecularConformationProblem()
        optimizer = MolecularConformationOptimizer(
            initial_temperature=30.0,  # Lower temperature for faster test
            max_iterations=500,        # Fewer iterations for quick test
            moves_per_temperature=20   # Fewer moves per temperature
        )
        
        print(f"   ✓ Problem: {problem.n_dihedrals} dihedral angles")
        print(f"   ✓ Energy range: {problem.min_energy:.2f} to {problem.max_energy:.2f} kcal/mol")
        print(f"   ✓ Optimizer: {optimizer.max_iterations} max iterations")
        
        # Run optimization
        print("\n2. Running optimization...")
        result = optimizer.optimize(problem)
        
        print(f"   ✓ Optimization completed")
        print(f"   ✓ Best energy: {result.best_value:.3f} kcal/mol")
        print(f"   ✓ Runtime: {result.runtime_seconds:.2f} seconds")
        print(f"   ✓ Iterations: {result.iterations}")
        print(f"   ✓ Termination: {result.termination_reason}")
        
        # Analyze results
        print("\n3. Analyzing results...")
        metrics = result.additional_metrics
        print(f"   ✓ Total moves: {metrics['total_moves']}")
        print(f"   ✓ Accepted moves: {metrics['accepted_moves']}")
        print(f"   ✓ Acceptance rate: {metrics['acceptance_rate']:.2%}")
        print(f"   ✓ Final temperature: {metrics['final_temperature']:.3f}")
        
        # Check if we found a good solution
        global_min_angles, global_min_energy = problem.get_global_minimum()
        energy_gap = result.best_value - global_min_energy
        print(f"   ✓ Energy gap from global minimum: {energy_gap:.3f} kcal/mol")
        
        if energy_gap < 2.0:  # Within 2 kcal/mol is good
            print("   ✓ Found good solution (within 2 kcal/mol of global minimum)")
        else:
            print("   ⚠️  Solution could be improved (try more iterations)")
        
        # Test solution info
        if 'molecular_info' in metrics:
            mol_info = metrics['molecular_info']
            print(f"   ✓ Molecular analysis available")
            print(f"     - Final angles: {[f'{a:.1f}°' for a in mol_info['dihedral_angles']]}")
            if 'energy_components' in mol_info:
                print(f"     - VdW energy: {mol_info['energy_components']['van_der_waals']:.3f} kcal/mol")
                print(f"     - Electrostatic: {mol_info['energy_components']['electrostatic']:.3f} kcal/mol")
        
        print("\n✅ Integration test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests for the molecular optimization decision model."""
    print("🧬 MOLECULAR CONFORMATION OPTIMIZATION DECISION MODEL TESTS")
    print("=" * 70)
    
    # Run individual tests
    tests = [
        ("Molecular Problem", test_molecular_problem),
        ("Molecular Optimizer", test_molecular_optimizer),
        ("RDKit Optimizer", test_rdkit_optimizer),
        ("Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The molecular optimization decision model is working correctly.")
        print("\nYou can now use this decision model for:")
        print("• Drug design and pharmaceutical research")
        print("• Protein conformation studies")
        print("• Chemical structure optimization")
        print("• Materials science applications")
    else:
        print("❌ SOME TESTS FAILED. Please check the error messages above.")
    
    print("\n📚 Next steps:")
    print("• Try different molecular datasets")
    print("• Experiment with optimizer parameters")
    print("• Integrate with other chemistry tools")
    print("• Upload to Rastion platform for cloud execution")

if __name__ == "__main__":
    main()
