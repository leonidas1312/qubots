#!/usr/bin/env python3
"""
Test script for Audio Optimization Problem and Optimizer integration.

This script demonstrates the complete workflow of audio signal optimization
using the qubots framework, including problem loading, optimization, and
result analysis.
"""

import sys
import os
from pathlib import Path

# Add examples directory to path for imports
examples_dir = Path(__file__).parent
sys.path.insert(0, str(examples_dir))

def test_audio_problem():
    """Test the audio optimization problem with sample data."""
    print("\n" + "=" * 60)
    print("TESTING AUDIO OPTIMIZATION PROBLEM")
    print("=" * 60)
    
    try:
        from audio_optimization_problem.qubot import AudioOptimizationProblem
        
        # Test 1: Default data
        print("\n📊 Test 1: Default Audio Signals")
        print("-" * 40)
        problem = AudioOptimizationProblem()
        
        print(f"Number of signals: {problem.n_signals}")
        print(f"Target SNR: {problem.target_snr_db} dB")
        print(f"Max THD: {problem.max_thd_percent}%")
        
        # Generate and evaluate a random solution
        solution = problem.random_solution()
        quality_score = problem.evaluate_solution(solution)
        
        print(f"Random solution quality: {-quality_score:.6f}")  # Negative because we minimize
        print(f"Is feasible: {problem.is_feasible(solution)}")
        
        # Get detailed solution info
        info = problem.get_solution_info(solution)
        print(f"Overall quality score: {info['overall_quality_score']:.4f}")
        print(f"Quality improvement: {info['average_quality_improvement']:.4f}")
        
        # Test 2: Custom CSV data
        print("\n🎵 Test 2: Music Signals Dataset")
        print("-" * 40)
        
        music_data_path = examples_dir / "audio_optimization_problem" / "datasets" / "music_signals.csv"
        if music_data_path.exists():
            problem_music = AudioOptimizationProblem(
                csv_file_path=str(music_data_path),
                target_snr_db=25.0,
                quality_weight=0.6,
                noise_weight=0.3,
                distortion_weight=0.1
            )
            
            print(f"Music signals loaded: {problem_music.n_signals}")
            solution_music = problem_music.random_solution()
            quality_music = problem_music.evaluate_solution(solution_music)
            print(f"Music solution quality: {-quality_music:.6f}")
            
            # Show signal details
            for i, signal in enumerate(problem_music.signals[:3]):  # Show first 3
                print(f"  Signal {signal.signal_id}: {signal.frequency_hz:.1f} Hz, "
                      f"Quality: {signal.target_quality:.3f}")
        
        print("\n✅ Audio Problem tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Audio Problem test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_optimizer():
    """Test the audio optimization optimizer."""
    print("\n" + "=" * 60)
    print("TESTING AUDIO OPTIMIZATION OPTIMIZER")
    print("=" * 60)
    
    try:
        from audio_optimization_optimizer.qubot import AudioOptimizationOptimizer
        
        # Test different algorithms
        algorithms = ["slsqp", "differential_evolution", "genetic"]
        
        for algorithm in algorithms:
            print(f"\n🔧 Testing {algorithm.upper()} Algorithm")
            print("-" * 40)
            
            optimizer = AudioOptimizationOptimizer(
                algorithm=algorithm,
                max_iterations=50,  # Reduced for testing
                population_size=20,
                create_plots=False,  # Disable plots for testing
                random_seed=42
            )
            
            print(f"Algorithm: {optimizer.algorithm}")
            print(f"Max iterations: {optimizer.max_iterations}")
            print(f"Population size: {optimizer.population_size}")
        
        print("\n✅ Audio Optimizer tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Audio Optimizer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between problem and optimizer with sample data."""
    print("\n" + "=" * 60)
    print("TESTING INTEGRATION WITH SAMPLE DATA")
    print("=" * 60)
    
    try:
        from audio_optimization_problem.qubot import AudioOptimizationProblem
        from audio_optimization_optimizer.qubot import AudioOptimizationOptimizer
        
        # Test with different datasets
        datasets = [
            ("audio_signals.csv", 20.0, "General Audio Signals"),
            ("music_signals.csv", 25.0, "Music Production Signals")
        ]
        
        for dataset_file, target_snr, description in datasets:
            print(f"\n🎼 Testing: {description}")
            print("-" * 40)
            
            dataset_path = examples_dir / "audio_optimization_problem" / "datasets" / dataset_file
            if not dataset_path.exists():
                print(f"⚠️  Dataset {dataset_file} not found, skipping...")
                continue
            
            # Create problem
            problem = AudioOptimizationProblem(
                csv_file_path=str(dataset_path),
                target_snr_db=target_snr,
                quality_weight=0.5,
                noise_weight=0.3,
                distortion_weight=0.2
            )
            
            # Create optimizer with automatic algorithm selection
            optimizer = AudioOptimizationOptimizer(
                algorithm="auto",
                max_iterations=100,  # Reduced for testing
                population_size=30,
                create_plots=False,  # Disable plots for testing
                time_limit=30.0,
                random_seed=42
            )
            
            print(f"Problem: {problem.n_signals} signals")
            print(f"Optimizer: {optimizer.algorithm} algorithm")
            
            # Run optimization
            print("Running optimization...")
            result = optimizer.optimize(problem)
            
            print(f"✅ Optimization completed!")
            print(f"   Algorithm used: {result.algorithm_used}")
            print(f"   Best quality: {-result.best_value:.6f}")
            print(f"   Quality improvement: {result.quality_improvement:.4f}")
            print(f"   Noise reduction: {result.noise_reduction:.4f}")
            print(f"   Distortion reduction: {result.distortion_reduction:.4f}")
            print(f"   Runtime: {result.runtime_seconds:.2f} seconds")
            print(f"   Iterations: {result.iterations}")
            
            # Verify solution is feasible
            is_feasible = problem.is_feasible(result.best_solution)
            print(f"   Solution feasible: {is_feasible}")
            
            if not is_feasible:
                print("⚠️  Warning: Solution violates constraints!")
        
        print("\n✅ Integration tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all audio optimization tests."""
    print("🎵 AUDIO OPTIMIZATION TESTING SUITE")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Audio Problem", test_audio_problem),
        ("Audio Optimizer", test_audio_optimizer),
        ("Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} tests...")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Audio optimization is working correctly.")
    else:
        print("💥 SOME TESTS FAILED! Please check the errors above.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
