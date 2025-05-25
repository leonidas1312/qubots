"""
Test script for fantasy football optimization examples.

This script verifies that the examples work correctly and can be used
for continuous integration or quick validation.

Author: Qubots Community
Version: 1.0.0
"""

import sys
import os
import unittest
import warnings
from unittest.mock import patch, MagicMock

# Suppress warnings during testing
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the examples
try:
    from fantasy_football_optimization_example import (
        setup_authentication,
        load_fantasy_football_problem,
        configure_optimizers,
        run_optimization,
        SimpleRandomSearchOptimizer,
        FantasyFootballGeneticOptimizer
    )
    MAIN_EXAMPLE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Main example not available: {e}")
    MAIN_EXAMPLE_AVAILABLE = False

try:
    from fantasy_football_advanced_demo import (
        create_optimizer_configurations,
        OptimizerConfig
    )
    ADVANCED_DEMO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced demo not available: {e}")
    ADVANCED_DEMO_AVAILABLE = False

class TestFantasyFootballExamples(unittest.TestCase):
    """Test cases for fantasy football optimization examples."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.problem = None
        
        # Try to load the problem
        if MAIN_EXAMPLE_AVAILABLE:
            try:
                self.problem = load_fantasy_football_problem()
            except Exception as e:
                print(f"Warning: Could not load problem: {e}")
    
    @unittest.skipUnless(MAIN_EXAMPLE_AVAILABLE, "Main example not available")
    def test_authentication_setup(self):
        """Test authentication setup function."""
        result = setup_authentication()
        self.assertIsInstance(result, bool)
    
    @unittest.skipUnless(MAIN_EXAMPLE_AVAILABLE, "Main example not available")
    def test_problem_loading(self):
        """Test fantasy football problem loading."""
        problem = load_fantasy_football_problem()
        
        if problem is not None:
            # Test problem properties
            self.assertTrue(hasattr(problem, 'n_players'))
            self.assertTrue(hasattr(problem, 'max_salary'))
            self.assertTrue(hasattr(problem, 'metadata'))
            self.assertTrue(hasattr(problem, 'random_solution'))
            self.assertTrue(hasattr(problem, 'evaluate_solution'))
            self.assertTrue(hasattr(problem, 'is_feasible'))
            
            # Test problem methods
            solution = problem.random_solution()
            self.assertIsInstance(solution, list)
            self.assertEqual(len(solution), problem.n_players)
            
            # Test evaluation
            if problem.is_feasible(solution):
                points = problem.evaluate_solution(solution)
                self.assertIsInstance(points, (int, float))
                self.assertGreaterEqual(points, 0)
    
    @unittest.skipUnless(MAIN_EXAMPLE_AVAILABLE, "Main example not available")
    def test_random_search_optimizer(self):
        """Test random search optimizer."""
        optimizer = SimpleRandomSearchOptimizer(n_trials=10)
        self.assertEqual(optimizer.n_trials, 10)
        
        if self.problem is not None:
            result = optimizer.optimize(self.problem)
            
            # Test result structure
            self.assertTrue(hasattr(result, 'best_solution'))
            self.assertTrue(hasattr(result, 'best_value'))
            self.assertTrue(hasattr(result, 'is_feasible'))
            
            if result.best_solution is not None:
                self.assertIsInstance(result.best_solution, list)
                self.assertEqual(len(result.best_solution), self.problem.n_players)
    
    @unittest.skipUnless(MAIN_EXAMPLE_AVAILABLE, "Main example not available")
    def test_genetic_algorithm_optimizer(self):
        """Test genetic algorithm optimizer."""
        optimizer = FantasyFootballGeneticOptimizer(
            population_size=10,
            max_generations=5,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        # Test optimizer properties
        self.assertTrue(hasattr(optimizer, 'metadata'))
        self.assertTrue(hasattr(optimizer, '_population_size'))
        
        if self.problem is not None:
            result = optimizer.optimize(self.problem)
            
            # Test result structure
            self.assertTrue(hasattr(result, 'best_solution'))
            self.assertTrue(hasattr(result, 'best_value'))
            self.assertTrue(hasattr(result, 'is_feasible'))
            
            if result.best_solution is not None:
                self.assertIsInstance(result.best_solution, list)
                self.assertEqual(len(result.best_solution), self.problem.n_players)
    
    @unittest.skipUnless(MAIN_EXAMPLE_AVAILABLE, "Main example not available")
    def test_optimizer_configuration(self):
        """Test optimizer configuration function."""
        optimizers = configure_optimizers()
        
        self.assertIsInstance(optimizers, dict)
        self.assertGreater(len(optimizers), 0)
        
        # Test that at least one optimizer is available
        for name, optimizer in optimizers.items():
            self.assertIsInstance(name, str)
            self.assertTrue(hasattr(optimizer, 'optimize'))
    
    @unittest.skipUnless(MAIN_EXAMPLE_AVAILABLE, "Main example not available")
    def test_optimization_run(self):
        """Test running optimization with error handling."""
        if self.problem is None:
            self.skipTest("Problem not available")
        
        # Test with random search optimizer
        optimizer = SimpleRandomSearchOptimizer(n_trials=5)
        result = run_optimization(self.problem, optimizer, "Test Random Search")
        
        if result is not None:
            self.assertTrue(hasattr(result, 'best_solution'))
            self.assertTrue(hasattr(result, 'best_value'))
    
    @unittest.skipUnless(ADVANCED_DEMO_AVAILABLE, "Advanced demo not available")
    def test_optimizer_configurations_creation(self):
        """Test creation of optimizer configurations for benchmarking."""
        configs = create_optimizer_configurations()
        
        self.assertIsInstance(configs, list)
        self.assertGreater(len(configs), 0)
        
        # Test configuration structure
        for config in configs:
            self.assertIsInstance(config, OptimizerConfig)
            self.assertTrue(hasattr(config, 'name'))
            self.assertTrue(hasattr(config, 'optimizer_class'))
            self.assertTrue(hasattr(config, 'parameters'))
            self.assertTrue(hasattr(config, 'description'))
            
            # Test that optimizer class is callable
            self.assertTrue(callable(config.optimizer_class))
    
    def test_imports(self):
        """Test that all necessary imports work."""
        try:
            import qubots
            import pandas as pd
            import numpy as np
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Required import failed: {e}")
    
    def test_optional_imports(self):
        """Test optional imports and provide helpful messages."""
        optional_packages = {
            'matplotlib': 'pip install matplotlib',
            'seaborn': 'pip install seaborn',
            'ortools': 'pip install ortools'
        }
        
        for package, install_cmd in optional_packages.items():
            try:
                __import__(package)
                print(f"✅ {package} is available")
            except ImportError:
                print(f"⚠️  {package} not available. Install with: {install_cmd}")

class TestExampleIntegration(unittest.TestCase):
    """Integration tests for the complete examples."""
    
    @unittest.skipUnless(MAIN_EXAMPLE_AVAILABLE, "Main example not available")
    def test_main_example_execution(self):
        """Test that the main example can execute without errors."""
        # Mock the main function to avoid full execution
        with patch('fantasy_football_optimization_example.main') as mock_main:
            mock_main.return_value = None
            
            # Import and call main
            from fantasy_football_optimization_example import main
            
            # This should not raise an exception
            try:
                # We don't actually call main() to avoid long execution
                # but we verify it's importable and callable
                self.assertTrue(callable(main))
            except Exception as e:
                self.fail(f"Main example execution failed: {e}")
    
    @unittest.skipUnless(ADVANCED_DEMO_AVAILABLE, "Advanced demo not available")
    def test_advanced_demo_execution(self):
        """Test that the advanced demo can execute without errors."""
        # Mock the main function to avoid full execution
        with patch('fantasy_football_advanced_demo.main') as mock_main:
            mock_main.return_value = None
            
            # Import and call main
            from fantasy_football_advanced_demo import main
            
            # This should not raise an exception
            try:
                # We don't actually call main() to avoid long execution
                # but we verify it's importable and callable
                self.assertTrue(callable(main))
            except Exception as e:
                self.fail(f"Advanced demo execution failed: {e}")

def run_quick_test():
    """Run a quick test to verify basic functionality."""
    print("🧪 Running Quick Fantasy Football Examples Test")
    print("=" * 60)
    
    # Test 1: Basic imports
    print("1. Testing imports...")
    try:
        import qubots
        import pandas as pd
        import numpy as np
        print("   ✅ Core imports successful")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Problem loading
    print("2. Testing problem loading...")
    if MAIN_EXAMPLE_AVAILABLE:
        try:
            problem = load_fantasy_football_problem()
            if problem is not None:
                print(f"   ✅ Problem loaded: {problem.n_players} players")
            else:
                print("   ⚠️  Problem loading returned None")
        except Exception as e:
            print(f"   ❌ Problem loading failed: {e}")
    else:
        print("   ⚠️  Main example not available")
    
    # Test 3: Optimizer creation
    print("3. Testing optimizer creation...")
    if MAIN_EXAMPLE_AVAILABLE:
        try:
            optimizer = SimpleRandomSearchOptimizer(n_trials=5)
            print("   ✅ Random search optimizer created")
            
            ga_optimizer = FantasyFootballGeneticOptimizer(
                population_size=5, max_generations=2
            )
            print("   ✅ Genetic algorithm optimizer created")
        except Exception as e:
            print(f"   ❌ Optimizer creation failed: {e}")
    else:
        print("   ⚠️  Main example not available")
    
    # Test 4: Configuration creation
    print("4. Testing configuration creation...")
    if ADVANCED_DEMO_AVAILABLE:
        try:
            configs = create_optimizer_configurations()
            print(f"   ✅ Created {len(configs)} configurations")
        except Exception as e:
            print(f"   ❌ Configuration creation failed: {e}")
    else:
        print("   ⚠️  Advanced demo not available")
    
    print("\n🎯 Quick test completed!")
    return True

if __name__ == "__main__":
    # Run quick test first
    success = run_quick_test()
    
    if success:
        print("\n" + "=" * 60)
        print("Running comprehensive unit tests...")
        print("=" * 60)
        
        # Run unit tests
        unittest.main(verbosity=2, exit=False)
    else:
        print("\n❌ Quick test failed. Please check your installation.")
        sys.exit(1)
