"""
Test script to verify the Qubots-Rastion integration works correctly.
"""

import sys
import os

# Add the qubots directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_imports():
    """Test that all integration components can be imported."""
    print("🧪 Testing imports...")

    try:
        # Test main qubots imports
        import qubots
        print("✅ qubots imported successfully")

        # Test rastion module import
        import qubots.rastion as rastion
        print("✅ qubots.rastion imported successfully")

        # Test individual components
        from qubots import (
            RastionClient, QubotPackager,
            load_qubots_model, upload_qubots_model,
            list_available_models, search_models
        )
        print("✅ Individual components imported successfully")

        # Test rastion convenience module
        from qubots.rastion import (
            authenticate, load_qubots_model as load_model,
            upload_model, discover_models
        )
        print("✅ Rastion convenience functions imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_client_creation():
    """Test that the Rastion client can be created."""
    print("\n🧪 Testing client creation...")

    try:
        from qubots.rastion_client import RastionClient, get_global_client

        # Test direct client creation
        client = RastionClient()
        print("✅ RastionClient created successfully")

        # Test global client
        global_client = get_global_client()
        print("✅ Global client retrieved successfully")

        # Test client methods exist
        assert hasattr(client, 'authenticate')
        assert hasattr(client, 'create_repository')
        assert hasattr(client, 'upload_file_to_repo')
        assert hasattr(client, 'search_repositories')
        print("✅ Client methods verified")

        return True

    except Exception as e:
        print(f"❌ Client creation failed: {e}")
        return False


def test_packager():
    """Test the QubotPackager functionality."""
    print("\n🧪 Testing QubotPackager...")

    try:
        from qubots import BaseProblem, ProblemMetadata, ProblemType, ObjectiveType, DifficultyLevel
        from qubots.rastion_client import QubotPackager

        # Create a simple test problem
        class TestProblem(BaseProblem):
            def __init__(self):
                super().__init__(
                    metadata=ProblemMetadata(
                        name="Test Problem",
                        description="A test problem for integration testing",
                        author="Test Author",
                        version="1.0.0",
                        problem_type=ProblemType.CONTINUOUS,
                        objective_type=ObjectiveType.MINIMIZE,
                        difficulty_level=DifficultyLevel.BEGINNER,
                        tags={"test", "demo"}
                    )
                )

            def _get_default_metadata(self):
                return ProblemMetadata(
                    name="Test Problem",
                    description="A test problem for integration testing",
                    author="Test Author",
                    version="1.0.0",
                    problem_type=ProblemType.CONTINUOUS,
                    objective_type=ObjectiveType.MINIMIZE,
                    difficulty_level=DifficultyLevel.BEGINNER,
                    tags={"test", "demo"}
                )

            def evaluate_solution(self, solution):
                return sum(x**2 for x in solution)

            def evaluate(self, solution):
                return sum(x**2 for x in solution)

            def is_valid(self, solution):
                return True

            def get_random_solution(self):
                import random
                return [random.random() for _ in range(5)]

        # Test packaging
        test_problem = TestProblem()
        package = QubotPackager.package_model(
            test_problem,
            "test_problem",
            "A test problem for verification"
        )

        # Verify package contents
        expected_files = {"qubot.py", "config.json", "requirements.txt", "README.md"}
        assert set(package.keys()) == expected_files
        print("✅ Package created with correct files")

        # Verify config.json content
        import json
        config = json.loads(package["config.json"])
        assert config["type"] == "problem"
        assert config["class_name"] == "TestProblem"
        print("✅ Config.json content verified")

        return True

    except Exception as e:
        print(f"❌ Packager test failed: {e}")
        return False


def test_convenience_interface():
    """Test the convenience interface functions."""
    print("\n🧪 Testing convenience interface...")

    try:
        import qubots.rastion as rastion

        # Test that functions exist and are callable
        assert callable(rastion.authenticate)
        assert callable(rastion.load_qubots_model)
        assert callable(rastion.upload_model)
        assert callable(rastion.discover_models)
        assert callable(rastion.search_models)
        print("✅ All convenience functions are callable")

        # Test authentication check (should be False initially)
        auth_status = rastion.is_authenticated()
        print(f"✅ Authentication status check: {auth_status}")

        return True

    except Exception as e:
        print(f"❌ Convenience interface test failed: {e}")
        return False


def test_cli_integration():
    """Test the CLI integration components."""
    print("\n🧪 Testing CLI integration...")

    try:
        from qubots.cli_integration import (
            load_model_from_file, validate_model,
            CLI_COMMANDS
        )

        # Test that CLI commands are defined
        expected_commands = {'upload', 'quick-upload', 'list', 'search', 'validate', 'usage'}
        assert set(CLI_COMMANDS.keys()) == expected_commands
        print("✅ CLI commands defined correctly")

        # Test that functions exist
        assert callable(load_model_from_file)
        assert callable(validate_model)
        print("✅ CLI functions are callable")

        return True

    except Exception as e:
        print(f"❌ CLI integration test failed: {e}")
        return False


def test_example_demo():
    """Test that the example demo can be imported."""
    print("\n🧪 Testing example demo...")

    try:
        # Import the demo module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'examples'))

        # Test import (but don't run the full demo)
        import rastion_integration_demo

        # Check that the demo classes exist
        assert hasattr(rastion_integration_demo, 'SimpleTSPProblem')
        assert hasattr(rastion_integration_demo, 'SimpleGeneticAlgorithm')
        assert hasattr(rastion_integration_demo, 'demonstrate_workflow')
        print("✅ Example demo imported successfully")

        return True

    except Exception as e:
        print(f"❌ Example demo test failed: {e}")
        return False


def run_all_tests():
    """Run all integration tests."""
    print("🚀 Running Qubots-Rastion Integration Tests")
    print("=" * 50)

    tests = [
        test_imports,
        test_client_creation,
        test_packager,
        test_convenience_interface,
        test_cli_integration,
        test_example_demo
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Integration is working correctly.")
        print("\n📖 Next steps:")
        print("1. Authenticate: rastion.authenticate('your_token')")
        print("2. Load models: model = rastion.load_qubots_model('model_name')")
        print("3. Upload models: rastion.upload_model(model, 'name', 'description')")
        print("4. See examples/rastion_integration_demo.py for full workflow")
    else:
        print("❌ Some tests failed. Please check the error messages above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
