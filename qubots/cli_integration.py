"""
Enhanced CLI integration for qubots with Rastion platform.
Extends the existing CLI with seamless model upload and management.
"""

import sys
import json
import importlib.util
from pathlib import Path
from typing import Optional, Union, Any, Dict

from .base_problem import BaseProblem
from .base_optimizer import BaseOptimizer
from .rastion_client import get_global_client, QubotPackager
from .rastion import authenticate, upload_model, discover_models


def load_model_from_file(file_path: str, class_name: str) -> Union[BaseProblem, BaseOptimizer]:
    """
    Load a qubots model from a Python file.
    
    Args:
        file_path: Path to the Python file
        class_name: Name of the class to instantiate
        
    Returns:
        Model instance
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the module
    spec = importlib.util.spec_from_file_location("temp_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get the class
    if not hasattr(module, class_name):
        raise ValueError(f"Class '{class_name}' not found in {file_path}")
    
    model_class = getattr(module, class_name)
    
    # Instantiate with default parameters
    try:
        return model_class()
    except TypeError:
        # If default constructor fails, try with empty dict
        try:
            return model_class(**{})
        except Exception as e:
            raise ValueError(f"Failed to instantiate {class_name}: {e}")


def upload_from_file(file_path: str, class_name: str, 
                    name: str, description: str,
                    requirements: Optional[str] = None,
                    private: bool = False) -> str:
    """
    Upload a qubots model from a Python file to the Rastion platform.
    
    Args:
        file_path: Path to the Python file containing the model
        class_name: Name of the class to upload
        name: Repository name
        description: Model description
        requirements: Comma-separated requirements
        private: Whether repository should be private
        
    Returns:
        Repository URL
    """
    # Load the model
    model = load_model_from_file(file_path, class_name)
    
    # Parse requirements
    req_list = None
    if requirements:
        req_list = [req.strip() for req in requirements.split(",") if req.strip()]
    
    # Upload the model
    return upload_model(model, name, description, req_list, private)


def interactive_upload():
    """Interactive upload wizard for qubots models."""
    print("🚀 Qubots Model Upload Wizard")
    print("=" * 40)
    
    # Check authentication
    client = get_global_client()
    if not client.is_authenticated():
        print("❌ Not authenticated with Rastion platform.")
        token = input("Please enter your Gitea token: ").strip()
        if not authenticate(token):
            print("❌ Authentication failed.")
            return
        print("✅ Authentication successful!")
    
    # Get file information
    file_path = input("📁 Python file path: ").strip()
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return
    
    class_name = input("🏷️  Class name: ").strip()
    name = input("📦 Repository name: ").strip()
    description = input("📝 Description: ").strip()
    
    requirements = input("📋 Requirements (comma-separated, optional): ").strip()
    private_input = input("🔒 Private repository? (y/N): ").strip().lower()
    private = private_input in ['y', 'yes']
    
    print("\n🔄 Uploading model...")
    
    try:
        url = upload_from_file(file_path, class_name, name, description, 
                              requirements if requirements else None, private)
        print(f"✅ Model uploaded successfully!")
        print(f"🌐 Repository URL: {url}")
        print(f"\n📖 Usage:")
        print(f"```python")
        print(f"import qubots.rastion as rastion")
        print(f"model = rastion.load_qubots_model('{name}')")
        print(f"```")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")


def quick_upload(file_path: str, class_name: str, name: str, description: str):
    """Quick upload without interactive prompts."""
    try:
        url = upload_from_file(file_path, class_name, name, description)
        print(f"✅ Uploaded: {url}")
        return url
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None


def list_models(username: Optional[str] = None):
    """List available models."""
    try:
        models = discover_models(username=username)
        
        if not models:
            print("No models found.")
            return
        
        print(f"📦 Available Models ({len(models)} found)")
        print("=" * 50)
        
        for model in models:
            print(f"🔹 {model['name']}")
            print(f"   👤 Owner: {model['owner']}")
            print(f"   📝 Description: {model.get('description', 'No description')}")
            print(f"   ⭐ Stars: {model.get('stars', 0)}")
            print(f"   🔗 URL: {model['url']}")
            print()
            
    except Exception as e:
        print(f"❌ Failed to list models: {e}")


def search_models_cli(query: str, limit: int = 10):
    """Search for models via CLI."""
    try:
        from .rastion import search_models
        models = search_models(query, limit=limit)
        
        if not models:
            print(f"No models found for query: '{query}'")
            return
        
        print(f"🔍 Search Results for '{query}' ({len(models)} found)")
        print("=" * 50)
        
        for model in models:
            print(f"🔹 {model['name']}")
            print(f"   👤 Owner: {model['owner']}")
            print(f"   📝 Description: {model.get('description', 'No description')}")
            print(f"   ⭐ Stars: {model.get('stars', 0)}")
            print()
            
    except Exception as e:
        print(f"❌ Search failed: {e}")


def validate_model(file_path: str, class_name: str) -> bool:
    """
    Validate that a Python file contains a valid qubots model.
    
    Args:
        file_path: Path to the Python file
        class_name: Name of the class to validate
        
    Returns:
        True if valid qubots model
    """
    try:
        model = load_model_from_file(file_path, class_name)
        
        # Check if it's a valid qubots model
        if not isinstance(model, (BaseProblem, BaseOptimizer)):
            print(f"❌ {class_name} is not a valid qubots model (must inherit from BaseProblem or BaseOptimizer)")
            return False
        
        # Check if it has required metadata
        if not hasattr(model, 'metadata'):
            print(f"⚠️  Warning: {class_name} doesn't have metadata attribute")
        
        model_type = "Problem" if isinstance(model, BaseProblem) else "Optimizer"
        print(f"✅ Valid qubots {model_type}: {class_name}")
        
        if hasattr(model, 'metadata'):
            metadata = model.metadata
            print(f"   📝 Name: {getattr(metadata, 'name', 'Unknown')}")
            print(f"   👤 Author: {getattr(metadata, 'author', 'Unknown')}")
            print(f"   🏷️  Version: {getattr(metadata, 'version', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


def show_usage_example(model_name: str, username: Optional[str] = None):
    """Show usage example for a model."""
    repo_id = f"{username}/{model_name}" if username else model_name
    
    print(f"📖 Usage Example for '{model_name}'")
    print("=" * 40)
    print("```python")
    print("import qubots.rastion as rastion")
    print()
    print("# Load the model")
    if username:
        print(f"model = rastion.load_qubots_model('{model_name}', username='{username}')")
    else:
        print(f"model = rastion.load_qubots_model('{model_name}')")
    print()
    print("# Use the model")
    print("# ... your optimization code here ...")
    print("```")


# CLI command mapping for integration with existing CLI
CLI_COMMANDS = {
    'upload': interactive_upload,
    'quick-upload': quick_upload,
    'list': list_models,
    'search': search_models_cli,
    'validate': validate_model,
    'usage': show_usage_example
}


if __name__ == "__main__":
    # Simple CLI interface for testing
    if len(sys.argv) < 2:
        print("Usage: python -m qubots.cli_integration <command> [args...]")
        print("Commands: upload, list, search, validate")
        sys.exit(1)
    
    command = sys.argv[1]
    if command in CLI_COMMANDS:
        if command == 'upload':
            interactive_upload()
        elif command == 'list':
            username = sys.argv[2] if len(sys.argv) > 2 else None
            list_models(username)
        elif command == 'search':
            if len(sys.argv) < 3:
                print("Usage: search <query>")
                sys.exit(1)
            search_models_cli(sys.argv[2])
        elif command == 'validate':
            if len(sys.argv) < 4:
                print("Usage: validate <file_path> <class_name>")
                sys.exit(1)
            validate_model(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
