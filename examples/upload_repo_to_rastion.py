#!/usr/bin/env python3
"""
Upload Repository to Rastion Platform

This script uploads a qubots model repository to the Rastion platform.
It supports uploading from a directory path containing qubot.py and config.json files.

Usage:
    python upload_repo_to_rastion.py <repo_path> [options]

Example:
    python upload_repo_to_rastion.py ./my_vrp_problem --name "my_vrp_problem" --description "My custom VRP problem"

Author: Qubots Community
Version: 1.0.0
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Optional, List

# Import qubots modules
try:
    import qubots.rastion_unified as rastion
except ImportError as e:
    print(f"Error importing qubots: {e}")
    print("Please install qubots: pip install qubots")
    sys.exit(1)


def validate_repo_structure(repo_path: Path) -> bool:
    """
    Validate that the repository has the required structure.

    Args:
        repo_path: Path to the repository directory

    Returns:
        True if valid structure, False otherwise
    """
    required_files = ["qubot.py", "config.json"]

    for file_name in required_files:
        file_path = repo_path / file_name
        if not file_path.exists():
            print(f"❌ Missing required file: {file_name}")
            return False

    # Validate config.json structure
    try:
        config_path = repo_path / "config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        required_config_fields = ["type", "entry_point", "class_name"]
        for field in required_config_fields:
            if field not in config:
                print(f"❌ Missing required field in config.json: {field}")
                return False

        # Validate type
        if config["type"] not in ["problem", "optimizer"]:
            print(f"❌ Invalid type in config.json: {config['type']}. Must be 'problem' or 'optimizer'")
            return False

    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config.json: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading config.json: {e}")
        return False

    print("✅ Repository structure is valid")
    return True


def sanitize_repo_name(name: str) -> str:
    """
    Sanitize repository name to comply with platform requirements.
    Only used for auto-detected names from metadata, not user-provided names.

    Args:
        name: Original repository name

    Returns:
        Sanitized repository name (alphanumeric, dashes, dots, underscores only)
    """
    import re

    # Convert to lowercase and replace spaces with underscores
    sanitized = name.lower().replace(" ", "_")

    # Remove any characters that aren't alphanumeric, dash, dot, or underscore
    sanitized = re.sub(r'[^a-z0-9\-\._]', '', sanitized)

    # Remove multiple consecutive underscores/dashes
    sanitized = re.sub(r'[_\-]+', '_', sanitized)

    # Remove leading/trailing underscores or dashes
    sanitized = sanitized.strip('_-')

    # Ensure it's not empty and doesn't start with a number
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"qubots_{sanitized}" if sanitized else "qubots_model"

    return sanitized


def get_repo_info(repo_path: Path) -> dict:
    """
    Extract repository information from config.json and requirements.txt.

    Args:
        repo_path: Path to the repository directory

    Returns:
        Dictionary containing repository information
    """
    config_path = repo_path / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    metadata = config.get("metadata", {})

    # Get the original name and sanitize it for repository naming (only for auto-detected names)
    original_name = metadata.get("name", repo_path.name)
    sanitized_name = sanitize_repo_name(original_name)

    # Read requirements from requirements.txt if it exists, otherwise use config.json
    requirements_file = repo_path / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, 'r', encoding='utf-8') as f:
            requirements = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
    else:
        requirements = config.get("requirements", ["qubots"])

    return {
        "name": sanitized_name,
        "original_name": original_name,
        "description": metadata.get("description", f"Qubots {config['type']} uploaded from {repo_path}"),
        "type": config["type"],
        "class_name": config["class_name"],
        "requirements": requirements
    }


def upload_repository(repo_path: str, repository_name: Optional[str] = None,
                     description: Optional[str] = None, private: bool = False,
                     overwrite: bool = False, requirements: Optional[List[str]] = None) -> str:
    """
    Upload a repository to the Rastion platform.

    Args:
        repo_path: Path to the repository directory
        repository_name: Name for the repository (auto-detected if None)
        description: Description for the repository (auto-detected if None)
        private: Whether the repository should be private
        overwrite: Whether to overwrite existing repository
        requirements: Python requirements (auto-detected if None)

    Returns:
        Repository URL
    """
    repo_path = Path(repo_path).resolve()

    # Validate repository structure
    if not validate_repo_structure(repo_path):
        raise ValueError("Invalid repository structure")

    # Get repository information
    repo_info = get_repo_info(repo_path)

    # Use provided values or fall back to auto-detected ones
    final_name = repository_name or repo_info["name"]
    final_description = description or repo_info["description"]
    final_requirements = requirements or repo_info["requirements"]

    # Only sanitize if using auto-detected name (not user-provided name)
    if repository_name is None:
        # Auto-detected name, apply sanitization
        final_name = sanitize_repo_name(final_name)
    # else: User-provided name, use as-is

    print(f"📦 Uploading repository: {final_name}")
    if repository_name is None and "original_name" in repo_info and repo_info["original_name"] != final_name:
        print(f"📝 Original name: {repo_info['original_name']} (sanitized to: {final_name})")
    print(f"📝 Description: {final_description}")
    print(f"🏷️  Type: {repo_info['type']}")
    print(f"🔧 Class: {repo_info['class_name']}")
    print(f"📋 Requirements: {final_requirements}")
    print(f"🔒 Private: {private}")
    print(f"🔄 Overwrite: {overwrite}")

    # Check for additional directories that will be included
    additional_dirs = ["instances", "data", "datasets", "examples", "tests"]
    found_dirs = []
    for dir_name in additional_dirs:
        dir_path = repo_path / dir_name
        if dir_path.exists() and dir_path.is_dir():
            file_count = len(list(dir_path.rglob("*")))
            found_dirs.append(f"{dir_name} ({file_count} files)")

    if found_dirs:
        print(f"📁 Additional directories: {', '.join(found_dirs)}")
    else:
        print("📁 No additional directories found")

    try:
        # Upload using qubots rastion module
        url = rastion.upload_qubots_model(
            path=str(repo_path),
            repository_name=final_name,
            description=final_description,
            requirements=final_requirements,
            private=private,
            overwrite=overwrite
        )

        print(f"✅ Successfully uploaded repository!")
        print(f"🌐 Repository URL: {url}")
        return url

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Upload failed: {error_msg}")

        # Provide helpful error messages
        if "AlphaDashDot" in error_msg:
            print("💡 Repository name contains invalid characters. Only letters, numbers, dashes, dots, and underscores are allowed.")
            if repository_name is None:
                print(f"💡 Suggested name: {sanitize_repo_name(final_name)}")
            else:
                print("💡 Please choose a different name with only valid characters.")
        elif "already exists" in error_msg.lower():
            print("💡 Repository already exists. Use --overwrite flag to update it.")
        elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            print("💡 Authentication failed. Please check your token.")
        elif "not found" in error_msg.lower():
            print("💡 Repository or user not found. Check the repository name and permissions.")

        raise


def main():
    """Main function to handle command line arguments and upload."""
    parser = argparse.ArgumentParser(
        description="Upload a qubots repository to the Rastion platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload with auto-detected name and description
  python upload_repo_to_rastion.py ./my_vrp_problem

  # Upload with custom name and description
  python upload_repo_to_rastion.py ./my_vrp_problem --name "custom_vrp" --description "My custom VRP implementation"

  # Upload as private repository with overwrite
  python upload_repo_to_rastion.py ./my_optimizer --private --overwrite

  # Upload with custom requirements
  python upload_repo_to_rastion.py ./my_problem --requirements "qubots,numpy>=1.20.0,scipy"
        """
    )

    parser.add_argument("repo_path", help="Path to the repository directory")
    parser.add_argument("--name", help="Repository name (auto-detected from config.json if not provided)")
    parser.add_argument("--description", help="Repository description (auto-detected from config.json if not provided)")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing repository")
    parser.add_argument("--requirements", help="Comma-separated list of Python requirements")
    parser.add_argument("--token", help="Rastion authentication token (can also be set via environment variable RASTION_TOKEN)")

    args = parser.parse_args()

    # Check if repository path exists
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        print(f"❌ Repository path does not exist: {repo_path}")
        sys.exit(1)

    if not repo_path.is_dir():
        print(f"❌ Repository path is not a directory: {repo_path}")
        sys.exit(1)

    # Parse requirements
    requirements = None
    if args.requirements:
        requirements = [req.strip() for req in args.requirements.split(",") if req.strip()]

    # Authentication
    token = args.token or os.getenv("RASTION_TOKEN")
    if not token:
        print("❌ Authentication token required!")
        print("Provide token via --token argument or RASTION_TOKEN environment variable")
        print("Get your token from: https://rastion.com/settings/tokens")
        sys.exit(1)

    # Authenticate
    print("🔐 Authenticating with Rastion platform...")
    if not rastion.authenticate(token):
        print("❌ Authentication failed!")
        print("Please check your token and try again")
        sys.exit(1)

    print("✅ Authentication successful!")

    try:
        # Upload repository
        url = upload_repository(
            repo_path=str(repo_path),
            repository_name=args.name,
            description=args.description,
            private=args.private,
            overwrite=args.overwrite,
            requirements=requirements
        )

        print(f"\n🎉 Upload completed successfully!")
        print(f"📍 You can view your repository at: {url}")
        print(f"🎮 Test it in the playground: https://rastion.com/playground")

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
