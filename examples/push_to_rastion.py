#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("ERROR: GITHUB_TOKEN not found in .env file.")
        sys.exit(1)

    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Automate creating and pushing a problem or solver repository to Rastion Hub."
    )
    parser.add_argument(
        "--type",
        choices=["problem", "solver"],
        required=True,
        help="Type of repository to push: 'problem' or 'solver'."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Name of the repository to create (e.g., trivial-qubo)."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="The Python file to push (e.g., trivial_qubo.py)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="The configuration file (JSON) for the repository (e.g., problem_config.json)."
    )
    parser.add_argument(
        "--org",
        default="Rastion",
        help="GitHub organization (default: Rastion)."
    )
    args = parser.parse_args()

    # Command to create the repository
    create_repo_cmd = [
        "rastion", "create_repo", args.repo,
        "--org", args.org,
        "--github-token", github_token
    ]
    print("Creating repository with command:")
    print(" ".join(create_repo_cmd))
    try:
        subprocess.run(create_repo_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to create repository:", e)
        sys.exit(1)

    # Decide which push command to use
    if args.type == "problem":
        push_cmd = [
            "rastion", "push_problem", args.repo,
            "--file", args.file,
            "--config", args.config,
            "--org", args.org,
            "--github-token", github_token
        ]
    else:  # solver
        push_cmd = [
            "rastion", "push_solver", args.repo,
            "--file", args.file,
            "--config", args.config,
            "--org", args.org,
            "--github-token", github_token
        ]

    print("Pushing repository contents with command:")
    print(" ".join(push_cmd))
    try:
        subprocess.run(push_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to push repository contents:", e)
        sys.exit(1)

    print("Repository pushed successfully!")

if __name__ == "__main__":
    main()
