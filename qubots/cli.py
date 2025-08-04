#!/usr/bin/env python3
"""
Qubots CLI tool for managing local Gitea setup and configuration.
"""

import argparse
import sys
import os
import json
import subprocess
from pathlib import Path
from typing import Optional

from .config_manager import get_config
from .local_gitea import get_local_gitea_client, setup_local_gitea
from .code_generator import CodeGenerator, WorkflowDefinition, WorkflowNode, WorkflowEdge
from .workflow_tester import WorkflowTester
from .workflow_exporter import WorkflowExporter


def cmd_setup(args):
    """Set up local Gitea instance."""
    print("🚀 Setting up local Qubots environment...")
    
    if setup_local_gitea():
        print("\n✅ Local Gitea setup completed!")
        print("📋 Next steps:")
        print("   1. Open http://localhost:3000 in your browser")
        print("   2. Complete the Gitea initial setup")
        print("   3. Create a user account")
        print("   4. Run: qubots auth --username YOUR_USERNAME --password YOUR_PASSWORD")
    else:
        print("❌ Setup failed. Please check Docker installation.")
        sys.exit(1)


def cmd_status(args):
    """Show status of local Gitea and configuration."""
    print("📊 Qubots Status")
    print("=" * 50)
    
    # Configuration status
    config = get_config()
    active_profile = config.get_active_profile()
    profile_info = config.get_profile_info()
    
    print(f"Active Profile: {active_profile}")
    print(f"Git Base URL: {profile_info['base_url']}")
    print(f"Authenticated: {'✅' if profile_info['authenticated'] else '❌'}")
    if profile_info['username']:
        print(f"Username: {profile_info['username']}")
    
    # Gitea availability
    if active_profile == "local":
        client = get_local_gitea_client()
        if client.is_available():
            print("Gitea Status: ✅ Available")
            
            # User info if authenticated
            if profile_info['authenticated']:
                user_info = client.get_user_info()
                if user_info:
                    print(f"User ID: {user_info['id']}")
                    print(f"Full Name: {user_info.get('full_name', 'N/A')}")
                    print(f"Email: {user_info.get('email', 'N/A')}")
        else:
            print("Gitea Status: ❌ Not available")
            print("Run 'qubots setup' to start local Gitea")
    
    # List repositories if authenticated
    if profile_info['authenticated'] and active_profile == "local":
        client = get_local_gitea_client()
        repos = client.list_repositories()
        print(f"\nRepositories: {len(repos)}")
        for repo in repos[:5]:  # Show first 5
            print(f"  - {repo['full_name']} ({'private' if repo['private'] else 'public'})")
        if len(repos) > 5:
            print(f"  ... and {len(repos) - 5} more")


def cmd_auth(args):
    """Authenticate with Gitea."""
    config = get_config()
    active_profile = config.get_active_profile()
    
    if active_profile != "local":
        print(f"❌ Authentication only supported for local profile. Current: {active_profile}")
        print("Run 'qubots profile set local' first")
        sys.exit(1)
    
    client = get_local_gitea_client()
    
    if not client.is_available():
        print("❌ Local Gitea is not available")
        print("Run 'qubots setup' first")
        sys.exit(1)
    
    username = args.username
    password = args.password
    
    if not username:
        username = input("Username: ")
    if not password:
        import getpass
        password = getpass.getpass("Password: ")
    
    print("🔐 Authenticating...")
    token = client.authenticate(username, password)
    
    if token:
        print("✅ Authentication successful!")
        print(f"Token saved for user: {username}")
    else:
        print("❌ Authentication failed")
        print("Please check your username and password")
        sys.exit(1)


def cmd_profile(args):
    """Manage configuration profiles."""
    config = get_config()
    
    if args.action == "list":
        profiles = config.list_profiles()
        active = config.get_active_profile()
        
        print("📋 Available Profiles:")
        for name, profile_config in profiles.items():
            marker = "✅" if name == active else "  "
            git_config = config.get_git_config(name)
            print(f"{marker} {name}: {git_config.base_url}")
    
    elif args.action == "set":
        if not args.name:
            print("❌ Profile name required")
            sys.exit(1)
        
        try:
            config.set_active_profile(args.name)
            print(f"✅ Active profile set to: {args.name}")
        except ValueError as e:
            print(f"❌ {e}")
            sys.exit(1)
    
    elif args.action == "create":
        if not args.name or not args.git_config:
            print("❌ Profile name and git-config required")
            sys.exit(1)
        
        try:
            config.create_profile(args.name, args.git_config)
            print(f"✅ Profile '{args.name}' created")
        except ValueError as e:
            print(f"❌ {e}")
            sys.exit(1)


def cmd_repo(args):
    """Manage repositories."""
    config = get_config()

    if not config.is_authenticated():
        print("❌ Not authenticated. Run 'qubots auth' first")
        sys.exit(1)

    client = get_local_gitea_client()

    if args.action == "list":
        repos = client.list_repositories()
        print(f"📚 Repositories ({len(repos)}):")
        for repo in repos:
            visibility = "🔒" if repo['private'] else "🌐"
            print(f"  {visibility} {repo['full_name']} - {repo.get('description', 'No description')}")

    elif args.action == "create":
        if not args.name:
            print("❌ Repository name required")
            sys.exit(1)

        success = client.create_repository(
            name=args.name,
            description=args.description or "",
            private=args.private
        )

        if success:
            print(f"✅ Repository '{args.name}' created")
        else:
            print(f"❌ Failed to create repository '{args.name}'")
            sys.exit(1)


def cmd_workflow(args):
    """Manage workflows."""
    if args.action == "validate":
        if not args.file:
            print("❌ Workflow file required")
            sys.exit(1)

        try:
            with open(args.file, 'r') as f:
                workflow_data = json.load(f)

            # Convert to WorkflowDefinition
            workflow = WorkflowDefinition(
                name=workflow_data.get('name', 'Unnamed'),
                description=workflow_data.get('description', ''),
                version=workflow_data.get('version', '1.0.0'),
                author=workflow_data.get('author', 'Unknown'),
                created_at=workflow_data.get('created_at', ''),
                nodes=[WorkflowNode(**node) for node in workflow_data.get('nodes', [])],
                edges=[WorkflowEdge(**edge) for edge in workflow_data.get('edges', [])],
                global_parameters=workflow_data.get('global_parameters', {}),
                metadata=workflow_data.get('metadata', {})
            )

            tester = WorkflowTester()
            results = tester.run_full_test_suite(workflow)
            report = tester.generate_test_report(results)

            print(report)

        except Exception as e:
            print(f"❌ Validation failed: {e}")
            sys.exit(1)

    elif args.action == "generate":
        if not args.file:
            print("❌ Workflow file required")
            sys.exit(1)

        try:
            with open(args.file, 'r') as f:
                workflow_data = json.load(f)

            # Convert to WorkflowDefinition
            workflow = WorkflowDefinition(
                name=workflow_data.get('name', 'Unnamed'),
                description=workflow_data.get('description', ''),
                version=workflow_data.get('version', '1.0.0'),
                author=workflow_data.get('author', 'Unknown'),
                created_at=workflow_data.get('created_at', ''),
                nodes=[WorkflowNode(**node) for node in workflow_data.get('nodes', [])],
                edges=[WorkflowEdge(**edge) for edge in workflow_data.get('edges', [])],
                global_parameters=workflow_data.get('global_parameters', {}),
                metadata=workflow_data.get('metadata', {})
            )

            generator = CodeGenerator()

            if args.format == "python":
                code = generator.generate_python_code(workflow)
                output_file = args.output or f"{workflow.name.lower().replace(' ', '_')}.py"
                with open(output_file, 'w') as f:
                    f.write(code)
                print(f"✅ Python code generated: {output_file}")

            elif args.format == "mcp":
                mcp_data = generator.generate_mcp_json(workflow)
                output_file = args.output or f"{workflow.name.lower().replace(' ', '_')}_mcp.json"
                with open(output_file, 'w') as f:
                    json.dump(mcp_data, f, indent=2)
                print(f"✅ MCP export generated: {output_file}")

        except Exception as e:
            print(f"❌ Code generation failed: {e}")
            sys.exit(1)

    elif args.action == "export":
        if not args.file:
            print("❌ Workflow file required")
            sys.exit(1)

        try:
            with open(args.file, 'r') as f:
                workflow_data = json.load(f)

            # Convert to WorkflowDefinition
            workflow = WorkflowDefinition(
                name=workflow_data.get('name', 'Unnamed'),
                description=workflow_data.get('description', ''),
                version=workflow_data.get('version', '1.0.0'),
                author=workflow_data.get('author', 'Unknown'),
                created_at=workflow_data.get('created_at', ''),
                nodes=[WorkflowNode(**node) for node in workflow_data.get('nodes', [])],
                edges=[WorkflowEdge(**edge) for edge in workflow_data.get('edges', [])],
                global_parameters=workflow_data.get('global_parameters', {}),
                metadata=workflow_data.get('metadata', {})
            )

            exporter = WorkflowExporter()

            if args.package:
                package_path = exporter.export_to_package(workflow)
                print(f"✅ Package exported: {package_path}")
            else:
                exports = exporter.export_for_sharing(workflow)
                print("✅ Workflow exported:")
                for format_name, file_path in exports.items():
                    print(f"  {format_name}: {file_path}")

        except Exception as e:
            print(f"❌ Export failed: {e}")
            sys.exit(1)


def cmd_component(args):
    """Manage components."""
    if args.action == "create":
        if not args.name or not args.type:
            print("❌ Component name and type required")
            sys.exit(1)

        try:
            generator = CodeGenerator()

            # Parse parameters
            parameters = {}
            if args.parameters:
                try:
                    parameters = json.loads(args.parameters)
                except json.JSONDecodeError:
                    print("❌ Invalid parameters JSON")
                    sys.exit(1)

            files = generator.generate_component_template(
                component_type=args.type,
                name=args.name,
                description=args.description or f"A {args.type} component",
                parameters=parameters
            )

            # Create component directory
            component_dir = Path(args.name.lower().replace(' ', '_'))
            component_dir.mkdir(exist_ok=True)

            # Write files
            for filename, content in files.items():
                file_path = component_dir / filename
                file_path.write_text(content)
                print(f"✅ Created: {file_path}")

            print(f"🎉 Component '{args.name}' created successfully!")
            print(f"📁 Directory: {component_dir}")

        except Exception as e:
            print(f"❌ Component creation failed: {e}")
            sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Qubots CLI - Manage local optimization framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up local Gitea instance")
    setup_parser.set_defaults(func=cmd_setup)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.set_defaults(func=cmd_status)
    
    # Auth command
    auth_parser = subparsers.add_parser("auth", help="Authenticate with Gitea")
    auth_parser.add_argument("--username", help="Gitea username")
    auth_parser.add_argument("--password", help="Gitea password")
    auth_parser.set_defaults(func=cmd_auth)
    
    # Profile command
    profile_parser = subparsers.add_parser("profile", help="Manage configuration profiles")
    profile_parser.add_argument("action", choices=["list", "set", "create"], help="Profile action")
    profile_parser.add_argument("--name", help="Profile name")
    profile_parser.add_argument("--git-config", choices=["local", "rastion"], help="Git configuration type")
    profile_parser.set_defaults(func=cmd_profile)
    
    # Repo command
    repo_parser = subparsers.add_parser("repo", help="Manage repositories")
    repo_parser.add_argument("action", choices=["list", "create"], help="Repository action")
    repo_parser.add_argument("--name", help="Repository name")
    repo_parser.add_argument("--description", help="Repository description")
    repo_parser.add_argument("--private", action="store_true", help="Create private repository")
    repo_parser.set_defaults(func=cmd_repo)

    # Workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Manage workflows")
    workflow_parser.add_argument("action", choices=["validate", "generate", "export"], help="Workflow action")
    workflow_parser.add_argument("--file", help="Workflow JSON file")
    workflow_parser.add_argument("--format", choices=["python", "mcp"], default="python", help="Output format")
    workflow_parser.add_argument("--output", help="Output file path")
    workflow_parser.add_argument("--package", action="store_true", help="Export as complete package")
    workflow_parser.set_defaults(func=cmd_workflow)

    # Component command
    component_parser = subparsers.add_parser("component", help="Manage components")
    component_parser.add_argument("action", choices=["create"], help="Component action")
    component_parser.add_argument("--name", help="Component name")
    component_parser.add_argument("--type", choices=["problem", "optimizer"], help="Component type")
    component_parser.add_argument("--description", help="Component description")
    component_parser.add_argument("--parameters", help="Parameters as JSON string")
    component_parser.set_defaults(func=cmd_component)
    
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
