"""
Workflow export system for qubots.
Handles export to various formats including MCP-compatible JSON.
"""

import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import asdict

from .code_generator import WorkflowDefinition, CodeGenerator
from .workflow_tester import WorkflowTester


class WorkflowExporter:
    """Export workflows to various formats."""
    
    def __init__(self, output_dir: str = "./exports"):
        """Initialize the workflow exporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.code_generator = CodeGenerator()
        self.workflow_tester = WorkflowTester()
    
    def export_to_mcp(self, workflow: WorkflowDefinition, 
                     include_code: bool = True,
                     include_tests: bool = False) -> Dict[str, Any]:
        """Export workflow to MCP-compatible format."""
        
        # Generate MCP JSON
        mcp_data = self.code_generator.generate_mcp_json(workflow)
        
        # Add additional MCP-specific metadata
        mcp_data.update({
            "mcp_version": "1.0.0",
            "qubots_version": "1.0.0",
            "exported_at": datetime.now().isoformat(),
            "export_format": "mcp",
            "compatibility": {
                "npx": True,
                "node": ">=16.0.0",
                "python": ">=3.8",
                "qubots": ">=1.0.0"
            }
        })
        
        # Add generated code if requested
        if include_code:
            python_code = self.code_generator.generate_python_code(workflow)
            mcp_data["generated_code"] = {
                "python": python_code,
                "language": "python",
                "executable": True
            }
        
        # Add test results if requested
        if include_tests:
            test_results = self.workflow_tester.run_full_test_suite(workflow)
            mcp_data["test_results"] = {
                "summary": self._summarize_test_results(test_results),
                "details": test_results,
                "tested_at": datetime.now().isoformat()
            }
        
        return mcp_data
    
    def export_to_package(self, workflow: WorkflowDefinition, 
                         package_name: Optional[str] = None) -> str:
        """Export workflow as a complete package."""
        
        if package_name is None:
            package_name = f"{workflow.name.lower().replace(' ', '-')}-workflow"
        
        package_dir = self.output_dir / package_name
        package_dir.mkdir(exist_ok=True)
        
        # Generate all files
        files = {}
        
        # 1. Workflow definition
        files["workflow.json"] = json.dumps(asdict(workflow), indent=2)
        
        # 2. Python code
        files["workflow.py"] = self.code_generator.generate_python_code(workflow)
        
        # 3. MCP export
        mcp_data = self.export_to_mcp(workflow, include_code=True, include_tests=True)
        files["mcp.json"] = json.dumps(mcp_data, indent=2)
        
        # 4. Package.json for NPX compatibility
        package_json = {
            "name": f"@qubots/{package_name}",
            "version": workflow.version,
            "description": workflow.description,
            "main": "workflow.py",
            "bin": {
                package_name: "./bin/run.js"
            },
            "scripts": {
                "start": "python workflow.py",
                "test": "python -m pytest tests/",
                "validate": "qubots-mcp workflow-validate workflow.json"
            },
            "keywords": ["qubots", "optimization", "workflow", "mcp"],
            "author": workflow.author,
            "license": "Apache-2.0",
            "dependencies": {
                "@qubots/mcp-tools": "^1.0.0"
            },
            "qubots": {
                "type": "workflow",
                "version": workflow.version,
                "nodes": len(workflow.nodes),
                "edges": len(workflow.edges)
            }
        }
        files["package.json"] = json.dumps(package_json, indent=2)
        
        # 5. NPX runner script
        npx_runner = '''#!/usr/bin/env node
const { spawn } = require('child_process');
const path = require('path');

const workflowPath = path.join(__dirname, '..', 'workflow.py');
const python = spawn('python', [workflowPath], { stdio: 'inherit' });

python.on('close', (code) => {
    process.exit(code);
});
'''
        files["bin/run.js"] = npx_runner
        
        # 6. README
        files["README.md"] = self._generate_package_readme(workflow, package_name)
        
        # 7. Requirements
        files["requirements.txt"] = "qubots>=1.0.0\nnumpy\nscipy"
        
        # 8. Test files
        test_results = self.workflow_tester.run_full_test_suite(workflow)
        files["tests/test_workflow.py"] = self._generate_test_file(workflow, test_results)
        
        # Write all files
        for file_path, content in files.items():
            full_path = package_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
        
        # Create ZIP package
        zip_path = self.output_dir / f"{package_name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in package_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(package_dir)
                    zipf.write(file_path, arcname)
        
        return str(zip_path)
    
    def export_for_sharing(self, workflow: WorkflowDefinition) -> Dict[str, str]:
        """Export workflow in multiple formats for easy sharing."""
        
        exports = {}
        
        # 1. JSON export
        json_path = self.output_dir / f"{workflow.name.lower().replace(' ', '-')}.json"
        json_path.write_text(json.dumps(asdict(workflow), indent=2))
        exports["json"] = str(json_path)
        
        # 2. Python code
        python_path = self.output_dir / f"{workflow.name.lower().replace(' ', '-')}.py"
        python_code = self.code_generator.generate_python_code(workflow)
        python_path.write_text(python_code)
        exports["python"] = str(python_path)
        
        # 3. MCP export
        mcp_path = self.output_dir / f"{workflow.name.lower().replace(' ', '-')}-mcp.json"
        mcp_data = self.export_to_mcp(workflow, include_code=True)
        mcp_path.write_text(json.dumps(mcp_data, indent=2))
        exports["mcp"] = str(mcp_path)
        
        # 4. NPX command file
        npx_path = self.output_dir / f"{workflow.name.lower().replace(' ', '-')}-npx.sh"
        npx_command = f'''#!/bin/bash
# Run this workflow using NPX
npx -y @qubots/mcp-tools workflow-execute {exports["json"]}
'''
        npx_path.write_text(npx_command)
        npx_path.chmod(0o755)
        exports["npx"] = str(npx_path)
        
        return exports
    
    def _summarize_test_results(self, test_results: Dict[str, List]) -> Dict[str, Any]:
        """Summarize test results for MCP export."""
        total_tests = sum(len(tests) for tests in test_results.values())
        passed_tests = sum(
            len([t for t in tests if t.result.value == "pass"])
            for tests in test_results.values()
        )
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "categories": {
                category: {
                    "total": len(tests),
                    "passed": len([t for t in tests if t.result.value == "pass"])
                }
                for category, tests in test_results.items()
            }
        }
    
    def _generate_package_readme(self, workflow: WorkflowDefinition, package_name: str) -> str:
        """Generate README for workflow package."""
        return f"""# {workflow.name}

{workflow.description}

## Overview

This is a Qubots optimization workflow package that can be executed locally or through NPX.

**Author:** {workflow.author}  
**Version:** {workflow.version}  
**Created:** {workflow.created_at}

## Components

- **Nodes:** {len(workflow.nodes)}
- **Connections:** {len(workflow.edges)}

### Problems
{chr(10).join(f"- {node.name} ({node.repository})" for node in workflow.nodes if node.type == "problem")}

### Optimizers
{chr(10).join(f"- {node.name} ({node.repository})" for node in workflow.nodes if node.type == "optimizer")}

## Usage

### Local Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run the workflow
python workflow.py
```

### NPX Execution

```bash
# Run directly with NPX
npx -y @qubots/{package_name}

# Or validate first
npx -y @qubots/mcp-tools workflow-validate workflow.json
npx -y @qubots/mcp-tools workflow-execute workflow.json
```

### MCP Integration

This workflow is MCP (Model Context Protocol) compatible and can be used with AI agents:

```bash
# Use with MCP tools
npx -y @qubots/mcp-tools component-search --type problem
npx -y @qubots/mcp-tools workflow-validate workflow.json
npx -y @qubots/mcp-tools code-generate workflow.json
```

## Files

- `workflow.json` - Workflow definition
- `workflow.py` - Generated Python code
- `mcp.json` - MCP-compatible export
- `package.json` - NPM package definition
- `requirements.txt` - Python dependencies
- `tests/` - Test suite

## Testing

```bash
# Run tests
python -m pytest tests/

# Validate workflow
qubots-mcp workflow-validate workflow.json
```

## License

Apache 2.0

## Generated by Qubots

This workflow was generated by the Qubots optimization framework.
Visit https://github.com/leonidas1312/qubots for more information.
"""
    
    def _generate_test_file(self, workflow: WorkflowDefinition, test_results: Dict) -> str:
        """Generate test file for workflow package."""
        return f'''"""
Test suite for {workflow.name}
Generated by Qubots Workflow Exporter
"""

import unittest
import json
from pathlib import Path


class TestWorkflow(unittest.TestCase):
    """Test cases for the workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.workflow_file = Path(__file__).parent.parent / "workflow.json"
        with open(self.workflow_file) as f:
            self.workflow_data = json.load(f)
    
    def test_workflow_structure(self):
        """Test workflow has valid structure."""
        self.assertIn("name", self.workflow_data)
        self.assertIn("nodes", self.workflow_data)
        self.assertIn("edges", self.workflow_data)
        self.assertGreater(len(self.workflow_data["nodes"]), 0)
    
    def test_node_ids_unique(self):
        """Test all node IDs are unique."""
        node_ids = [node["id"] for node in self.workflow_data["nodes"]]
        self.assertEqual(len(node_ids), len(set(node_ids)))
    
    def test_edges_valid(self):
        """Test all edges reference valid nodes."""
        node_ids = set(node["id"] for node in self.workflow_data["nodes"])
        for edge in self.workflow_data["edges"]:
            self.assertIn(edge["source"], node_ids)
            self.assertIn(edge["target"], node_ids)
    
    def test_has_optimization_components(self):
        """Test workflow has both problems and optimizers."""
        node_types = [node["type"] for node in self.workflow_data["nodes"]]
        self.assertIn("problem", node_types)
        self.assertIn("optimizer", node_types)


if __name__ == "__main__":
    unittest.main()
'''
