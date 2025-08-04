"""
Advanced code generation system for qubots workflows.
Generates Python scripts, MCP-compatible JSON, and component templates.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from jinja2 import Environment, FileSystemLoader, Template


@dataclass
class WorkflowNode:
    """Represents a node in the workflow."""
    id: str
    type: str  # 'problem', 'optimizer', 'data'
    name: str
    repository: str
    parameters: Dict[str, Any]
    position: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowEdge:
    """Represents an edge/connection in the workflow."""
    id: str
    source: str
    target: str
    source_handle: Optional[str] = None
    target_handle: Optional[str] = None


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    name: str
    description: str
    version: str
    author: str
    created_at: str
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]
    global_parameters: Dict[str, Any]
    metadata: Dict[str, Any]


class CodeGenerator:
    """Advanced code generator for qubots workflows."""
    
    def __init__(self, template_dir: Optional[str] = None):
        """Initialize the code generator."""
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Create default templates if they don't exist
        self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default code generation templates."""
        templates = {
            "workflow_main.py.j2": self._get_main_template(),
            "problem_template.py.j2": self._get_problem_template(),
            "optimizer_template.py.j2": self._get_optimizer_template(),
            "config_template.json.j2": self._get_config_template(),
            "mcp_schema.json.j2": self._get_mcp_template(),
            "test_template.py.j2": self._get_test_template()
        }
        
        for filename, content in templates.items():
            template_path = self.template_dir / filename
            if not template_path.exists():
                template_path.write_text(content)
    
    def generate_python_code(self, workflow: WorkflowDefinition) -> str:
        """Generate Python code from workflow definition."""
        template = self.jinja_env.get_template("workflow_main.py.j2")
        
        # Prepare template context
        context = {
            "workflow": workflow,
            "problems": [n for n in workflow.nodes if n.type == "problem"],
            "optimizers": [n for n in workflow.nodes if n.type == "optimizer"],
            "data_sources": [n for n in workflow.nodes if n.type == "data"],
            "has_connections": len(workflow.edges) > 0,
            "timestamp": datetime.now().isoformat(),
            "imports": self._get_required_imports(workflow),
            "execution_graph": self._build_execution_graph(workflow)
        }
        
        return template.render(**context)
    
    def generate_mcp_json(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Generate MCP-compatible JSON export."""
        template = self.jinja_env.get_template("mcp_schema.json.j2")
        
        # Build MCP tools from workflow nodes
        tools = []
        for node in workflow.nodes:
            tool = {
                "name": f"qubots_{node.type}_{node.id}",
                "description": f"{node.name} - {node.type} component",
                "inputSchema": {
                    "type": "object",
                    "properties": self._convert_parameters_to_schema(node.parameters),
                    "required": self._get_required_parameters(node.parameters)
                }
            }
            tools.append(tool)
        
        # MCP-compatible structure
        mcp_data = {
            "name": f"qubots-workflow-{workflow.name.lower().replace(' ', '-')}",
            "version": workflow.version,
            "description": workflow.description,
            "author": workflow.author,
            "license": "Apache-2.0",
            "main": "workflow.py",
            "type": "module",
            "exports": {
                "tools": tools
            },
            "qubots": {
                "workflow": asdict(workflow),
                "execution_order": self._get_execution_order(workflow),
                "compatibility": "1.0.0"
            }
        }
        
        return mcp_data
    
    def generate_component_template(self, 
                                  component_type: str, 
                                  name: str, 
                                  description: str,
                                  parameters: Dict[str, Any]) -> Dict[str, str]:
        """Generate a new component template."""
        if component_type not in ["problem", "optimizer"]:
            raise ValueError("Component type must be 'problem' or 'optimizer'")
        
        # Generate Python code
        template_name = f"{component_type}_template.py.j2"
        template = self.jinja_env.get_template(template_name)
        
        python_code = template.render(
            name=name,
            description=description,
            class_name=self._to_class_name(name),
            parameters=parameters,
            timestamp=datetime.now().isoformat()
        )
        
        # Generate config.json
        config_template = self.jinja_env.get_template("config_template.json.j2")
        config_json = config_template.render(
            type=component_type,
            name=name,
            description=description,
            class_name=self._to_class_name(name),
            parameters=parameters
        )
        
        # Generate test file
        test_template = self.jinja_env.get_template("test_template.py.j2")
        test_code = test_template.render(
            name=name,
            class_name=self._to_class_name(name),
            component_type=component_type,
            parameters=parameters
        )
        
        return {
            "qubot.py": python_code,
            "config.json": config_json,
            "test_qubot.py": test_code,
            "requirements.txt": "qubots\nnumpy\nscipy",
            "README.md": self._generate_readme(name, description, component_type)
        }
    
    def _get_required_imports(self, workflow: WorkflowDefinition) -> List[str]:
        """Get required imports for the workflow."""
        imports = ["from qubots import AutoProblem, AutoOptimizer"]
        
        # Add specific imports based on components
        for node in workflow.nodes:
            if "numpy" in str(node.parameters):
                imports.append("import numpy as np")
            if "pandas" in str(node.parameters):
                imports.append("import pandas as pd")
        
        return list(set(imports))
    
    def _build_execution_graph(self, workflow: WorkflowDefinition) -> List[Dict[str, Any]]:
        """Build execution graph from workflow edges."""
        graph = []
        
        # Create adjacency list
        adj_list = {}
        for edge in workflow.edges:
            if edge.source not in adj_list:
                adj_list[edge.source] = []
            adj_list[edge.source].append(edge.target)
        
        # Build execution steps
        for node in workflow.nodes:
            step = {
                "node_id": node.id,
                "type": node.type,
                "name": node.name,
                "dependencies": [e.source for e in workflow.edges if e.target == node.id],
                "outputs": adj_list.get(node.id, [])
            }
            graph.append(step)
        
        return graph
    
    def _convert_parameters_to_schema(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert qubots parameters to JSON schema."""
        schema = {}
        for param_name, param_value in parameters.items():
            if isinstance(param_value, (int, float)):
                schema[param_name] = {"type": "number", "default": param_value}
            elif isinstance(param_value, str):
                schema[param_name] = {"type": "string", "default": param_value}
            elif isinstance(param_value, bool):
                schema[param_name] = {"type": "boolean", "default": param_value}
            elif isinstance(param_value, list):
                schema[param_name] = {"type": "array", "default": param_value}
            else:
                schema[param_name] = {"type": "object", "default": param_value}
        
        return schema
    
    def _get_required_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Get list of required parameters."""
        # For now, consider all parameters optional
        # In the future, this could be based on parameter metadata
        return []
    
    def _get_execution_order(self, workflow: WorkflowDefinition) -> List[str]:
        """Get topological execution order of nodes."""
        # Simple topological sort
        in_degree = {node.id: 0 for node in workflow.nodes}
        
        for edge in workflow.edges:
            in_degree[edge.target] += 1
        
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            for edge in workflow.edges:
                if edge.source == node_id:
                    in_degree[edge.target] -= 1
                    if in_degree[edge.target] == 0:
                        queue.append(edge.target)
        
        return result
    
    def _to_class_name(self, name: str) -> str:
        """Convert name to Python class name."""
        return ''.join(word.capitalize() for word in name.replace('-', ' ').replace('_', ' ').split())
    
    def _generate_readme(self, name: str, description: str, component_type: str) -> str:
        """Generate README for component."""
        return f"""# {name}

{description}

## Type
{component_type.capitalize()}

## Usage

```python
from qubots import Auto{component_type.capitalize()}

# Load the {component_type}
{component_type} = Auto{component_type.capitalize()}.from_repo("your-username/{name.lower().replace(' ', '-')}")

# Use the {component_type}
# ... your code here ...
```

## Parameters

See `config.json` for available parameters and their descriptions.

## Installation

```bash
pip install qubots
```

## License

Apache 2.0
"""

    def _get_main_template(self) -> str:
        """Get the main workflow template."""
        return '''#!/usr/bin/env python3
"""
{{ workflow.name }}
{{ workflow.description }}

Generated by Qubots Code Generator
Created: {{ timestamp }}
Author: {{ workflow.author }}
"""

{% for import in imports %}
{{ import }}
{% endfor %}
import json
import time
from pathlib import Path


def main():
    """Main workflow execution function."""
    print("🚀 Starting {{ workflow.name }}")
    print("=" * 50)

    start_time = time.time()

    {% if problems %}
    # Load Problems
    {% for problem in problems %}
    print("📊 Loading {{ problem.name }}...")
    {{ problem.id }} = AutoProblem.from_repo("{{ problem.repository }}"{% if problem.parameters %},
        override_params={{ problem.parameters | tojson }}{% endif %})
    print(f"✅ Loaded: {{{ problem.id }}.metadata.name}")
    {% endfor %}

    {% endif %}
    {% if optimizers %}
    # Load Optimizers
    {% for optimizer in optimizers %}
    print("🧠 Loading {{ optimizer.name }}...")
    {{ optimizer.id }} = AutoOptimizer.from_repo("{{ optimizer.repository }}"{% if optimizer.parameters %},
        override_params={{ optimizer.parameters | tojson }}{% endif %})
    print(f"✅ Loaded: {{{ optimizer.id }}.metadata.name}")
    {% endfor %}

    {% endif %}
    {% if problems and optimizers %}
    # Execute Optimization
    print("🔄 Running optimization...")
    {% for step in execution_graph %}
    {% if step.type == "optimizer" and step.dependencies %}
    {% set problem_id = step.dependencies[0] %}
    result_{{ step.node_id }} = {{ step.node_id }}.optimize({{ problem_id }})
    print(f"📈 {{ step.name }} completed:")
    print(f"   Best value: {result_{{ step.node_id }}.best_value}")
    print(f"   Runtime: {result_{{ step.node_id }}.runtime_seconds:.3f}s")
    print(f"   Status: {result_{{ step.node_id }}.termination_reason}")
    {% endif %}
    {% endfor %}
    {% else %}
    print("⚠️  Add both problems and optimizers to run optimization")
    {% endif %}

    total_time = time.time() - start_time
    print(f"\\n⏱️  Total execution time: {total_time:.3f} seconds")
    print("✅ Workflow completed successfully!")


if __name__ == "__main__":
    main()
'''

    def _get_problem_template(self) -> str:
        """Get the problem component template."""
        return '''"""
{{ name }}
{{ description }}

Generated by Qubots Template Generator
Created: {{ timestamp }}
"""

import numpy as np
from qubots import BaseProblem, ProblemMetadata, ProblemType, ObjectiveType, DifficultyLevel


class {{ class_name }}(BaseProblem):
    """{{ description }}"""

    def __init__(self{% if parameters %}, {% for param, value in parameters.items() %}{{ param }}={{ value | tojson }}{% if not loop.last %}, {% endif %}{% endfor %}{% endif %}):
        """Initialize the problem."""
        {% for param in parameters.keys() %}
        self.{{ param }} = {{ param }}
        {% endfor %}
        super().__init__()

    def _get_default_metadata(self):
        """Get problem metadata."""
        return ProblemMetadata(
            name="{{ name }}",
            description="{{ description }}",
            problem_type=ProblemType.DISCRETE,  # Adjust as needed
            objective_type=ObjectiveType.MINIMIZE,  # Adjust as needed
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            domain="general",
            tags={"custom", "generated"},
            dimension=100,  # Adjust as needed
            constraints_count=0,
            evaluation_complexity="O(n)",
            memory_complexity="O(n)"
        )

    def random_solution(self):
        """Generate a random solution."""
        # TODO: Implement random solution generation
        return np.random.random(100)  # Placeholder

    def evaluate_solution(self, solution):
        """Evaluate a solution."""
        # TODO: Implement solution evaluation
        return np.sum(solution)  # Placeholder

    def is_feasible(self, solution):
        """Check if solution is feasible."""
        # TODO: Implement feasibility check
        return True  # Placeholder

    def get_bounds(self):
        """Get variable bounds."""
        # TODO: Implement bounds
        return [(0, 1) for _ in range(100)]  # Placeholder
'''

    def _get_optimizer_template(self) -> str:
        """Get the optimizer component template."""
        return '''"""
{{ name }}
{{ description }}

Generated by Qubots Template Generator
Created: {{ timestamp }}
"""

import numpy as np
from qubots import BaseOptimizer, OptimizerMetadata, OptimizerType, OptimizerFamily


class {{ class_name }}(BaseOptimizer):
    """{{ description }}"""

    def __init__(self{% if parameters %}, {% for param, value in parameters.items() %}{{ param }}={{ value | tojson }}{% if not loop.last %}, {% endif %}{% endfor %}{% endif %}):
        """Initialize the optimizer."""
        {% for param in parameters.keys() %}
        self.{{ param }} = {{ param }}
        {% endfor %}
        super().__init__()

    def _get_default_metadata(self):
        """Get optimizer metadata."""
        return OptimizerMetadata(
            name="{{ name }}",
            description="{{ description }}",
            optimizer_type=OptimizerType.METAHEURISTIC,  # Adjust as needed
            optimizer_family=OptimizerFamily.EVOLUTIONARY,  # Adjust as needed
            problem_types={"discrete", "continuous"},
            complexity_class="polynomial",
            deterministic=False,
            parallel_capable=True,
            memory_requirements="O(n)",
            typical_iterations=1000
        )

    def optimize(self, problem):
        """Optimize the given problem."""
        # TODO: Implement optimization algorithm

        # Placeholder implementation
        best_solution = problem.random_solution()
        best_value = problem.evaluate_solution(best_solution)

        for iteration in range(100):  # Simple random search
            candidate = problem.random_solution()
            if problem.is_feasible(candidate):
                value = problem.evaluate_solution(candidate)
                if value < best_value:  # Assuming minimization
                    best_solution = candidate
                    best_value = value

        return self._create_result(
            best_solution=best_solution,
            best_value=best_value,
            iterations=100,
            termination_reason="max_iterations"
        )
'''

    def _get_config_template(self) -> str:
        """Get the config.json template."""
        return '''{
  "type": "{{ type }}",
  "entry_point": "qubot",
  "class_name": "{{ class_name }}",
  "default_params": {{ parameters | tojson(indent=2) }},
  "metadata": {
    "name": "{{ name }}",
    "description": "{{ description }}",
    "domain": "general",
    "tags": ["custom", "generated"],
    "difficulty": "intermediate"
  },
  "parameters": {
    {% for param, value in parameters.items() %}
    "{{ param }}": {
      "type": "{% if value is number %}number{% elif value is string %}string{% elif value is boolean %}boolean{% else %}object{% endif %}",
      "default": {{ value | tojson }},
      "description": "{{ param | replace('_', ' ') | title }} parameter"
    }{% if not loop.last %},{% endif %}
    {% endfor %}
  }
}'''

    def _get_mcp_template(self) -> str:
        """Get the MCP schema template."""
        return '''{
  "name": "{{ name }}",
  "version": "{{ version }}",
  "description": "{{ description }}",
  "author": "{{ author }}",
  "license": "Apache-2.0",
  "type": "module",
  "main": "workflow.py",
  "exports": {
    "tools": {{ tools | tojson(indent=2) }}
  },
  "qubots": {{ qubots | tojson(indent=2) }},
  "mcp": {
    "version": "1.0.0",
    "compatibility": ["npx", "node", "python"],
    "runtime": {
      "python": {
        "version": ">=3.8",
        "dependencies": ["qubots", "numpy", "scipy"]
      }
    }
  }
}'''

    def _get_test_template(self) -> str:
        """Get the test template."""
        return '''"""
Test suite for {{ name }}
Generated by Qubots Template Generator
"""

import unittest
import numpy as np
from qubot import {{ class_name }}


class Test{{ class_name }}(unittest.TestCase):
    """Test cases for {{ class_name }}."""

    def setUp(self):
        """Set up test fixtures."""
        self.{{ component_type }} = {{ class_name }}()

    def test_initialization(self):
        """Test {{ component_type }} initialization."""
        self.assertIsNotNone(self.{{ component_type }})
        self.assertIsNotNone(self.{{ component_type }}.metadata)

    {% if component_type == "problem" %}
    def test_random_solution(self):
        """Test random solution generation."""
        solution = self.{{ component_type }}.random_solution()
        self.assertIsNotNone(solution)

    def test_evaluate_solution(self):
        """Test solution evaluation."""
        solution = self.{{ component_type }}.random_solution()
        value = self.{{ component_type }}.evaluate_solution(solution)
        self.assertIsInstance(value, (int, float))

    def test_feasibility_check(self):
        """Test feasibility checking."""
        solution = self.{{ component_type }}.random_solution()
        feasible = self.{{ component_type }}.is_feasible(solution)
        self.assertIsInstance(feasible, bool)
    {% endif %}

    {% if component_type == "optimizer" %}
    def test_optimization(self):
        """Test optimization process."""
        # Create a simple test problem
        from qubots.examples import SimpleTestProblem
        problem = SimpleTestProblem()

        result = self.{{ component_type }}.optimize(problem)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.best_solution)
        self.assertIsNotNone(result.best_value)
    {% endif %}

    def test_metadata(self):
        """Test metadata properties."""
        metadata = self.{{ component_type }}.metadata
        self.assertEqual(metadata.name, "{{ name }}")
        self.assertIsInstance(metadata.description, str)


if __name__ == "__main__":
    unittest.main()
'''
