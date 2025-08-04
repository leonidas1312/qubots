"""
Qubots API Backend
Provides REST API for the web interface
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
import json
import asyncio
from pathlib import Path

# Add the parent directory to Python path to import qubots
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from qubots import (
        AutoProblem, AutoOptimizer, 
        get_config, get_local_gitea_client,
        is_local_mode
    )
except ImportError as e:
    print(f"Warning: Could not import qubots: {e}")
    # Create mock classes for development
    class AutoProblem:
        @classmethod
        def from_repo(cls, repo_id, **kwargs):
            return None
    
    class AutoOptimizer:
        @classmethod
        def from_repo(cls, repo_id, **kwargs):
            return None
    
    def get_config():
        return None
    
    def get_local_gitea_client():
        return None
    
    def is_local_mode():
        return True

app = FastAPI(
    title="Qubots API",
    description="REST API for Qubots optimization workflow platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ComponentInfo(BaseModel):
    id: str
    name: str
    type: str
    description: str
    domain: str
    difficulty: str
    tags: List[str]
    author: str
    downloads: int
    rating: float
    last_updated: str
    repository: str

class WorkflowNode(BaseModel):
    id: str
    type: str
    config: Dict[str, Any]
    parameters: Dict[str, Any]
    position: Dict[str, float]

class WorkflowEdge(BaseModel):
    source: str
    target: str
    id: str

class Workflow(BaseModel):
    name: str
    description: str
    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]

class ExecutionRequest(BaseModel):
    workflow: Workflow
    parameters: Optional[Dict[str, Any]] = {}

# Mock data
MOCK_COMPONENTS = [
    ComponentInfo(
        id="portfolio-problem",
        name="Portfolio Optimization Problem",
        type="problem",
        description="Markowitz mean-variance portfolio optimization problem",
        domain="finance",
        difficulty="intermediate",
        tags=["portfolio", "markowitz", "finance", "risk"],
        author="Qubots Team",
        downloads=1250,
        rating=4.8,
        last_updated="2 days ago",
        repository="examples/portfolio_optimization_problem"
    ),
    ComponentInfo(
        id="genetic-optimizer",
        name="Genetic Algorithm Optimizer",
        type="optimizer",
        description="Evolutionary optimization algorithm with customizable operators",
        domain="metaheuristic",
        difficulty="intermediate",
        tags=["genetic", "evolutionary", "metaheuristic"],
        author="Qubots Team",
        downloads=2100,
        rating=4.9,
        last_updated="3 days ago",
        repository="examples/genetic_optimizer"
    ),
    ComponentInfo(
        id="tsp-problem",
        name="Traveling Salesman Problem",
        type="problem",
        description="Classic TSP optimization problem with various distance metrics",
        domain="routing",
        difficulty="intermediate",
        tags=["tsp", "routing", "combinatorial"],
        author="Qubots Team",
        downloads=890,
        rating=4.6,
        last_updated="1 week ago",
        repository="examples/tsp"
    ),
    ComponentInfo(
        id="ortools-optimizer",
        name="OR-Tools Solver",
        type="optimizer",
        description="Google OR-Tools optimization solver with constraint programming",
        domain="exact",
        difficulty="advanced",
        tags=["exact", "linear", "constraint", "google"],
        author="Qubots Team",
        downloads=1680,
        rating=4.7,
        last_updated="5 days ago",
        repository="examples/ortools_optimizer"
    )
]

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Qubots API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "mode": "local" if is_local_mode() else "cloud"}

@app.get("/api/components", response_model=List[ComponentInfo])
async def get_components(
    type: Optional[str] = None,
    domain: Optional[str] = None,
    search: Optional[str] = None
):
    """Get available components"""
    components = MOCK_COMPONENTS.copy()
    
    # Filter by type
    if type and type != "all":
        components = [c for c in components if c.type == type]
    
    # Filter by domain
    if domain and domain != "all":
        components = [c for c in components if c.domain == domain]
    
    # Filter by search term
    if search:
        search_lower = search.lower()
        components = [
            c for c in components 
            if (search_lower in c.name.lower() or 
                search_lower in c.description.lower() or
                any(search_lower in tag.lower() for tag in c.tags))
        ]
    
    return components

@app.get("/api/components/{component_id}", response_model=ComponentInfo)
async def get_component(component_id: str):
    """Get specific component details"""
    component = next((c for c in MOCK_COMPONENTS if c.id == component_id), None)
    if not component:
        raise HTTPException(status_code=404, detail="Component not found")
    return component

@app.post("/api/components/{component_id}/install")
async def install_component(component_id: str):
    """Install a component"""
    component = next((c for c in MOCK_COMPONENTS if c.id == component_id), None)
    if not component:
        raise HTTPException(status_code=404, detail="Component not found")
    
    # Simulate installation
    await asyncio.sleep(1)
    
    return {"message": f"Component {component.name} installed successfully"}

@app.post("/api/workflows/execute")
async def execute_workflow(request: ExecutionRequest):
    """Execute a workflow"""
    try:
        # Simulate workflow execution
        await asyncio.sleep(2)
        
        # Mock execution result
        result = {
            "status": "completed",
            "execution_id": "exec_123456",
            "runtime_seconds": 2.34,
            "best_solution": [1, 0, 1, 1, 0],
            "best_value": 42.5,
            "termination_reason": "optimal_found",
            "iterations": 150,
            "nodes_processed": len(request.workflow.nodes),
            "edges_processed": len(request.workflow.edges)
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

@app.post("/api/workflows/validate")
async def validate_workflow(workflow: Workflow):
    """Validate a workflow"""
    errors = []
    warnings = []
    
    # Basic validation
    if not workflow.nodes:
        errors.append("Workflow must contain at least one node")
    
    # Check for problems and optimizers
    problems = [n for n in workflow.nodes if n.type == "problem"]
    optimizers = [n for n in workflow.nodes if n.type == "optimizer"]
    
    if not problems:
        warnings.append("No problem nodes found")
    if not optimizers:
        warnings.append("No optimizer nodes found")
    
    # Check connections
    node_ids = {n.id for n in workflow.nodes}
    for edge in workflow.edges:
        if edge.source not in node_ids:
            errors.append(f"Edge source '{edge.source}' not found in nodes")
        if edge.target not in node_ids:
            errors.append(f"Edge target '{edge.target}' not found in nodes")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

@app.post("/api/workflows/generate-code")
async def generate_code(workflow: Workflow):
    """Generate Python code from workflow"""
    try:
        # Generate Python code
        code = generate_python_code(workflow)
        
        return {
            "code": code,
            "language": "python",
            "filename": f"{workflow.name.lower().replace(' ', '_')}.py"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

@app.get("/api/system/status")
async def get_system_status():
    """Get system status"""
    try:
        config = get_config()
        if config:
            profile_info = config.get_profile_info()
            return {
                "profile": profile_info["name"],
                "git_url": profile_info["base_url"],
                "authenticated": profile_info["authenticated"],
                "local_mode": is_local_mode()
            }
        else:
            return {
                "profile": "unknown",
                "git_url": "unknown",
                "authenticated": False,
                "local_mode": True
            }
    except Exception as e:
        return {
            "profile": "error",
            "git_url": "error",
            "authenticated": False,
            "local_mode": True,
            "error": str(e)
        }

def generate_python_code(workflow: Workflow) -> str:
    """Generate Python code from workflow definition"""
    if not workflow.nodes:
        return """# Empty workflow
# Add some components to generate code!

from qubots import AutoProblem, AutoOptimizer

# Your workflow will appear here once you add components
"""

    code = f'''#!/usr/bin/env python3
"""
{workflow.name}
{workflow.description}

Auto-generated from Qubots visual workflow designer
"""

from qubots import AutoProblem, AutoOptimizer
import json

def main():
    """Main workflow execution function."""
    
'''

    # Find problems and optimizers
    problems = [node for node in workflow.nodes if node.type == 'problem']
    optimizers = [node for node in workflow.nodes if node.type == 'optimizer']

    # Generate problem loading code
    for i, problem in enumerate(problems):
        var_name = f"problem_{i + 1}"
        repo_id = problem.config.get('repository', f'examples/{problem.id}')
        
        code += f'    # Load {problem.config.get("name", "Problem")}\n'
        code += f'    {var_name} = AutoProblem.from_repo("{repo_id}"'
        
        if problem.parameters:
            params_str = json.dumps(problem.parameters, indent=8).replace('\n', '\n        ')
            code += f',\n        override_params={params_str}'
        
        code += ')\n'
        code += f'    print(f"Loaded problem: {{{var_name}.metadata.name}}")\n\n'

    # Generate optimizer loading code
    for i, optimizer in enumerate(optimizers):
        var_name = f"optimizer_{i + 1}"
        repo_id = optimizer.config.get('repository', f'examples/{optimizer.id}')
        
        code += f'    # Load {optimizer.config.get("name", "Optimizer")}\n'
        code += f'    {var_name} = AutoOptimizer.from_repo("{repo_id}"'
        
        if optimizer.parameters:
            params_str = json.dumps(optimizer.parameters, indent=8).replace('\n', '\n        ')
            code += f',\n        override_params={params_str}'
        
        code += ')\n'
        code += f'    print(f"Loaded optimizer: {{{var_name}.metadata.name}}")\n\n'

    # Generate optimization execution code
    if problems and optimizers:
        code += '    # Run optimization\n'
        code += '    result = optimizer_1.optimize(problem_1)\n'
        code += '    \n'
        code += '    # Display results\n'
        code += '    print(f"Best solution: {result.best_solution}")\n'
        code += '    print(f"Best value: {result.best_value}")\n'
        code += '    print(f"Runtime: {result.runtime_seconds:.3f} seconds")\n'
        code += '    print(f"Status: {result.termination_reason}")\n'
    else:
        code += '    # Add both problems and optimizers to run optimization\n'
        code += '    print("Workflow loaded successfully!")\n'

    code += '''

if __name__ == "__main__":
    main()
'''

    return code

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
