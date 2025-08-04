"""
Integration testing framework for qubots workflows.
Validates workflow structure, component compatibility, and execution.
"""

import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .code_generator import WorkflowDefinition, WorkflowNode, WorkflowEdge
from .auto_problem import AutoProblem
from .auto_optimizer import AutoOptimizer


class TestResult(Enum):
    """Test result status."""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class TestCase:
    """Individual test case."""
    name: str
    description: str
    test_function: str
    expected_result: TestResult
    timeout: float = 30.0
    parameters: Dict[str, Any] = None


@dataclass
class TestReport:
    """Test execution report."""
    test_name: str
    result: TestResult
    message: str
    execution_time: float
    details: Dict[str, Any] = None


class WorkflowTester:
    """Integration testing framework for workflows."""
    
    def __init__(self):
        """Initialize the workflow tester."""
        self.test_cases = []
        self.reports = []
    
    def validate_workflow_structure(self, workflow: WorkflowDefinition) -> List[TestReport]:
        """Validate the basic structure of a workflow."""
        reports = []
        
        # Test 1: Workflow has nodes
        reports.append(self._run_test(
            "workflow_has_nodes",
            "Workflow contains at least one node",
            lambda: len(workflow.nodes) > 0,
            "Workflow must contain at least one node"
        ))
        
        # Test 2: Node IDs are unique
        node_ids = [node.id for node in workflow.nodes]
        reports.append(self._run_test(
            "unique_node_ids",
            "All node IDs are unique",
            lambda: len(node_ids) == len(set(node_ids)),
            "Duplicate node IDs found"
        ))
        
        # Test 3: Edge references valid nodes
        valid_node_ids = set(node_ids)
        for edge in workflow.edges:
            reports.append(self._run_test(
                f"edge_valid_{edge.id}",
                f"Edge {edge.id} references valid nodes",
                lambda e=edge: e.source in valid_node_ids and e.target in valid_node_ids,
                f"Edge {edge.id} references invalid nodes"
            ))
        
        # Test 4: No circular dependencies
        reports.append(self._run_test(
            "no_circular_dependencies",
            "Workflow has no circular dependencies",
            lambda: self._check_no_cycles(workflow),
            "Circular dependencies detected"
        ))
        
        # Test 5: Has optimization flow (problem -> optimizer)
        problems = [n for n in workflow.nodes if n.type == "problem"]
        optimizers = [n for n in workflow.nodes if n.type == "optimizer"]
        
        reports.append(self._run_test(
            "has_problems",
            "Workflow contains problem nodes",
            lambda: len(problems) > 0,
            "No problem nodes found"
        ))
        
        reports.append(self._run_test(
            "has_optimizers",
            "Workflow contains optimizer nodes",
            lambda: len(optimizers) > 0,
            "No optimizer nodes found"
        ))
        
        return reports
    
    def validate_component_compatibility(self, workflow: WorkflowDefinition) -> List[TestReport]:
        """Validate that components are compatible with each other."""
        reports = []
        
        # Check each edge for compatibility
        for edge in workflow.edges:
            source_node = next((n for n in workflow.nodes if n.id == edge.source), None)
            target_node = next((n for n in workflow.nodes if n.id == edge.target), None)
            
            if source_node and target_node:
                reports.append(self._run_test(
                    f"compatibility_{edge.id}",
                    f"Components {source_node.name} -> {target_node.name} are compatible",
                    lambda s=source_node, t=target_node: self._check_compatibility(s, t),
                    f"Incompatible connection: {source_node.type} -> {target_node.type}"
                ))
        
        return reports
    
    def validate_component_loading(self, workflow: WorkflowDefinition) -> List[TestReport]:
        """Validate that all components can be loaded."""
        reports = []
        
        for node in workflow.nodes:
            if node.type == "problem":
                reports.append(self._run_test(
                    f"load_problem_{node.id}",
                    f"Can load problem: {node.name}",
                    lambda n=node: self._try_load_problem(n),
                    f"Failed to load problem: {node.repository}"
                ))
            elif node.type == "optimizer":
                reports.append(self._run_test(
                    f"load_optimizer_{node.id}",
                    f"Can load optimizer: {node.name}",
                    lambda n=node: self._try_load_optimizer(n),
                    f"Failed to load optimizer: {node.repository}"
                ))
        
        return reports
    
    def validate_workflow_execution(self, workflow: WorkflowDefinition) -> List[TestReport]:
        """Validate that the workflow can execute successfully."""
        reports = []
        
        try:
            # Load all components
            loaded_components = {}
            
            for node in workflow.nodes:
                if node.type == "problem":
                    component = AutoProblem.from_repo(
                        node.repository,
                        override_params=node.parameters
                    )
                    loaded_components[node.id] = component
                elif node.type == "optimizer":
                    component = AutoOptimizer.from_repo(
                        node.repository,
                        override_params=node.parameters
                    )
                    loaded_components[node.id] = component
            
            reports.append(TestReport(
                test_name="load_all_components",
                result=TestResult.PASS,
                message="All components loaded successfully",
                execution_time=0.0
            ))
            
            # Execute optimization if possible
            problems = [n for n in workflow.nodes if n.type == "problem"]
            optimizers = [n for n in workflow.nodes if n.type == "optimizer"]
            
            if problems and optimizers:
                problem = loaded_components[problems[0].id]
                optimizer = loaded_components[optimizers[0].id]
                
                start_time = time.time()
                result = optimizer.optimize(problem)
                execution_time = time.time() - start_time
                
                reports.append(TestReport(
                    test_name="execute_optimization",
                    result=TestResult.PASS,
                    message=f"Optimization completed in {execution_time:.3f}s",
                    execution_time=execution_time,
                    details={
                        "best_value": result.best_value,
                        "iterations": result.iterations,
                        "termination_reason": result.termination_reason
                    }
                ))
            
        except Exception as e:
            reports.append(TestReport(
                test_name="workflow_execution",
                result=TestResult.ERROR,
                message=f"Execution failed: {str(e)}",
                execution_time=0.0,
                details={"error": traceback.format_exc()}
            ))
        
        return reports
    
    def run_full_test_suite(self, workflow: WorkflowDefinition) -> Dict[str, List[TestReport]]:
        """Run the complete test suite on a workflow."""
        test_suite = {
            "structure": self.validate_workflow_structure(workflow),
            "compatibility": self.validate_component_compatibility(workflow),
            "loading": self.validate_component_loading(workflow),
            "execution": self.validate_workflow_execution(workflow)
        }
        
        return test_suite
    
    def generate_test_report(self, test_results: Dict[str, List[TestReport]]) -> str:
        """Generate a human-readable test report."""
        report = []
        report.append("🧪 Qubots Workflow Test Report")
        report.append("=" * 50)
        report.append("")
        
        total_tests = sum(len(tests) for tests in test_results.values())
        passed_tests = sum(
            len([t for t in tests if t.result == TestResult.PASS])
            for tests in test_results.values()
        )
        
        report.append(f"📊 Summary: {passed_tests}/{total_tests} tests passed")
        report.append("")
        
        for category, tests in test_results.items():
            report.append(f"## {category.title()} Tests")
            report.append("")
            
            for test in tests:
                icon = {
                    TestResult.PASS: "✅",
                    TestResult.FAIL: "❌",
                    TestResult.SKIP: "⏭️",
                    TestResult.ERROR: "💥"
                }[test.result]
                
                report.append(f"{icon} {test.test_name}: {test.message}")
                
                if test.details:
                    for key, value in test.details.items():
                        report.append(f"   {key}: {value}")
            
            report.append("")
        
        return "\n".join(report)
    
    def _run_test(self, name: str, description: str, test_func, error_message: str) -> TestReport:
        """Run a single test."""
        start_time = time.time()
        
        try:
            result = test_func()
            execution_time = time.time() - start_time
            
            if result:
                return TestReport(
                    test_name=name,
                    result=TestResult.PASS,
                    message=description,
                    execution_time=execution_time
                )
            else:
                return TestReport(
                    test_name=name,
                    result=TestResult.FAIL,
                    message=error_message,
                    execution_time=execution_time
                )
        
        except Exception as e:
            execution_time = time.time() - start_time
            return TestReport(
                test_name=name,
                result=TestResult.ERROR,
                message=f"Test error: {str(e)}",
                execution_time=execution_time,
                details={"error": traceback.format_exc()}
            )
    
    def _check_no_cycles(self, workflow: WorkflowDefinition) -> bool:
        """Check if workflow has no circular dependencies."""
        # Build adjacency list
        adj_list = {}
        for node in workflow.nodes:
            adj_list[node.id] = []
        
        for edge in workflow.edges:
            adj_list[edge.source].append(edge.target)
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for neighbor in adj_list.get(node_id, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node in workflow.nodes:
            if node.id not in visited:
                if has_cycle(node.id):
                    return False
        
        return True
    
    def _check_compatibility(self, source: WorkflowNode, target: WorkflowNode) -> bool:
        """Check if two components are compatible."""
        # Define compatibility rules
        compatibility_rules = {
            ("problem", "optimizer"): True,
            ("data", "problem"): True,
            ("data", "optimizer"): True,
            ("optimizer", "optimizer"): False,  # Generally not allowed
            ("problem", "problem"): False,      # Generally not allowed
        }
        
        return compatibility_rules.get((source.type, target.type), False)
    
    def _try_load_problem(self, node: WorkflowNode) -> bool:
        """Try to load a problem component."""
        try:
            AutoProblem.from_repo(node.repository, override_params=node.parameters)
            return True
        except Exception:
            return False
    
    def _try_load_optimizer(self, node: WorkflowNode) -> bool:
        """Try to load an optimizer component."""
        try:
            AutoOptimizer.from_repo(node.repository, override_params=node.parameters)
            return True
        except Exception:
            return False
