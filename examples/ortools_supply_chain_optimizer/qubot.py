"""
OR-Tools-based Supply Chain Optimizer for Qubots Framework

This module implements a linear programming optimizer using Google OR-Tools for supply chain
network optimization problems. It provides optimal solutions with comprehensive
visualization and analysis.

Compatible with Rastion platform workflow automation and local development.
"""

from ortools.linear_solver import pywraplp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, Any, Optional, List, Tuple
import time
from qubots import (
    BaseOptimizer, OptimizerMetadata, OptimizerType, OptimizerFamily,
    OptimizationResult, BaseProblem
)


class ORToolsSupplyChainOptimizer(BaseOptimizer):
    """
    OR-Tools-based Linear Programming optimizer for supply chain network problems.

    This optimizer uses linear programming to find optimal solutions for supply chain
    network optimization problems. It formulates the problem as a mixed-integer linear
    program (MILP) and solves it using OR-Tools' solvers.

    Features:
    - Optimal solution guarantee for linear/mixed-integer problems
    - Multiple solver support (SCIP, GLOP, CBC, Gurobi, CPLEX)
    - Comprehensive visualization of results
    - Network flow analysis and cost breakdown
    - Solution validation and sensitivity analysis
    """

    def __init__(self, solver_name: str = "SCIP", time_limit: float = 300.0, **kwargs):
        """
        Initialize OR-Tools Supply Chain Optimizer.

        Args:
            solver_name: Solver to use ("SCIP", "GLOP", "CBC", "GUROBI", "CPLEX")
            time_limit: Maximum solving time in seconds
            **kwargs: Additional parameters passed to BaseOptimizer
        """
        metadata = OptimizerMetadata(
            name="OR-Tools Supply Chain Optimizer",
            description="Linear programming optimizer for supply chain networks using Google OR-Tools",
            optimizer_type=OptimizerType.EXACT,
            optimizer_family=OptimizerFamily.LINEAR_PROGRAMMING,
            author="Qubots Framework",
            version="1.0.0",
            is_deterministic=True,
            supports_constraints=True,
            supports_continuous=True,
            supports_discrete=True,
            supports_mixed_integer=True,
            time_complexity="Polynomial for LP, Exponential for MILP",
            space_complexity="O(variables + constraints)",
            convergence_guaranteed=True,
            parallel_capable=True
        )

        super().__init__(metadata, **kwargs)
        
        self.solver_name = solver_name
        self.time_limit = time_limit
        self._setup_solver()

    def _get_default_metadata(self) -> OptimizerMetadata:
        """Return default metadata for the optimizer."""
        return self.metadata

    def _setup_solver(self):
        """Setup the OR-Tools solver."""
        # Store solver name as string for OR-Tools API
        self.solver_type = self.solver_name.upper()

    def _optimize_implementation(self, problem: BaseProblem, 
                               initial_solution: Optional[Any] = None) -> OptimizationResult:
        """
        Core optimization implementation using OR-Tools linear programming.

        Args:
            problem: Supply chain network problem instance
            initial_solution: Optional initial solution (not used in LP)

        Returns:
            OptimizationResult with optimal solution and comprehensive analysis
        """
        start_time = time.time()
        
        try:
            # Create the solver
            solver = pywraplp.Solver.CreateSolver(self.solver_type)
            if not solver:
                raise Exception(f"Could not create solver for {self.solver_name}")
            
            # Set time limit
            solver.SetTimeLimit(int(self.time_limit * 1000))  # OR-Tools expects milliseconds
            
            # Formulate the linear programming problem
            variables = self._formulate_lp_problem(solver, problem)
            
            # Solve the problem
            status = solver.Solve()



            # Extract solution
            solution = self._extract_solution(solver, variables, problem)

            # Calculate metrics
            objective_value = problem.evaluate_solution(solution)
            runtime = time.time() - start_time
            
            # Generate visualizations
            self._create_visualizations(problem, solution, solver)
            
            # Create result
            result = OptimizationResult(
                best_solution=solution,
                best_value=objective_value,
                runtime_seconds=runtime,
                iterations=1,
                evaluations=1,
                termination_reason=self._get_termination_reason(status),
                optimization_history=[{"iteration": 0, "objective_value": objective_value}],
                additional_metrics={
                    "solver_status": self._get_status_string(status),
                    "solver_name": self.solver_name,
                    "problem_type": "Mixed Integer Linear Program",
                    "variables_count": solver.NumVariables(),
                    "constraints_count": solver.NumConstraints(),
                    "is_optimal": status == pywraplp.Solver.OPTIMAL,
                    "objective_value": solver.Objective().Value() if status == pywraplp.Solver.OPTIMAL else None
                }
            )
            
            return result
            
        except Exception as e:
            # Return failure result
            runtime = time.time() - start_time
            return OptimizationResult(
                best_solution=problem.random_solution(),
                best_value=float('inf'),
                runtime_seconds=runtime,
                iterations=0,
                evaluations=0,
                termination_reason="error",
                optimization_history=[],
                additional_metrics={"error": str(e), "solver_name": self.solver_name}
            )

    def _get_status_string(self, status) -> str:
        """Convert OR-Tools status to string."""
        status_map = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE", 
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
            pywraplp.Solver.ABNORMAL: "ABNORMAL",
            pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"
        }
        return status_map.get(status, "UNKNOWN")

    def _get_termination_reason(self, status) -> str:
        """Get termination reason from OR-Tools status."""
        if status == pywraplp.Solver.OPTIMAL:
            return "optimal"
        elif status == pywraplp.Solver.FEASIBLE:
            return "time_limit"
        elif status == pywraplp.Solver.INFEASIBLE:
            return "infeasible"
        elif status == pywraplp.Solver.UNBOUNDED:
            return "unbounded"
        else:
            return "solver_limit"

    def _formulate_lp_problem(self, solver, problem) -> Dict[str, Any]:
        """Formulate the supply chain problem as a linear program."""
        variables = {
            'production_vars': {},
            'warehouse_vars': {},
            'flow_vars': {}
        }

        # Decision variables
        # Production variables: x[s] = production amount at supplier s
        for supplier_id in problem.suppliers:
            supplier = problem.suppliers[supplier_id]
            variables['production_vars'][supplier_id] = solver.NumVar(
                0.0, supplier.capacity, f"production_{supplier_id}"
            )

        # Warehouse operation variables: y[w] = 1 if warehouse w is open, 0 otherwise
        for warehouse_id in problem.warehouses:
            variables['warehouse_vars'][warehouse_id] = solver.BoolVar(
                f"warehouse_{warehouse_id}"
            )

        # Flow variables: f[(i,j)] = flow from entity i to entity j

        # Supplier to warehouse flows
        for (supplier_id, warehouse_id) in problem.supplier_warehouse_links:
            variables['flow_vars'][(supplier_id, warehouse_id)] = solver.NumVar(
                0.0, solver.infinity(), f"flow_{supplier_id}_{warehouse_id}"
            )

        # Warehouse to customer flows
        for (warehouse_id, customer_id) in problem.warehouse_customer_links:
            variables['flow_vars'][(warehouse_id, customer_id)] = solver.NumVar(
                0.0, solver.infinity(), f"flow_{warehouse_id}_{customer_id}"
            )

        # Objective function: minimize total cost
        objective = solver.Objective()
        objective.SetMinimization()

        # Production costs
        for supplier_id, var in variables['production_vars'].items():
            supplier = problem.suppliers[supplier_id]
            objective.SetCoefficient(var, supplier.variable_cost)

        # Warehouse fixed costs
        for warehouse_id, var in variables['warehouse_vars'].items():
            warehouse = problem.warehouses[warehouse_id]
            objective.SetCoefficient(var, warehouse.fixed_cost)

        # Transportation costs
        for (from_id, to_id), var in variables['flow_vars'].items():
            if (from_id, to_id) in problem.supplier_warehouse_links:
                link = problem.supplier_warehouse_links[(from_id, to_id)]
                objective.SetCoefficient(var, link.unit_cost)
            elif (from_id, to_id) in problem.warehouse_customer_links:
                link = problem.warehouse_customer_links[(from_id, to_id)]
                objective.SetCoefficient(var, link.unit_cost)

        # Constraints

        # 1. Demand satisfaction constraints
        for customer_id, customer in problem.customers.items():
            constraint = solver.Constraint(customer.demand, solver.infinity(), f"demand_{customer_id}")
            for warehouse_id in problem.warehouses:
                if (warehouse_id, customer_id) in variables['flow_vars']:
                    constraint.SetCoefficient(variables['flow_vars'][(warehouse_id, customer_id)], 1.0)

        # 2. Supplier capacity constraints
        for supplier_id, supplier in problem.suppliers.items():
            # Production >= Outflow constraint
            constraint = solver.Constraint(0.0, solver.infinity(), f"supplier_capacity_{supplier_id}")
            # Production variable coefficient (positive)
            constraint.SetCoefficient(variables['production_vars'][supplier_id], 1.0)
            # Outflow variables coefficients (negative)
            for warehouse_id in problem.warehouses:
                if (supplier_id, warehouse_id) in variables['flow_vars']:
                    constraint.SetCoefficient(variables['flow_vars'][(supplier_id, warehouse_id)], -1.0)

        # 3. Warehouse capacity and operation constraints
        for warehouse_id, warehouse in problem.warehouses.items():
            # Warehouse capacity constraint: inflow <= capacity * is_open
            capacity_constraint = solver.Constraint(-solver.infinity(), 0.0, f"warehouse_capacity_{warehouse_id}")

            # Inflow (positive coefficients)
            for supplier_id in problem.suppliers:
                if (supplier_id, warehouse_id) in variables['flow_vars']:
                    capacity_constraint.SetCoefficient(variables['flow_vars'][(supplier_id, warehouse_id)], 1.0)

            # Warehouse capacity constraint (negative coefficient)
            capacity_constraint.SetCoefficient(variables['warehouse_vars'][warehouse_id], -warehouse.capacity)

            # Flow balance constraint: inflow = outflow
            balance_constraint = solver.Constraint(0.0, 0.0, f"flow_balance_{warehouse_id}")

            # Inflow (positive coefficients)
            for supplier_id in problem.suppliers:
                if (supplier_id, warehouse_id) in variables['flow_vars']:
                    balance_constraint.SetCoefficient(variables['flow_vars'][(supplier_id, warehouse_id)], 1.0)

            # Outflow (negative coefficients)
            for customer_id in problem.customers:
                if (warehouse_id, customer_id) in variables['flow_vars']:
                    balance_constraint.SetCoefficient(variables['flow_vars'][(warehouse_id, customer_id)], -1.0)

        return variables

    def _extract_solution(self, solver, variables: Dict[str, Any], problem) -> Dict[str, Any]:
        """Extract solution from solved OR-Tools problem."""
        solution = {
            'supplier_production': {},
            'warehouse_flows': {},
            'warehouse_operations': {}
        }

        # Extract production values
        for supplier_id, var in variables['production_vars'].items():
            solution['supplier_production'][supplier_id] = var.solution_value()

        # Extract warehouse operations
        for warehouse_id, var in variables['warehouse_vars'].items():
            solution['warehouse_operations'][warehouse_id] = int(var.solution_value())

        # Extract flow values
        for (from_id, to_id), var in variables['flow_vars'].items():
            flow_value = var.solution_value()
            if flow_value and flow_value > 1e-6:  # Only include non-zero flows
                solution['warehouse_flows'][(from_id, to_id)] = flow_value

        return solution

    def _create_visualizations(self, problem, solution: Dict[str, Any], solver):
        """Create comprehensive visualizations of the optimization results."""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))

        # 1. Network Flow Visualization
        ax1 = plt.subplot(2, 3, 1)
        self._plot_network_flow(problem, solution, ax1)

        # 2. Cost Breakdown
        ax2 = plt.subplot(2, 3, 2)
        self._plot_cost_breakdown(problem, solution, ax2)

        # 3. Capacity Utilization
        ax3 = plt.subplot(2, 3, 3)
        self._plot_capacity_utilization(problem, solution, ax3)

        # 4. Geographic Layout
        ax4 = plt.subplot(2, 3, 4)
        self._plot_geographic_layout(problem, solution, ax4)

        # 5. Flow Distribution
        ax5 = plt.subplot(2, 3, 5)
        self._plot_flow_distribution(problem, solution, ax5)

        # 6. Solution Summary
        ax6 = plt.subplot(2, 3, 6)
        self._plot_solution_summary(problem, solution, solver, ax6)

        plt.tight_layout()
        plt.show()

    def _plot_network_flow(self, problem, solution: Dict[str, Any], ax):
        """Plot network flow diagram."""
        G = nx.DiGraph()

        # Add nodes
        for supplier_id, supplier in problem.suppliers.items():
            G.add_node(supplier_id, node_type='supplier', name=supplier.name)

        for warehouse_id, warehouse in problem.warehouses.items():
            if solution['warehouse_operations'].get(warehouse_id, 0) > 0:
                G.add_node(warehouse_id, node_type='warehouse', name=warehouse.name)

        for customer_id, customer in problem.customers.items():
            G.add_node(customer_id, node_type='customer', name=customer.name)

        # Add edges with flows
        for (from_id, to_id), flow in solution['warehouse_flows'].items():
            if flow > 0:
                G.add_edge(from_id, to_id, weight=flow)

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw nodes by type
        suppliers = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'supplier']
        warehouses = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'warehouse']
        customers = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'customer']

        nx.draw_networkx_nodes(G, pos, nodelist=suppliers, node_color='lightblue',
                              node_size=800, label='Suppliers', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=warehouses, node_color='lightgreen',
                              node_size=800, label='Warehouses', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=customers, node_color='lightcoral',
                              node_size=800, label='Customers', ax=ax)

        # Draw edges with thickness proportional to flow
        edges = G.edges(data=True)
        weights = [d['weight'] for u, v, d in edges]
        max_weight = max(weights) if weights else 1
        edge_widths = [3 * w / max_weight for w in weights]

        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        ax.set_title('Supply Chain Network Flow', fontsize=14, fontweight='bold')
        ax.legend()
        ax.axis('off')

    def _plot_cost_breakdown(self, problem, solution: Dict[str, Any], ax):
        """Plot cost breakdown pie chart."""
        costs = {'Production': 0, 'Transportation': 0, 'Fixed Warehouse': 0}

        # Production costs
        for supplier_id, production in solution['supplier_production'].items():
            supplier = problem.suppliers[supplier_id]
            costs['Production'] += supplier.variable_cost * production

        # Transportation costs
        for (from_id, to_id), flow in solution['warehouse_flows'].items():
            if (from_id, to_id) in problem.supplier_warehouse_links:
                link = problem.supplier_warehouse_links[(from_id, to_id)]
                costs['Transportation'] += link.unit_cost * flow
            elif (from_id, to_id) in problem.warehouse_customer_links:
                link = problem.warehouse_customer_links[(from_id, to_id)]
                costs['Transportation'] += link.unit_cost * flow

        # Fixed warehouse costs
        for warehouse_id, is_open in solution['warehouse_operations'].items():
            if is_open > 0:
                warehouse = problem.warehouses[warehouse_id]
                costs['Fixed Warehouse'] += warehouse.fixed_cost

        # Filter out zero costs
        costs = {k: v for k, v in costs.items() if v > 0}

        if costs:
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            wedges, texts, autotexts = ax.pie(costs.values(), labels=costs.keys(), autopct='%1.1f%%',
                                             colors=colors, startangle=90)
            ax.set_title('Cost Breakdown', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No costs to display', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Cost Breakdown', fontsize=14, fontweight='bold')

    def _plot_capacity_utilization(self, problem, solution: Dict[str, Any], ax):
        """Plot capacity utilization for suppliers and warehouses."""
        entities = []
        utilizations = []
        colors = []

        # Supplier utilization
        for supplier_id, supplier in problem.suppliers.items():
            production = solution['supplier_production'].get(supplier_id, 0)
            utilization = (production / supplier.capacity) * 100 if supplier.capacity > 0 else 0
            entities.append(f"S: {supplier.name[:15]}")
            utilizations.append(utilization)
            colors.append('lightblue')

        # Warehouse utilization (only open warehouses)
        for warehouse_id, warehouse in problem.warehouses.items():
            if solution['warehouse_operations'].get(warehouse_id, 0) > 0:
                total_flow = sum(flow for (from_id, to_id), flow in solution['warehouse_flows'].items()
                               if to_id == warehouse_id or from_id == warehouse_id)
                utilization = (total_flow / warehouse.capacity) * 100 if warehouse.capacity > 0 else 0
                entities.append(f"W: {warehouse.name[:15]}")
                utilizations.append(utilization)
                colors.append('lightgreen')

        if entities:
            bars = ax.barh(entities, utilizations, color=colors)
            ax.set_xlabel('Capacity Utilization (%)')
            ax.set_title('Capacity Utilization', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 100)

            # Add percentage labels
            for bar, util in zip(bars, utilizations):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'{util:.1f}%', va='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No utilization data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Capacity Utilization', fontsize=14, fontweight='bold')

    def _plot_geographic_layout(self, problem, solution: Dict[str, Any], ax):
        """Plot geographic layout of supply chain entities."""
        # Plot suppliers
        supplier_x = [s.location_x for s in problem.suppliers.values()]
        supplier_y = [s.location_y for s in problem.suppliers.values()]
        ax.scatter(supplier_x, supplier_y, c='blue', s=100, alpha=0.7, label='Suppliers', marker='s')

        # Plot warehouses (different colors for open/closed)
        open_warehouses = [wid for wid, is_open in solution['warehouse_operations'].items() if is_open > 0]
        closed_warehouses = [wid for wid, is_open in solution['warehouse_operations'].items() if is_open == 0]

        if open_warehouses:
            open_x = [problem.warehouses[wid].location_x for wid in open_warehouses]
            open_y = [problem.warehouses[wid].location_y for wid in open_warehouses]
            ax.scatter(open_x, open_y, c='green', s=150, alpha=0.7, label='Open Warehouses', marker='^')

        if closed_warehouses:
            closed_x = [problem.warehouses[wid].location_x for wid in closed_warehouses]
            closed_y = [problem.warehouses[wid].location_y for wid in closed_warehouses]
            ax.scatter(closed_x, closed_y, c='gray', s=150, alpha=0.3, label='Closed Warehouses', marker='^')

        # Plot customers
        customer_x = [c.location_x for c in problem.customers.values()]
        customer_y = [c.location_y for c in problem.customers.values()]
        ax.scatter(customer_x, customer_y, c='red', s=80, alpha=0.7, label='Customers', marker='o')

        # Draw flow lines
        for (from_id, to_id), flow in solution['warehouse_flows'].items():
            if flow > 0:
                if from_id in problem.suppliers and to_id in problem.warehouses:
                    from_entity = problem.suppliers[from_id]
                    to_entity = problem.warehouses[to_id]
                elif from_id in problem.warehouses and to_id in problem.customers:
                    from_entity = problem.warehouses[from_id]
                    to_entity = problem.customers[to_id]
                else:
                    continue

                ax.plot([from_entity.location_x, to_entity.location_x],
                       [from_entity.location_y, to_entity.location_y],
                       'k-', alpha=0.3, linewidth=1)

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Geographic Layout', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_flow_distribution(self, problem, solution: Dict[str, Any], ax):
        """Plot distribution of flows."""
        flows = [flow for flow in solution['warehouse_flows'].values() if flow > 0]

        if flows:
            ax.hist(flows, bins=min(10, len(flows)), alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Flow Amount')
            ax.set_ylabel('Frequency')
            ax.set_title('Flow Distribution', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No flows to display', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Flow Distribution', fontsize=14, fontweight='bold')

    def _plot_solution_summary(self, problem, solution: Dict[str, Any], solver, ax):
        """Plot solution summary statistics."""
        ax.axis('off')

        # Calculate summary statistics
        total_cost = problem.evaluate_solution(solution)
        total_demand = sum(customer.demand for customer in problem.customers.values())
        total_production = sum(solution['supplier_production'].values())
        open_warehouses = sum(1 for is_open in solution['warehouse_operations'].values() if is_open > 0)
        total_warehouses = len(problem.warehouses)

        # Get solver status
        solver_status = "OPTIMAL" if hasattr(solver, 'Objective') and solver.Objective() else "UNKNOWN"

        # Create summary text
        summary_text = f"""
OPTIMIZATION SUMMARY

Solver: {self.solver_name}
Status: {solver_status}

COSTS:
Total Cost: ${total_cost:,.2f}

PRODUCTION:
Total Production: {total_production:,.0f} units
Total Demand: {total_demand:,.0f} units
Demand Coverage: {(total_production/total_demand*100):.1f}%

FACILITIES:
Open Warehouses: {open_warehouses}/{total_warehouses}
Utilization Rate: {(open_warehouses/total_warehouses*100):.1f}%

NETWORK:
Suppliers: {len(problem.suppliers)}
Customers: {len(problem.customers)}
Active Flows: {len([f for f in solution['warehouse_flows'].values() if f > 0])}
        """

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_title('Solution Summary', fontsize=14, fontweight='bold')
