"""
Heuristic Supply Chain Optimizer for Qubots Framework

This module implements a custom heuristic optimizer combining greedy construction
with local search improvement for supply chain network optimization problems.
Provides fast, high-quality solutions with comprehensive visualization.

Compatible with Rastion platform workflow automation and local development.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, Any, Optional, List, Tuple
import time
import random
from copy import deepcopy
from qubots import (
    BaseOptimizer, OptimizerMetadata, OptimizerType, OptimizerFamily,
    OptimizationResult, BaseProblem
)


class HeuristicSupplyChainOptimizer(BaseOptimizer):
    """
    Heuristic optimizer for supply chain network problems using greedy construction
    and local search improvement.

    This optimizer uses a multi-phase approach:
    1. Greedy Construction: Build initial solution using cost-based heuristics
    2. Local Search: Improve solution through neighborhood exploration
    3. Perturbation: Escape local optima through strategic modifications

    Features:
    - Fast execution for large-scale problems
    - High-quality solutions close to optimal
    - Adaptive parameter tuning during search
    - Comprehensive visualization of search progress
    - Multiple neighborhood operators for diversification
    """

    def __init__(self,
                 max_iterations: int = 1000,
                 local_search_iterations: int = 100,
                 perturbation_strength: float = 0.2,
                 improvement_threshold: float = 0.01,
                 random_seed: int = None,
                 **kwargs):
        """
        Initialize Heuristic Supply Chain Optimizer.

        Args:
            max_iterations: Maximum number of main iterations
            local_search_iterations: Iterations for local search phase
            perturbation_strength: Strength of perturbation (0.0 to 1.0)
            improvement_threshold: Minimum improvement to continue search
            random_seed: Random seed for reproducibility
            **kwargs: Additional parameters passed to BaseOptimizer
        """
        metadata = OptimizerMetadata(
            name="Heuristic Supply Chain Optimizer",
            description="Greedy construction with local search for supply chain optimization",
            optimizer_type=OptimizerType.HEURISTIC,
            optimizer_family=OptimizerFamily.LOCAL_SEARCH,
            author="Qubots Framework",
            version="1.0.0",
            is_deterministic=False,
            supports_constraints=True,
            supports_continuous=True,
            supports_discrete=True,
            supports_mixed_integer=True,
            time_complexity="O(iterations * local_search * problem_size)",
            space_complexity="O(problem_size)",
            convergence_guaranteed=False,
            parallel_capable=False
        )

        super().__init__(metadata, **kwargs)

        self.max_iterations = max_iterations
        self.local_search_iterations = local_search_iterations
        self.perturbation_strength = perturbation_strength
        self.improvement_threshold = improvement_threshold
        self.random_seed = random_seed

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def _get_default_metadata(self) -> OptimizerMetadata:
        """Return default metadata for the optimizer."""
        return self.metadata

    def _optimize_implementation(self, problem: BaseProblem,
                               initial_solution: Optional[Any] = None) -> OptimizationResult:
        """
        Core optimization implementation using heuristic approach.

        Args:
            problem: Supply chain network problem instance
            initial_solution: Optional initial solution to start from

        Returns:
            OptimizationResult with best solution found and search analytics
        """
        start_time = time.time()

        try:
            # Initialize tracking
            self.convergence_data = []
            self.iteration_times = []
            self.improvement_history = []

            # Phase 1: Greedy Construction
            if initial_solution is None:
                current_solution = self._greedy_construction(problem)
            else:
                current_solution = deepcopy(initial_solution)

            current_cost = problem.evaluate_solution(current_solution)
            best_solution = deepcopy(current_solution)
            best_cost = current_cost

            self.convergence_data.append(best_cost)

            # Phase 2: Iterative Improvement
            no_improvement_count = 0

            for iteration in range(self.max_iterations):
                iter_start = time.time()

                # Local search improvement
                improved_solution, improved_cost = self._local_search(
                    problem, current_solution, current_cost
                )

                # Accept improvement
                if improved_cost < current_cost:
                    improvement = (current_cost - improved_cost) / current_cost
                    current_solution = improved_solution
                    current_cost = improved_cost
                    no_improvement_count = 0

                    # Update best solution
                    if improved_cost < best_cost:
                        best_solution = deepcopy(improved_solution)
                        best_cost = improved_cost

                    self.improvement_history.append(improvement)
                else:
                    no_improvement_count += 1
                    self.improvement_history.append(0.0)

                self.convergence_data.append(best_cost)
                self.iteration_times.append(time.time() - iter_start)

                # Perturbation if stuck in local optimum
                if no_improvement_count >= 20:
                    current_solution = self._perturbation(problem, current_solution)
                    current_cost = problem.evaluate_solution(current_solution)
                    no_improvement_count = 0

                # Early termination if improvement is too small
                if (len(self.improvement_history) >= 10 and
                    max(self.improvement_history[-10:]) < self.improvement_threshold):
                    break

            runtime = time.time() - start_time

            # Generate visualizations
            self._create_visualizations(problem, best_solution, self.convergence_data)

            # Create optimization history
            optimization_history = []
            for i, cost in enumerate(self.convergence_data):
                optimization_history.append({
                    "iteration": i,
                    "objective_value": cost,
                    "improvement": self.improvement_history[i] if i < len(self.improvement_history) else 0.0
                })

            # Create result
            result = OptimizationResult(
                best_solution=best_solution,
                best_value=best_cost,
                runtime_seconds=runtime,
                iterations=len(self.convergence_data),
                evaluations=len(self.convergence_data) * self.local_search_iterations,
                termination_reason="convergence" if len(self.improvement_history) >= 10 and max(self.improvement_history[-10:]) < self.improvement_threshold else "max_iterations",
                optimization_history=optimization_history,
                additional_metrics={
                    "algorithm": "Greedy + Local Search",
                    "final_iteration": len(self.convergence_data),
                    "total_improvements": sum(1 for imp in self.improvement_history if imp > 0),
                    "average_iteration_time": np.mean(self.iteration_times) if self.iteration_times else 0,
                    "best_improvement": max(self.improvement_history) if self.improvement_history else 0,
                    "convergence_rate": self._calculate_convergence_rate()
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
                additional_metrics={"error": str(e), "algorithm": "Greedy + Local Search"}
            )

    def _greedy_construction(self, problem) -> Dict[str, Any]:
        """Construct initial solution using greedy heuristics."""
        solution = {
            'supplier_production': {},
            'warehouse_flows': {},
            'warehouse_operations': {}
        }

        # Step 1: Select warehouses to open based on cost-effectiveness
        warehouse_scores = []
        for warehouse_id, warehouse in problem.warehouses.items():
            # Calculate average distance to customers
            avg_distance = np.mean([
                problem.warehouse_customer_links[(warehouse_id, customer_id)].unit_cost
                for customer_id in problem.customers
                if (warehouse_id, customer_id) in problem.warehouse_customer_links
            ])

            # Score based on capacity per cost and location advantage
            score = warehouse.capacity / (warehouse.fixed_cost + avg_distance * 100)
            warehouse_scores.append((warehouse_id, score))

        # Sort by score and select top warehouses
        warehouse_scores.sort(key=lambda x: x[1], reverse=True)

        # Open warehouses greedily until demand can be satisfied
        total_capacity_needed = sum(customer.demand for customer in problem.customers.values())
        opened_capacity = 0

        for warehouse_id, score in warehouse_scores:
            solution['warehouse_operations'][warehouse_id] = 1
            opened_capacity += problem.warehouses[warehouse_id].capacity

            if opened_capacity >= total_capacity_needed * 1.2:  # 20% buffer
                break

        # Close remaining warehouses
        for warehouse_id in problem.warehouses:
            if warehouse_id not in solution['warehouse_operations']:
                solution['warehouse_operations'][warehouse_id] = 0

        # Step 2: Assign customers to warehouses (minimum cost assignment)
        open_warehouses = [wid for wid, is_open in solution['warehouse_operations'].items() if is_open]

        for customer_id, customer in problem.customers.items():
            # Find cheapest open warehouse for this customer
            best_warehouse = None
            best_cost = float('inf')

            for warehouse_id in open_warehouses:
                if (warehouse_id, customer_id) in problem.warehouse_customer_links:
                    cost = problem.warehouse_customer_links[(warehouse_id, customer_id)].unit_cost
                    if cost < best_cost:
                        best_cost = cost
                        best_warehouse = warehouse_id

            if best_warehouse:
                solution['warehouse_flows'][(best_warehouse, customer_id)] = customer.demand

        # Step 3: Assign suppliers to warehouses (minimum cost assignment)
        for warehouse_id in open_warehouses:
            # Calculate total demand for this warehouse
            warehouse_demand = sum(
                flow for (from_id, to_id), flow in solution['warehouse_flows'].items()
                if from_id == warehouse_id
            )

            if warehouse_demand > 0:
                # Find cheapest supplier for this warehouse
                best_supplier = None
                best_cost = float('inf')

                for supplier_id in problem.suppliers:
                    if (supplier_id, warehouse_id) in problem.supplier_warehouse_links:
                        cost = problem.supplier_warehouse_links[(supplier_id, warehouse_id)].unit_cost
                        if cost < best_cost:
                            best_cost = cost
                            best_supplier = supplier_id

                if best_supplier:
                    solution['warehouse_flows'][(best_supplier, warehouse_id)] = warehouse_demand

        # Step 4: Set supplier production levels
        for supplier_id in problem.suppliers:
            total_supply = sum(
                flow for (from_id, to_id), flow in solution['warehouse_flows'].items()
                if from_id == supplier_id
            )
            solution['supplier_production'][supplier_id] = min(
                total_supply, problem.suppliers[supplier_id].capacity
            )

        return solution

    def _local_search(self, problem, solution: Dict[str, Any], current_cost: float) -> Tuple[Dict[str, Any], float]:
        """Perform local search to improve the current solution."""
        best_solution = deepcopy(solution)
        best_cost = current_cost

        for _ in range(self.local_search_iterations):
            # Try different neighborhood operators
            operators = [
                self._swap_warehouse_assignment,
                self._reassign_customer,
                self._adjust_supplier_allocation,
                self._toggle_warehouse
            ]

            operator = random.choice(operators)
            neighbor_solution = operator(problem, best_solution)

            if neighbor_solution:
                neighbor_cost = problem.evaluate_solution(neighbor_solution)
                if neighbor_cost < best_cost:
                    best_solution = neighbor_solution
                    best_cost = neighbor_cost

        return best_solution, best_cost

    def _swap_warehouse_assignment(self, problem, solution: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Swap warehouse assignments for customers."""
        new_solution = deepcopy(solution)

        # Get customers with flows
        customer_flows = [(from_id, to_id, flow) for (from_id, to_id), flow in solution['warehouse_flows'].items()
                         if to_id in problem.customers]

        if len(customer_flows) < 2:
            return None

        # Select two random customer assignments
        flow1, flow2 = random.sample(customer_flows, 2)
        warehouse1, customer1, demand1 = flow1
        warehouse2, customer2, demand2 = flow2

        # Check if swap is possible
        if ((warehouse2, customer1) in problem.warehouse_customer_links and
            (warehouse1, customer2) in problem.warehouse_customer_links):

            # Remove old flows
            del new_solution['warehouse_flows'][(warehouse1, customer1)]
            del new_solution['warehouse_flows'][(warehouse2, customer2)]

            # Add new flows
            new_solution['warehouse_flows'][(warehouse2, customer1)] = demand1
            new_solution['warehouse_flows'][(warehouse1, customer2)] = demand2

            return new_solution

        return None

    def _reassign_customer(self, problem, solution: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Reassign a customer to a different warehouse."""
        new_solution = deepcopy(solution)

        # Get open warehouses
        open_warehouses = [wid for wid, is_open in solution['warehouse_operations'].items() if is_open]

        if len(open_warehouses) < 2:
            return None

        # Select random customer with current assignment
        customer_flows = [(from_id, to_id, flow) for (from_id, to_id), flow in solution['warehouse_flows'].items()
                         if to_id in problem.customers]

        if not customer_flows:
            return None

        current_warehouse, customer_id, demand = random.choice(customer_flows)

        # Find alternative warehouses
        alternative_warehouses = [wid for wid in open_warehouses
                                if wid != current_warehouse and
                                (wid, customer_id) in problem.warehouse_customer_links]

        if not alternative_warehouses:
            return None

        new_warehouse = random.choice(alternative_warehouses)

        # Update flows
        del new_solution['warehouse_flows'][(current_warehouse, customer_id)]
        new_solution['warehouse_flows'][(new_warehouse, customer_id)] = demand

        return new_solution

    def _adjust_supplier_allocation(self, problem, solution: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Adjust supplier allocation to warehouses."""
        new_solution = deepcopy(solution)

        # Get supplier-warehouse flows
        supplier_flows = [(from_id, to_id, flow) for (from_id, to_id), flow in solution['warehouse_flows'].items()
                         if from_id in problem.suppliers]

        if not supplier_flows:
            return None

        supplier_id, warehouse_id, current_flow = random.choice(supplier_flows)

        # Find alternative suppliers for this warehouse
        alternative_suppliers = [sid for sid in problem.suppliers
                               if sid != supplier_id and
                               (sid, warehouse_id) in problem.supplier_warehouse_links]

        if not alternative_suppliers:
            return None

        new_supplier = random.choice(alternative_suppliers)

        # Check capacity constraints
        if problem.suppliers[new_supplier].capacity >= current_flow:
            # Update flows
            del new_solution['warehouse_flows'][(supplier_id, warehouse_id)]
            new_solution['warehouse_flows'][(new_supplier, warehouse_id)] = current_flow

            # Update production
            new_solution['supplier_production'][supplier_id] -= current_flow
            new_solution['supplier_production'][new_supplier] = new_solution['supplier_production'].get(new_supplier, 0) + current_flow

            return new_solution

        return None

    def _toggle_warehouse(self, problem, solution: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Toggle warehouse open/close status."""
        new_solution = deepcopy(solution)

        # Select random warehouse
        warehouse_id = random.choice(list(problem.warehouses.keys()))
        current_status = solution['warehouse_operations'].get(warehouse_id, 0)

        if current_status == 0:
            # Try to open warehouse
            new_solution['warehouse_operations'][warehouse_id] = 1
        else:
            # Try to close warehouse (if it has no critical flows)
            warehouse_flows = [(from_id, to_id, flow) for (from_id, to_id), flow in solution['warehouse_flows'].items()
                             if from_id == warehouse_id or to_id == warehouse_id]

            if not warehouse_flows:  # Only close if no flows
                new_solution['warehouse_operations'][warehouse_id] = 0
            else:
                return None

        return new_solution

    def _perturbation(self, problem, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Apply perturbation to escape local optima."""
        new_solution = deepcopy(solution)

        # Number of perturbations based on strength
        num_perturbations = max(1, int(self.perturbation_strength * len(problem.warehouses)))

        for _ in range(num_perturbations):
            # Random perturbation operators
            operators = [
                self._random_warehouse_toggle,
                self._random_customer_reassignment,
                self._random_supplier_change
            ]

            operator = random.choice(operators)
            perturbed = operator(problem, new_solution)
            if perturbed:
                new_solution = perturbed

        return new_solution

    def _random_warehouse_toggle(self, problem, solution: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Randomly toggle warehouse status."""
        new_solution = deepcopy(solution)
        warehouse_id = random.choice(list(problem.warehouses.keys()))
        current_status = solution['warehouse_operations'].get(warehouse_id, 0)
        new_solution['warehouse_operations'][warehouse_id] = 1 - current_status
        return new_solution

    def _random_customer_reassignment(self, problem, solution: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Randomly reassign customers to warehouses."""
        return self._reassign_customer(problem, solution)

    def _random_supplier_change(self, problem, solution: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Randomly change supplier assignments."""
        return self._adjust_supplier_allocation(problem, solution)

    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate based on improvement history."""
        if len(self.convergence_data) < 2:
            return 0.0

        initial_cost = self.convergence_data[0]
        final_cost = self.convergence_data[-1]

        if initial_cost == 0:
            return 0.0

        return (initial_cost - final_cost) / initial_cost

    def _create_visualizations(self, problem, solution: Dict[str, Any], convergence_data: List[float]):
        """Create comprehensive visualizations of the optimization results."""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))

        # 1. Convergence Plot
        ax1 = plt.subplot(2, 3, 1)
        self._plot_convergence(convergence_data, ax1)

        # 2. Network Visualization
        ax2 = plt.subplot(2, 3, 2)
        self._plot_network(problem, solution, ax2)

        # 3. Cost Analysis
        ax3 = plt.subplot(2, 3, 3)
        self._plot_cost_analysis(problem, solution, ax3)

        # 4. Improvement History
        ax4 = plt.subplot(2, 3, 4)
        self._plot_improvement_history(ax4)

        # 5. Warehouse Utilization
        ax5 = plt.subplot(2, 3, 5)
        self._plot_warehouse_utilization(problem, solution, ax5)

        # 6. Algorithm Performance
        ax6 = plt.subplot(2, 3, 6)
        self._plot_algorithm_performance(ax6)

        plt.tight_layout()
        plt.show()

    def _plot_convergence(self, convergence_data: List[float], ax):
        """Plot convergence curve."""
        if convergence_data:
            ax.plot(convergence_data, 'b-', linewidth=2, alpha=0.8)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Best Cost')
            ax.set_title('Convergence Curve', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add improvement annotations
            if len(convergence_data) > 1:
                improvement = (convergence_data[0] - convergence_data[-1]) / convergence_data[0] * 100
                ax.text(0.02, 0.98, f'Improvement: {improvement:.1f}%',
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No convergence data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Convergence Curve', fontsize=14, fontweight='bold')

    def _plot_network(self, problem, solution: Dict[str, Any], ax):
        """Plot supply chain network."""
        # Create network graph
        G = nx.DiGraph()

        # Add nodes
        for supplier_id in problem.suppliers:
            G.add_node(supplier_id, node_type='supplier')

        for warehouse_id, is_open in solution['warehouse_operations'].items():
            if is_open > 0:
                G.add_node(warehouse_id, node_type='warehouse')

        for customer_id in problem.customers:
            G.add_node(customer_id, node_type='customer')

        # Add edges
        for (from_id, to_id), flow in solution['warehouse_flows'].items():
            if flow > 0:
                G.add_edge(from_id, to_id, weight=flow)

        if G.nodes():
            # Layout
            pos = nx.spring_layout(G, k=2, iterations=50)

            # Draw nodes by type
            suppliers = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'supplier']
            warehouses = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'warehouse']
            customers = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'customer']

            if suppliers:
                nx.draw_networkx_nodes(G, pos, nodelist=suppliers, node_color='lightblue',
                                      node_size=600, label='Suppliers', ax=ax)
            if warehouses:
                nx.draw_networkx_nodes(G, pos, nodelist=warehouses, node_color='lightgreen',
                                      node_size=800, label='Warehouses', ax=ax)
            if customers:
                nx.draw_networkx_nodes(G, pos, nodelist=customers, node_color='lightcoral',
                                      node_size=500, label='Customers', ax=ax)

            # Draw edges
            if G.edges():
                nx.draw_networkx_edges(G, pos, alpha=0.6, ax=ax)

            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
            ax.legend()

        ax.set_title('Supply Chain Network', fontsize=14, fontweight='bold')
        ax.axis('off')

    def _plot_cost_analysis(self, problem, solution: Dict[str, Any], ax):
        """Plot cost breakdown analysis."""
        costs = {'Production': 0, 'Transportation': 0, 'Fixed Warehouse': 0}

        # Production costs
        for supplier_id, production in solution['supplier_production'].items():
            if supplier_id in problem.suppliers:
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
            if is_open > 0 and warehouse_id in problem.warehouses:
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

    def _plot_improvement_history(self, ax):
        """Plot improvement history over iterations."""
        if self.improvement_history:
            # Plot improvement percentages
            improvements = [imp * 100 for imp in self.improvement_history if imp > 0]
            if improvements:
                ax.bar(range(len(improvements)), improvements, alpha=0.7, color='green')
                ax.set_xlabel('Improvement Event')
                ax.set_ylabel('Improvement (%)')
                ax.set_title('Improvement History', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No improvements recorded', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Improvement History', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No improvement data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Improvement History', fontsize=14, fontweight='bold')

    def _plot_warehouse_utilization(self, problem, solution: Dict[str, Any], ax):
        """Plot warehouse utilization rates."""
        warehouses = []
        utilizations = []
        colors = []

        for warehouse_id, warehouse in problem.warehouses.items():
            is_open = solution['warehouse_operations'].get(warehouse_id, 0)

            if is_open > 0:
                # Calculate utilization
                total_flow = sum(flow for (from_id, to_id), flow in solution['warehouse_flows'].items()
                               if from_id == warehouse_id or to_id == warehouse_id)
                utilization = (total_flow / warehouse.capacity) * 100 if warehouse.capacity > 0 else 0
                warehouses.append(warehouse.name[:15])
                utilizations.append(utilization)
                colors.append('lightgreen')
            else:
                warehouses.append(warehouse.name[:15])
                utilizations.append(0)
                colors.append('lightgray')

        if warehouses:
            bars = ax.barh(warehouses, utilizations, color=colors)
            ax.set_xlabel('Utilization (%)')
            ax.set_title('Warehouse Utilization', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 100)

            # Add percentage labels
            for bar, util in zip(bars, utilizations):
                if util > 0:
                    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                           f'{util:.1f}%', va='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No warehouse data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Warehouse Utilization', fontsize=14, fontweight='bold')

    def _plot_algorithm_performance(self, ax):
        """Plot algorithm performance metrics."""
        ax.axis('off')

        # Calculate performance metrics
        total_iterations = len(self.convergence_data)
        total_improvements = sum(1 for imp in self.improvement_history if imp > 0)
        avg_iteration_time = np.mean(self.iteration_times) if self.iteration_times else 0
        convergence_rate = self._calculate_convergence_rate()

        # Create performance summary
        performance_text = f"""
ALGORITHM PERFORMANCE

Method: Greedy + Local Search
Total Iterations: {total_iterations}
Improvements Found: {total_improvements}
Improvement Rate: {(total_improvements/total_iterations*100):.1f}%

Timing:
Avg Iteration Time: {avg_iteration_time:.4f}s
Total Runtime: {sum(self.iteration_times):.2f}s

Convergence:
Convergence Rate: {convergence_rate:.1%}
Final Cost Reduction: {convergence_rate:.1%}

Parameters:
Max Iterations: {self.max_iterations}
Local Search Iters: {self.local_search_iterations}
Perturbation Strength: {self.perturbation_strength}
        """

        ax.text(0.05, 0.95, performance_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_title('Algorithm Performance', fontsize=14, fontweight='bold')