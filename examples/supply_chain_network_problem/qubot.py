"""
Supply Chain Network Optimization Problem for Qubots Framework

This module implements a comprehensive supply chain network optimization problem
that models multi-echelon supply chains with suppliers, warehouses, and customers.
The problem minimizes total cost while satisfying demand and capacity constraints.

Compatible with Rastion platform workflow automation and local development.
"""

import pandas as pd
import numpy as np
import io
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from qubots import (
    BaseProblem, ProblemMetadata, ProblemType, ObjectiveType,
    DifficultyLevel, EvaluationResult
)


@dataclass
class SupplyChainEntity:
    """Represents an entity in the supply chain network."""
    entity_id: str
    name: str
    location_x: float
    location_y: float
    capacity: float
    fixed_cost: float
    variable_cost: float
    demand: float = 0.0


@dataclass
class SupplyChainLink:
    """Represents a connection between entities in the supply chain."""
    from_entity: str
    to_entity: str
    unit_cost: float


class SupplyChainNetworkProblem(BaseProblem):
    """
    Supply Chain Network Optimization Problem for modular qubots architecture.

    This problem models a multi-echelon supply chain network with:
    - Suppliers with production capacities and costs
    - Warehouses with storage capacities and operating costs
    - Customers with demand requirements
    - Transportation costs between entities

    Solution Format:
        Dictionary with keys:
        - 'supplier_production': Dict[supplier_id, production_amount]
        - 'warehouse_flows': Dict[(from_id, to_id), flow_amount]
        - 'warehouse_operations': Dict[warehouse_id, is_open (0/1)]

    Objective: Minimize total cost (production + transportation + fixed costs)
    """

    def __init__(self,
                 csv_data: str = None,
                 dataset_content: str = None,
                 **kwargs):
        """
        Initialize Supply Chain Network Problem.

        Args:
            csv_data: CSV data as string (for custom data)
            dataset_content: Pre-loaded dataset content from platform
            **kwargs: Additional parameters passed to BaseProblem
        """
        # Create metadata
        metadata = ProblemMetadata(
            name="Supply Chain Network Optimization",
            description="Multi-echelon supply chain network optimization with cost minimization",
            problem_type=ProblemType.MIXED_INTEGER,
            objective_type=ObjectiveType.MINIMIZE,
            difficulty_level=DifficultyLevel.ADVANCED,
            domain="supply_chain",
            author="Qubots Framework",
            version="1.0.0"
        )

        super().__init__(metadata, **kwargs)

        # Load and parse data
        self._load_supply_chain_data(csv_data or dataset_content)
        
        # Calculate derived properties
        self._calculate_network_properties()

    def _get_default_metadata(self) -> ProblemMetadata:
        """Return default metadata for the problem."""
        return self.metadata

    def _load_supply_chain_data(self, data: str):
        """Load and parse supply chain data from CSV."""
        if data is None:
            # Use default embedded data
            data = self._get_default_data()

        # Parse CSV data
        df = pd.read_csv(io.StringIO(data))

        # Check if data was loaded
        if df.empty:
            print(f"Warning: No data loaded from CSV. Data length: {len(data) if data else 0}")
            return
        
        # Initialize collections
        self.suppliers = {}
        self.warehouses = {}
        self.customers = {}
        self.supplier_warehouse_links = {}
        self.warehouse_customer_links = {}

        # Parse entities
        for idx, row in df.iterrows():
            entity_type = row['entity_type']
            entity_id = row['entity_id']

            if entity_type == 'supplier':
                self.suppliers[entity_id] = SupplyChainEntity(
                    entity_id=entity_id,
                    name=row['name'],
                    location_x=float(row['location_x']),
                    location_y=float(row['location_y']),
                    capacity=float(row['capacity']),
                    fixed_cost=float(row['fixed_cost']),
                    variable_cost=float(row['variable_cost'])
                )
            elif entity_type == 'warehouse':
                self.warehouses[entity_id] = SupplyChainEntity(
                    entity_id=entity_id,
                    name=row['name'],
                    location_x=float(row['location_x']),
                    location_y=float(row['location_y']),
                    capacity=float(row['capacity']),
                    fixed_cost=float(row['fixed_cost']),
                    variable_cost=float(row['variable_cost'])
                )
            elif entity_type == 'customer':
                self.customers[entity_id] = SupplyChainEntity(
                    entity_id=entity_id,
                    name=row['name'],
                    location_x=float(row['location_x']),
                    location_y=float(row['location_y']),
                    capacity=0,
                    fixed_cost=0,
                    variable_cost=0,
                    demand=float(row['demand']) if row['demand'] and row['demand'] != '' else 0.0
                )
            elif entity_type == 'supplier_warehouse':
                # For links: supplier_id, warehouse_id, and unit_cost are in their respective columns
                supplier_id = row['supplier_id']
                warehouse_id = row['warehouse_id']
                unit_cost = float(row['unit_cost']) if row['unit_cost'] and row['unit_cost'] != '' else 0.0
                key = (supplier_id, warehouse_id)
                self.supplier_warehouse_links[key] = SupplyChainLink(
                    from_entity=supplier_id,
                    to_entity=warehouse_id,
                    unit_cost=unit_cost
                )
            elif entity_type == 'warehouse_customer':
                # For warehouse_customer links: warehouse_id is in supplier_id column, customer_id is in customer_id column
                warehouse_id = row['supplier_id']  # warehouse_id is actually in the supplier_id column
                customer_id = row['customer_id']
                unit_cost = float(row['unit_cost']) if row['unit_cost'] and row['unit_cost'] != '' else 0.0
                key = (warehouse_id, customer_id)
                self.warehouse_customer_links[key] = SupplyChainLink(
                    from_entity=warehouse_id,
                    to_entity=customer_id,
                    unit_cost=unit_cost
                )

    def _calculate_network_properties(self):
        """Calculate derived network properties."""
        self.total_demand = sum(customer.demand for customer in self.customers.values())
        self.total_supplier_capacity = sum(supplier.capacity for supplier in self.suppliers.values())
        self.total_warehouse_capacity = sum(warehouse.capacity for warehouse in self.warehouses.values())
        
        # Update metadata with problem dimensions
        self.metadata.dimension = (len(self.suppliers) + len(self.warehouses) + 
                                 len(self.supplier_warehouse_links) + len(self.warehouse_customer_links))

    def _get_default_data(self) -> str:
        """Return default supply chain data."""
        return """entity_type,entity_id,name,location_x,location_y,capacity,fixed_cost,variable_cost,demand,supplier_id,warehouse_id,customer_id,unit_cost
supplier,S1,Global Materials Inc,10,50,5000,50000,12.5,0,,,,,
supplier,S2,Regional Supply Co,30,80,3000,35000,15.2,0,,,,,
warehouse,W1,Central Distribution,40,45,4000,80000,8.5,0,,,,,
warehouse,W2,North Hub,35,75,3500,70000,9.2,0,,,,,
customer,C1,Metro Retail Chain,25,65,0,0,0,850,,,,,
customer,C2,Downtown Stores,45,70,0,0,0,620,,,,,
customer,C3,Suburban Markets,55,50,0,0,0,480,,,,,
supplier_warehouse,SW1,S1-W1 Link,,,,,,,S1,W1,,15.5
supplier_warehouse,SW2,S1-W2 Link,,,,,,,S1,W2,,18.2
supplier_warehouse,SW3,S2-W1 Link,,,,,,,S2,W1,,19.8
supplier_warehouse,SW4,S2-W2 Link,,,,,,,S2,W2,,14.6
warehouse_customer,WC1,W1-C1 Link,,,,,,,W1,,C1,12.8
warehouse_customer,WC2,W1-C2 Link,,,,,,,W1,,C2,8.5
warehouse_customer,WC3,W1-C3 Link,,,,,,,W1,,C3,14.2
warehouse_customer,WC4,W2-C1 Link,,,,,,,W2,,C1,9.2
warehouse_customer,WC5,W2-C2 Link,,,,,,,W2,,C2,11.8
warehouse_customer,WC6,W2-C3 Link,,,,,,,W2,,C3,17.5"""

    def evaluate_solution(self, solution: Dict[str, Any]) -> float:
        """
        Evaluate supply chain network solution.

        Args:
            solution: Dictionary containing:
                - 'supplier_production': Dict[supplier_id, production_amount]
                - 'warehouse_flows': Dict[(from_id, to_id), flow_amount]
                - 'warehouse_operations': Dict[warehouse_id, is_open (0/1)]

        Returns:
            Total cost with penalties for constraint violations
        """
        try:
            supplier_production = solution.get('supplier_production', {})
            warehouse_flows = solution.get('warehouse_flows', {})
            warehouse_operations = solution.get('warehouse_operations', {})

            total_cost = 0.0
            penalty = 0.0

            # Production costs
            for supplier_id, production in supplier_production.items():
                if supplier_id in self.suppliers:
                    supplier = self.suppliers[supplier_id]
                    total_cost += supplier.variable_cost * production

            # Warehouse fixed costs
            for warehouse_id, is_open in warehouse_operations.items():
                if warehouse_id in self.warehouses and is_open > 0:
                    warehouse = self.warehouses[warehouse_id]
                    total_cost += warehouse.fixed_cost

            # Transportation costs
            for (from_id, to_id), flow in warehouse_flows.items():
                if (from_id, to_id) in self.supplier_warehouse_links:
                    link = self.supplier_warehouse_links[(from_id, to_id)]
                    total_cost += link.unit_cost * flow
                elif (from_id, to_id) in self.warehouse_customer_links:
                    link = self.warehouse_customer_links[(from_id, to_id)]
                    total_cost += link.unit_cost * flow

            # Constraint penalties
            penalty += self._check_capacity_constraints(supplier_production, warehouse_flows, warehouse_operations)
            penalty += self._check_demand_constraints(warehouse_flows)
            penalty += self._check_flow_balance_constraints(supplier_production, warehouse_flows, warehouse_operations)

            return total_cost + penalty

        except Exception as e:
            return 1e6  # Large penalty for invalid solutions

    def _check_capacity_constraints(self, supplier_production, warehouse_flows, warehouse_operations) -> float:
        """Check capacity constraints and return penalty."""
        penalty = 0.0

        # Supplier capacity constraints
        for supplier_id, production in supplier_production.items():
            if supplier_id in self.suppliers:
                capacity = self.suppliers[supplier_id].capacity
                if production > capacity:
                    penalty += 1000 * (production - capacity)

        # Warehouse capacity constraints
        for warehouse_id, is_open in warehouse_operations.items():
            if warehouse_id in self.warehouses and is_open > 0:
                warehouse = self.warehouses[warehouse_id]
                # Only check inflow (throughput) against capacity
                inflow = sum(flow for (from_id, to_id), flow in warehouse_flows.items()
                           if to_id == warehouse_id)
                if inflow > warehouse.capacity:
                    penalty += 1000 * (inflow - warehouse.capacity)

        return penalty

    def _check_demand_constraints(self, warehouse_flows) -> float:
        """Check demand satisfaction constraints and return penalty."""
        penalty = 0.0

        for customer_id, customer in self.customers.items():
            total_delivered = sum(flow for (from_id, to_id), flow in warehouse_flows.items() 
                                if to_id == customer_id)
            if total_delivered < customer.demand:
                penalty += 1000 * (customer.demand - total_delivered)

        return penalty

    def _check_flow_balance_constraints(self, supplier_production, warehouse_flows, warehouse_operations) -> float:
        """Check flow balance constraints and return penalty."""
        penalty = 0.0

        # Warehouse flow balance
        for warehouse_id in self.warehouses:
            if warehouse_operations.get(warehouse_id, 0) > 0:
                inflow = sum(flow for (from_id, to_id), flow in warehouse_flows.items() 
                           if to_id == warehouse_id)
                outflow = sum(flow for (from_id, to_id), flow in warehouse_flows.items() 
                            if from_id == warehouse_id)
                if abs(inflow - outflow) > 1e-6:
                    penalty += 1000 * abs(inflow - outflow)

        return penalty

    def random_solution(self) -> Dict[str, Any]:
        """Generate a random feasible solution."""
        solution = {
            'supplier_production': {},
            'warehouse_flows': {},
            'warehouse_operations': {}
        }

        # Random warehouse operations (50% chance each warehouse is open)
        for warehouse_id in self.warehouses:
            solution['warehouse_operations'][warehouse_id] = np.random.choice([0, 1])

        # Random supplier production (within capacity)
        for supplier_id, supplier in self.suppliers.items():
            solution['supplier_production'][supplier_id] = np.random.uniform(0, supplier.capacity)

        # Random flows (simplified - just satisfy basic demand)
        total_demand = self.total_demand
        open_warehouses = [wid for wid, is_open in solution['warehouse_operations'].items() if is_open]
        
        if open_warehouses:
            # Distribute demand among open warehouses
            for customer_id, customer in self.customers.items():
                warehouse_id = np.random.choice(open_warehouses)
                if (warehouse_id, customer_id) in self.warehouse_customer_links:
                    solution['warehouse_flows'][(warehouse_id, customer_id)] = customer.demand

            # Balance warehouse flows
            for warehouse_id in open_warehouses:
                outflow = sum(flow for (from_id, to_id), flow in solution['warehouse_flows'].items() 
                            if from_id == warehouse_id)
                if outflow > 0:
                    # Find a supplier to provide this flow
                    available_suppliers = [sid for sid in self.suppliers 
                                         if (sid, warehouse_id) in self.supplier_warehouse_links]
                    if available_suppliers:
                        supplier_id = np.random.choice(available_suppliers)
                        solution['warehouse_flows'][(supplier_id, warehouse_id)] = outflow

        return solution

    def is_feasible(self, solution: Dict[str, Any]) -> bool:
        """Check if solution is feasible."""
        try:
            penalty = (self._check_capacity_constraints(
                solution.get('supplier_production', {}),
                solution.get('warehouse_flows', {}),
                solution.get('warehouse_operations', {})
            ) + self._check_demand_constraints(solution.get('warehouse_flows', {})) +
            self._check_flow_balance_constraints(
                solution.get('supplier_production', {}),
                solution.get('warehouse_flows', {}),
                solution.get('warehouse_operations', {})
            ))
            return penalty < 1e-6
        except:
            return False

    def get_solution_summary(self, solution: Dict[str, Any]) -> str:
        """Get a human-readable summary of the solution."""
        try:
            supplier_production = solution.get('supplier_production', {})
            warehouse_flows = solution.get('warehouse_flows', {})
            warehouse_operations = solution.get('warehouse_operations', {})

            summary = []
            summary.append("=== Supply Chain Network Solution ===")
            
            # Supplier production
            summary.append("\nSupplier Production:")
            for supplier_id, production in supplier_production.items():
                if production > 0:
                    supplier_name = self.suppliers[supplier_id].name
                    summary.append(f"  {supplier_name}: {production:.1f} units")

            # Open warehouses
            summary.append("\nOpen Warehouses:")
            for warehouse_id, is_open in warehouse_operations.items():
                if is_open > 0:
                    warehouse_name = self.warehouses[warehouse_id].name
                    summary.append(f"  {warehouse_name}")

            # Customer deliveries
            summary.append("\nCustomer Deliveries:")
            for (from_id, to_id), flow in warehouse_flows.items():
                if to_id in self.customers and flow > 0:
                    customer_name = self.customers[to_id].name
                    warehouse_name = self.warehouses[from_id].name
                    summary.append(f"  {customer_name}: {flow:.1f} units from {warehouse_name}")

            return "\n".join(summary)

        except Exception as e:
            return f"Error generating summary: {str(e)}"
