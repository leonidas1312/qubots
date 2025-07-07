# Molecular Conformation Optimizer

A specialized simulated annealing optimizer designed for molecular conformation problems. This optimizer uses chemistry-aware moves and adaptive parameters to efficiently explore conformational energy landscapes.

## 🧬 Overview

This optimizer is specifically designed for finding low-energy molecular conformations by optimizing dihedral angles. It incorporates chemical knowledge into the optimization process through specialized move types and temperature schedules.

### Key Features
- **Chemistry-Aware Moves**: Three types of moves tailored for molecular systems
- **Adaptive Parameters**: Step size and temperature adapt based on acceptance rates
- **Multiple Move Types**: Single angle, coupled angles, and random walk moves
- **Convergence Detection**: Energy-based stopping criteria
- **Detailed Reporting**: Comprehensive optimization statistics and molecular information

## 🔬 Algorithm Details

### Simulated Annealing for Molecules

The optimizer uses a modified simulated annealing approach with:

1. **Temperature Schedule**: Exponential cooling with adaptive rates
2. **Move Generation**: Chemistry-informed perturbations of dihedral angles
3. **Acceptance Criterion**: Metropolis criterion with energy-based probability
4. **Convergence**: Energy variance monitoring over sliding windows

### Move Types

1. **Single Angle (60% probability)**
   - Perturbs one dihedral angle at a time
   - Most common move for local optimization

2. **Coupled Angles (30% probability)**
   - Simultaneously modifies two adjacent dihedral angles
   - Mimics correlated motions in real molecules

3. **Random Walk (10% probability)**
   - Small perturbations to all dihedral angles
   - Helps escape local minima

## 🚀 Usage

### Basic Usage

```python
from qubots import AutoProblem, AutoOptimizer

# Load molecular problem
problem = AutoProblem.from_repo("examples/molecular_conformation_problem")

# Load optimizer with default settings
optimizer = AutoOptimizer.from_repo("examples/molecular_conformation_optimizer")

# Run optimization
result = optimizer.optimize(problem)

print(f"Best energy: {result.best_value:.3f} kcal/mol")
print(f"Optimization time: {result.runtime_seconds:.2f} seconds")
print(f"Acceptance rate: {result.additional_metrics['acceptance_rate']:.2%}")
```

### Custom Parameters

```python
from qubots import AutoOptimizer

# High-temperature exploration
optimizer = AutoOptimizer.from_repo("examples/molecular_conformation_optimizer",
                                   override_params={
                                       "initial_temperature": 200.0,
                                       "cooling_rate": 0.98,
                                       "step_size": 20.0,
                                       "max_iterations": 20000
                                   })
```

### Fine-Tuned Optimization

```python
# Precision optimization for final refinement
optimizer = AutoOptimizer.from_repo("examples/molecular_conformation_optimizer",
                                   override_params={
                                       "initial_temperature": 50.0,
                                       "final_temperature": 0.01,
                                       "step_size": 5.0,
                                       "energy_tolerance": 0.001,
                                       "convergence_window": 1000
                                   })
```

## 🎯 Parameter Guide

### Temperature Parameters
- `initial_temperature`: Higher values (100-500) for broad exploration
- `final_temperature`: Lower values (0.01-0.1) for precise convergence
- `cooling_rate`: Slower cooling (0.98-0.999) for thorough search

### Move Parameters
- `step_size`: Larger steps (20-30°) for exploration, smaller (5-10°) for refinement
- `adaptive_step_size`: Enable for automatic tuning based on acceptance rates
- `moves_per_temperature`: More moves (200-500) for complex molecules

### Convergence Parameters
- `energy_tolerance`: Tighter tolerance (0.001-0.01) for high precision
- `convergence_window`: Larger windows (1000-2000) for stable convergence

## 📊 Performance Analysis

### Monitoring Optimization

```python
result = optimizer.optimize(problem)

# Access detailed metrics
metrics = result.additional_metrics
print(f"Total moves attempted: {metrics['total_moves']}")
print(f"Moves accepted: {metrics['accepted_moves']}")
print(f"Final temperature: {metrics['final_temperature']:.3f}")

# Plot energy history
import matplotlib.pyplot as plt
plt.plot(metrics['energy_history'])
plt.xlabel('Recent Iterations')
plt.ylabel('Energy (kcal/mol)')
plt.title('Energy Evolution')
plt.show()
```

### Molecular Analysis

```python
# Get detailed molecular information
if 'molecular_info' in result.additional_metrics:
    mol_info = result.additional_metrics['molecular_info']
    print(f"Final dihedral angles: {mol_info['dihedral_angles']}")
    print(f"Van der Waals energy: {mol_info['energy_components']['van_der_waals']:.3f}")
    print(f"Electrostatic energy: {mol_info['energy_components']['electrostatic']:.3f}")
    print(f"Is local minimum: {mol_info['is_local_minimum']}")
```

## 🔧 Algorithm Tuning

### For Different Molecule Types

**Small Molecules (< 5 dihedral angles)**
```python
params = {
    "initial_temperature": 50.0,
    "step_size": 10.0,
    "max_iterations": 5000
}
```

**Medium Molecules (5-15 dihedral angles)**
```python
params = {
    "initial_temperature": 100.0,
    "step_size": 15.0,
    "max_iterations": 10000
}
```

**Large Molecules (> 15 dihedral angles)**
```python
params = {
    "initial_temperature": 200.0,
    "step_size": 20.0,
    "max_iterations": 20000,
    "moves_per_temperature": 200
}
```

## 🧪 Integration Examples

### Drug Design Workflow

```python
# Load drug molecule conformation problem
drug_problem = AutoProblem.from_repo("examples/molecular_conformation_problem",
                                    override_params={
                                        "csv_file_path": "drug_conformations.csv"
                                    })

# Multi-stage optimization
stages = [
    {"initial_temperature": 300.0, "step_size": 25.0},  # Exploration
    {"initial_temperature": 100.0, "step_size": 15.0},  # Refinement
    {"initial_temperature": 30.0, "step_size": 5.0}     # Fine-tuning
]

best_result = None
for i, stage_params in enumerate(stages):
    print(f"Stage {i+1}: {stage_params}")
    optimizer = AutoOptimizer.from_repo("examples/molecular_conformation_optimizer",
                                       override_params=stage_params)
    result = optimizer.optimize(drug_problem)
    
    if best_result is None or result.best_value < best_result.best_value:
        best_result = result

print(f"Final best energy: {best_result.best_value:.3f} kcal/mol")
```

## 📈 Expected Performance

- **Convergence**: Typically 2000-10000 iterations for most molecules
- **Acceptance Rate**: Target ~40% for optimal exploration/exploitation balance
- **Energy Improvement**: Usually finds conformations within 1-5 kcal/mol of global minimum
- **Runtime**: Scales approximately O(n × iterations) where n is number of dihedral angles

## 🔬 Chemical Validation

The optimizer includes several chemistry-aware features:

1. **Periodic Boundaries**: Proper handling of 0°/360° angle equivalence
2. **Realistic Perturbations**: Step sizes based on typical rotational barriers
3. **Correlated Moves**: Coupled angle moves reflect real molecular flexibility
4. **Energy Landscapes**: Designed for typical molecular energy surface characteristics
