# 🧬 RDKit Molecular Optimizer

A sophisticated molecular conformation optimizer using RDKit's molecular mechanics force fields and conformational sampling algorithms.

## 🎯 Overview

This optimizer leverages RDKit's powerful chemistry toolkit to find optimal molecular conformations through:
- **Force Field Optimization**: MMFF94 and UFF force fields for realistic energy calculations
- **Conformational Sampling**: ETKDG algorithm for diverse conformer generation
- **Energy Minimization**: Gradient-based optimization with molecular mechanics
- **Chemical Constraints**: Automatic enforcement of chemical validity

## 🔬 Features

### Force Fields
- **MMFF94**: Merck Molecular Force Field (default, high accuracy)
- **UFF**: Universal Force Field (broader coverage, lower accuracy)

### Algorithms
- **ETKDG**: Experimental-Torsion-Knowledge Distance Geometry for conformer generation
- **Gradient Descent**: Force field-based energy minimization
- **Conformer Clustering**: RMSD-based pruning of similar structures

### Capabilities
- Multiple conformer generation and ranking
- Chemical validity enforcement
- Reproducible results with random seeds
- Configurable optimization parameters

## 📦 Dependencies

```bash
pip install rdkit
```

## 🚀 Usage

### Basic Usage

```python
from qubots import AutoProblem, AutoOptimizer

# Load molecular problem
problem = AutoProblem.from_repo("examples/molecular_conformation_problem")

# Load RDKit optimizer with default settings
optimizer = AutoOptimizer.from_repo("examples/rdkit_molecular_optimizer")

# Run optimization
result = optimizer.optimize(problem)

print(f"Best Energy: {result.best_value:.3f} kcal/mol")
print(f"Conformers Generated: {result.algorithm_info['conformers_generated']}")
print(f"Force Field: {result.algorithm_info['force_field']}")
```

### Force Field Selection

```python
# Use MMFF94 for high accuracy
optimizer = AutoOptimizer.from_repo("examples/rdkit_molecular_optimizer",
                                   override_params={
                                       "force_field": "MMFF94",
                                       "num_conformers": 200,
                                       "max_iterations": 2000
                                   })

# Use UFF for broader molecular coverage
optimizer = AutoOptimizer.from_repo("examples/rdkit_molecular_optimizer",
                                   override_params={
                                       "force_field": "UFF",
                                       "num_conformers": 150
                                   })
```

### High-Throughput Screening

```python
# Fast screening with fewer conformers
optimizer = AutoOptimizer.from_repo("examples/rdkit_molecular_optimizer",
                                   override_params={
                                       "num_conformers": 50,
                                       "max_iterations": 500,
                                       "optimize_conformers": False  # Skip optimization for speed
                                   })
```

### Precision Optimization

```python
# High-precision optimization
optimizer = AutoOptimizer.from_repo("examples/rdkit_molecular_optimizer",
                                   override_params={
                                       "num_conformers": 500,
                                       "max_iterations": 5000,
                                       "energy_tolerance": 1e-8,
                                       "prune_rms_thresh": 0.05  # Tighter clustering
                                   })
```

## ⚙️ Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force_field` | string | "MMFF94" | Force field ("MMFF94" or "UFF") |
| `num_conformers` | integer | 100 | Number of conformers to generate |
| `max_iterations` | integer | 1000 | Max optimization iterations per conformer |
| `energy_tolerance` | number | 1e-6 | Energy convergence tolerance |
| `rmsd_threshold` | number | 0.5 | RMSD threshold for clustering |
| `random_seed` | integer | 42 | Random seed for reproducibility |
| `use_random_coords` | boolean | true | Use random initial coordinates |
| `optimize_conformers` | boolean | true | Optimize generated conformers |
| `prune_rms_thresh` | number | 0.1 | RMSD threshold for pruning |

## 🎯 Algorithm Details

### Conformer Generation
1. **ETKDG Sampling**: Generate diverse conformers using distance geometry
2. **Random Coordinates**: Optional random initial coordinate generation
3. **Pruning**: Remove similar conformers based on RMSD threshold

### Energy Optimization
1. **Force Field Setup**: Initialize MMFF94 or UFF parameters
2. **Gradient Descent**: Minimize energy using molecular mechanics
3. **Convergence**: Stop when energy change < tolerance

### Best Conformer Selection
1. **Energy Calculation**: Compute force field energy for each conformer
2. **Ranking**: Sort conformers by energy
3. **Selection**: Return lowest energy conformation

## 🔬 Chemistry Integration

### Molecular Mechanics
- Realistic bond, angle, and torsion potentials
- Van der Waals and electrostatic interactions
- Chemical constraint enforcement

### Force Field Comparison
- **MMFF94**: Higher accuracy, limited to common organic molecules
- **UFF**: Broader coverage, includes metals and unusual atoms

## 📊 Performance

### Typical Performance
- **Small molecules** (< 50 atoms): 1-10 seconds
- **Medium molecules** (50-200 atoms): 10-60 seconds  
- **Large molecules** (> 200 atoms): 1-10 minutes

### Scaling
- Linear with number of conformers
- Quadratic with molecular size
- Depends on force field complexity

## 🧪 Example Results

```python
result = optimizer.optimize(problem)

print(f"Optimization Results:")
print(f"  Best Energy: {result.best_value:.3f} kcal/mol")
print(f"  Runtime: {result.runtime_seconds:.2f} seconds")
print(f"  Conformers: {result.algorithm_info['conformers_generated']}")
print(f"  Force Field: {result.algorithm_info['force_field']}")
print(f"  RDKit Energy: {result.algorithm_info['rdkit_energy']:.3f}")
```

## 🔗 Integration

### With Other Optimizers
```python
# Compare with simulated annealing
sa_optimizer = AutoOptimizer.from_repo("examples/molecular_conformation_optimizer")
rdkit_optimizer = AutoOptimizer.from_repo("examples/rdkit_molecular_optimizer")

sa_result = sa_optimizer.optimize(problem)
rdkit_result = rdkit_optimizer.optimize(problem)

print(f"SA Energy: {sa_result.best_value:.3f}")
print(f"RDKit Energy: {rdkit_result.best_value:.3f}")
```

### With Different Problems
```python
# Works with any molecular conformation problem
problems = [
    "examples/molecular_conformation_problem",
    "examples/protein_folding_problem",
    "examples/drug_design_problem"
]

for problem_repo in problems:
    problem = AutoProblem.from_repo(problem_repo)
    result = optimizer.optimize(problem)
    print(f"{problem_repo}: {result.best_value:.3f}")
```

## 📚 References

- [RDKit Documentation](https://www.rdkit.org/docs/)
- [MMFF94 Force Field](https://pubs.acs.org/doi/10.1021/ja00124a002)
- [ETKDG Algorithm](https://pubs.acs.org/doi/10.1021/acs.jcim.5b00654)
- [Conformational Sampling](https://pubs.acs.org/doi/10.1021/ci100031x)
