# Molecular Conformation Optimization Problem

A chemistry-focused optimization problem for finding the lowest energy molecular conformations by optimizing dihedral angles. This problem is essential for drug design, protein folding studies, and chemical structure optimization.

## 🧬 Problem Overview

Molecular conformation optimization involves finding the arrangement of atoms in a molecule that minimizes the total energy. This problem focuses on optimizing dihedral (torsion) angles, which are critical for determining molecular shape and stability.

### Key Applications
- **Drug Design**: Finding bioactive conformations of pharmaceutical compounds
- **Protein Folding**: Understanding protein structure and stability
- **Chemical Synthesis**: Predicting reaction pathways and intermediates
- **Materials Science**: Designing polymers and molecular materials

## 🔬 Features

- **CSV-Based Molecular Data**: Load conformational data from CSV files
- **Energy Interpolation**: Continuous optimization using nearest-neighbor interpolation
- **Multiple Energy Components**: Van der Waals, electrostatic, and total energies
- **Chemical Validation**: Ensures conformations are chemically reasonable
- **Local Minimum Detection**: Identifies stable conformational states
- **Realistic Constraints**: Handles periodic boundary conditions for dihedral angles

## 📊 Dataset Format

The problem accepts CSV files with molecular conformation data:

```csv
dihedral_1,dihedral_2,dihedral_3,energy,bond_length_C1_C2,van_der_waals_energy,electrostatic_energy
90.0,90.0,0.0,0.4,1.54,-3.9,-1.4
60.0,60.0,0.0,1.7,1.54,-3.4,-0.9
120.0,120.0,0.0,1.7,1.54,-3.4,-0.8
```

### Required Columns
- `dihedral_1`: First dihedral angle (degrees)
- `energy`: Total molecular energy (kcal/mol)

### Optional Columns
- `dihedral_2`, `dihedral_3`, etc.: Additional dihedral angles
- `bond_length_*`: Bond lengths (Angstroms)
- `van_der_waals_energy`: Van der Waals interaction energy
- `electrostatic_energy`: Electrostatic interaction energy

## 🚀 Usage

### Basic Usage

```python
from qubots import AutoProblem

# Load with default butane dataset
problem = AutoProblem.from_repo("examples/molecular_conformation_problem")

# Generate and evaluate a random conformation
solution = problem.random_solution()
energy = problem.evaluate_solution(solution)

print(f"Molecular energy: {energy:.2f} kcal/mol")
print(f"Dihedral angles: {solution}")
```

### Custom Molecular Data

```python
from qubots import AutoProblem

# CSV data as string
molecular_data = """dihedral_1,dihedral_2,energy,van_der_waals_energy,electrostatic_energy
0.0,0.0,5.2,-2.1,0.3
90.0,90.0,0.4,-3.9,-1.4
180.0,180.0,4.9,-2.3,0.1"""

problem = AutoProblem.from_repo("examples/molecular_conformation_problem", 
                               override_params={
                                   "csv_data": molecular_data,
                                   "energy_tolerance": 0.05
                               })
```

### Loading from File

```python
from qubots import AutoProblem

problem = AutoProblem.from_repo("examples/molecular_conformation_problem",
                               override_params={
                                   "csv_file_path": "path/to/your/molecule.csv"
                               })
```

### Detailed Analysis

```python
# Get detailed solution information
solution = problem.random_solution()
info = problem.get_solution_info(solution)

print(f"Total energy: {info['total_energy']:.2f} kcal/mol")
print(f"Van der Waals: {info['energy_components']['van_der_waals']:.2f} kcal/mol")
print(f"Electrostatic: {info['energy_components']['electrostatic']:.2f} kcal/mol")
print(f"Is local minimum: {info['is_local_minimum']}")
print(f"Strain energy: {info['strain_energy']:.2f} kcal/mol")

# Find the global minimum
global_min_angles, global_min_energy = problem.get_global_minimum()
print(f"Global minimum: {global_min_energy:.2f} kcal/mol at {global_min_angles}")
```

## 🎯 Optimization Objectives

- **Primary**: Minimize total molecular energy
- **Secondary**: Find chemically stable conformations
- **Constraints**: Dihedral angles in [0°, 360°) range

## 🔧 Parameters

- `energy_tolerance`: Convergence tolerance (default: 0.1 kcal/mol)
- `angle_precision`: Dihedral angle precision (default: 1.0 degrees)
- `csv_data`: Molecular data as CSV string
- `csv_file_path`: Path to CSV file with molecular data

## 📈 Example Molecules

The package includes sample data for:
- **Butane (C₄H₁₀)**: Simple alkane with rotational barriers
- **Ethane derivatives**: Various substituted ethanes
- **Small peptides**: Amino acid conformations

## 🧪 Integration with Optimizers

This problem works with various optimization algorithms:

```python
from qubots import AutoProblem, AutoOptimizer

# Load molecular problem
problem = AutoProblem.from_repo("examples/molecular_conformation_problem")

# Try different optimizers
optimizers = [
    "examples/simulated_annealing_optimizer",
    "examples/genetic_algorithm_optimizer", 
    "examples/particle_swarm_optimizer"
]

for opt_name in optimizers:
    optimizer = AutoOptimizer.from_repo(opt_name)
    result = optimizer.optimize(problem)
    print(f"{opt_name}: {result.best_value:.2f} kcal/mol")
```

## 📚 Chemical Background

Molecular conformations are different spatial arrangements of atoms that can be interconverted by rotation around single bonds. The energy landscape is determined by:

1. **Steric Interactions**: Repulsion between atoms in close proximity
2. **Van der Waals Forces**: Weak attractive forces between atoms
3. **Electrostatic Interactions**: Coulombic forces between charged groups
4. **Torsional Strain**: Energy barriers to rotation around bonds

Finding the global minimum energy conformation is crucial for understanding molecular behavior and designing new compounds.
