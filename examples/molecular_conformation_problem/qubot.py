"""
Molecular Conformation Optimization Problem for Qubots Framework

This problem implements molecular conformation optimization for finding the lowest energy
conformations of molecules by optimizing dihedral angles. It reads molecular data from
CSV files containing conformational energies and structural parameters.

The problem accepts CSV data with columns:
- dihedral_1, dihedral_2, dihedral_3: Dihedral angles in degrees
- energy: Total molecular energy in kcal/mol
- bond_length_*: Bond lengths in Angstroms
- van_der_waals_energy: Van der Waals interaction energy
- electrostatic_energy: Electrostatic interaction energy

Compatible with Rastion platform workflow automation and local development.
"""

import numpy as np
import pandas as pd
import math
from typing import List, Tuple, Optional, Dict, Any, Union
from qubots import (
    BaseProblem, ProblemMetadata, ProblemType, ObjectiveType,
    DifficultyLevel, EvaluationResult
)


class MolecularConformationProblem(BaseProblem):
    """
    Molecular conformation optimization problem for finding minimum energy conformations.
    
    This problem optimizes molecular conformations by adjusting dihedral angles to minimize
    the total molecular energy. It supports both CSV data input and interpolation for
    continuous optimization of conformational space.
    
    Features:
    - CSV-based molecular data input
    - Energy interpolation for continuous optimization
    - Multiple energy components (van der Waals, electrostatic)
    - Realistic chemical constraints
    - Solution validation and analysis
    """
    
    def __init__(self, 
                 csv_data: str = None,
                 csv_file_path: str = None,
                 energy_tolerance: float = 0.1,
                 angle_precision: float = 1.0,
                 **kwargs):
        """
        Initialize molecular conformation optimization problem.
        
        Args:
            csv_data: CSV content as string
            csv_file_path: Path to CSV file (alternative to csv_data)
            energy_tolerance: Energy tolerance for convergence (kcal/mol)
            angle_precision: Precision for dihedral angles (degrees)
            **kwargs: Additional parameters
        """
        self.energy_tolerance = energy_tolerance
        self.angle_precision = angle_precision
        
        # Load molecular data from CSV
        self.molecular_data = self._load_molecular_data(csv_data, csv_file_path)
        self.n_dihedrals = self._count_dihedral_angles()
        
        # Extract energy landscape information
        self.min_energy = self.molecular_data['energy'].min()
        self.max_energy = self.molecular_data['energy'].max()
        self.energy_range = self.max_energy - self.min_energy
        
        # Initialize base class (metadata will be set via _get_default_metadata)
        super().__init__()

    def _get_default_metadata(self) -> ProblemMetadata:
        """Return default metadata for molecular conformation problem."""
        return ProblemMetadata(
            name="Molecular Conformation Optimization",
            description=f"Find minimum energy conformation by optimizing {getattr(self, 'n_dihedrals', 'N')} dihedral angles",
            problem_type=ProblemType.CONTINUOUS,
            objective_type=ObjectiveType.MINIMIZE,
            domain="chemistry",
            tags={"molecular_dynamics", "conformation", "energy_minimization", "chemistry", "optimization"},
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            dimension=getattr(self, 'n_dihedrals', 0),
            variable_bounds=[(0.0, 360.0)] * getattr(self, 'n_dihedrals', 0),  # Dihedral angles in degrees
            constraints_count=0,  # No hard constraints
            evaluation_complexity="O(n)",
            memory_complexity="O(n)"
        )

    def _load_molecular_data(self, csv_data: str = None, csv_file_path: str = None) -> pd.DataFrame:
        """Load molecular conformation data from CSV."""
        if csv_data:
            # Parse CSV data from string
            from io import StringIO
            df = pd.read_csv(StringIO(csv_data), comment='#')
        elif csv_file_path:
            # Load from file
            df = pd.read_csv(csv_file_path, comment='#')
        else:
            # Use default butane dataset
            default_data = self._get_default_butane_data()
            from io import StringIO
            df = pd.read_csv(StringIO(default_data), comment='#')
        
        return df
    
    def _count_dihedral_angles(self) -> int:
        """Count the number of dihedral angles in the dataset."""
        dihedral_cols = [col for col in self.molecular_data.columns if col.startswith('dihedral_')]
        return len(dihedral_cols)
    
    def _get_default_butane_data(self) -> str:
        """Return default butane conformation data."""
        return """dihedral_1,dihedral_2,dihedral_3,energy,bond_length_C1_C2,bond_length_C2_C3,bond_length_C3_C4,van_der_waals_energy,electrostatic_energy
90.0,90.0,0.0,0.4,1.54,1.54,1.54,-3.9,-1.4
60.0,60.0,0.0,1.7,1.54,1.54,1.54,-3.4,-0.9
120.0,120.0,0.0,1.7,1.54,1.54,1.54,-3.4,-0.8
180.0,180.0,0.0,4.9,1.54,1.54,1.54,-2.3,0.1"""
    
    def evaluate_solution(self, solution: Union[List[float], np.ndarray], verbose: bool = False) -> Union[float, EvaluationResult]:
        """
        Evaluate molecular energy for given dihedral angles.
        
        Args:
            solution: List or array of dihedral angles in degrees
            verbose: Whether to return detailed evaluation result
            
        Returns:
            Energy value (float) or detailed EvaluationResult
        """
        if isinstance(solution, list):
            solution = np.array(solution)
        
        # Normalize angles to [0, 360) range
        normalized_angles = solution % 360.0
        
        # Interpolate energy from the dataset
        energy = self._interpolate_energy(normalized_angles)
        
        # Calculate additional metrics
        is_feasible = self._is_valid_conformation(normalized_angles)
        
        if verbose:
            # Get detailed energy components
            vdw_energy, elec_energy = self._interpolate_energy_components(normalized_angles)
            
            return EvaluationResult(
                objective_value=energy,
                is_feasible=is_feasible,
                additional_metrics={
                    "dihedral_angles": normalized_angles.tolist(),
                    "van_der_waals_energy": vdw_energy,
                    "electrostatic_energy": elec_energy,
                    "total_energy": energy,
                    "energy_above_minimum": energy - self.min_energy,
                    "is_local_minimum": self._is_local_minimum(normalized_angles),
                    "strain_energy": max(0, energy - self.min_energy)
                }
            )
        
        return energy
    
    def _interpolate_energy(self, angles: np.ndarray) -> float:
        """Interpolate energy for given dihedral angles using nearest neighbor."""
        min_distance = float('inf')
        closest_energy = self.max_energy
        
        for _, row in self.molecular_data.iterrows():
            # Calculate distance in dihedral space
            data_angles = np.array([row[f'dihedral_{i+1}'] for i in range(self.n_dihedrals)])
            
            # Handle periodic boundary conditions for angles
            angle_diff = np.abs(angles - data_angles)
            angle_diff = np.minimum(angle_diff, 360.0 - angle_diff)
            
            distance = np.sqrt(np.sum(angle_diff**2))
            
            if distance < min_distance:
                min_distance = distance
                closest_energy = row['energy']
        
        return closest_energy
    
    def _interpolate_energy_components(self, angles: np.ndarray) -> Tuple[float, float]:
        """Interpolate van der Waals and electrostatic energy components."""
        min_distance = float('inf')
        closest_vdw = 0.0
        closest_elec = 0.0
        
        for _, row in self.molecular_data.iterrows():
            data_angles = np.array([row[f'dihedral_{i+1}'] for i in range(self.n_dihedrals)])
            angle_diff = np.abs(angles - data_angles)
            angle_diff = np.minimum(angle_diff, 360.0 - angle_diff)
            distance = np.sqrt(np.sum(angle_diff**2))
            
            if distance < min_distance:
                min_distance = distance
                closest_vdw = row.get('van_der_waals_energy', 0.0)
                closest_elec = row.get('electrostatic_energy', 0.0)
        
        return closest_vdw, closest_elec
    
    def _is_valid_conformation(self, angles: np.ndarray) -> bool:
        """Check if the conformation is chemically valid."""
        # All angles should be within [0, 360) range
        if np.any(angles < 0) or np.any(angles >= 360):
            return False
        
        # Check for severe steric clashes (very high energy)
        energy = self._interpolate_energy(angles)
        return energy < (self.min_energy + 2 * self.energy_range)
    
    def _is_local_minimum(self, angles: np.ndarray, delta: float = 5.0) -> bool:
        """Check if the conformation is a local energy minimum."""
        current_energy = self._interpolate_energy(angles)
        
        # Check neighboring conformations
        for i in range(self.n_dihedrals):
            for direction in [-delta, delta]:
                neighbor_angles = angles.copy()
                neighbor_angles[i] = (neighbor_angles[i] + direction) % 360.0
                neighbor_energy = self._interpolate_energy(neighbor_angles)
                
                if neighbor_energy < current_energy:
                    return False
        
        return True
    
    def random_solution(self) -> List[float]:
        """Generate a random valid conformation."""
        return [np.random.uniform(0, 360) for _ in range(self.n_dihedrals)]
    
    def get_neighbor_solution(self, solution: List[float], step_size: float = 10.0) -> List[float]:
        """Generate a neighboring solution by small angle perturbations."""
        neighbor = solution.copy()
        
        # Randomly select which dihedral to modify
        dihedral_idx = np.random.randint(0, self.n_dihedrals)
        
        # Apply small random perturbation
        perturbation = np.random.uniform(-step_size, step_size)
        neighbor[dihedral_idx] = (neighbor[dihedral_idx] + perturbation) % 360.0
        
        return neighbor
    
    def get_solution_info(self, solution: List[float]) -> Dict[str, Any]:
        """Get detailed information about a solution."""
        result = self.evaluate_solution(solution, verbose=True)
        
        return {
            "dihedral_angles": result.additional_metrics["dihedral_angles"],
            "total_energy": result.objective_value,
            "energy_components": {
                "van_der_waals": result.additional_metrics["van_der_waals_energy"],
                "electrostatic": result.additional_metrics["electrostatic_energy"]
            },
            "is_local_minimum": result.additional_metrics["is_local_minimum"],
            "strain_energy": result.additional_metrics["strain_energy"],
            "is_feasible": result.is_feasible
        }
    
    def get_global_minimum(self) -> Tuple[List[float], float]:
        """Get the known global minimum from the dataset."""
        min_idx = self.molecular_data['energy'].idxmin()
        min_row = self.molecular_data.loc[min_idx]
        
        angles = [min_row[f'dihedral_{i+1}'] for i in range(self.n_dihedrals)]
        energy = min_row['energy']
        
        return angles, energy
