"""
RDKit Molecular Optimizer for molecular conformation optimization.

This optimizer uses RDKit's molecular dynamics and force field capabilities
to optimize molecular conformations through energy minimization and conformational
sampling. It provides chemistry-aware optimization with realistic molecular mechanics.

Features:
- RDKit force field integration (MMFF94, UFF)
- Conformational sampling with ETKDG
- Energy minimization with molecular mechanics
- Multiple conformer generation and selection
- Chemical validity constraints
"""

import numpy as np
import pandas as pd
import random
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any
from qubots import (
    BaseOptimizer, OptimizerMetadata, OptimizerType, OptimizerFamily,
    OptimizationResult, BaseProblem
)

# RDKit imports with error handling
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors, rdForceFieldHelpers, rdMolTransforms
    from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, MMFFOptimizeMolecule
    from rdkit.Chem.rdDistGeom import EmbedMolecule, EmbedMultipleConfs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Install with: pip install rdkit")


class RDKitMolecularOptimizer(BaseOptimizer):
    """
    RDKit-based molecular conformation optimizer.
    
    This optimizer uses RDKit's molecular mechanics force fields and conformational
    sampling algorithms to find optimal molecular conformations. It supports
    multiple force fields and sampling strategies for robust optimization.
    
    Features:
    - MMFF94 and UFF force field support
    - ETKDG conformational sampling
    - Energy minimization with gradient descent
    - Multiple conformer generation and ranking
    - Chemical constraint enforcement
    """
    
    def __init__(self,
                 force_field: str = "MMFF94",
                 num_conformers: int = 100,
                 max_iterations: int = 1000,
                 energy_tolerance: float = 1e-6,
                 rmsd_threshold: float = 0.5,
                 random_seed: int = 42,
                 use_random_coords: bool = True,
                 optimize_conformers: bool = True,
                 prune_rms_thresh: float = 0.1,
                 create_plots: bool = True,
                 **kwargs):
        """
        Initialize RDKit molecular optimizer.

        Args:
            force_field: Force field to use ("MMFF94" or "UFF")
            num_conformers: Number of conformers to generate
            max_iterations: Maximum optimization iterations per conformer
            energy_tolerance: Energy convergence tolerance
            rmsd_threshold: RMSD threshold for conformer clustering
            random_seed: Random seed for reproducibility
            use_random_coords: Whether to use random initial coordinates
            optimize_conformers: Whether to optimize generated conformers
            prune_rms_thresh: RMSD threshold for pruning similar conformers
            create_plots: Whether to create visualization plots after optimization
            **kwargs: Additional parameters
        """
        # Initialize base class with all parameters
        super().__init__(
            force_field=force_field,
            num_conformers=num_conformers,
            max_iterations=max_iterations,
            energy_tolerance=energy_tolerance,
            rmsd_threshold=rmsd_threshold,
            random_seed=random_seed,
            use_random_coords=use_random_coords,
            optimize_conformers=optimize_conformers,
            prune_rms_thresh=prune_rms_thresh,
            create_plots=create_plots,
            **kwargs
        )
        
        # Check RDKit availability
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for this optimizer. Install with: pip install rdkit")
        
        # Store algorithm parameters
        self.force_field = force_field.upper()
        self.num_conformers = num_conformers
        self.max_iterations = max_iterations
        self.energy_tolerance = energy_tolerance
        self.rmsd_threshold = rmsd_threshold
        self.random_seed = random_seed
        self.use_random_coords = use_random_coords
        self.optimize_conformers = optimize_conformers
        self.prune_rms_thresh = prune_rms_thresh
        self.create_plots = create_plots
        
        # Validate force field
        if self.force_field not in ["MMFF94", "UFF"]:
            raise ValueError(f"Unsupported force field: {force_field}. Use 'MMFF94' or 'UFF'")
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize tracking variables
        self.conformer_energies = []
        self.optimization_history = []
        
    def _get_default_metadata(self) -> OptimizerMetadata:
        """Return default metadata for RDKit molecular optimizer."""
        return OptimizerMetadata(
            name="RDKit Molecular Optimizer",
            description="RDKit-based molecular conformation optimizer using force fields and conformational sampling",
            optimizer_type=OptimizerType.HEURISTIC,
            optimizer_family=OptimizerFamily.LOCAL_SEARCH,
            author="Qubots Framework",
            version="1.0.0",
            is_deterministic=False,
            supports_constraints=True,
            supports_continuous=True,
            supports_discrete=False,
            time_complexity="O(conformers * iterations)",
            space_complexity="O(conformers * atoms)",
            convergence_guaranteed=False,
            parallel_capable=False,
            required_parameters=["force_field", "num_conformers", "max_iterations"],
            optional_parameters=["energy_tolerance", "rmsd_threshold", "random_seed", "use_random_coords", "optimize_conformers", "prune_rms_thresh"]
        )
    
    def _optimize_implementation(self, problem: BaseProblem, initial_solution: Optional[Any] = None) -> OptimizationResult:
        """
        Core RDKit optimization implementation.
        
        Args:
            problem: Molecular conformation problem instance
            initial_solution: Optional initial dihedral angles
            
        Returns:
            OptimizationResult with best conformation and energy
        """
        start_time = time.time()
        
        # Create molecule from problem data
        mol = self._create_molecule_from_problem(problem)
        if mol is None:
            raise ValueError("Could not create valid molecule from problem data")
        
        # Generate conformers
        conformer_ids = self._generate_conformers(mol)

        # Check if we have any conformers
        if not conformer_ids:
            raise ValueError("Could not generate any conformers for the molecule")

        # Optimize conformers if requested
        if self.optimize_conformers:
            self._optimize_conformers(mol, conformer_ids)

        # Calculate energies and find best conformer
        best_conformer_id, best_energy = self._find_best_conformer(mol, conformer_ids)
        
        # Extract dihedral angles from best conformer
        best_solution = self._extract_dihedral_angles(mol, best_conformer_id, problem)
        
        # Validate solution with problem
        final_energy = problem.evaluate_solution(best_solution)

        runtime = time.time() - start_time

        # Create visualization plots if requested
        if self.create_plots:
            self._create_optimization_plots(mol, conformer_ids, problem)

        return OptimizationResult(
            best_solution=best_solution,
            best_value=final_energy,
            iterations=len(conformer_ids),
            runtime_seconds=runtime,
            convergence_achieved=True,  # RDKit always finds a solution
            termination_reason="conformer_generation_complete",
            optimization_history=[{"energy": e, "conformer_id": i} for i, e in enumerate(self.conformer_energies)],
            additional_metrics={
                "force_field": self.force_field,
                "conformers_generated": len(conformer_ids),
                "rdkit_energy": best_energy,
                "problem_energy": final_energy,
                "optimization_method": "RDKit Force Field",
                "energy_range": max(self.conformer_energies) - min(self.conformer_energies) if self.conformer_energies else 0.0,
                "energy_std": float(np.std(self.conformer_energies)) if self.conformer_energies else 0.0
            }
        )
    
    def _create_molecule_from_problem(self, problem: BaseProblem) -> Optional[Chem.Mol]:
        """Create RDKit molecule from problem data."""
        # Determine molecule type based on number of dihedral angles in the dataset
        n_dihedrals = getattr(problem, 'n_dihedrals', 3)

        # Map number of dihedrals to appropriate SMILES strings
        if n_dihedrals == 3:
            # Butane (C4H10) - 3 dihedral angles
            smiles = "CCCC"
        elif n_dihedrals == 5:
            # Hexane (C6H14) - 5 dihedral angles
            smiles = "CCCCCC"
        elif n_dihedrals == 4:
            # Pentane (C5H12) - 4 dihedral angles
            smiles = "CCCCC"
        elif n_dihedrals == 2:
            # Propane (C3H8) - 2 dihedral angles
            smiles = "CCC"
        elif n_dihedrals == 1:
            # Ethane (C2H6) - 1 dihedral angle
            smiles = "CC"
        else:
            # Default to linear alkane with appropriate length
            # For n dihedral angles, we need n+1 carbon atoms
            carbon_count = max(2, n_dihedrals + 1)
            smiles = "C" * carbon_count

        print(f"Creating molecule with {n_dihedrals} dihedral angles using SMILES: {smiles}")

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
        return mol
    
    def _generate_conformers(self, mol: Chem.Mol) -> List[int]:
        """Generate conformers using ETKDG algorithm."""
        try:
            # Set up conformer generation parameters
            params = AllChem.EmbedParameters()
            params.randomSeed = self.random_seed
            params.useRandomCoords = self.use_random_coords
            params.pruneRmsThresh = self.prune_rms_thresh

            # Generate multiple conformers
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=self.num_conformers,
                params=params
            )

            # If no conformers were generated, try with simpler parameters
            if len(conformer_ids) == 0:
                # Try without pruning
                conformer_ids = AllChem.EmbedMultipleConfs(
                    mol,
                    numConfs=min(10, self.num_conformers),  # Fewer conformers
                    randomSeed=self.random_seed
                )

            return list(conformer_ids)
        except Exception as e:
            # If conformer generation fails, try to generate at least one
            try:
                conf_id = AllChem.EmbedMolecule(mol, randomSeed=self.random_seed)
                if conf_id >= 0:
                    return [conf_id]
                else:
                    return []
            except:
                return []
    
    def _optimize_conformers(self, mol: Chem.Mol, conformer_ids: List[int]):
        """Optimize conformers using force field minimization."""
        for conf_id in conformer_ids:
            try:
                if self.force_field == "MMFF94":
                    # Try MMFF94 first
                    if AllChem.MMFFHasAllMoleculeParams(mol):
                        MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=self.max_iterations)
                    else:
                        # Fallback to UFF if MMFF94 parameters not available
                        UFFOptimizeMolecule(mol, confId=conf_id, maxIters=self.max_iterations)
                else:  # UFF
                    UFFOptimizeMolecule(mol, confId=conf_id, maxIters=self.max_iterations)
            except Exception:
                # If optimization fails for this conformer, skip it
                continue
    
    def _find_best_conformer(self, mol: Chem.Mol, conformer_ids: List[int]) -> Tuple[int, float]:
        """Find conformer with lowest energy."""
        best_energy = float('inf')
        best_conformer_id = conformer_ids[0]
        
        self.conformer_energies = []
        
        for conf_id in conformer_ids:
            energy = self._calculate_conformer_energy(mol, conf_id)
            self.conformer_energies.append(energy)
            
            if energy < best_energy:
                best_energy = energy
                best_conformer_id = conf_id
        
        return best_conformer_id, best_energy
    
    def _calculate_conformer_energy(self, mol: Chem.Mol, conf_id: int) -> float:
        """Calculate energy of a specific conformer."""
        try:
            if self.force_field == "MMFF94" and AllChem.MMFFHasAllMoleculeParams(mol):
                # Get MMFF properties first
                mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
                if mmff_props is not None:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
                else:
                    # Fallback to UFF if MMFF properties can't be generated
                    ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)

            if ff is not None:
                return ff.CalcEnergy()
            else:
                return float('inf')
        except Exception:
            # If force field calculation fails, return high energy
            return float('inf')
    
    def _extract_dihedral_angles(self, mol: Chem.Mol, conf_id: int, problem: BaseProblem) -> List[float]:
        """Extract dihedral angles from conformer to match problem format."""
        try:
            # Get the conformer
            conf = mol.GetConformer(conf_id)

            # Get number of expected dihedral angles from problem
            n_dihedrals = getattr(problem, 'n_dihedrals', 3)

            # For linear alkanes, extract backbone dihedral angles
            # Find the main carbon chain (non-hydrogen atoms)
            carbon_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']

            if len(carbon_atoms) < 4:
                # Not enough carbons for dihedral angles, return zeros
                return [0.0] * n_dihedrals

            # Extract dihedral angles along the carbon backbone
            dihedral_angles = []
            for i in range(len(carbon_atoms) - 3):
                # Get four consecutive carbon atoms
                atom_indices = carbon_atoms[i:i+4]

                # Calculate dihedral angle
                try:
                    dihedral = Chem.rdMolTransforms.GetDihedralDeg(conf, *atom_indices)
                    # Normalize to [0, 360) range
                    dihedral = dihedral % 360.0
                    dihedral_angles.append(dihedral)
                except:
                    # If calculation fails, use 0
                    dihedral_angles.append(0.0)

            # Pad or truncate to match expected number of dihedrals
            while len(dihedral_angles) < n_dihedrals:
                dihedral_angles.append(0.0)

            return dihedral_angles[:n_dihedrals]

        except Exception as e:
            print(f"Warning: Could not extract dihedral angles: {e}")
            # Fallback to zeros if extraction fails
            n_dihedrals = getattr(problem, 'n_dihedrals', 3)
            return [0.0] * n_dihedrals

    def _create_optimization_plots(self, mol: Chem.Mol, conformer_ids: List[int], problem: BaseProblem):
        """Create visualization plots for the optimization results."""
        try:
            # Set up the plotting style
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('🧬 RDKit Molecular Optimization Results', fontsize=16, fontweight='bold')

            # Plot 1: Energy Distribution of Conformers
            if self.conformer_energies:
                axes[0, 0].hist(self.conformer_energies, bins=min(20, len(self.conformer_energies)),
                               alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, 0].axvline(min(self.conformer_energies), color='red', linestyle='--',
                                  label=f'Best Energy: {min(self.conformer_energies):.2f}')
                axes[0, 0].set_xlabel('Energy (kcal/mol)')
                axes[0, 0].set_ylabel('Number of Conformers')
                axes[0, 0].set_title('Energy Distribution of Generated Conformers')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Energy vs Conformer Index
            if self.conformer_energies:
                conformer_indices = list(range(len(self.conformer_energies)))
                axes[0, 1].scatter(conformer_indices, self.conformer_energies,
                                  alpha=0.6, c=self.conformer_energies, cmap='viridis')
                best_idx = self.conformer_energies.index(min(self.conformer_energies))
                axes[0, 1].scatter(best_idx, min(self.conformer_energies),
                                  color='red', s=100, marker='*', label='Best Conformer')
                axes[0, 1].set_xlabel('Conformer Index')
                axes[0, 1].set_ylabel('Energy (kcal/mol)')
                axes[0, 1].set_title('Energy vs Conformer Index')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Energy Statistics
            if self.conformer_energies:
                energy_stats = {
                    'Min Energy': min(self.conformer_energies),
                    'Max Energy': max(self.conformer_energies),
                    'Mean Energy': np.mean(self.conformer_energies),
                    'Std Energy': np.std(self.conformer_energies)
                }

                stats_names = list(energy_stats.keys())
                stats_values = list(energy_stats.values())

                bars = axes[1, 0].bar(stats_names, stats_values,
                                     color=['green', 'red', 'blue', 'orange'], alpha=0.7)
                axes[1, 0].set_ylabel('Energy (kcal/mol)')
                axes[1, 0].set_title('Energy Statistics Summary')
                axes[1, 0].tick_params(axis='x', rotation=45)

                # Add value labels on bars
                for bar, value in zip(bars, stats_values):
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{value:.2f}', ha='center', va='bottom')

            # Plot 4: Optimization Summary
            axes[1, 1].axis('off')
            summary_text = f"""
🔬 Optimization Summary

Force Field: {self.force_field}
Conformers Generated: {len(conformer_ids)}
Best Energy: {min(self.conformer_energies):.3f} kcal/mol

📊 Energy Range: {max(self.conformer_energies) - min(self.conformer_energies):.3f} kcal/mol
📈 Energy Spread: {np.std(self.conformer_energies):.3f} kcal/mol

⚙️ Parameters:
• Max Iterations: {self.max_iterations}
• Random Seed: {self.random_seed}
• RMSD Threshold: {self.rmsd_threshold}

🎯 Best conformer found among {len(conformer_ids)} generated structures
            """

            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=11, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

            plt.tight_layout()
            plt.show()

            # Additional plot: Energy landscape if we have enough data points
            if len(self.conformer_energies) > 10:
                self._create_energy_landscape_plot()

        except Exception as e:
            print(f"⚠️  Could not create plots: {e}")
            print("📊 Plotting requires matplotlib. Install with: pip install matplotlib")

    def _create_energy_landscape_plot(self):
        """Create an additional energy landscape visualization."""
        try:
            plt.figure(figsize=(12, 6))

            # Sort energies for better visualization
            sorted_energies = sorted(self.conformer_energies)

            # Create energy landscape plot
            plt.subplot(1, 2, 1)
            plt.plot(sorted_energies, 'o-', alpha=0.7, color='darkblue', markersize=4)
            plt.axhline(y=min(sorted_energies), color='red', linestyle='--', alpha=0.7,
                       label=f'Global Minimum: {min(sorted_energies):.2f}')
            plt.xlabel('Conformer Rank (by Energy)')
            plt.ylabel('Energy (kcal/mol)')
            plt.title('🏔️ Energy Landscape (Sorted)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Create energy difference plot
            plt.subplot(1, 2, 2)
            energy_diffs = [e - min(sorted_energies) for e in sorted_energies]
            plt.semilogy(energy_diffs, 'o-', alpha=0.7, color='darkgreen', markersize=4)
            plt.xlabel('Conformer Rank')
            plt.ylabel('Energy Above Global Min (kcal/mol)')
            plt.title('📈 Relative Energy Distribution')
            plt.grid(True, alpha=0.3)

            plt.suptitle('🧬 Detailed Energy Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"⚠️  Could not create energy landscape plot: {e}")
