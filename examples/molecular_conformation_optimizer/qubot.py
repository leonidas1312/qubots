"""
Molecular Conformation Optimizer for Qubots Framework

This module implements a specialized simulated annealing optimizer for molecular
conformation problems. It uses chemistry-aware moves and cooling schedules
optimized for energy landscape exploration in conformational space.

Features:
- Chemistry-aware dihedral angle moves
- Adaptive temperature scheduling
- Multiple move types (single angle, coupled angles)
- Energy-based acceptance criteria
- Convergence detection for molecular systems

Compatible with Rastion platform workflow automation and molecular problems.
"""

import numpy as np
import random
import time
import math
from typing import List, Tuple, Optional, Dict, Any
from qubots import (
    BaseOptimizer, OptimizerMetadata, OptimizerType, OptimizerFamily,
    OptimizationResult, BaseProblem
)


class MolecularConformationOptimizer(BaseOptimizer):
    """
    Simulated annealing optimizer specialized for molecular conformation problems.
    
    This optimizer uses chemistry-aware moves and temperature schedules designed
    for exploring molecular energy landscapes. It includes multiple move types
    and adaptive parameters for efficient conformational sampling.
    
    Features:
    - Chemistry-aware dihedral angle perturbations
    - Exponential and linear cooling schedules
    - Multiple move types (single, coupled, random)
    - Energy-based convergence detection
    - Adaptive step sizes based on acceptance rates
    """
    
    def __init__(self,
                 initial_temperature: float = 100.0,
                 final_temperature: float = 0.1,
                 cooling_rate: float = 0.95,
                 max_iterations: int = 10000,
                 moves_per_temperature: int = 100,
                 step_size: float = 15.0,
                 adaptive_step_size: bool = True,
                 move_type_probabilities: Dict[str, float] = None,
                 convergence_window: int = 500,
                 energy_tolerance: float = 0.01,
                 **kwargs):
        """
        Initialize molecular conformation optimizer.
        
        Args:
            initial_temperature: Starting temperature for annealing
            final_temperature: Final temperature (stopping criterion)
            cooling_rate: Temperature reduction factor (0 < rate < 1)
            max_iterations: Maximum number of iterations
            moves_per_temperature: Number of moves at each temperature
            step_size: Initial step size for dihedral angle changes (degrees)
            adaptive_step_size: Whether to adapt step size based on acceptance
            move_type_probabilities: Probabilities for different move types
            convergence_window: Window size for convergence detection
            energy_tolerance: Energy tolerance for convergence
            **kwargs: Additional parameters
        """
        # Set default move type probabilities
        if move_type_probabilities is None:
            move_type_probabilities = {
                "single_angle": 0.6,    # Change one dihedral angle
                "coupled_angles": 0.3,  # Change two related angles
                "random_walk": 0.1      # Random perturbation
            }
        
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.moves_per_temperature = moves_per_temperature
        self.step_size = step_size
        self.initial_step_size = step_size
        self.adaptive_step_size = adaptive_step_size
        self.move_type_probabilities = move_type_probabilities
        self.convergence_window = convergence_window
        self.energy_tolerance = energy_tolerance
        
        # Initialize tracking variables
        self.current_temperature = initial_temperature
        self.accepted_moves = 0
        self.total_moves = 0
        self.energy_history = []
        
        # Initialize base class with all parameters
        super().__init__(
            initial_temperature=initial_temperature,
            final_temperature=final_temperature,
            cooling_rate=cooling_rate,
            max_iterations=max_iterations,
            moves_per_temperature=moves_per_temperature,
            step_size=step_size,
            adaptive_step_size=adaptive_step_size,
            convergence_window=convergence_window,
            energy_tolerance=energy_tolerance,
            **kwargs
        )

    def _get_default_metadata(self) -> OptimizerMetadata:
        """Return default metadata for molecular conformation optimizer."""
        return OptimizerMetadata(
            name="Molecular Conformation Optimizer",
            description="Simulated annealing optimizer specialized for molecular conformation problems with chemistry-aware moves",
            optimizer_type=OptimizerType.METAHEURISTIC,
            optimizer_family=OptimizerFamily.LOCAL_SEARCH,
            author="Qubots Framework",
            version="1.0.0",
            is_deterministic=False,
            supports_constraints=False,
            supports_continuous=True,
            supports_discrete=False,
            time_complexity="O(iterations * moves_per_temp)",
            space_complexity="O(n)",
            convergence_guaranteed=False,
            parallel_capable=False,
            required_parameters=["initial_temperature", "cooling_rate", "max_iterations"],
            optional_parameters=["final_temperature", "moves_per_temperature", "step_size", "adaptive_step_size", "convergence_window", "energy_tolerance"]
        )

    def _optimize_implementation(self, problem: BaseProblem, initial_solution: Optional[Any] = None) -> OptimizationResult:
        """
        Core simulated annealing implementation for molecular conformation optimization.
        
        Args:
            problem: Molecular conformation problem instance
            initial_solution: Optional initial conformation
            
        Returns:
            OptimizationResult with best conformation found
        """
        start_time = time.time()
        
        # Initialize solution
        if initial_solution is not None:
            current_solution = list(initial_solution)
        else:
            current_solution = problem.random_solution()
        
        current_energy = problem.evaluate_solution(current_solution)
        
        # Track best solution
        best_solution = current_solution.copy()
        best_energy = current_energy
        
        # Reset tracking variables
        self.current_temperature = self.initial_temperature
        self.accepted_moves = 0
        self.total_moves = 0
        self.energy_history = [current_energy]
        
        iteration = 0
        temperature_moves = 0
        
        print(f"Starting molecular conformation optimization...")
        print(f"Initial energy: {current_energy:.3f} kcal/mol")
        print(f"Initial temperature: {self.current_temperature:.2f}")
        
        while (iteration < self.max_iterations and 
               self.current_temperature > self.final_temperature):
            
            # Generate new candidate solution
            candidate_solution = self._generate_candidate_move(current_solution, problem)
            candidate_energy = problem.evaluate_solution(candidate_solution)
            
            # Accept or reject move
            if self._accept_move(current_energy, candidate_energy):
                current_solution = candidate_solution
                current_energy = candidate_energy
                self.accepted_moves += 1
                
                # Update best solution
                if current_energy < best_energy:
                    best_solution = current_solution.copy()
                    best_energy = current_energy
                    print(f"New best energy: {best_energy:.3f} kcal/mol at iteration {iteration}")
            
            self.total_moves += 1
            temperature_moves += 1
            self.energy_history.append(current_energy)
            
            # Cool down temperature
            if temperature_moves >= self.moves_per_temperature:
                self._update_temperature()
                self._adapt_step_size()
                temperature_moves = 0
                
                if iteration % 1000 == 0:
                    acceptance_rate = self.accepted_moves / max(1, self.total_moves)
                    print(f"Iteration {iteration}: T={self.current_temperature:.2f}, "
                          f"Energy={current_energy:.3f}, Acceptance={acceptance_rate:.2%}")
            
            iteration += 1
            
            # Check for convergence
            if self._check_convergence():
                print(f"Converged at iteration {iteration}")
                break
        
        runtime = time.time() - start_time
        
        # Create detailed result
        final_info = problem.get_solution_info(best_solution) if hasattr(problem, 'get_solution_info') else {}
        
        print(f"Optimization completed in {runtime:.2f} seconds")
        print(f"Best energy found: {best_energy:.3f} kcal/mol")
        print(f"Total moves: {self.total_moves}, Accepted: {self.accepted_moves}")
        print(f"Final acceptance rate: {self.accepted_moves/max(1,self.total_moves):.2%}")
        
        return OptimizationResult(
            best_solution=best_solution,
            best_value=best_energy,
            runtime_seconds=runtime,
            iterations=iteration,
            termination_reason="converged" if self._check_convergence() else "max_iterations",
            additional_metrics={
                "final_temperature": self.current_temperature,
                "total_moves": self.total_moves,
                "accepted_moves": self.accepted_moves,
                "acceptance_rate": self.accepted_moves / max(1, self.total_moves),
                "energy_history": self.energy_history[-100:],  # Last 100 energies
                "molecular_info": final_info
            }
        )
    
    def _generate_candidate_move(self, current_solution: List[float], problem: BaseProblem) -> List[float]:
        """Generate a new candidate solution using chemistry-aware moves."""
        candidate = current_solution.copy()
        n_dihedrals = len(candidate)
        
        # Select move type based on probabilities
        move_type = np.random.choice(
            list(self.move_type_probabilities.keys()),
            p=list(self.move_type_probabilities.values())
        )
        
        if move_type == "single_angle":
            # Modify a single dihedral angle
            angle_idx = random.randint(0, n_dihedrals - 1)
            perturbation = np.random.normal(0, self.step_size)
            candidate[angle_idx] = (candidate[angle_idx] + perturbation) % 360.0
            
        elif move_type == "coupled_angles" and n_dihedrals > 1:
            # Modify two adjacent dihedral angles (common in chemistry)
            angle_idx = random.randint(0, n_dihedrals - 2)
            perturbation1 = np.random.normal(0, self.step_size)
            perturbation2 = np.random.normal(0, self.step_size * 0.5)  # Smaller for coupled
            
            candidate[angle_idx] = (candidate[angle_idx] + perturbation1) % 360.0
            candidate[angle_idx + 1] = (candidate[angle_idx + 1] + perturbation2) % 360.0
            
        elif move_type == "random_walk":
            # Random walk move - modify all angles slightly
            for i in range(n_dihedrals):
                perturbation = np.random.normal(0, self.step_size * 0.3)
                candidate[i] = (candidate[i] + perturbation) % 360.0
        
        return candidate
    
    def _accept_move(self, current_energy: float, candidate_energy: float) -> bool:
        """Determine whether to accept a move using Metropolis criterion."""
        if candidate_energy < current_energy:
            return True
        
        # Calculate acceptance probability
        energy_diff = candidate_energy - current_energy
        probability = math.exp(-energy_diff / self.current_temperature)
        
        return random.random() < probability
    
    def _update_temperature(self):
        """Update temperature using exponential cooling schedule."""
        self.current_temperature *= self.cooling_rate
    
    def _adapt_step_size(self):
        """Adapt step size based on recent acceptance rate."""
        if not self.adaptive_step_size:
            return
        
        if self.total_moves > 0:
            recent_acceptance = self.accepted_moves / self.total_moves
            
            # Adjust step size to maintain ~40% acceptance rate
            target_acceptance = 0.4
            if recent_acceptance > target_acceptance * 1.2:
                self.step_size *= 1.1  # Increase step size
            elif recent_acceptance < target_acceptance * 0.8:
                self.step_size *= 0.9  # Decrease step size
            
            # Keep step size within reasonable bounds
            self.step_size = max(1.0, min(self.step_size, self.initial_step_size * 3))
    
    def _check_convergence(self) -> bool:
        """Check if the optimization has converged."""
        if len(self.energy_history) < self.convergence_window:
            return False
        
        # Check if energy has stabilized
        recent_energies = self.energy_history[-self.convergence_window:]
        energy_std = np.std(recent_energies)
        
        return energy_std < self.energy_tolerance
