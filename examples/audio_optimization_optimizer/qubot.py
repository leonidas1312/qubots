"""
Audio Optimization Optimizer for Qubots Framework

This optimizer implements advanced signal processing optimization algorithms
for audio enhancement problems. Uses multiple optimization strategies including
genetic algorithms, particle swarm optimization, and gradient-based methods
with automatic algorithm selection.

Features:
- Multiple optimization algorithms (GA, PSO, Differential Evolution, SLSQP)
- Automatic algorithm selection based on problem characteristics
- Real-time visualization of optimization progress
- Audio quality metrics tracking
- Signal processing parameter optimization
- Constraint handling for audio processing bounds

Compatible with Rastion platform workflow automation and local development.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from qubots import (
    BaseOptimizer, OptimizerMetadata, OptimizerType, OptimizerFamily,
    OptimizationResult, BaseProblem
)


@dataclass
class AudioOptimizationResult:
    """Extended result class for audio optimization with detailed metrics."""
    best_solution: Dict[str, Any]
    best_value: float
    iterations: int
    runtime_seconds: float
    termination_reason: str
    algorithm_used: str
    quality_improvement: float
    noise_reduction: float
    distortion_reduction: float
    signal_analyses: List[Dict[str, Any]]
    convergence_history: List[float]


class AudioOptimizationOptimizer(BaseOptimizer):
    """
    Audio Signal Processing Optimizer using multiple algorithms.
    
    Supports:
    - Genetic Algorithm (GA) - for global optimization with discrete parameters
    - Particle Swarm Optimization (PSO) - for continuous parameter optimization
    - Differential Evolution (DE) - for robust global optimization
    - SLSQP - for gradient-based local optimization
    - Automatic algorithm selection based on problem characteristics
    
    Features:
    - Multi-objective optimization (quality, noise, distortion)
    - Real-time convergence visualization
    - Audio processing constraint handling
    - Signal quality metrics tracking
    """
    
    def __init__(self,
                 algorithm: str = "auto",
                 max_iterations: int = 1000,
                 population_size: int = 50,
                 tolerance: float = 1e-6,
                 time_limit: float = 120.0,
                 random_seed: int = 42,
                 create_plots: bool = True,
                 convergence_window: int = 50,
                 **kwargs):
        """
        Initialize audio optimization optimizer.
        
        Args:
            algorithm: Optimization algorithm ("auto", "genetic", "pso", "differential_evolution", "slsqp")
            max_iterations: Maximum number of iterations
            population_size: Population size for population-based algorithms
            tolerance: Convergence tolerance
            time_limit: Maximum optimization time in seconds
            random_seed: Random seed for reproducibility
            create_plots: Whether to create visualization plots
            convergence_window: Window size for convergence detection
            **kwargs: Additional algorithm-specific parameters
        """
        self.algorithm = algorithm
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.tolerance = tolerance
        self.time_limit = time_limit
        self.random_seed = random_seed
        self.create_plots = create_plots
        self.convergence_window = convergence_window
        
        # Set random seed
        np.random.seed(self.random_seed)
        
        # Store parameters for result
        self._parameters = {
            'algorithm': algorithm,
            'max_iterations': max_iterations,
            'population_size': population_size,
            'tolerance': tolerance,
            'time_limit': time_limit,
            'random_seed': random_seed
        }
        
        # Initialize optimizer metadata
        super().__init__()

    def _get_default_metadata(self):
        """Return default metadata for the audio optimization optimizer."""
        return OptimizerMetadata(
            name="Audio Optimization Optimizer",
            description="Multi-algorithm optimizer for audio signal processing enhancement",
            optimizer_type=OptimizerType.METAHEURISTIC,
            optimizer_family=OptimizerFamily.EVOLUTIONARY,
            author="Qubots Framework",
            version="1.0.0",
            is_deterministic=False,
            supports_constraints=True,
            supports_multi_objective=True,
            supports_continuous=True,
            supports_discrete=False,
            supports_mixed_integer=False,
            time_complexity="O(iterations * population_size)",
            space_complexity="O(population_size * dimension)",
            convergence_guaranteed=False,
            parallel_capable=False
        )
    
    def _optimize_implementation(self, problem: BaseProblem, initial_solution: Optional[Any] = None) -> OptimizationResult:
        """
        Core optimization implementation with automatic algorithm selection.
        
        Args:
            problem: Audio optimization problem instance
            initial_solution: Optional initial solution
            
        Returns:
            OptimizationResult with audio optimization details
        """
        # Get detailed audio result
        audio_result = self._optimize_audio(problem, initial_solution)
        
        # Convert to base OptimizationResult
        return OptimizationResult(
            best_solution=audio_result.best_solution,
            best_value=audio_result.best_value,
            iterations=audio_result.iterations,
            runtime_seconds=audio_result.runtime_seconds,
            termination_reason=audio_result.termination_reason,
            optimization_history=audio_result.convergence_history,
            parameter_values=self._parameters,
            additional_metrics={
                'algorithm_used': audio_result.algorithm_used,
                'quality_improvement': audio_result.quality_improvement,
                'noise_reduction': audio_result.noise_reduction,
                'distortion_reduction': audio_result.distortion_reduction
            }
        )

    def optimize(self, problem) -> AudioOptimizationResult:
        """Public interface that returns detailed audio optimization result."""
        return self._optimize_audio(problem)
    
    def _optimize_audio(self, problem: BaseProblem, initial_solution: Optional[Any] = None) -> AudioOptimizationResult:
        """
        Main audio optimization method with algorithm selection.
        
        Args:
            problem: Audio optimization problem instance
            initial_solution: Optional initial solution
            
        Returns:
            AudioOptimizationResult with detailed metrics
        """
        start_time = time.time()
        
        # Select algorithm automatically if needed
        selected_algorithm = self._select_algorithm(problem)
        
        # Initialize tracking
        convergence_history = []
        best_value = float('inf')
        best_solution = None
        iterations = 0
        
        try:
            # Run optimization based on selected algorithm
            if selected_algorithm == "genetic":
                result = self._run_genetic_algorithm(problem, convergence_history)
            elif selected_algorithm == "pso":
                result = self._run_particle_swarm(problem, convergence_history)
            elif selected_algorithm == "differential_evolution":
                result = self._run_differential_evolution(problem, convergence_history)
            elif selected_algorithm == "slsqp":
                result = self._run_slsqp(problem, convergence_history)
            else:
                raise ValueError(f"Unknown algorithm: {selected_algorithm}")
            
            best_solution = result['x']
            best_value = result['fun']
            iterations = result.get('nit', len(convergence_history))
            termination_reason = result.get('message', 'Optimization completed')
            
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            # Fallback to random solution
            best_solution = problem.random_solution()
            best_value = problem.evaluate_solution(best_solution)
            termination_reason = f"Optimization failed: {e}"
        
        runtime = time.time() - start_time
        
        # Get detailed solution analysis
        solution_info = problem.get_solution_info(best_solution)
        
        # Create visualization plots if requested
        if self.create_plots and convergence_history:
            self._create_optimization_plots(convergence_history, solution_info, selected_algorithm)
        
        return AudioOptimizationResult(
            best_solution=best_solution,
            best_value=best_value,
            iterations=iterations,
            runtime_seconds=runtime,
            termination_reason=termination_reason,
            algorithm_used=selected_algorithm,
            quality_improvement=solution_info.get('average_quality_improvement', 0.0),
            noise_reduction=np.mean([s['noise_reduction'] for s in solution_info.get('signal_analyses', [])]),
            distortion_reduction=np.mean([s['distortion_reduction'] for s in solution_info.get('signal_analyses', [])]),
            signal_analyses=solution_info.get('signal_analyses', []),
            convergence_history=convergence_history
        )

    def _select_algorithm(self, problem: BaseProblem) -> str:
        """
        Automatically select the best algorithm based on problem characteristics.

        Args:
            problem: Audio optimization problem instance

        Returns:
            Selected algorithm name
        """
        if self.algorithm != "auto":
            return self.algorithm

        # Get problem characteristics
        n_signals = len(problem.signals) if hasattr(problem, 'signals') else 5

        # Algorithm selection logic
        if n_signals <= 3:
            return "slsqp"  # Fast gradient-based for small problems
        elif n_signals <= 10:
            return "differential_evolution"  # Robust global optimization
        else:
            return "genetic"  # Population-based for large problems

    def _run_genetic_algorithm(self, problem: BaseProblem, convergence_history: List[float]) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization.

        Args:
            problem: Audio optimization problem instance
            convergence_history: List to store convergence values

        Returns:
            Optimization result dictionary
        """
        # Initialize population
        population = []
        for _ in range(self.population_size):
            individual = problem.random_solution()
            population.append(individual)

        best_individual = None
        best_fitness = float('inf')

        for generation in range(self.max_iterations):
            # Evaluate population
            fitness_scores = []
            for individual in population:
                fitness = problem.evaluate_solution(individual)
                fitness_scores.append(fitness)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()

            convergence_history.append(best_fitness)

            # Check convergence
            if len(convergence_history) > self.convergence_window:
                recent_improvement = (convergence_history[-self.convergence_window] -
                                    convergence_history[-1])
                if recent_improvement < self.tolerance:
                    break

            # Selection, crossover, and mutation
            new_population = []

            # Elitism: keep best individuals
            elite_size = max(1, self.population_size // 10)
            elite_indices = np.argsort(fitness_scores)[:elite_size]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                # Crossover
                child = self._crossover_audio_solution(parent1, parent2)

                # Mutation
                child = self._mutate_audio_solution(child, problem)

                new_population.append(child)

            population = new_population

        return {
            'x': best_individual,
            'fun': best_fitness,
            'nit': len(convergence_history),
            'message': 'Genetic algorithm optimization completed'
        }

    def _run_differential_evolution(self, problem: BaseProblem, convergence_history: List[float]) -> Dict[str, Any]:
        """
        Run differential evolution optimization.

        Args:
            problem: Audio optimization problem instance
            convergence_history: List to store convergence values

        Returns:
            Optimization result dictionary
        """
        # Create bounds for differential evolution
        bounds = self._create_bounds_for_de(problem)

        # Callback to track convergence
        def callback(xk, convergence):
            fitness = problem.evaluate_solution(self._de_vector_to_solution(xk, problem))
            convergence_history.append(fitness)

        # Run differential evolution
        result = differential_evolution(
            func=lambda x: problem.evaluate_solution(self._de_vector_to_solution(x, problem)),
            bounds=bounds,
            maxiter=self.max_iterations,
            popsize=max(15, self.population_size // 10),
            tol=self.tolerance,
            seed=self.random_seed,
            callback=callback
        )

        # Convert result back to audio solution format
        best_solution = self._de_vector_to_solution(result.x, problem)

        return {
            'x': best_solution,
            'fun': result.fun,
            'nit': result.nit,
            'message': result.message
        }

    def _run_slsqp(self, problem: BaseProblem, convergence_history: List[float]) -> Dict[str, Any]:
        """
        Run SLSQP optimization.

        Args:
            problem: Audio optimization problem instance
            convergence_history: List to store convergence values

        Returns:
            Optimization result dictionary
        """
        # Create initial solution and bounds
        initial_solution = problem.random_solution()
        bounds = self._create_bounds_for_slsqp(problem)
        initial_vector = self._solution_to_vector(initial_solution)

        # Callback to track convergence
        def callback(xk):
            solution = self._vector_to_solution(xk, problem)
            fitness = problem.evaluate_solution(solution)
            convergence_history.append(fitness)

        # Run SLSQP
        result = minimize(
            fun=lambda x: problem.evaluate_solution(self._vector_to_solution(x, problem)),
            x0=initial_vector,
            method='SLSQP',
            bounds=bounds,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.tolerance,
                'disp': False
            },
            callback=callback
        )

        # Convert result back to audio solution format
        best_solution = self._vector_to_solution(result.x, problem)

        return {
            'x': best_solution,
            'fun': result.fun,
            'nit': result.nit,
            'message': result.message
        }

    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float], tournament_size: int = 3) -> Dict:
        """Tournament selection for genetic algorithm."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()

    def _crossover_audio_solution(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover operation for audio solutions."""
        child = {'signal_parameters': []}

        for i, (p1_params, p2_params) in enumerate(zip(parent1['signal_parameters'], parent2['signal_parameters'])):
            child_params = {}
            for key in p1_params:
                if key == 'signal_id':
                    child_params[key] = p1_params[key]
                else:
                    # Blend crossover
                    alpha = np.random.random()
                    child_params[key] = alpha * p1_params[key] + (1 - alpha) * p2_params[key]
            child['signal_parameters'].append(child_params)

        return child

    def _mutate_audio_solution(self, solution: Dict, problem: BaseProblem, mutation_rate: float = 0.1) -> Dict:
        """Mutation operation for audio solutions."""
        mutated = {'signal_parameters': []}

        for params in solution['signal_parameters']:
            mutated_params = {}
            for key, value in params.items():
                if key == 'signal_id':
                    mutated_params[key] = value
                elif np.random.random() < mutation_rate:
                    # Add Gaussian noise
                    noise_std = abs(value) * 0.1  # 10% of current value
                    mutated_params[key] = value + np.random.normal(0, noise_std)
                else:
                    mutated_params[key] = value
            mutated['signal_parameters'].append(mutated_params)

        return mutated

    def _create_bounds_for_de(self, problem: BaseProblem) -> List[tuple]:
        """Create bounds for differential evolution."""
        bounds = []
        n_signals = len(problem.signals) if hasattr(problem, 'signals') else 5

        # For each signal, create bounds for all parameters
        for _ in range(n_signals):
            bounds.extend([
                (0.1, 2.0),      # gain
                (-12.0, 12.0),   # eq_low
                (-12.0, 12.0),   # eq_mid
                (-12.0, 12.0),   # eq_high
                (-60.0, -20.0),  # noise_gate_threshold
                (1.0, 10.0),     # compressor_ratio
                (-180.0, 180.0), # phase_correction
                (20.0, 22050.0)  # filter_cutoff
            ])

        return bounds

    def _create_bounds_for_slsqp(self, problem: BaseProblem) -> List[tuple]:
        """Create bounds for SLSQP optimization."""
        return self._create_bounds_for_de(problem)

    def _de_vector_to_solution(self, vector: np.ndarray, problem: BaseProblem) -> Dict[str, Any]:
        """Convert DE vector to audio solution format."""
        solution = {'signal_parameters': []}
        n_signals = len(problem.signals) if hasattr(problem, 'signals') else 5
        params_per_signal = 8

        for i in range(n_signals):
            start_idx = i * params_per_signal
            signal_id = problem.signals[i].signal_id if hasattr(problem, 'signals') else f'signal_{i+1:03d}'

            params = {
                'signal_id': signal_id,
                'gain': vector[start_idx],
                'eq_low': vector[start_idx + 1],
                'eq_mid': vector[start_idx + 2],
                'eq_high': vector[start_idx + 3],
                'noise_gate_threshold': vector[start_idx + 4],
                'compressor_ratio': vector[start_idx + 5],
                'phase_correction': vector[start_idx + 6],
                'filter_cutoff': vector[start_idx + 7]
            }
            solution['signal_parameters'].append(params)

        return solution

    def _solution_to_vector(self, solution: Dict[str, Any]) -> np.ndarray:
        """Convert audio solution to vector format."""
        vector = []
        for params in solution['signal_parameters']:
            vector.extend([
                params['gain'],
                params['eq_low'],
                params['eq_mid'],
                params['eq_high'],
                params['noise_gate_threshold'],
                params['compressor_ratio'],
                params['phase_correction'],
                params['filter_cutoff']
            ])
        return np.array(vector)

    def _vector_to_solution(self, vector: np.ndarray, problem: BaseProblem) -> Dict[str, Any]:
        """Convert vector to audio solution format."""
        return self._de_vector_to_solution(vector, problem)

    def _create_optimization_plots(self, convergence_history: List[float],
                                 solution_info: Dict[str, Any], algorithm: str):
        """Create visualization plots for optimization results."""
        try:
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")

            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Audio Optimization Results - {algorithm.upper()}', fontsize=16, fontweight='bold')

            # Plot 1: Convergence history
            ax1.plot(convergence_history, linewidth=2, color='#2E86AB')
            ax1.set_title('Optimization Convergence', fontweight='bold')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Objective Value')
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')

            # Plot 2: Signal quality comparison
            if 'signal_analyses' in solution_info:
                signal_ids = [s['signal_id'] for s in solution_info['signal_analyses']]
                original_quality = [s['original_quality'] for s in solution_info['signal_analyses']]
                processed_quality = [s['processed_quality'] for s in solution_info['signal_analyses']]

                x = np.arange(len(signal_ids))
                width = 0.35

                ax2.bar(x - width/2, original_quality, width, label='Original', alpha=0.8, color='#A23B72')
                ax2.bar(x + width/2, processed_quality, width, label='Optimized', alpha=0.8, color='#F18F01')

                ax2.set_title('Signal Quality Comparison', fontweight='bold')
                ax2.set_xlabel('Signal ID')
                ax2.set_ylabel('Quality Score')
                ax2.set_xticks(x)
                ax2.set_xticklabels(signal_ids, rotation=45)
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            # Plot 3: Processing parameters heatmap
            if 'signal_analyses' in solution_info:
                param_names = ['Gain', 'EQ Low', 'EQ Mid', 'EQ High', 'Noise Gate', 'Compressor', 'Phase', 'Filter']
                param_data = []

                for analysis in solution_info['signal_analyses']:
                    params = analysis['processing_parameters']
                    param_values = [
                        params['gain'],
                        params['eq_low'],
                        params['eq_mid'],
                        params['eq_high'],
                        params['noise_gate_threshold'],
                        params['compressor_ratio'],
                        params['phase_correction'],
                        params['filter_cutoff'] / 1000  # Convert to kHz for better visualization
                    ]
                    param_data.append(param_values)

                if param_data:
                    param_matrix = np.array(param_data).T
                    im = ax3.imshow(param_matrix, cmap='RdYlBu_r', aspect='auto')
                    ax3.set_title('Processing Parameters Heatmap', fontweight='bold')
                    ax3.set_xlabel('Signal ID')
                    ax3.set_ylabel('Parameter')
                    ax3.set_yticks(range(len(param_names)))
                    ax3.set_yticklabels(param_names)
                    ax3.set_xticks(range(len(signal_ids)))
                    ax3.set_xticklabels(signal_ids, rotation=45)
                    plt.colorbar(im, ax=ax3)

            # Plot 4: Objective breakdown
            if 'signal_analyses' in solution_info:
                quality_scores = [s['processed_quality'] for s in solution_info['signal_analyses']]
                noise_scores = [s['noise_reduction'] for s in solution_info['signal_analyses']]
                distortion_scores = [s['distortion_reduction'] for s in solution_info['signal_analyses']]

                x = np.arange(len(signal_ids))
                width = 0.25

                ax4.bar(x - width, quality_scores, width, label='Quality', alpha=0.8, color='#2E86AB')
                ax4.bar(x, noise_scores, width, label='Noise Reduction', alpha=0.8, color='#A23B72')
                ax4.bar(x + width, distortion_scores, width, label='Distortion Reduction', alpha=0.8, color='#F18F01')

                ax4.set_title('Objective Components', fontweight='bold')
                ax4.set_xlabel('Signal ID')
                ax4.set_ylabel('Score')
                ax4.set_xticks(x)
                ax4.set_xticklabels(signal_ids, rotation=45)
                ax4.legend()
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            warnings.warn(f"Failed to create plots: {e}")

