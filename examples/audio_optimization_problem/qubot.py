"""
Audio Optimization Problem for Qubots Framework

This problem implements audio signal enhancement optimization that reads
audio signal data from CSV files and optimizes signal processing parameters
to maximize audio quality while minimizing noise and distortion.

The problem accepts CSV data with columns:
- signal_id: Signal identifier
- frequency_hz: Signal frequency in Hz
- amplitude: Signal amplitude (0-1)
- phase_deg: Phase shift in degrees
- noise_level: Noise level (0-1)
- snr_db: Signal-to-noise ratio in dB
- thd_percent: Total harmonic distortion percentage
- bandwidth_hz: Signal bandwidth in Hz
- sample_rate_hz: Sample rate in Hz
- duration_ms: Signal duration in milliseconds
- target_quality: Target quality score (0-1)

Compatible with Rastion platform workflow automation and local development.
"""

import io
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from qubots import (
    BaseProblem, ProblemMetadata, ProblemType, ObjectiveType,
    DifficultyLevel, EvaluationResult
)


@dataclass
class AudioSignal:
    """Data class for audio signal parameters."""
    signal_id: str
    frequency_hz: float
    amplitude: float
    phase_deg: float
    noise_level: float
    snr_db: float
    thd_percent: float
    bandwidth_hz: float
    sample_rate_hz: float
    duration_ms: float
    target_quality: float


class AudioOptimizationProblem(BaseProblem):
    """
    Audio Signal Enhancement Optimization Problem for Qubots Framework.
    
    Optimizes audio signal processing parameters to maximize quality while
    minimizing noise and distortion. The problem formulation includes:
    
    - Objective: Maximize overall audio quality score
    - Constraints: 
      - Signal amplitude bounds (0 ≤ amplitude ≤ 1)
      - Frequency response within bandwidth limits
      - Phase coherence requirements
      - Noise reduction targets
      - Distortion minimization
    
    Features:
    - CSV-based audio signal data input
    - Multi-objective optimization (quality vs. noise vs. distortion)
    - Realistic audio signal processing constraints
    - Support for various audio signal types (music, speech, etc.)
    """
    
    def __init__(self, 
                 csv_data: str = None,
                 csv_file_path: str = None,
                 target_snr_db: float = 20.0,
                 max_thd_percent: float = 1.0,
                 quality_weight: float = 0.5,
                 noise_weight: float = 0.3,
                 distortion_weight: float = 0.2,
                 **kwargs):
        """
        Initialize audio optimization problem.
        
        Args:
            csv_data: CSV content as string
            csv_file_path: Path to CSV file (alternative to csv_data)
            target_snr_db: Target signal-to-noise ratio in dB (default: 20.0)
            max_thd_percent: Maximum allowed THD percentage (default: 1.0)
            quality_weight: Weight for quality objective (default: 0.5)
            noise_weight: Weight for noise reduction objective (default: 0.3)
            distortion_weight: Weight for distortion minimization (default: 0.2)
            **kwargs: Additional parameters
        """
        self.target_snr_db = target_snr_db
        self.max_thd_percent = max_thd_percent
        self.quality_weight = quality_weight
        self.noise_weight = noise_weight
        self.distortion_weight = distortion_weight
        
        # Load audio signal data from CSV
        self.signals = self._load_signal_data(csv_data, csv_file_path)
        self.n_signals = len(self.signals)
        
        # Initialize problem metadata
        super().__init__()

    def _get_default_metadata(self):
        """Return default metadata for the audio optimization problem."""
        return ProblemMetadata(
            name="Audio Optimization Problem",
            description="Audio signal enhancement optimization for quality maximization",
            problem_type=ProblemType.CONTINUOUS,
            objective_type=ObjectiveType.MINIMIZE,  # We minimize negative quality
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            domain="audio_processing",
            tags={"audio", "signal_processing", "optimization", "quality", "noise_reduction"},
            dimension=self.n_signals * 8,  # 8 parameters per signal
            constraints_count=self.n_signals * 6,  # 6 constraints per signal
            evaluation_complexity="O(n)",
            memory_complexity="O(n)"
        )
    
    def _load_signal_data(self, csv_data: str = None, csv_file_path: str = None) -> List[AudioSignal]:
        """Load audio signal data from CSV source."""
        if csv_data:
            # Load from string data
            df = pd.read_csv(io.StringIO(csv_data))
        elif csv_file_path:
            # Load from file path
            df = pd.read_csv(csv_file_path)
        else:
            # Use default sample data
            df = self._create_default_data()
        
        # Validate required columns
        required_cols = ['signal_id', 'frequency_hz', 'amplitude', 'phase_deg', 
                        'noise_level', 'snr_db', 'thd_percent', 'bandwidth_hz',
                        'sample_rate_hz', 'duration_ms', 'target_quality']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert to AudioSignal objects
        signals = []
        for _, row in df.iterrows():
            signal = AudioSignal(
                signal_id=str(row['signal_id']),
                frequency_hz=float(row['frequency_hz']),
                amplitude=float(row['amplitude']),
                phase_deg=float(row['phase_deg']),
                noise_level=float(row['noise_level']),
                snr_db=float(row['snr_db']),
                thd_percent=float(row['thd_percent']),
                bandwidth_hz=float(row['bandwidth_hz']),
                sample_rate_hz=float(row['sample_rate_hz']),
                duration_ms=float(row['duration_ms']),
                target_quality=float(row['target_quality'])
            )
            signals.append(signal)
        
        if len(signals) < 1:
            raise ValueError("Audio optimization requires at least 1 signal")
        
        return signals
    
    def _create_default_data(self) -> pd.DataFrame:
        """Create default sample audio signal data for demonstration."""
        return pd.DataFrame({
            'signal_id': ['signal_001', 'signal_002', 'signal_003', 'signal_004', 'signal_005'],
            'frequency_hz': [440.0, 880.0, 1760.0, 220.0, 110.0],
            'amplitude': [0.8, 0.7, 0.6, 0.9, 0.5],
            'phase_deg': [0, 45, 90, 135, 180],
            'noise_level': [0.05, 0.08, 0.12, 0.03, 0.15],
            'snr_db': [26.02, 21.94, 16.48, 30.46, 10.46],
            'thd_percent': [0.5, 0.8, 1.2, 0.3, 2.0],
            'bandwidth_hz': [20, 25, 30, 15, 35],
            'sample_rate_hz': [44100, 44100, 44100, 44100, 44100],
            'duration_ms': [1000, 1000, 1000, 1000, 1000],
            'target_quality': [0.95, 0.92, 0.88, 0.97, 0.80]
        })

    def random_solution(self) -> Dict[str, Any]:
        """
        Generate a random audio processing solution.

        Returns:
            Dictionary with audio processing parameters for each signal
        """
        solution = {
            'signal_parameters': []
        }

        for signal in self.signals:
            # Generate random processing parameters within reasonable bounds
            signal_params = {
                'signal_id': signal.signal_id,
                'gain': np.random.uniform(0.1, 2.0),  # Gain adjustment
                'eq_low': np.random.uniform(-12.0, 12.0),  # Low frequency EQ (dB)
                'eq_mid': np.random.uniform(-12.0, 12.0),  # Mid frequency EQ (dB)
                'eq_high': np.random.uniform(-12.0, 12.0),  # High frequency EQ (dB)
                'noise_gate_threshold': np.random.uniform(-60.0, -20.0),  # Noise gate (dB)
                'compressor_ratio': np.random.uniform(1.0, 10.0),  # Compression ratio
                'phase_correction': np.random.uniform(-180.0, 180.0),  # Phase correction (degrees)
                'filter_cutoff': np.random.uniform(20.0, signal.sample_rate_hz / 2),  # Filter cutoff (Hz)
            }
            solution['signal_parameters'].append(signal_params)

        return solution

    def evaluate_solution(self, solution: Union[List[Dict], Dict[str, Any]]) -> float:
        """
        Evaluate audio processing solution.

        Args:
            solution: Audio processing parameters as dict with 'signal_parameters' key

        Returns:
            Negative quality score (lower is better for minimization)
        """
        # Extract signal parameters from solution
        if isinstance(solution, dict) and 'signal_parameters' in solution:
            signal_params = solution['signal_parameters']
        elif isinstance(solution, list):
            signal_params = solution
        else:
            return 1e6  # Large penalty for invalid format

        if len(signal_params) != self.n_signals:
            return 1e6  # Large penalty for wrong number of signals

        total_quality = 0.0
        total_penalty = 0.0

        for i, params in enumerate(signal_params):
            signal = self.signals[i]

            # Calculate quality metrics
            quality_score = self._calculate_quality_score(signal, params)
            noise_score = self._calculate_noise_reduction(signal, params)
            distortion_score = self._calculate_distortion_reduction(signal, params)

            # Apply constraint penalties
            penalty = self._calculate_constraint_penalties(signal, params)

            # Weighted combination of objectives
            combined_score = (
                self.quality_weight * quality_score +
                self.noise_weight * noise_score +
                self.distortion_weight * distortion_score
            )

            total_quality += combined_score
            total_penalty += penalty

        # Return negative quality (for minimization) plus penalties
        return -(total_quality / self.n_signals) + total_penalty

    def _calculate_quality_score(self, signal: AudioSignal, params: Dict[str, Any]) -> float:
        """Calculate quality improvement score for a signal."""
        # Base quality from signal characteristics
        base_quality = signal.target_quality

        # Quality improvements from processing
        gain_factor = min(params.get('gain', 1.0), 2.0)  # Cap gain at 2x
        eq_balance = 1.0 - (abs(params.get('eq_low', 0)) +
                           abs(params.get('eq_mid', 0)) +
                           abs(params.get('eq_high', 0))) / 36.0  # Penalty for extreme EQ

        # Phase coherence improvement
        phase_improvement = 1.0 - abs(params.get('phase_correction', 0)) / 180.0

        # Filter effectiveness
        filter_cutoff = params.get('filter_cutoff', signal.frequency_hz)
        filter_effectiveness = 1.0 if filter_cutoff > signal.frequency_hz else 0.5

        quality_improvement = (gain_factor * eq_balance * phase_improvement * filter_effectiveness)
        return base_quality * quality_improvement

    def _calculate_noise_reduction(self, signal: AudioSignal, params: Dict[str, Any]) -> float:
        """Calculate noise reduction score for a signal."""
        # Noise gate effectiveness
        gate_threshold = params.get('noise_gate_threshold', -40.0)
        noise_reduction = max(0, min(1.0, (signal.noise_level * 100 + gate_threshold) / 40.0))

        # Compression helps with noise
        compression_ratio = params.get('compressor_ratio', 1.0)
        compression_benefit = min(0.3, (compression_ratio - 1.0) / 10.0)

        return noise_reduction + compression_benefit

    def _calculate_distortion_reduction(self, signal: AudioSignal, params: Dict[str, Any]) -> float:
        """Calculate distortion reduction score for a signal."""
        # Gain control reduces distortion
        gain = params.get('gain', 1.0)
        gain_distortion = max(0, gain - 1.0) * 0.1  # Penalty for high gain

        # EQ can introduce distortion if extreme
        eq_distortion = (abs(params.get('eq_low', 0)) +
                        abs(params.get('eq_mid', 0)) +
                        abs(params.get('eq_high', 0))) / 36.0

        # Compression can reduce distortion
        compression_ratio = params.get('compressor_ratio', 1.0)
        compression_benefit = min(0.2, (compression_ratio - 1.0) / 20.0)

        base_distortion_reduction = 1.0 - signal.thd_percent / 10.0
        processing_impact = 1.0 - gain_distortion - eq_distortion + compression_benefit

        return max(0, base_distortion_reduction * processing_impact)

    def _calculate_constraint_penalties(self, signal: AudioSignal, params: Dict[str, Any]) -> float:
        """Calculate penalties for constraint violations."""
        penalty = 0.0

        # Gain constraints
        gain = params.get('gain', 1.0)
        if gain < 0.1 or gain > 2.0:
            penalty += 100.0 * abs(gain - np.clip(gain, 0.1, 2.0))

        # EQ constraints (reasonable range)
        for eq_param in ['eq_low', 'eq_mid', 'eq_high']:
            eq_value = params.get(eq_param, 0.0)
            if abs(eq_value) > 12.0:
                penalty += 10.0 * (abs(eq_value) - 12.0)

        # Noise gate constraints
        gate_threshold = params.get('noise_gate_threshold', -40.0)
        if gate_threshold < -60.0 or gate_threshold > -20.0:
            penalty += 50.0 * abs(gate_threshold - np.clip(gate_threshold, -60.0, -20.0))

        # Compression ratio constraints
        compression_ratio = params.get('compressor_ratio', 1.0)
        if compression_ratio < 1.0 or compression_ratio > 10.0:
            penalty += 20.0 * abs(compression_ratio - np.clip(compression_ratio, 1.0, 10.0))

        # Filter cutoff constraints
        filter_cutoff = params.get('filter_cutoff', signal.frequency_hz)
        nyquist_freq = signal.sample_rate_hz / 2
        if filter_cutoff < 20.0 or filter_cutoff > nyquist_freq:
            penalty += 30.0 * abs(filter_cutoff - np.clip(filter_cutoff, 20.0, nyquist_freq))

        return penalty

    def is_feasible(self, solution: Union[List[Dict], Dict[str, Any]]) -> bool:
        """
        Check if solution satisfies all constraints.

        Args:
            solution: Audio processing parameters

        Returns:
            True if solution is feasible, False otherwise
        """
        # Extract signal parameters from solution
        if isinstance(solution, dict) and 'signal_parameters' in solution:
            signal_params = solution['signal_parameters']
        elif isinstance(solution, list):
            signal_params = solution
        else:
            return False

        if len(signal_params) != self.n_signals:
            return False

        for i, params in enumerate(signal_params):
            signal = self.signals[i]

            # Check all constraints
            gain = params.get('gain', 1.0)
            if gain < 0.1 or gain > 2.0:
                return False

            # EQ constraints
            for eq_param in ['eq_low', 'eq_mid', 'eq_high']:
                eq_value = params.get(eq_param, 0.0)
                if abs(eq_value) > 12.0:
                    return False

            # Noise gate constraints
            gate_threshold = params.get('noise_gate_threshold', -40.0)
            if gate_threshold < -60.0 or gate_threshold > -20.0:
                return False

            # Compression ratio constraints
            compression_ratio = params.get('compressor_ratio', 1.0)
            if compression_ratio < 1.0 or compression_ratio > 10.0:
                return False

            # Filter cutoff constraints
            filter_cutoff = params.get('filter_cutoff', signal.frequency_hz)
            nyquist_freq = signal.sample_rate_hz / 2
            if filter_cutoff < 20.0 or filter_cutoff > nyquist_freq:
                return False

        return True

    def get_solution_info(self, solution: Union[List[Dict], Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get detailed information about a solution.

        Args:
            solution: Audio processing parameters

        Returns:
            Dictionary with solution analysis
        """
        # Extract signal parameters from solution
        if isinstance(solution, dict) and 'signal_parameters' in solution:
            signal_params = solution['signal_parameters']
        elif isinstance(solution, list):
            signal_params = solution
        else:
            return {'error': 'Invalid solution format'}

        if len(signal_params) != self.n_signals:
            return {'error': 'Wrong number of signal parameters'}

        total_quality = 0.0
        signal_analyses = []

        for i, params in enumerate(signal_params):
            signal = self.signals[i]

            # Calculate individual scores
            quality_score = self._calculate_quality_score(signal, params)
            noise_score = self._calculate_noise_reduction(signal, params)
            distortion_score = self._calculate_distortion_reduction(signal, params)
            penalty = self._calculate_constraint_penalties(signal, params)

            # Combined score for this signal
            combined_score = (
                self.quality_weight * quality_score +
                self.noise_weight * noise_score +
                self.distortion_weight * distortion_score
            )

            total_quality += combined_score

            signal_analysis = {
                'signal_id': signal.signal_id,
                'original_quality': signal.target_quality,
                'processed_quality': quality_score,
                'noise_reduction': noise_score,
                'distortion_reduction': distortion_score,
                'combined_score': combined_score,
                'constraint_penalty': penalty,
                'processing_parameters': params,
                'is_feasible': penalty == 0
            }
            signal_analyses.append(signal_analysis)

        overall_score = total_quality / self.n_signals

        return {
            'overall_quality_score': overall_score,
            'average_quality_improvement': sum(s['processed_quality'] - s['original_quality']
                                             for s in signal_analyses) / self.n_signals,
            'total_constraint_violations': sum(s['constraint_penalty'] for s in signal_analyses),
            'is_feasible': self.is_feasible(solution),
            'signal_analyses': signal_analyses,
            'objective_weights': {
                'quality': self.quality_weight,
                'noise_reduction': self.noise_weight,
                'distortion_reduction': self.distortion_weight
            }
        }

