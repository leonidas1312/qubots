# Audio Optimization Optimizer

A comprehensive multi-algorithm optimizer for audio signal processing enhancement in the Qubots framework. Uses advanced optimization methods including Genetic Algorithms, Particle Swarm Optimization, Differential Evolution, and SLSQP with automatic algorithm selection and real-time visualization.

## 🎯 Overview

This optimizer solves audio signal enhancement problems using multiple state-of-the-art algorithms:

- **Genetic Algorithm (GA)**: Population-based evolutionary optimization for complex audio parameter spaces
- **Particle Swarm Optimization (PSO)**: Swarm intelligence for continuous audio parameter optimization
- **Differential Evolution (DE)**: Robust global optimization for audio signal processing
- **SLSQP**: Sequential Least Squares Programming for fast, gradient-based optimization
- **Automatic Selection**: Intelligent algorithm choice based on problem characteristics

## 🔧 Algorithms

### Genetic Algorithm
- **Best for**: Large problems with many audio signals (>10)
- **Strengths**: Global optimization, handles discrete parameters well
- **Features**: Tournament selection, blend crossover, Gaussian mutation

### Particle Swarm Optimization
- **Best for**: Continuous audio parameter optimization
- **Strengths**: Fast convergence, good exploration-exploitation balance
- **Features**: Adaptive inertia, cognitive and social learning

### Differential Evolution
- **Best for**: Medium-sized problems (3-10 signals)
- **Strengths**: Robust global optimization, parameter-free
- **Features**: Mutation, crossover, and selection operations

### SLSQP
- **Best for**: Small problems (≤3 signals)
- **Strengths**: Fast local optimization, handles constraints well
- **Features**: Gradient-based, sequential quadratic programming

### Automatic Selection
The optimizer automatically selects the best algorithm based on:
- Number of audio signals in the problem
- Problem complexity and constraint structure
- Expected performance characteristics


## 📊 Visualization Features

The optimizer provides comprehensive visualization:

### Convergence Plot
- Real-time optimization progress
- Logarithmic scale for better visualization
- Convergence detection indicators

### Signal Quality Comparison
- Before/after quality scores for each signal
- Bar chart comparison
- Quality improvement metrics

### Processing Parameters Heatmap
- Visual representation of optimized parameters
- Color-coded parameter values
- Easy identification of parameter patterns

### Objective Components Breakdown
- Individual objective scores (quality, noise, distortion)
- Multi-objective optimization visualization
- Component contribution analysis


## 📊 Result Format

The optimizer returns an `AudioOptimizationResult` with:

```python
{
    "best_solution": {
        "signal_parameters": [
            {
                "signal_id": "signal_001",
                "gain": 1.2,
                "eq_low": -2.0,
                "eq_mid": 3.5,
                "eq_high": -1.0,
                "noise_gate_threshold": -35.0,
                "compressor_ratio": 3.0,
                "phase_correction": 15.0,
                "filter_cutoff": 8000.0
            }
            # ... more signals
        ]
    },
    "best_value": -0.856,                    # Negative quality score
    "quality_improvement": 0.125,            # Average quality improvement
    "noise_reduction": 0.342,               # Average noise reduction
    "distortion_reduction": 0.198,          # Average distortion reduction
    "algorithm_used": "differential_evolution",
    "runtime_seconds": 45.2,
    "iterations": 1247,
    "convergence_history": [...]             # Optimization progress
}
```

## ⚙️ Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm` | string | "auto" | Optimization algorithm to use |
| `max_iterations` | integer | 1000 | Maximum number of iterations |
| `population_size` | integer | 50 | Population size for population-based algorithms |
| `tolerance` | number | 1e-6 | Convergence tolerance |
| `time_limit` | number | 120.0 | Maximum optimization time (seconds) |
| `random_seed` | integer | 42 | Random seed for reproducibility |
| `create_plots` | boolean | true | Enable visualization plots |
| `convergence_window` | integer | 50 | Window size for convergence detection |

## 🎵 Compatible Problems

This optimizer works with:
- Audio signal enhancement problems
- Music production optimization
- Speech processing enhancement
- Podcast audio optimization
- Live sound processing
- Audio restoration problems


