# Audio Optimization Problem

An audio signal enhancement optimization problem implementation for the Qubots framework. This problem reads audio signal data from CSV files and optimizes signal processing parameters to maximize audio quality while minimizing noise and distortion.

## 🎵 Problem Description

The audio optimization problem implements a multi-objective signal processing optimization:

- **Primary Objective**: Maximize overall audio quality score
- **Secondary Objectives**: 
  - Minimize noise levels
  - Minimize harmonic distortion
- **Constraints**: 
  - Signal gain bounds (0.1 ≤ gain ≤ 2.0)
  - EQ adjustment limits (±12 dB)
  - Noise gate threshold range (-60 to -20 dB)
  - Compression ratio limits (1.0 to 10.0)
  - Filter cutoff frequency bounds (20 Hz to Nyquist frequency)

## 🎛️ Signal Processing Parameters

The optimization controls the following audio processing parameters for each signal:

- **Gain**: Overall signal amplification (0.1 - 2.0x)
- **EQ (3-band)**: Low, mid, and high frequency adjustments (±12 dB)
- **Noise Gate**: Threshold for noise reduction (-60 to -20 dB)
- **Compressor**: Dynamic range compression ratio (1.0 - 10.0)
- **Phase Correction**: Phase adjustment in degrees (±180°)
- **Filter Cutoff**: Low-pass filter cutoff frequency (Hz)

## 📁 CSV Input Format

The problem accepts CSV data with the following columns:

### Required Columns
- `signal_id`: Signal identifier (string)
- `frequency_hz`: Primary signal frequency in Hz (float)
- `amplitude`: Signal amplitude 0-1 (float)
- `phase_deg`: Phase shift in degrees (float)
- `noise_level`: Noise level 0-1 (float)
- `snr_db`: Signal-to-noise ratio in dB (float)
- `thd_percent`: Total harmonic distortion percentage (float)
- `bandwidth_hz`: Signal bandwidth in Hz (float)
- `sample_rate_hz`: Sample rate in Hz (float)
- `duration_ms`: Signal duration in milliseconds (float)
- `target_quality`: Target quality score 0-1 (float)

### Example CSV
```csv
signal_id,frequency_hz,amplitude,phase_deg,noise_level,snr_db,thd_percent,bandwidth_hz,sample_rate_hz,duration_ms,target_quality
signal_001,440.0,0.8,0,0.05,26.02,0.5,20,44100,1000,0.95
signal_002,880.0,0.7,45,0.08,21.94,0.8,25,44100,1000,0.92
signal_003,1760.0,0.6,90,0.12,16.48,1.2,30,44100,1000,0.88
```

## 📊 Solution Analysis

Get detailed solution information:

```python
solution = problem.random_solution()
info = problem.get_solution_info(solution)

print(f"Overall quality score: {info['overall_quality_score']:.4f}")
print(f"Quality improvement: {info['average_quality_improvement']:.4f}")
print(f"Is feasible: {info['is_feasible']}")

# Analyze individual signals
for signal_info in info['signal_analyses']:
    print(f"Signal {signal_info['signal_id']}: "
          f"Quality {signal_info['processed_quality']:.3f}, "
          f"Noise reduction {signal_info['noise_reduction']:.3f}")
```

## ⚙️ Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_snr_db` | number | 20.0 | Target signal-to-noise ratio in dB |
| `max_thd_percent` | number | 1.0 | Maximum allowed THD percentage |
| `quality_weight` | number | 0.5 | Weight for quality objective |
| `noise_weight` | number | 0.3 | Weight for noise reduction objective |
| `distortion_weight` | number | 0.2 | Weight for distortion minimization |
| `csv_data` | string | null | CSV content as string |
| `csv_file_path` | string | null | Path to CSV file |

## 🎯 Solution Format

Solutions should be provided as:

```python
{
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
        },
        # ... more signals
    ]
}
```

## 🔧 Compatible Optimizers

This problem works with:
- Genetic algorithms
- Particle swarm optimization
- Simulated annealing
- Gradient-based optimizers
- Multi-objective optimizers (NSGA-II, etc.)

## 📈 Applications

- Music mastering and production
- Speech enhancement
- Podcast audio optimization
- Live sound processing
- Audio restoration
- Broadcast audio quality control
