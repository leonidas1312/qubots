# Playground Testing Guide

This guide explains how to test your qubots repositories for playground consistency before uploading to Rastion. The testing tools ensure that your optimization will work the same way in the Rastion playground as it does locally.

## Overview

The Rastion playground runs your optimization in a containerized environment with only the dependencies specified in your `requirements.txt` files. This can sometimes lead to differences compared to your local environment where you might have additional packages installed.

## Testing Tools

### 1. `playground_consistency_test.py` - Quick Consistency Check

**Purpose**: Compare local optimization results with playground execution results.

**Usage**:
```bash
# Test TSP problem with HiGHS solver
python playground_consistency_test.py tsp highs_tsp_solver

# Test MaxCut problem with OR-Tools optimizer
python playground_consistency_test.py maxcut_problem ortools_maxcut_optimizer

# Test with custom number of iterations
python playground_consistency_test.py tsp highs_tsp_solver --iterations 5
```

**What it does**:
- Loads your models locally and runs optimization
- Simulates playground execution using `execute_playground_optimization()`
- Compares results and provides consistency analysis
- Gives clear recommendations about upload readiness

### 2. `test_playground_execution.py` - Comprehensive Testing

**Purpose**: Comprehensive testing including container simulation and playground integration.

**Usage**:
```bash
# Full test suite
python test_playground_execution.py tsp highs_tsp_solver --iterations 3

# Test only container simulation
python test_playground_execution.py tsp highs_tsp_solver --container-only

# Test only playground integration
python test_playground_execution.py tsp highs_tsp_solver --playground-only

# Quiet mode
python test_playground_execution.py tsp highs_tsp_solver --quiet
```

**What it does**:
- Creates isolated environments that simulate container conditions
- Tests dependency installation in isolation
- Compares multiple execution methods
- Provides detailed analysis and recommendations

### 3. `run_playground_tests.py` - Test Suite Runner

**Purpose**: Run comprehensive tests across multiple problem-optimizer pairs.

**Usage**:
```bash
# Test all predefined pairs
python run_playground_tests.py

# Test specific pair
python run_playground_tests.py --problem tsp --optimizer highs_tsp_solver

# Run only consistency tests
python run_playground_tests.py --consistency-only

# Custom iterations
python run_playground_tests.py --iterations 5
```

**What it does**:
- Runs multiple test scripts in sequence
- Tests predefined problem-optimizer pairs
- Provides comprehensive summary and recommendations

## Recommended Testing Workflow

### Step 1: Quick Consistency Check
Start with the quick consistency test to get immediate feedback:

```bash
python playground_consistency_test.py your_problem your_optimizer
```

**Expected output**:
- ✅ EXCELLENT: Results are highly consistent - safe to upload to Rastion!
- ✅ GOOD: Results are reasonably consistent - should work well in Rastion
- ⚠️ FAIR: Some variation detected - test thoroughly in Rastion playground
- ❌ POOR: Significant differences detected - investigate before uploading

### Step 2: Comprehensive Testing (if needed)
If the quick test shows variations, run comprehensive testing:

```bash
python test_playground_execution.py your_problem your_optimizer --iterations 3
```

### Step 3: Fix Issues (if any)
Common issues and solutions:

**Missing Dependencies**:
- Add missing packages to `requirements.txt`
- Ensure version compatibility

**Environment Differences**:
- Check for hardcoded paths
- Verify random seed handling
- Review algorithm parameters

**Import Errors**:
- Ensure all imports are available in requirements
- Check for local-only packages

### Step 4: Upload to Rastion
Once tests pass consistently:

```bash
python upload_repo_to_rastion.py your_problem --token YOUR_TOKEN
python upload_repo_to_rastion.py your_optimizer --token YOUR_TOKEN
```

## Understanding Test Results

### Consistency Levels

- **EXCELLENT** (< 1% difference): Results are nearly identical
- **GOOD** (< 5% difference): Minor variations, likely due to randomness
- **FAIR** (< 10% difference): Some differences, investigate causes
- **POOR** (≥ 10% difference): Significant issues, fix before uploading

### Common Causes of Inconsistency

1. **Random Seed Differences**: Algorithms using randomness may produce different results
2. **Missing Dependencies**: Required packages not in requirements.txt
3. **Version Mismatches**: Different package versions between local and container
4. **Environment Variables**: Missing or different environment settings
5. **File Paths**: Hardcoded paths that don't exist in container

## Best Practices

### Repository Structure
Ensure your repository has:
```
your_repo/
├── qubot.py          # Main implementation
├── config.json       # Configuration
├── requirements.txt  # Dependencies
└── README.md         # Documentation
```

### Requirements.txt
- Include all necessary dependencies
- Specify version ranges when needed
- Test with minimal dependencies

### Code Guidelines
- Avoid hardcoded file paths
- Use relative imports
- Handle randomness consistently
- Make algorithms deterministic when possible

## Troubleshooting

### Test Fails to Run
- Check that qubots is installed: `pip install qubots`
- Verify repository structure and config.json
- Ensure Python path is correct

### Inconsistent Results
- Review requirements.txt for missing packages
- Check for random seed handling
- Verify algorithm parameters
- Test with deterministic settings

### Container Simulation Errors
- Check dependency installation logs
- Verify package compatibility
- Review import statements

## Example Test Session

```bash
# Test TSP with HiGHS solver
$ python playground_consistency_test.py tsp highs_tsp_solver

🎯 Qubots Playground Consistency Test
========================================
📁 Problem: tsp
🔧 Optimizer: highs_tsp_solver
🔄 Iterations: 3

📁 Loading local models...
✅ Loaded TSPProblem and HiGHSTSPSolver

🏠 Running local optimization (3 iterations)...
  ✅ Iteration 1: cost=2845.2341, time=0.12s
  ✅ Iteration 2: cost=2834.1234, time=0.11s
  ✅ Iteration 3: cost=2851.3456, time=0.13s

🎮 Running playground optimization (3 iterations)...
  ✅ Iteration 1: cost=2847.1234, time=0.15s
  ✅ Iteration 2: cost=2836.2345, time=0.14s
  ✅ Iteration 3: cost=2849.3456, time=0.16s

======================================================================
🎯 PLAYGROUND CONSISTENCY TEST RESULTS
======================================================================

🏠 LOCAL OPTIMIZATION:
   ✅ Successful: 3
   ❌ Failed: 0
   💰 Average cost: 2843.5677
   ⏱️  Average time: 0.12s

🎮 PLAYGROUND OPTIMIZATION:
   ✅ Successful: 3
   ❌ Failed: 0
   💰 Average cost: 2844.2345
   ⏱️  Average time: 0.15s

📊 CONSISTENCY ANALYSIS:
   💰 Cost difference: 0.02%

💡 RECOMMENDATION:
   ✅ EXCELLENT: Results are highly consistent - safe to upload to Rastion!

======================================================================

🎉 Consistency test PASSED! You can safely upload to Rastion.
```

This indicates your repositories are ready for upload to Rastion!
