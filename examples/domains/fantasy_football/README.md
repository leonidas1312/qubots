# Fantasy Football Optimization Examples

This directory contains comprehensive examples demonstrating how to use the qubots framework for fantasy football optimization, including integration with the Rastion platform.

## 📁 Files Overview

### Core Examples
- **`fantasy_football_optimization_example.py`** - Main comprehensive example
- **`fantasy_football_advanced_demo.py`** - Advanced benchmarking and analysis
- **`README_FANTASY_FOOTBALL.md`** - This documentation file

### Generated Output Files
- **`fantasy_football_benchmark_results.png`** - Performance visualization charts
- **`fantasy_football_benchmark_report.txt`** - Detailed benchmark report

## 🚀 Quick Start

### Prerequisites

1. **Install qubots framework:**
   ```bash
   pip install qubots
   ```

2. **Install optional dependencies for advanced features:**
   ```bash
   pip install matplotlib seaborn ortools
   ```

3. **Set up Rastion authentication (optional):**
   ```python
   import qubots.rastion as rastion
   rastion.authenticate("your_gitea_token_here")
   ```
   Get your token from: https://hub.rastion.com

### Running the Basic Example

```bash
cd examples
python fantasy_football_optimization_example.py
```

This will:
1. Load the fantasy football problem from Rastion (or local fallback)
2. Configure multiple optimization solvers
3. Run optimizations and compare results
4. Display detailed lineup analysis

### Running the Advanced Demo

```bash
cd examples
python fantasy_football_advanced_demo.py
```

This will:
1. Run comprehensive benchmarks across multiple configurations
2. Generate statistical analysis and performance metrics
3. Create visualization charts
4. Generate a detailed report

## 📊 What You'll Learn

### 1. Loading Problems from Rastion Platform

```python
import qubots.rastion as rastion

# Load the fantasy football problem
problem = rastion.load_qubots_model("fantasy_football_problem")

# Access problem properties
print(f"Players: {problem.n_players}")
print(f"Salary cap: ${problem.max_salary:,}")
```

### 2. Configuring Multiple Optimizers

The examples demonstrate several optimization approaches:

- **Random Search**: Simple baseline algorithm
- **Genetic Algorithm**: Evolutionary optimization
- **OR-Tools Integration**: Exact/heuristic solvers (when available)
- **Custom Optimizers**: Domain-specific implementations

### 3. Running Optimization

```python
# Run optimization
result = optimizer.optimize(problem)

# Access results
print(f"Best points: {result.best_value:.2f}")
print(f"Feasible: {result.is_feasible}")
print(f"Runtime: {result.runtime_seconds:.2f}s")
```

### 4. Analyzing Results

```python
# Get detailed lineup information
lineup_df = problem.get_lineup_summary(result.best_solution)
print(lineup_df)

# Calculate statistics
total_salary = lineup_df['DK.salary'].sum()
total_points = lineup_df['DK.points'].sum()
```

## 🔧 Customization Options

### Optimizer Parameters

You can customize optimizer behavior:

```python
# Random Search
optimizer = SimpleRandomSearchOptimizer(n_trials=5000)

# Genetic Algorithm
optimizer = FantasyFootballGeneticOptimizer(
    population_size=100,
    max_generations=200,
    mutation_rate=0.1,
    crossover_rate=0.8
)
```

### Problem Configuration

Modify problem constraints:

```python
# Load with custom parameters
problem = FantasyFootballProblem(
    max_salary=60000,  # Custom salary cap
    excluded_players=["Player Name"]  # Exclude specific players
)
```

### Benchmarking Configuration

Adjust benchmark parameters in the advanced demo:

```python
# Number of independent runs per configuration
num_runs = 5

# Add custom optimizer configurations
configs.append(OptimizerConfig(
    name="Custom Optimizer",
    optimizer_class=YourCustomOptimizer,
    parameters={"param1": value1, "param2": value2},
    description="Your custom optimization approach"
))
```

## 📈 Expected Output

### Basic Example Output

```
🏈 Fantasy Football Optimization with Qubots
============================================================

🔐 Setting up Rastion Authentication
==================================================
✅ Authentication configured (using demo mode)

🏈 Loading Fantasy Football Problem from Rastion
==================================================
✅ Successfully loaded: Fantasy Football Lineup Optimization
   Number of players: 442
   Salary cap: $50,000

⚙️  Configuring Optimization Solvers
==================================================
✅ Random Search optimizer created successfully
✅ Local Genetic Algorithm optimizer created successfully

🚀 Running Random Search Optimization
==================================================
✅ Random Search completed successfully!
   Runtime: 0.45 seconds
   Best value: 156.32 points
   Feasible: Yes

📋 Random Search - Lineup Analysis
==================================================
🏆 Selected Players:
     Name Pos Team  DK.points  DK.salary
0  Player1  QB  KC      25.4       8500
1  Player2  RB  SF      18.2       7200
...

📊 Lineup Statistics:
   Total Players: 9
   Total Salary: $49,800
   Salary Remaining: $200
   Total Projected Points: 156.32
```

### Advanced Demo Output

The advanced demo generates:

1. **Console Output**: Real-time progress and summary statistics
2. **Visualization Charts**: Performance comparison plots
3. **Detailed Report**: Comprehensive analysis in text format

## 🛠️ Troubleshooting

### Common Issues

1. **Rastion Connection Failed**
   - The examples include fallback to local problem instances
   - Ensure internet connectivity for Rastion platform access

2. **Missing Dependencies**
   ```bash
   pip install matplotlib seaborn pandas numpy
   ```

3. **OR-Tools Not Available**
   ```bash
   pip install ortools
   ```

4. **Memory Issues with Large Populations**
   - Reduce `population_size` and `max_generations` parameters
   - Use fewer benchmark runs in advanced demo

### Performance Tips

1. **For Quick Testing**: Use smaller parameter values
   ```python
   optimizer = FantasyFootballGeneticOptimizer(
       population_size=20,
       max_generations=50
   )
   ```

2. **For Best Results**: Use larger parameter values
   ```python
   optimizer = FantasyFootballGeneticOptimizer(
       population_size=100,
       max_generations=200
   )
   ```

3. **For Benchmarking**: Adjust number of runs based on time constraints
   ```python
   num_runs = 3  # Quick benchmark
   num_runs = 10  # Thorough benchmark
   ```

## 🔗 Integration with Other Optimizers

### Using OR-Tools

```python
# Load OR-Tools optimizer from Rastion
try:
    ortools_optimizer = AutoOptimizer.from_repo("Rastion/ortools_optimizer")
    result = ortools_optimizer.optimize(problem)
except Exception as e:
    print(f"OR-Tools not available: {e}")
```

### Using CasADi

```python
# Load CasADi optimizer from Rastion
try:
    casadi_optimizer = AutoOptimizer.from_repo("Rastion/casadi_optimizer")
    result = casadi_optimizer.optimize(problem)
except Exception as e:
    print(f"CasADi not available: {e}")
```

### Creating Custom Optimizers

```python
from qubots import BaseOptimizer, OptimizerMetadata

class MyFantasyOptimizer(BaseOptimizer):
    def __init__(self, **params):
        metadata = OptimizerMetadata(
            name="My Custom Fantasy Optimizer",
            description="Custom optimization for fantasy football",
            # ... other metadata
        )
        super().__init__(metadata, **params)
    
    def _optimize_implementation(self, problem, initial_solution=None):
        # Your optimization logic here
        pass
```

## 📚 Next Steps

1. **Experiment with Parameters**: Try different optimizer configurations
2. **Add Custom Constraints**: Modify the problem for specific requirements
3. **Upload to Rastion**: Share your optimizers with the community
4. **Integrate with Real Data**: Use live player data and projections
5. **Multi-Objective Optimization**: Consider multiple objectives (risk, upside, etc.)

## 🤝 Contributing

To contribute improvements or new optimizers:

1. Fork the repository
2. Create your optimization algorithm
3. Add comprehensive tests and documentation
4. Submit a pull request

## 📞 Support

For questions or issues:
- Check the main qubots documentation
- Visit the Rastion platform: https://hub.rastion.com
- Open an issue in the repository

---

Happy optimizing! 🏈🚀
