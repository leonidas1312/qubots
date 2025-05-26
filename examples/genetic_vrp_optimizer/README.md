# Genetic Algorithm VRP Optimizer - Qubots Implementation

A sophisticated Genetic Algorithm optimizer specifically designed for Vehicle Routing Problems (VRP) using the qubots framework, optimized for the Rastion platform playground.

## 🧬 Algorithm Overview

This optimizer uses evolutionary computation principles to solve Vehicle Routing Problems by evolving a population of candidate solutions over multiple generations. The algorithm incorporates advanced genetic operations specifically tailored for VRP solution structures.

### Key Features

- **Population-Based Evolution**: Maintains diverse population of VRP solutions
- **Tournament Selection**: Robust parent selection mechanism
- **Route-Based Crossover**: VRP-specific crossover preserving route structure
- **Multiple Mutation Operators**: Swap, insert, and route exchange mutations
- **Elitism**: Preserves best solutions across generations
- **Adaptive Parameters**: Dynamic adjustment based on population diversity
- **Solution Repair**: Ensures all customers are served exactly once

## 🎯 Algorithm Components

### Selection Strategy
- **Tournament Selection**: Selects parents through competitive tournaments
- **Configurable Tournament Size**: Balance between selection pressure and diversity
- **Elitism**: Automatically preserves top-performing individuals

### Crossover Operations
- **Route-Based Crossover**: Exchanges complete routes between parents
- **Customer Preservation**: Ensures all customers remain in solution
- **Adaptive Crossover Rate**: Adjusts based on evolution progress

### Mutation Operators
1. **Swap Mutation**: Exchanges two customers within or between routes
2. **Insert Mutation**: Moves customer from one route to another
3. **Route Exchange**: Swaps segments between different routes

### Adaptive Mechanisms
- **Diversity Monitoring**: Tracks population diversity over generations
- **Parameter Adaptation**: Adjusts mutation/crossover rates automatically
- **Convergence Detection**: Identifies when to intensify or diversify search

## 🔧 Configuration Parameters

### Core Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `population_size` | 50 | 10-200 | Number of individuals in population |
| `generations` | 100 | 10-1000 | Maximum evolution generations |
| `crossover_rate` | 0.8 | 0.0-1.0 | Probability of crossover operation |
| `mutation_rate` | 0.1 | 0.0-0.5 | Probability of mutation operation |
| `elite_size` | 5 | 1-20 | Best individuals preserved per generation |

### Advanced Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `tournament_size` | 3 | 2-10 | Tournament selection size |
| `diversity_threshold` | 0.1 | 0.01-0.5 | Diversity trigger for adaptation |
| `adaptive_parameters` | true | boolean | Enable parameter adaptation |

## 🚀 Usage Examples

### Basic Usage

```python
from qubot import GeneticVRPOptimizer
import qubots.rastion as rastion

# Load VRP problem
problem = rastion.load_qubots_model("vehicle_routing_problem")

# Create optimizer with default settings
optimizer = GeneticVRPOptimizer()

# Run optimization
result = optimizer.optimize(problem)

print(f"Best solution cost: {result.best_value:.2f}")
print(f"Generations: {result.iterations}")
print(f"Runtime: {result.runtime_seconds:.2f} seconds")
```

### Custom Configuration

```python
# Create optimizer with custom parameters
optimizer = GeneticVRPOptimizer(
    population_size=100,        # Larger population for better diversity
    generations=200,            # More generations for better convergence
    crossover_rate=0.9,         # High crossover rate
    mutation_rate=0.15,         # Higher mutation for exploration
    elite_size=10,              # Preserve more elite solutions
    tournament_size=5,          # Larger tournaments for selection pressure
    adaptive_parameters=True    # Enable adaptive mechanisms
)

result = optimizer.optimize(problem)
```

### Parameter Tuning for Different Problem Sizes

```python
# Small problems (< 20 customers)
small_optimizer = GeneticVRPOptimizer(
    population_size=30,
    generations=50,
    mutation_rate=0.2
)

# Medium problems (20-50 customers)
medium_optimizer = GeneticVRPOptimizer(
    population_size=50,
    generations=100,
    mutation_rate=0.1
)

# Large problems (> 50 customers)
large_optimizer = GeneticVRPOptimizer(
    population_size=100,
    generations=200,
    mutation_rate=0.05,
    elite_size=15
)
```

## 🎮 Rastion Playground Integration

### Loading in Playground

```python
import qubots.rastion as rastion

# Load the genetic VRP optimizer
optimizer = rastion.load_qubots_model("genetic_vrp_optimizer")

# Load a VRP problem
problem = rastion.load_qubots_model("vehicle_routing_problem")

# Run optimization with real-time progress
result = optimizer.optimize(problem)
```

### Real-Time Parameter Adjustment

The playground supports dynamic parameter modification:

- **Population Size**: Adjust for exploration vs. speed trade-off
- **Generations**: Control optimization duration
- **Crossover Rate**: Balance exploitation vs. exploration
- **Mutation Rate**: Fine-tune search diversity
- **Elite Size**: Preserve more/fewer best solutions

### Progress Monitoring

The optimizer provides real-time feedback:

```python
# Access convergence history
convergence = result.convergence_history
diversity = result.diversity_history

# Monitor progress during optimization
print(f"Generation 10: Best fitness = {convergence[10]:.2f}")
print(f"Final diversity: {diversity[-1]:.3f}")
```

## 📊 Performance Analysis

### Convergence Tracking

```python
# Analyze optimization progress
result = optimizer.optimize(problem)

print("Convergence Analysis:")
print(f"  Initial fitness: {result.convergence_history[0]:.2f}")
print(f"  Final fitness: {result.convergence_history[-1]:.2f}")
print(f"  Improvement: {result.convergence_history[0] - result.convergence_history[-1]:.2f}")

# Plot convergence (if matplotlib available)
import matplotlib.pyplot as plt
plt.plot(result.convergence_history)
plt.title("Genetic Algorithm Convergence")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.show()
```

### Diversity Analysis

```python
# Monitor population diversity
diversity_history = result.diversity_history

print("Diversity Analysis:")
print(f"  Initial diversity: {diversity_history[0]:.3f}")
print(f"  Final diversity: {diversity_history[-1]:.3f}")
print(f"  Average diversity: {sum(diversity_history)/len(diversity_history):.3f}")
```

## 🔍 Algorithm Insights

### When to Use This Optimizer

**Best For:**
- Medium to large VRP instances (20+ customers)
- Problems requiring good solution quality over speed
- Scenarios where multiple good solutions are valuable
- Complex VRP variants with additional constraints

**Consider Alternatives For:**
- Very small problems (< 10 customers) - use exact methods
- Real-time applications requiring sub-second response
- Problems with very tight time constraints

### Parameter Guidelines

**For Better Solution Quality:**
- Increase `population_size` (50-100)
- Increase `generations` (200-500)
- Enable `adaptive_parameters`
- Use moderate `mutation_rate` (0.05-0.15)

**For Faster Convergence:**
- Decrease `population_size` (20-40)
- Increase `crossover_rate` (0.9-1.0)
- Increase `elite_size` (10-20% of population)
- Use higher `tournament_size` (5-7)

## 🚀 Advanced Features

### Custom Fitness Functions

The optimizer works with any VRP problem implementing the qubots interface:

```python
# Works with custom VRP variants
custom_vrp = CustomVRPWithTimeWindows()
result = optimizer.optimize(custom_vrp)
```

### Multi-Objective Optimization

```python
# Configure for multi-objective scenarios
multi_obj_optimizer = GeneticVRPOptimizer(
    population_size=100,
    diversity_threshold=0.2,  # Maintain higher diversity
    adaptive_parameters=True
)
```

## 📈 Benchmarking Results

Typical performance on standard VRP instances:

| Problem Size | Population | Generations | Avg. Runtime | Solution Quality |
|--------------|------------|-------------|--------------|------------------|
| 20 customers | 30 | 50 | 5-10s | 95-98% of best known |
| 50 customers | 50 | 100 | 30-60s | 90-95% of best known |
| 100 customers | 100 | 200 | 2-5min | 85-92% of best known |

## 🔗 Compatible Problems

- **Vehicle Routing Problem (VRP)**
- **Capacitated VRP (CVRP)**
- **VRP with Time Windows (VRPTW)**
- **Multi-Depot VRP (MDVRP)**
- **VRP with Pickup and Delivery (VRPPD)**

---

**Algorithm Type**: Metaheuristic (Evolutionary)
**Complexity**: O(G × P × N²) where G=generations, P=population, N=customers
**Memory Usage**: O(P × N) for population storage
**Parallelizable**: Yes (population evaluation can be parallelized)
