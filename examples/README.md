# Qubots Examples

Welcome to the qubots examples directory! This collection demonstrates the power and versatility of the qubots optimization framework across various domains and use cases.

## 📁 Directory Structure

### 🎯 [Basic Examples](basic/)
Fundamental qubots usage patterns and simple optimization problems:
- **Getting Started**: Basic problem and optimizer creation
- **Core Concepts**: Understanding qubots architecture
- **Simple Algorithms**: Random search, hill climbing, basic genetic algorithms

### 🏭 [Domain-Specific Examples](domains/)
Real-world optimization problems organized by domain:

#### 🚛 [Routing and Logistics](domains/routing/)
- **Vehicle Routing Problem (VRP)**: Multi-vehicle delivery optimization
- **Traveling Salesman Problem (TSP)**: Classic route optimization
- **Supply Chain Optimization**: Warehouse and distribution planning
- **Last-Mile Delivery**: Urban delivery optimization

#### ⏰ [Scheduling](domains/scheduling/)
- **Job Shop Scheduling**: Manufacturing and production planning
- **Resource Allocation**: Optimal resource assignment
- **Project Scheduling**: Timeline and dependency management
- **Workforce Scheduling**: Staff allocation and shift planning

#### 📦 [Logistics](domains/logistics/)
- **Inventory Optimization**: Stock level management
- **Warehouse Layout**: Storage optimization
- **Transportation Planning**: Multi-modal transport optimization
- **Distribution Network**: Hub and spoke optimization

#### 💰 [Finance](domains/finance/)
- **Portfolio Optimization**: Risk-return optimization
- **Asset Allocation**: Investment strategy optimization
- **Risk Management**: Financial risk minimization
- **Algorithmic Trading**: Trading strategy optimization

#### ⚡ [Energy](domains/energy/)
- **Power Grid Optimization**: Energy distribution planning
- **Renewable Energy**: Solar and wind optimization
- **Energy Storage**: Battery and storage optimization
- **Smart Grid**: Demand response optimization

#### 🏈 [Fantasy Football](domains/fantasy_football/)
- **Lineup Optimization**: Daily fantasy sports optimization
- **3-File Structure**: Production deployment examples
- **Multi-Contest**: Tournament optimization strategies
- **Player Analysis**: Statistical optimization approaches

### 🌐 [Rastion Integration](rastion_integration/)
Platform integration examples and workflows:
- **Model Upload/Download**: Sharing optimization models
- **Collaborative Development**: Team-based optimization projects
- **Playground Integration**: Interactive optimization testing
- **Production Deployment**: Real-world deployment patterns

## 🚀 Quick Start

### Running Your First Example

1. **Install qubots**:
```bash
pip install qubots[all]
```

2. **Try a basic example**:
```bash
cd basic/
python getting_started_example.py
```

3. **Explore domain examples**:
```bash
cd domains/routing/
python vehicle_routing_tutorial.py
```

### Prerequisites by Domain

#### Basic Examples
- Python 3.9+
- qubots library
- numpy, matplotlib (optional for visualization)

#### Routing/Logistics
```bash
pip install qubots[routing]  # Includes OR-Tools
```

#### Finance
```bash
pip install qubots[finance]  # Includes financial libraries
```

#### Energy
```bash
pip install qubots[energy]  # Includes energy optimization tools
```

#### All Domains
```bash
pip install qubots[all]  # Includes all optional dependencies
```

## 📚 Learning Path

### 1. Beginners (New to Optimization)
Start here to learn optimization fundamentals:

1. **[Basic Examples](basic/)** - Core concepts and simple problems
2. **[Getting Started Tutorial](../docs/tutorials/getting_started.md)** - Step-by-step guide
3. **[Fantasy Football](domains/fantasy_football/)** - Fun, relatable optimization problem
4. **[Routing](domains/routing/)** - Classic optimization problems

### 2. Intermediate (Some Optimization Experience)
Build on existing knowledge:

1. **[Custom Optimizers Tutorial](../docs/tutorials/custom_optimizers.md)** - Create your own algorithms
2. **[Scheduling](domains/scheduling/)** - Complex constraint handling
3. **[Finance](domains/finance/)** - Multi-objective optimization
4. **[Rastion Integration](rastion_integration/)** - Platform collaboration

### 3. Advanced (Optimization Experts)
Explore cutting-edge applications:

1. **[Energy](domains/energy/)** - Large-scale optimization
2. **[Logistics](domains/logistics/)** - Multi-level optimization
3. **[Benchmarking Guide](../docs/guides/benchmarking.md)** - Performance analysis
4. **[Production Deployment](../docs/guides/deployment.md)** - Real-world deployment

## 🎯 Example Categories

### By Problem Type

#### **Continuous Optimization**
- Portfolio optimization (Finance)
- Energy distribution (Energy)
- Parameter tuning (Basic)

#### **Discrete Optimization**
- Job scheduling (Scheduling)
- Resource allocation (Logistics)
- Fantasy lineups (Fantasy Football)

#### **Combinatorial Optimization**
- Vehicle routing (Routing)
- Traveling salesman (Routing)
- Warehouse layout (Logistics)

#### **Multi-Objective Optimization**
- Risk-return tradeoff (Finance)
- Cost-quality tradeoff (Logistics)
- Performance-efficiency (Energy)

### By Algorithm Type

#### **Exact Algorithms**
- Linear programming examples
- Integer programming examples
- Dynamic programming examples

#### **Metaheuristics**
- Genetic algorithms
- Simulated annealing
- Particle swarm optimization
- Tabu search

#### **Hybrid Approaches**
- Matheuristics
- Machine learning + optimization
- Multi-stage optimization

## 🔧 Running Examples

### Local Execution

```bash
# Navigate to example directory
cd domains/routing/

# Run example
python vehicle_routing_tutorial.py

# Run with custom parameters
python vehicle_routing_tutorial.py --num_vehicles 5 --num_customers 20
```

### Jupyter Notebooks

Many examples include Jupyter notebook versions:

```bash
# Install Jupyter
pip install jupyter

# Start notebook server
jupyter notebook

# Open example notebook
# Navigate to domains/routing/vehicle_routing_tutorial.ipynb
```

### Docker Execution

```bash
# Build qubots Docker image
docker build -t qubots .

# Run example in container
docker run -v $(pwd):/workspace qubots python examples/domains/routing/vehicle_routing_tutorial.py
```

## 🌐 Rastion Platform Integration

Most examples support Rastion platform integration:

### Authentication Setup

```python
import qubots.rastion as rastion

# Set up authentication (one-time)
rastion.authenticate("your_gitea_token")
```

### Loading Models from Platform

```python
# Load pre-built models
problem = rastion.load_qubots_model("traveling_salesman_problem")
optimizer = rastion.load_qubots_model("genetic_algorithm_tsp")

# Run optimization
result = optimizer.optimize(problem)
```

### Uploading Your Models

```python
# Share your optimizers
url = rastion.upload_model(
    model=my_optimizer,
    name="my_custom_algorithm",
    description="Novel optimization approach"
)
```

## 📊 Performance and Benchmarking

### Benchmark Suites

Each domain includes benchmark problems:

```python
from qubots import BenchmarkSuite

# Create benchmark suite
suite = BenchmarkSuite()
suite.add_optimizer("Algorithm A", OptimizerA())
suite.add_optimizer("Algorithm B", OptimizerB())

# Run benchmarks
results = suite.run_benchmarks(problem, num_runs=10)
suite.generate_report(results, "benchmark_report.html")
```

### Performance Metrics

Examples demonstrate various performance metrics:
- **Solution Quality**: Objective function value
- **Runtime**: Execution time
- **Convergence**: Improvement over time
- **Robustness**: Performance across multiple runs
- **Scalability**: Performance vs. problem size

## 🤝 Contributing Examples

We welcome new examples! Please follow these guidelines:

### Example Structure

```
domain_name/
├── README.md                 # Domain overview and examples list
├── basic_example.py         # Simple, well-commented example
├── advanced_example.py      # Complex, real-world example
├── tutorial.md             # Step-by-step tutorial
├── data/                   # Sample data files
├── tests/                  # Example validation tests
└── notebooks/              # Jupyter notebook versions
```

### Code Standards

- **Clear Documentation**: Comprehensive docstrings and comments
- **Error Handling**: Robust error handling and validation
- **Testing**: Include test cases and validation
- **Dependencies**: Minimal, well-specified dependencies
- **Performance**: Efficient implementations with timing

### Submission Process

1. **Fork Repository**: Create your own fork
2. **Create Example**: Follow the structure above
3. **Test Thoroughly**: Ensure examples work correctly
4. **Submit PR**: Create pull request with description
5. **Review Process**: Address feedback and iterate

## 📞 Getting Help

### Documentation
- **[Main Documentation](../docs/)**: Comprehensive guides and tutorials
- **[API Reference](../docs/api/)**: Complete API documentation
- **[Tutorials](../docs/tutorials/)**: Step-by-step learning materials

### Community Support
- **[Rastion Platform](https://rastion.com)**: Community discussions and model sharing
- **[GitHub Issues](https://github.com/Rastion/qubots/issues)**: Bug reports and feature requests
- **[Stack Overflow](https://stackoverflow.com/questions/tagged/qubots)**: Q&A with the community

### Example-Specific Help

Each domain directory includes:
- **README.md**: Domain overview and getting started
- **troubleshooting.md**: Common issues and solutions
- **FAQ.md**: Frequently asked questions

## 🎉 Featured Examples

### 🏆 Most Popular
1. **[Fantasy Football Optimization](domains/fantasy_football/)** - Fun introduction to optimization
2. **[Vehicle Routing](domains/routing/)** - Classic logistics problem
3. **[Portfolio Optimization](domains/finance/)** - Financial applications

### 🆕 Recently Added
1. **[Energy Grid Optimization](domains/energy/)** - Smart grid applications
2. **[Workforce Scheduling](domains/scheduling/)** - HR optimization
3. **[Supply Chain Resilience](domains/logistics/)** - Risk-aware logistics

### 🔬 Advanced Techniques
1. **[Multi-Objective Optimization](domains/finance/multi_objective_portfolio.py)** - Pareto optimization
2. **[Hybrid Algorithms](domains/routing/hybrid_vrp_solver.py)** - Combining exact and heuristic methods
3. **[Real-Time Optimization](domains/energy/real_time_grid_optimization.py)** - Dynamic optimization

---

Start exploring and happy optimizing! 🚀

For questions or suggestions, please open an issue or join our community on the Rastion platform.
