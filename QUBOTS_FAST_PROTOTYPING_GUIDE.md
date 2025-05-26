# Qubots Fast Prototyping with Rastion Platform

## Overview

This system provides **fast prototyping** capabilities where users can quickly test optimization approaches using the **qubots framework** as the core optimization engine with built-in dashboards and visualization.

## Architecture Philosophy

### **Qubots as the Optimization Framework**
- **Built-in dashboards** and visualization capabilities
- **Standardized result formats** for web integration
- **Automatic plotting** and analysis tools
- **Lightweight execution** with rich output

### **Rastion as the Platform**
- **Repository management** and selection
- **Parameter configuration** interface
- **Result display** using qubots' dashboard data
- **Fast environment provisioning**

### **Ultra-Lightweight Containers**
- **Minimal footprint** (~100MB vs 4GB)
- **Fast startup** (seconds vs minutes)
- **Simple execution** via HTTP API
- **No heavy development tools**

## Key Components

### 1. **Qubots Dashboard Module** (`qubots/dashboard.py`)

```python
from qubots import QubotsAutoDashboard

# Automatic optimization with dashboard
result = QubotsAutoDashboard.auto_optimize_with_dashboard(
    problem=my_problem,
    optimizer=my_optimizer,
    problem_name="TSP",
    optimizer_name="Genetic Algorithm"
)

# Rich dashboard result with visualizations
print(result.to_json())
```

**Features:**
- **Automatic visualization generation** (convergence plots, solution displays)
- **Standardized result format** for web integration
- **Multiple chart types** (Plotly, matplotlib, text fallbacks)
- **Performance metrics** and metadata collection

### 2. **Ultra-Lightweight Container** (< 100MB)

```dockerfile
FROM python:3.11-alpine
# Install qubots + minimal dependencies
# Simple HTTP API for execution
# No Jupyter, VS Code, or heavy tools
```

**Benefits:**
- **Fast startup** (< 10 seconds)
- **Low resource usage** (256MB RAM, 0.2 CPU)
- **Quick scaling** for multiple users
- **Cost effective** deployment

### 3. **Rastion Integration**

```tsx
// React component displays qubots dashboard results
<QubotsPlayground />
```

**Features:**
- **Repository selection** with qubots type detection
- **Parameter configuration** via JSON
- **Real-time execution** monitoring
- **Rich result display** using qubots visualizations

## User Workflow

### 1. **Repository Selection**
- User selects **problem repository** (marked with `qubot_type: "problem"`)
- User selects **optimizer repository** (marked with `qubot_type: "optimizer"`)
- System validates compatibility

### 2. **Fast Environment Creation**
- **Lightweight container** spins up in seconds
- **Repositories cloned** automatically
- **Qubots framework** ready for execution

### 3. **Parameter Configuration**
- **Problem parameters** via JSON (e.g., `{"cities": 20, "seed": 42}`)
- **Optimizer parameters** via JSON (e.g., `{"population_size": 100}`)
- **Real-time validation** of JSON syntax

### 4. **Optimization Execution**
- **Single API call** to container
- **Qubots handles** optimization + visualization
- **Dashboard results** returned automatically

### 5. **Rich Results Display**
- **Convergence plots** (interactive Plotly charts)
- **Performance metrics** (execution time, iterations, best value)
- **Solution visualization** (problem-specific displays)
- **Export capabilities** for further analysis

## Implementation

### Build Ultra-Lightweight Image

```powershell
# Copy qubots library
Copy-Item -Recurse ../qubots ./playground-environment/qubots

# Build minimal image
docker build -f ./playground-environment/Dockerfile.lightweight -t qubots-playground:lightweight ./playground-environment

# Result: ~100MB image vs 4GB
```

### Deploy to Production

```powershell
# Deploy with lightweight containers
.\deploy-backend-production.ps1
.\deploy-frontend-production.ps1
```

### Container Configuration

```yaml
resources:
  limits:
    cpu: '0.5'      # vs 2 CPU
    memory: '512Mi'  # vs 4Gi
  requests:
    cpu: '100m'     # vs 500m
    memory: '128Mi'  # vs 1Gi
```

## Benefits

### **For Users**
- **Fast prototyping** - test ideas in seconds
- **Rich visualizations** - automatic dashboard generation
- **Easy parameter tuning** - JSON-based configuration
- **No setup required** - everything handled by platform

### **For Platform**
- **Cost effective** - 8x less resources per environment
- **Scalable** - support more concurrent users
- **Fast provisioning** - environments ready in seconds
- **Maintainable** - simple, focused containers

### **For Qubots Framework**
- **Showcase capabilities** - built-in dashboards highlight features
- **Easy adoption** - users see value immediately
- **Standardized output** - consistent result formats
- **Framework focus** - qubots handles optimization, not infrastructure

## Example Usage

### Repository Structure

```
my-tsp-problem/
├── config.json          # {"qubot_type": "problem"}
├── tsp_problem.py       # BaseProblem implementation
└── README.md

my-genetic-optimizer/
├── config.json          # {"qubot_type": "optimizer"}  
├── genetic_optimizer.py # BaseOptimizer implementation
└── README.md
```

### Execution Flow

```python
# In lightweight container
from qubots import execute_playground_optimization

result = execute_playground_optimization(
    problem_name="my-tsp-problem",
    optimizer_name="my-genetic-optimizer",
    problem_params={"cities": 20},
    optimizer_params={"population_size": 100, "generations": 50}
)

# Returns rich dashboard result with:
# - Convergence plot
# - Best solution visualization  
# - Performance metrics
# - Execution metadata
```

### Frontend Display

```tsx
// Automatic rendering of qubots dashboard results
{dashboardResult.visualizations?.map(viz => (
  <Plot data={viz.data} layout={viz.layout} />
))}
```

## Deployment

### Resource Requirements

| Component | CPU | Memory | Storage | Startup Time |
|-----------|-----|--------|---------|--------------|
| Old System | 2 cores | 4GB | 4GB image | 2-3 minutes |
| New System | 0.5 cores | 512MB | 100MB image | 10 seconds |
| **Savings** | **75%** | **87%** | **97%** | **90%** |

### Scaling Capacity

- **Old**: 5 concurrent environments per node
- **New**: 40+ concurrent environments per node
- **8x improvement** in user capacity

## Next Steps

1. **Enhanced Visualizations** - Add more chart types to qubots dashboard
2. **Problem-Specific Displays** - Custom visualizations for TSP, scheduling, etc.
3. **Benchmark Integration** - Built-in performance comparison tools
4. **Template System** - Pre-configured optimization templates
5. **Community Sharing** - Share successful optimization configurations

This approach positions **qubots as the optimization framework** while **Rastion provides the platform** for fast, efficient prototyping with rich, automatic dashboards.
