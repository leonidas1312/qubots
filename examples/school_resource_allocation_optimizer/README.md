# School Resource Allocation Optimizer

A research-grade OR-Tools based optimizer for solving school scheduling problems using constraint programming. This optimizer finds optimal teacher-subject-classroom-timeslot assignments while respecting all constraints and minimizing costs.

## 🔧 Algorithm Overview

This optimizer uses **Google OR-Tools CP-SAT solver**, a state-of-the-art constraint programming solver that excels at scheduling and resource allocation problems.

### Key Features
- **Exact Solutions**: Finds optimal or proven near-optimal solutions
- **Constraint Programming**: Naturally handles complex logical constraints
- **Multi-Objective**: Balances cost, quality, and resource utilization
- **Scalable**: Handles real-world school sizes efficiently
- **Parallel Processing**: Uses multiple search workers for faster solving

### Algorithm Type
- **Family**: Constraint Programming
- **Type**: Exact Algorithm
- **Solver**: OR-Tools CP-SAT
- **Guarantees**: Optimal solutions when found, proven bounds otherwise

## 🎯 Optimization Objectives

### Primary Objective
**Minimize Total Cost** while satisfying all constraints:
```
Total Cost = Teacher Costs + Classroom Costs - Quality Bonus
```

### Multi-Objective Components
1. **Cost Minimization** (weight: configurable)
   - Teacher hourly rates
   - Classroom operational costs

2. **Quality Maximization** (weight: configurable)
   - Teacher experience levels
   - Subject-teacher compatibility

3. **Constraint Satisfaction** (hard constraints)
   - No teacher double-booking
   - No room double-booking
   - Subject hour requirements
   - Teacher qualifications
   - Room type requirements

## 🔒 Constraints Handled

### Hard Constraints (Must be satisfied)
- **Teacher Availability**: No teacher in multiple places simultaneously
- **Room Availability**: No room hosting multiple classes simultaneously
- **Teacher Qualifications**: Teachers only assigned to subjects they can teach
- **Room Requirements**: Subjects assigned to appropriate room types
- **Capacity Limits**: Class sizes within room capacity
- **Subject Hours**: All subjects receive required weekly hours
- **Daily Limits**: Teachers don't exceed daily hour limits

### Soft Constraints (Optimized)
- **Cost Efficiency**: Minimize total operational costs
- **Quality Matching**: Assign experienced teachers to important subjects
- **Resource Utilization**: Efficient use of available resources

## ⚙️ Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_solve_time_seconds` | float | 300.0 | Maximum solving time |
| `num_search_workers` | int | 4 | Parallel search workers |
| `emphasis` | string | "balanced" | Search strategy |
| `enable_logging` | bool | false | Detailed solver logging |

### Search Emphasis Options
- **"feasibility"**: Find any valid solution quickly
- **"optimality"**: Find the best possible solution
- **"balanced"**: Compromise between speed and quality

## 🎨 Visualization Features

### Comprehensive Analysis Plots
1. **Teacher Workload Distribution**
   - Bar chart showing hours assigned per teacher
   - Identifies workload imbalances
   - Helps ensure fair distribution

2. **Classroom Utilization**
   - Room usage efficiency analysis
   - Identifies underutilized resources
   - Supports facility planning

3. **Subject Coverage Analysis**
   - Required vs. assigned hours comparison
   - Ensures curriculum requirements are met
   - Highlights coverage gaps

4. **Schedule Heatmap**
   - Visual overview of time slot assignments
   - Quick identification of busy/free periods
   - Pattern analysis for optimization

### Summary Statistics
- Total assignments and costs
- Constraint violation reports
- Resource utilization metrics
- Solution quality indicators

## 🔍 Solution Analysis

### Detailed Metrics
- **Solver Status**: OPTIMAL, FEASIBLE, INFEASIBLE, etc.
- **Solution Quality**: Objective value and bounds
- **Constraint Violations**: Detailed violation reports
- **Resource Utilization**: Teacher and room usage statistics
- **Cost Breakdown**: Teacher vs. classroom costs

### Feasibility Checking
- Automatic constraint violation detection
- Detailed error reporting for infeasible problems
- Suggestions for constraint relaxation

## 🏆 Advantages

### vs. Manual Scheduling
- **Speed**: Minutes vs. weeks of manual work
- **Optimality**: Mathematically optimal solutions
- **Consistency**: Objective, repeatable results
- **Completeness**: Considers all constraints simultaneously

### vs. Heuristic Methods
- **Guarantees**: Proven optimal or bounded solutions
- **Reliability**: Always finds feasible solution if one exists
- **Quality**: Superior solution quality
- **Transparency**: Clear constraint handling

### vs. Other Solvers
- **Performance**: State-of-the-art CP-SAT solver
- **Scalability**: Handles real-world problem sizes
- **Robustness**: Mature, well-tested implementation
- **Support**: Active development and community

## 🔧 Technical Requirements

### Dependencies
```
ortools>=9.0.0
matplotlib>=3.0.0
numpy>=1.18.0
pandas>=1.0.0
```

## 🎓 Educational Applications

### Research Applications
- **Operations Research**: Real-world scheduling case study
- **Algorithm Comparison**: Benchmark against other methods
- **Constraint Programming**: Educational example of CP modeling
- **Optimization Theory**: Multi-objective optimization example

### Practical Applications
- **School Administration**: Automated timetable generation
- **Resource Planning**: Optimal facility utilization
- **Cost Management**: Budget-conscious scheduling
- **Quality Improvement**: Better teacher-subject matching

This optimizer represents the state-of-the-art in school scheduling technology, bringing research-grade optimization capabilities to educational institutions of all sizes.
