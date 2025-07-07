# Furniture Arrangement Optimizer

A sophisticated simulated annealing optimizer for furniture arrangement problems with comprehensive real-time visualizations and spatial optimization capabilities.

## 🎯 Overview

This optimizer uses simulated annealing to find optimal furniture arrangements in living rooms. It features specialized move types for spatial optimization and provides rich visualizations to help you understand the optimization process and results.

## ✨ Key Features

- **Simulated Annealing Algorithm** with adaptive temperature scheduling
- **Specialized Spatial Moves** (translation, rotation, swapping)
- **Real-time Visualizations** showing optimization progress
- **Comprehensive Room Layouts** with clearance zones and traffic flow
- **Solution Quality Metrics** tracking space utilization and accessibility
- **Before/After Comparisons** to visualize improvements


## 🎨 Visualization Features

The optimizer provides comprehensive visualizations throughout the optimization process:

### 1. Initial Layout
- Shows the random starting arrangement
- Room boundaries, doors, and windows
- Initial furniture placement

### 2. Progress Tracking
- Real-time cost evolution
- Temperature schedule visualization  
- Current best arrangement
- Cost vs temperature scatter plot

### 3. Final Results
- Optimized furniture layout with labels
- Clearance zones and traffic flow indicators
- Solution quality metrics summary
- Acceptance rate and convergence analysis

### 4. Detailed Room Layout
- 2D floor plan with accurate furniture dimensions
- Clearance zones (red areas showing required space)
- Furniture orientation indicators (arrows for rotated pieces)
- Room features (doors, windows, electrical outlets)

## ⚙️ Algorithm Details

### Simulated Annealing Process
1. **Initialization**: Start with random furniture arrangement
2. **Temperature Schedule**: Geometric cooling with configurable rate
3. **Move Generation**: Three types of spatial moves
4. **Acceptance Criteria**: Metropolis criterion with temperature-based probability
5. **Termination**: When temperature drops below threshold or max iterations reached

### Move Types

#### Translation Moves
- Small random position adjustments (±20cm by default)
- Respects room boundaries
- Maintains furniture within valid placement areas

#### Rotation Moves  
- 90-degree rotations for rotatable furniture
- Automatically adjusts position if needed to stay in bounds
- Only applied to furniture that can rotate

#### Swapping Moves
- Exchange positions of two furniture pieces
- Helps escape local optima
- Maintains individual furniture orientations

### Temperature Schedule
- **Initial Temperature**: High enough to accept most moves initially
- **Cooling Rate**: Geometric reduction (typically 0.95)
- **Minimum Temperature**: Stopping criterion (typically 1.0)

## 📊 Performance Metrics

The optimizer tracks comprehensive metrics:

| Metric | Description |
|--------|-------------|
| `best_value` | Lowest cost achieved |
| `iterations` | Total optimization iterations |
| `runtime_seconds` | Optimization time |
| `acceptance_rate` | Percentage of moves accepted |
| `space_utilization` | Percentage of room area used |
| `is_valid_arrangement` | Whether final solution is feasible |
| `final_temperature` | Temperature at termination |

## 🔧 Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `initial_temperature` | float | 1000.0 | 100-5000 | Starting temperature |
| `cooling_rate` | float | 0.95 | 0.8-0.99 | Temperature reduction factor |
| `min_temperature` | float | 1.0 | 0.1-10 | Stopping temperature |
| `max_iterations` | int | 5000 | 1000-20000 | Maximum iterations |
| `moves_per_temperature` | int | 50 | 10-200 | Moves per temperature level |
| `position_step_size` | float | 20.0 | 5-100 | Maximum position change (cm) |
| `create_plots` | bool | true | - | Enable visualizations |
| `save_plots` | bool | false | - | Save plots to files |
| `plot_interval` | int | 500 | 100-2000 | Iterations between plot updates |
| `random_seed` | int | null | - | Seed for reproducible results |

## 📈 Applications

- **Home Interior Design**: Optimize your living room layout
- **Real Estate Staging**: Create appealing arrangements for property showings
- **Furniture Retail**: Design optimal showroom displays
- **Office Planning**: Arrange furniture in office common areas
- **Accessibility Design**: Ensure adequate clearances and pathways
- **Space Planning Research**: Study optimal furniture arrangement patterns

Perfect for anyone who wants to create beautiful, functional, and efficient living spaces!
