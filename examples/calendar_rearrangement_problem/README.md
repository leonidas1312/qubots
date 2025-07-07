# Calendar Rearrangement Problem

A qubots optimization problem for rearranging calendar meetings to free up a specific day while minimizing disruption and respecting scheduling constraints.

## 🎯 Problem Overview

This problem helps you optimize your weekly calendar by moving meetings from a target day (e.g., Wednesday) to other available days. It's perfect for:

- **Flexible work arrangements**: Creating a dedicated day off or focus day
- **Meeting optimization**: Reducing context switching by consolidating meetings
- **Work-life balance**: Freeing up specific days for deep work or personal time
- **Dynamic scheduling**: Adapting to changing weekly priorities

## 📊 Input Data Format

The problem accepts CSV data with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `meeting_id` | string | Unique identifier for the meeting |
| `meeting_name` | string | Name/title of the meeting |
| `duration_hours` | float | Duration in hours (0.5 to 8.0) |
| `priority` | int | Priority level (1-5, where 5 is highest) |
| `current_day` | string | Current day of the week |
| `flexible` | bool | Whether the meeting can be moved |
| `participants` | int | Number of participants |

## 🎯 Solution Format

Solutions are represented as dictionaries with an `assignments` key:

```python
solution = {
    'assignments': [0, 1, 2, 0]  # Day indices for each moveable meeting
}

# Where day indices map to available_days:
# 0 = Monday, 1 = Tuesday, 2 = Thursday, 3 = Friday
```

## 📈 Objective Function

The problem minimizes total rescheduling cost:

```
Total Cost = Σ(Base Cost + Priority Cost) + Capacity Penalties

Where:
- Base Cost = rescheduling_weight × (1 + 0.1 × participants)
- Priority Cost = priority_weight × meeting_priority
- Capacity Penalties = 1000 × (excess_hours)²
```

## 🔍 Constraints

1. **Capacity Constraint**: Each day cannot exceed `max_hours_per_day`
2. **Assignment Constraint**: Each moveable meeting must be assigned to exactly one available day
3. **Flexibility Constraint**: Only meetings marked as `flexible=True` can be moved

## 🧪 Testing

The problem includes comprehensive validation:

- **Data validation**: Ensures CSV data has required columns and valid values
- **Constraint checking**: Validates capacity and assignment constraints
- **Solution feasibility**: Checks if solutions respect all constraints
- **Cost calculation**: Accurate objective function evaluation

## 🔧 Advanced Features

- **Greedy fallback**: Automatic fallback to greedy solution if random generation fails
- **Priority weighting**: Higher priority meetings cost more to reschedule
- **Participant scaling**: Larger meetings are harder to reschedule
- **Flexible constraints**: Easy to modify available days and capacity limits

Perfect for integration with calendar APIs and automated scheduling systems!
