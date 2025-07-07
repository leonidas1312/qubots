# Furniture Arrangement Problem

A comprehensive furniture arrangement optimization problem for the qubots framework that helps you find the optimal layout for your living room furniture.

## 🏠 Overview

This problem models the challenge of arranging furniture in a living room to maximize:
- **Space utilization** (minimize wasted space)
- **Accessibility** (ensure clear pathways and traffic flow)
- **Aesthetic appeal** (proper groupings, symmetry, focal points)

The optimization considers realistic constraints like furniture dimensions, clearance requirements, wall placement needs, and door/window accessibility.

## 📊 Problem Features

- **Multi-objective optimization** with configurable weights
- **Realistic furniture database** with 40+ common living room pieces
- **Multiple room configurations** (small apartments to large family rooms)
- **Physical constraints** (no overlaps, clearances, wall requirements)
- **Functional requirements** (TV viewing angles, conversation areas)
- **Aesthetic considerations** (balance, symmetry, focal points)


## 📋 Available Furniture

The problem includes 40+ realistic furniture pieces:

### Seating
- 3-seat sofa, 2-seat loveseat, armchairs, recliners, ottomans

### Tables  
- Coffee tables (rectangular/round), side tables, console tables

### Storage
- TV stands, bookshelves, filing cabinets, storage baskets

### Electronics
- TVs (43", 55", 65"), gaming consoles, sound systems

### Lighting & Decor
- Floor lamps, table lamps, plants, rugs, artwork, mirrors

### Office
- Desks, office chairs, printer stands

## 🏠 Room Configurations

| Room Type | Dimensions | Area | Description |
|-----------|------------|------|-------------|
| Small Square | 350×350 cm | 12.3 m² | Compact apartment living room |
| Medium Rect | 450×350 cm | 15.8 m² | Standard family home living room |
| Large Rect | 550×400 cm | 22.0 m² | Spacious family room |
| Open Plan | 600×500 cm | 30.0 m² | Open concept living area |
| Studio Apt | 400×300 cm | 12.0 m² | Studio apartment space |
| Cozy Cottage | 380×320 cm | 12.2 m² | Charming cottage style room |

## 🎯 Solution Format

Each solution is a list of furniture placements:

```python
solution = [
    {
        'furniture_id': 'sofa_3seat',
        'x': 50.0,        # X position in cm
        'y': 100.0,       # Y position in cm  
        'rotation': 0     # Rotation in degrees (0, 90, 180, 270)
    },
    {
        'furniture_id': 'coffee_table_rect',
        'x': 80.0,
        'y': 200.0,
        'rotation': 0
    }
    # ... more furniture pieces
]
```

## 🔧 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `room_config` | string | "medium_rect" | Room configuration ID |
| `furniture_selection` | list | null | Custom furniture list (null for default) |
| `space_weight` | float | 0.4 | Weight for space utilization objective |
| `accessibility_weight` | float | 0.3 | Weight for accessibility objective |
| `aesthetic_weight` | float | 0.3 | Weight for aesthetic objective |
| `min_walkway_width` | float | 80.0 | Minimum walkway width in cm |


## 📚 Applications

- **Home interior design** - Optimize your living room layout
- **Real estate staging** - Create appealing arrangements for showings  
- **Furniture store displays** - Design optimal showroom layouts
- **Office space planning** - Arrange furniture in office common areas
- **Accessibility compliance** - Ensure adequate clearances and pathways

## 🎨 Visualization

When used with the furniture arrangement optimizer, you'll get:
- 2D floor plan visualizations
- Clearance zone indicators  
- Traffic flow analysis
- Before/after comparisons
- Real-time optimization progress

Perfect for visualizing and validating your furniture arrangements!
