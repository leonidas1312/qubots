"""
Furniture Arrangement Optimization Problem for Qubots Framework.

This problem optimizes the placement of furniture in a living room to maximize
space utilization, accessibility, and aesthetic appeal while respecting
physical and functional constraints.
"""

import numpy as np
import pandas as pd
import random
import math
from typing import Dict, List, Tuple, Any, Union, Optional
from dataclasses import dataclass
import ast

from qubots.base_problem import (
    BaseProblem, ProblemMetadata, ProblemType, ObjectiveType, 
    DifficultyLevel, EvaluationResult
)


@dataclass
class FurniturePiece:
    """Represents a piece of furniture with its properties."""
    id: str
    name: str
    width: float  # cm
    depth: float  # cm
    height: float  # cm
    type: str
    can_rotate: bool
    wall_required: bool
    clearance_front: float
    clearance_back: float
    clearance_sides: float
    aesthetic_weight: float
    functional_weight: float
    cost: float
    color: str
    material: str


@dataclass
class Room:
    """Represents a room with its dimensions and features."""
    id: str
    name: str
    width: float  # cm
    depth: float  # cm
    door_x: float
    door_y: float
    door_width: float
    windows: List[Tuple[str, float, float]]  # (wall, start, end)
    electrical_outlets: List[Tuple[float, float]]
    room_type: str
    description: str


@dataclass
class FurniturePlacement:
    """Represents the placement of a furniture piece."""
    furniture_id: str
    x: float  # cm from room origin
    y: float  # cm from room origin
    rotation: float  # degrees (0, 90, 180, 270)


class FurnitureArrangementProblem(BaseProblem):
    """
    Furniture Arrangement Optimization Problem for modular qubots architecture.
    
    This problem models the optimal placement of furniture in a living room considering:
    - Physical constraints (no overlaps, clearances, wall requirements)
    - Functional requirements (accessibility, viewing angles, traffic flow)
    - Aesthetic considerations (symmetry, groupings, focal points)
    - Space utilization efficiency
    
    Solution Format:
        List of FurniturePlacement objects, each containing:
        - furniture_id: ID of the furniture piece
        - x, y: Position coordinates in cm
        - rotation: Rotation angle in degrees (0, 90, 180, 270)
    
    Objective: Minimize total cost = space_penalty + accessibility_penalty + aesthetic_penalty
    """
    
    def __init__(self, 
                 room_config: str = "medium_rect",
                 furniture_selection: Optional[List[str]] = None,
                 furniture_csv_data: Optional[str] = None,
                 room_csv_data: Optional[str] = None,
                 furniture_csv_file: Optional[str] = None,
                 room_csv_file: Optional[str] = None,
                 space_weight: float = 0.4,
                 accessibility_weight: float = 0.3,
                 aesthetic_weight: float = 0.3,
                 min_walkway_width: float = 80.0,  # cm
                 **kwargs):
        
        # Load furniture data
        if furniture_csv_data:
            furniture_df = pd.read_csv(pd.StringIO(furniture_csv_data))
        elif furniture_csv_file:
            furniture_df = pd.read_csv(furniture_csv_file)
        else:
            # Use default dataset
            import os
            default_path = os.path.join(os.path.dirname(__file__), 'datasets', 'living_room_furniture.csv')
            furniture_df = pd.read_csv(default_path)
        
        # Load room data
        if room_csv_data:
            room_df = pd.read_csv(pd.StringIO(room_csv_data))
        elif room_csv_file:
            room_df = pd.read_csv(room_csv_file)
        else:
            # Use default dataset
            import os
            default_path = os.path.join(os.path.dirname(__file__), 'datasets', 'room_configurations.csv')
            room_df = pd.read_csv(default_path)
        
        # Parse room configuration
        room_row = room_df[room_df['room_id'] == room_config].iloc[0]
        
        # Parse electrical outlets from string representation
        outlets_str = room_row['electrical_outlets']
        electrical_outlets = ast.literal_eval(outlets_str) if outlets_str else []
        
        # Create room object
        self.room = Room(
            id=room_row['room_id'],
            name=room_row['name'],
            width=room_row['width_cm'],
            depth=room_row['depth_cm'],
            door_x=room_row['door_x_cm'],
            door_y=room_row['door_y_cm'],
            door_width=room_row['door_width_cm'],
            windows=[
                (room_row['window_1_wall'], room_row['window_1_start_cm'], room_row['window_1_end_cm']),
                (room_row['window_2_wall'], room_row['window_2_start_cm'], room_row['window_2_end_cm'])
            ] if pd.notna(room_row['window_2_wall']) else [
                (room_row['window_1_wall'], room_row['window_1_start_cm'], room_row['window_1_end_cm'])
            ],
            electrical_outlets=electrical_outlets,
            room_type=room_row['room_type'],
            description=room_row['description']
        )
        
        # Select furniture pieces
        if furniture_selection:
            furniture_df = furniture_df[furniture_df['furniture_id'].isin(furniture_selection)]
        else:
            # Default selection for a typical living room
            default_selection = [
                'sofa_3seat', 'coffee_table_rect', 'tv_stand_large', 'tv_55inch',
                'armchair_1', 'side_table_1', 'floor_lamp_1', 'plant_large',
                'rug_large', 'bookshelf_tall'
            ]
            furniture_df = furniture_df[furniture_df['furniture_id'].isin(default_selection)]
        
        # Create furniture objects
        self.furniture_pieces = {}
        for _, row in furniture_df.iterrows():
            piece = FurniturePiece(
                id=row['furniture_id'],
                name=row['name'],
                width=row['width_cm'],
                depth=row['depth_cm'],
                height=row['height_cm'],
                type=row['type'],
                can_rotate=row['can_rotate'],
                wall_required=row['wall_required'],
                clearance_front=row['clearance_front_cm'],
                clearance_back=row['clearance_back_cm'],
                clearance_sides=row['clearance_sides_cm'],
                aesthetic_weight=row['aesthetic_weight'],
                functional_weight=row['functional_weight'],
                cost=row['cost_usd'],
                color=row['color'],
                material=row['material']
            )
            self.furniture_pieces[piece.id] = piece
        
        # Problem parameters
        self.space_weight = space_weight
        self.accessibility_weight = accessibility_weight
        self.aesthetic_weight = aesthetic_weight
        self.min_walkway_width = min_walkway_width
        
        # Create metadata
        metadata = ProblemMetadata(
            name=f"Furniture Arrangement - {self.room.name}",
            description=f"Optimize furniture placement in {self.room.description.lower()}",
            domain="interior_design",
            problem_type=ProblemType.COMBINATORIAL,
            objective_type=ObjectiveType.MINIMIZE,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            tags={"furniture", "interior_design", "spatial_optimization", "2d_packing", "real_world"},
            author="Qubots Framework",
            version="1.0.0"
        )

        # Initialize base problem
        super().__init__(metadata, **kwargs)

    def _get_default_metadata(self) -> ProblemMetadata:
        """Get default metadata for the furniture arrangement problem."""
        return self.metadata
    
    def random_solution(self) -> List[Dict[str, Any]]:
        """Generate a random furniture arrangement."""
        placements = []
        
        for furniture_id, piece in self.furniture_pieces.items():
            # Random position within room bounds
            max_x = self.room.width - piece.width
            max_y = self.room.depth - piece.depth
            
            if max_x <= 0 or max_y <= 0:
                continue  # Furniture too large for room
            
            x = random.uniform(0, max_x)
            y = random.uniform(0, max_y)
            
            # Random rotation if allowed
            if piece.can_rotate:
                rotation = random.choice([0, 90, 180, 270])
            else:
                rotation = 0
            
            placements.append({
                'furniture_id': furniture_id,
                'x': x,
                'y': y,
                'rotation': rotation
            })
        
        return placements

    def evaluate_solution(self, solution: List[Dict[str, Any]]) -> float:
        """
        Evaluate furniture arrangement solution.

        Returns total cost = space_penalty + accessibility_penalty + aesthetic_penalty
        Lower values are better.
        """
        if not solution:
            return 1e6  # Empty solution penalty

        # Parse solution into placement objects
        placements = []
        for item in solution:
            if isinstance(item, dict):
                placements.append(FurniturePlacement(
                    furniture_id=item['furniture_id'],
                    x=item['x'],
                    y=item['y'],
                    rotation=item['rotation']
                ))
            else:
                return 1e6  # Invalid solution format

        # Check basic feasibility
        feasibility_penalty = self._check_feasibility(placements)
        if feasibility_penalty > 0:
            return 1e6 + feasibility_penalty

        # Calculate individual penalty components
        space_penalty = self._calculate_space_penalty(placements)
        accessibility_penalty = self._calculate_accessibility_penalty(placements)
        aesthetic_penalty = self._calculate_aesthetic_penalty(placements)

        # Weighted total cost
        total_cost = (
            self.space_weight * space_penalty +
            self.accessibility_weight * accessibility_penalty +
            self.aesthetic_weight * aesthetic_penalty
        )

        return total_cost

    def _check_feasibility(self, placements: List[FurniturePlacement]) -> float:
        """Check if placement is feasible (no overlaps, within bounds)."""
        penalty = 0.0

        for i, placement in enumerate(placements):
            piece = self.furniture_pieces[placement.furniture_id]

            # Get actual dimensions considering rotation
            width, depth = self._get_rotated_dimensions(piece, placement.rotation)

            # Check room bounds
            if (placement.x < 0 or placement.y < 0 or
                placement.x + width > self.room.width or
                placement.y + depth > self.room.depth):
                penalty += 1000

            # Check wall requirements
            if piece.wall_required:
                if not self._is_against_wall(placement, width, depth):
                    penalty += 500

            # Check overlaps with other furniture
            for j, other_placement in enumerate(placements):
                if i != j:
                    if self._check_overlap(placement, other_placement):
                        penalty += 1000

        return penalty

    def _get_rotated_dimensions(self, piece: FurniturePiece, rotation: float) -> Tuple[float, float]:
        """Get furniture dimensions considering rotation."""
        if rotation in [90, 270]:
            return piece.depth, piece.width
        return piece.width, piece.depth

    def _is_against_wall(self, placement: FurniturePlacement, width: float, depth: float) -> bool:
        """Check if furniture is against a wall."""
        tolerance = 5.0  # cm

        # Check each wall
        against_north = placement.y <= tolerance
        against_south = placement.y + depth >= self.room.depth - tolerance
        against_west = placement.x <= tolerance
        against_east = placement.x + width >= self.room.width - tolerance

        return against_north or against_south or against_west or against_east

    def _check_overlap(self, placement1: FurniturePlacement, placement2: FurniturePlacement) -> bool:
        """Check if two furniture pieces overlap."""
        piece1 = self.furniture_pieces[placement1.furniture_id]
        piece2 = self.furniture_pieces[placement2.furniture_id]

        w1, d1 = self._get_rotated_dimensions(piece1, placement1.rotation)
        w2, d2 = self._get_rotated_dimensions(piece2, placement2.rotation)

        # Rectangle overlap check
        return not (placement1.x + w1 <= placement2.x or
                   placement2.x + w2 <= placement1.x or
                   placement1.y + d1 <= placement2.y or
                   placement2.y + d2 <= placement1.y)

    def _calculate_space_penalty(self, placements: List[FurniturePlacement]) -> float:
        """Calculate penalty for poor space utilization."""
        total_furniture_area = 0

        for placement in placements:
            piece = self.furniture_pieces[placement.furniture_id]
            width, depth = self._get_rotated_dimensions(piece, placement.rotation)
            total_furniture_area += width * depth

        room_area = self.room.width * self.room.depth
        utilization = total_furniture_area / room_area

        # Penalty for both under and over-utilization
        optimal_utilization = 0.4  # 40% is typically good for living rooms
        penalty = abs(utilization - optimal_utilization) * 1000

        return penalty

    def _calculate_accessibility_penalty(self, placements: List[FurniturePlacement]) -> float:
        """Calculate penalty for poor accessibility and traffic flow."""
        penalty = 0.0

        # Check door accessibility
        door_blocked = self._is_door_blocked(placements)
        if door_blocked:
            penalty += 2000

        # Check walkway widths
        narrow_walkways = self._count_narrow_walkways(placements)
        penalty += narrow_walkways * 500

        # Check furniture-specific accessibility requirements
        for placement in placements:
            piece = self.furniture_pieces[placement.furniture_id]

            # Check clearance requirements
            if not self._has_adequate_clearance(placement, placements):
                penalty += 300 * piece.functional_weight

        return penalty

    def _is_door_blocked(self, placements: List[FurniturePlacement]) -> bool:
        """Check if door area is blocked."""
        door_area = {
            'x': self.room.door_x,
            'y': self.room.door_y,
            'width': self.room.door_width,
            'depth': 100  # Door swing area
        }

        for placement in placements:
            piece = self.furniture_pieces[placement.furniture_id]
            width, depth = self._get_rotated_dimensions(piece, placement.rotation)

            # Check if furniture intersects door area
            if not (placement.x + width <= door_area['x'] or
                   door_area['x'] + door_area['width'] <= placement.x or
                   placement.y + depth <= door_area['y'] or
                   door_area['y'] + door_area['depth'] <= placement.y):
                return True

        return False

    def _count_narrow_walkways(self, placements: List[FurniturePlacement]) -> int:
        """Count walkways that are too narrow."""
        # Simplified check - count furniture pieces too close to each other
        narrow_count = 0

        for i, placement1 in enumerate(placements):
            for j, placement2 in enumerate(placements):
                if i >= j:
                    continue

                distance = self._min_distance_between_furniture(placement1, placement2)
                if distance < self.min_walkway_width:
                    narrow_count += 1

        return narrow_count

    def _min_distance_between_furniture(self, placement1: FurniturePlacement,
                                       placement2: FurniturePlacement) -> float:
        """Calculate minimum distance between two furniture pieces."""
        piece1 = self.furniture_pieces[placement1.furniture_id]
        piece2 = self.furniture_pieces[placement2.furniture_id]

        w1, d1 = self._get_rotated_dimensions(piece1, placement1.rotation)
        w2, d2 = self._get_rotated_dimensions(piece2, placement2.rotation)

        # Calculate center-to-center distance
        center1_x = placement1.x + w1 / 2
        center1_y = placement1.y + d1 / 2
        center2_x = placement2.x + w2 / 2
        center2_y = placement2.y + d2 / 2

        # Calculate edge-to-edge distance
        dx = max(0, abs(center1_x - center2_x) - (w1 + w2) / 2)
        dy = max(0, abs(center1_y - center2_y) - (d1 + d2) / 2)

        return math.sqrt(dx**2 + dy**2)

    def _has_adequate_clearance(self, placement: FurniturePlacement,
                               all_placements: List[FurniturePlacement]) -> bool:
        """Check if furniture has adequate clearance space."""
        piece = self.furniture_pieces[placement.furniture_id]
        width, depth = self._get_rotated_dimensions(piece, placement.rotation)

        # Define clearance zones based on rotation
        clearance_zones = self._get_clearance_zones(placement, width, depth, piece)

        # Check if clearance zones are free
        for zone in clearance_zones:
            for other_placement in all_placements:
                if other_placement.furniture_id == placement.furniture_id:
                    continue

                if self._zone_intersects_furniture(zone, other_placement):
                    return False

        return True

    def _get_clearance_zones(self, placement: FurniturePlacement, width: float,
                           depth: float, piece: FurniturePiece) -> List[Dict]:
        """Get required clearance zones for furniture piece."""
        zones = []

        # Front clearance (considering rotation)
        if piece.clearance_front > 0:
            if placement.rotation == 0:  # Front faces north
                zones.append({
                    'x': placement.x, 'y': placement.y - piece.clearance_front,
                    'width': width, 'depth': piece.clearance_front
                })
            elif placement.rotation == 90:  # Front faces east
                zones.append({
                    'x': placement.x + width, 'y': placement.y,
                    'width': piece.clearance_front, 'depth': depth
                })
            elif placement.rotation == 180:  # Front faces south
                zones.append({
                    'x': placement.x, 'y': placement.y + depth,
                    'width': width, 'depth': piece.clearance_front
                })
            elif placement.rotation == 270:  # Front faces west
                zones.append({
                    'x': placement.x - piece.clearance_front, 'y': placement.y,
                    'width': piece.clearance_front, 'depth': depth
                })

        return zones

    def _zone_intersects_furniture(self, zone: Dict, placement: FurniturePlacement) -> bool:
        """Check if clearance zone intersects with furniture."""
        piece = self.furniture_pieces[placement.furniture_id]
        width, depth = self._get_rotated_dimensions(piece, placement.rotation)

        return not (zone['x'] + zone['width'] <= placement.x or
                   placement.x + width <= zone['x'] or
                   zone['y'] + zone['depth'] <= placement.y or
                   placement.y + depth <= zone['y'])

    def _calculate_aesthetic_penalty(self, placements: List[FurniturePlacement]) -> float:
        """Calculate penalty for poor aesthetic arrangement."""
        penalty = 0.0

        # Check for balanced arrangement
        penalty += self._check_balance(placements) * 200

        # Check for proper groupings (conversation areas)
        penalty += self._check_groupings(placements) * 300

        # Check for focal point arrangement
        penalty += self._check_focal_points(placements) * 250

        # Check for symmetry where appropriate
        penalty += self._check_symmetry(placements) * 150

        return penalty

    def _check_balance(self, placements: List[FurniturePlacement]) -> float:
        """Check visual balance of furniture arrangement."""
        # Calculate center of mass for furniture
        total_weight = 0
        weighted_x = 0
        weighted_y = 0

        for placement in placements:
            piece = self.furniture_pieces[placement.furniture_id]
            width, depth = self._get_rotated_dimensions(piece, placement.rotation)

            # Use aesthetic weight as mass
            mass = piece.aesthetic_weight * width * depth
            center_x = placement.x + width / 2
            center_y = placement.y + depth / 2

            total_weight += mass
            weighted_x += mass * center_x
            weighted_y += mass * center_y

        if total_weight == 0:
            return 1.0

        # Center of mass
        com_x = weighted_x / total_weight
        com_y = weighted_y / total_weight

        # Room center
        room_center_x = self.room.width / 2
        room_center_y = self.room.depth / 2

        # Distance from room center (normalized)
        distance = math.sqrt((com_x - room_center_x)**2 + (com_y - room_center_y)**2)
        max_distance = math.sqrt(room_center_x**2 + room_center_y**2)

        return distance / max_distance

    def _check_groupings(self, placements: List[FurniturePlacement]) -> float:
        """Check for proper furniture groupings (conversation areas)."""
        penalty = 0.0

        # Find seating furniture
        seating_placements = [p for p in placements
                            if self.furniture_pieces[p.furniture_id].type == 'seating']

        if len(seating_placements) < 2:
            return 0.0

        # Check if seating forms conversation groups
        for i, seat1 in enumerate(seating_placements):
            for j, seat2 in enumerate(seating_placements):
                if i >= j:
                    continue

                distance = self._min_distance_between_furniture(seat1, seat2)

                # Ideal conversation distance is 180-300 cm
                if distance < 150 or distance > 350:
                    penalty += 0.5

        return penalty

    def _check_focal_points(self, placements: List[FurniturePlacement]) -> float:
        """Check arrangement around focal points (TV, fireplace, etc.)."""
        penalty = 0.0

        # Find TV placement
        tv_placement = None
        for placement in placements:
            if 'tv' in placement.furniture_id.lower():
                tv_placement = placement
                break

        if tv_placement is None:
            return 0.0

        # Check if seating faces TV
        seating_placements = [p for p in placements
                            if self.furniture_pieces[p.furniture_id].type == 'seating']

        for seat_placement in seating_placements:
            if not self._is_facing_towards(seat_placement, tv_placement):
                penalty += 0.3

        return penalty

    def _is_facing_towards(self, seat_placement: FurniturePlacement,
                          target_placement: FurniturePlacement) -> bool:
        """Check if seating is oriented towards target."""
        # Simplified check based on relative positions and rotation
        seat_piece = self.furniture_pieces[seat_placement.furniture_id]
        seat_width, seat_depth = self._get_rotated_dimensions(seat_piece, seat_placement.rotation)

        target_piece = self.furniture_pieces[target_placement.furniture_id]
        target_width, target_depth = self._get_rotated_dimensions(target_piece, target_placement.rotation)

        # Calculate centers
        seat_center_x = seat_placement.x + seat_width / 2
        seat_center_y = seat_placement.y + seat_depth / 2
        target_center_x = target_placement.x + target_width / 2
        target_center_y = target_placement.y + target_depth / 2

        # Calculate angle from seat to target
        dx = target_center_x - seat_center_x
        dy = target_center_y - seat_center_y
        angle_to_target = math.degrees(math.atan2(dy, dx))

        # Normalize angles
        angle_to_target = (angle_to_target + 360) % 360
        seat_facing_angle = seat_placement.rotation

        # Check if seat is roughly facing target (within 45 degrees)
        angle_diff = abs(angle_to_target - seat_facing_angle)
        angle_diff = min(angle_diff, 360 - angle_diff)

        return angle_diff <= 45

    def _check_symmetry(self, placements: List[FurniturePlacement]) -> float:
        """Check for symmetrical arrangements where appropriate."""
        # Simple symmetry check - compare left and right halves
        room_center_x = self.room.width / 2

        left_weight = 0
        right_weight = 0

        for placement in placements:
            piece = self.furniture_pieces[placement.furniture_id]
            width, depth = self._get_rotated_dimensions(piece, placement.rotation)
            center_x = placement.x + width / 2

            weight = piece.aesthetic_weight
            if center_x < room_center_x:
                left_weight += weight
            else:
                right_weight += weight

        total_weight = left_weight + right_weight
        if total_weight == 0:
            return 0.0

        # Calculate imbalance
        balance_ratio = abs(left_weight - right_weight) / total_weight
        return balance_ratio

    def is_valid_solution(self, solution: List[Dict[str, Any]]) -> bool:
        """Check if solution is valid (no overlaps, within bounds)."""
        if not solution:
            return False

        try:
            placements = []
            for item in solution:
                placements.append(FurniturePlacement(
                    furniture_id=item['furniture_id'],
                    x=item['x'],
                    y=item['y'],
                    rotation=item['rotation']
                ))

            return self._check_feasibility(placements) == 0
        except:
            return False

    def get_solution_summary(self, solution: List[Dict[str, Any]]) -> str:
        """Get human-readable summary of the solution."""
        if not solution:
            return "Empty solution"

        try:
            placements = []
            for item in solution:
                placements.append(FurniturePlacement(
                    furniture_id=item['furniture_id'],
                    x=item['x'],
                    y=item['y'],
                    rotation=item['rotation']
                ))

            total_cost = self.evaluate_solution(solution)
            is_valid = self.is_valid_solution(solution)

            # Calculate space utilization
            total_furniture_area = 0
            for placement in placements:
                piece = self.furniture_pieces[placement.furniture_id]
                width, depth = self._get_rotated_dimensions(piece, placement.rotation)
                total_furniture_area += width * depth

            room_area = self.room.width * self.room.depth
            utilization = (total_furniture_area / room_area) * 100

            summary = f"""
Furniture Arrangement Summary:
- Room: {self.room.name} ({self.room.width}×{self.room.depth} cm)
- Furniture pieces: {len(placements)}
- Total cost: {total_cost:.2f}
- Valid arrangement: {is_valid}
- Space utilization: {utilization:.1f}%
- Furniture list: {', '.join([self.furniture_pieces[p.furniture_id].name for p in placements])}
            """.strip()

            return summary
        except Exception as e:
            return f"Error analyzing solution: {str(e)}"
