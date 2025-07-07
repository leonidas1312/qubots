"""
Furniture Arrangement Optimizer for Qubots Framework.

This optimizer uses simulated annealing to find optimal furniture arrangements
with comprehensive visualizations showing the optimization process and final layout.
"""

import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import copy

from qubots.base_optimizer import (
    BaseOptimizer, OptimizerMetadata, OptimizerType, OptimizerFamily,
    OptimizationResult
)


class FurnitureArrangementOptimizer(BaseOptimizer):
    """
    Simulated Annealing Optimizer for Furniture Arrangement Problems.
    
    Uses simulated annealing with specialized moves for spatial optimization:
    - Position adjustments (small translations)
    - Rotation changes (90-degree increments)
    - Furniture swapping
    - Random restarts for exploration
    
    Features comprehensive visualizations including:
    - Room layout with furniture placement
    - Clearance zones and traffic flow
    - Optimization progress tracking
    - Before/after comparisons
    """
    
    def __init__(self,
                 initial_temperature: float = 1000.0,
                 cooling_rate: float = 0.95,
                 min_temperature: float = 1.0,
                 max_iterations: int = 5000,
                 moves_per_temperature: int = 50,
                 position_step_size: float = 20.0,  # cm
                 create_plots: bool = True,
                 save_plots: bool = False,
                 plot_interval: int = 500,
                 random_seed: Optional[int] = None,
                 **kwargs):
        
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.moves_per_temperature = moves_per_temperature
        self.position_step_size = position_step_size
        self.create_plots = create_plots
        self.save_plots = save_plots
        self.plot_interval = plot_interval
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Create metadata
        metadata = OptimizerMetadata(
            name="Furniture Arrangement Optimizer",
            description="Simulated annealing optimizer for furniture placement with spatial visualizations",
            optimizer_type=OptimizerType.METAHEURISTIC,
            optimizer_family=OptimizerFamily.LOCAL_SEARCH,
            author="Qubots Framework",
            version="1.0.0",
            is_deterministic=False,
            supports_constraints=True,
            supports_continuous=False,
            supports_discrete=True,
            time_complexity="O(iterations × moves_per_temp)",
            space_complexity="O(furniture_count)",
            convergence_guaranteed=False,
            parallel_capable=False
        )

        # Initialize base optimizer
        super().__init__(metadata, **kwargs)

    def _get_default_metadata(self) -> OptimizerMetadata:
        """Get default metadata for the furniture arrangement optimizer."""
        return self.metadata
    
    def _optimize_implementation(self, problem, initial_solution=None) -> OptimizationResult:
        """
        Optimize furniture arrangement using simulated annealing.
        """
        # Initialize tracking variables
        best_solution = None
        best_cost = float('inf')
        current_solution = initial_solution if initial_solution is not None else problem.random_solution()
        current_cost = problem.evaluate_solution(current_solution)
        
        # Tracking for visualization
        cost_history = []
        temperature_history = []
        acceptance_history = []
        
        # Initial visualization
        if self.create_plots:
            self._create_initial_plot(problem, current_solution)
        
        # Simulated annealing loop
        temperature = self.initial_temperature
        iteration = 0
        accepted_moves = 0
        
        while temperature > self.min_temperature and iteration < self.max_iterations:
            for _ in range(self.moves_per_temperature):
                # Generate neighbor solution
                neighbor_solution = self._generate_neighbor(current_solution, problem)
                neighbor_cost = problem.evaluate_solution(neighbor_solution)
                
                # Accept or reject move
                if self._accept_move(current_cost, neighbor_cost, temperature):
                    current_solution = neighbor_solution
                    current_cost = neighbor_cost
                    accepted_moves += 1
                    
                    # Update best solution
                    if current_cost < best_cost:
                        best_solution = copy.deepcopy(current_solution)
                        best_cost = current_cost
                
                iteration += 1
                
                # Record progress
                cost_history.append(current_cost)
                temperature_history.append(temperature)
                acceptance_history.append(accepted_moves / max(1, iteration))
                
                # Periodic visualization
                if self.create_plots and iteration % self.plot_interval == 0:
                    self._create_progress_plot(problem, current_solution, iteration, 
                                             cost_history, temperature_history)
            
            # Cool down
            temperature *= self.cooling_rate
        
        # Final visualization
        if self.create_plots:
            self._create_final_plots(problem, best_solution, cost_history, 
                                   temperature_history, acceptance_history)
        
        # Calculate additional metrics
        acceptance_rate = accepted_moves / max(1, iteration)
        
        return OptimizationResult(
            best_solution=best_solution,
            best_value=best_cost,
            iterations=iteration,
            evaluations=iteration,
            termination_reason="temperature_threshold" if temperature <= self.min_temperature else "max_iterations",
            convergence_achieved=best_cost < 1000,  # Reasonable arrangement found
            additional_metrics={
                'final_temperature': temperature,
                'acceptance_rate': acceptance_rate,
                'total_moves_evaluated': iteration,
                'moves_accepted': accepted_moves,
                'space_utilization': self._calculate_space_utilization(problem, best_solution),
                'is_valid_arrangement': problem.is_valid_solution(best_solution) if best_solution else False
            }
        )
    
    def _generate_neighbor(self, solution: List[Dict], problem) -> List[Dict]:
        """Generate a neighbor solution using various move types."""
        neighbor = copy.deepcopy(solution)
        
        if not neighbor:
            return problem.random_solution()
        
        # Choose move type randomly
        move_type = random.choice(['translate', 'rotate', 'swap'])
        
        if move_type == 'translate':
            # Translate a random furniture piece
            idx = random.randint(0, len(neighbor) - 1)
            furniture_id = neighbor[idx]['furniture_id']
            piece = problem.furniture_pieces[furniture_id]
            
            # Get current dimensions
            width, depth = problem._get_rotated_dimensions(piece, neighbor[idx]['rotation'])
            
            # Generate new position
            max_x = max(0, problem.room.width - width)
            max_y = max(0, problem.room.depth - depth)
            
            # Small random step
            new_x = neighbor[idx]['x'] + random.uniform(-self.position_step_size, self.position_step_size)
            new_y = neighbor[idx]['y'] + random.uniform(-self.position_step_size, self.position_step_size)
            
            # Clamp to room bounds
            neighbor[idx]['x'] = max(0, min(max_x, new_x))
            neighbor[idx]['y'] = max(0, min(max_y, new_y))
            
        elif move_type == 'rotate' and len(neighbor) > 0:
            # Rotate a random furniture piece
            idx = random.randint(0, len(neighbor) - 1)
            furniture_id = neighbor[idx]['furniture_id']
            piece = problem.furniture_pieces[furniture_id]
            
            if piece.can_rotate:
                # Rotate by 90 degrees
                current_rotation = neighbor[idx]['rotation']
                new_rotation = (current_rotation + 90) % 360
                neighbor[idx]['rotation'] = new_rotation
                
                # Adjust position if needed to stay in bounds
                width, depth = problem._get_rotated_dimensions(piece, new_rotation)
                max_x = max(0, problem.room.width - width)
                max_y = max(0, problem.room.depth - depth)
                
                neighbor[idx]['x'] = min(neighbor[idx]['x'], max_x)
                neighbor[idx]['y'] = min(neighbor[idx]['y'], max_y)
        
        elif move_type == 'swap' and len(neighbor) > 1:
            # Swap positions of two furniture pieces
            idx1, idx2 = random.sample(range(len(neighbor)), 2)
            
            # Swap positions
            neighbor[idx1]['x'], neighbor[idx2]['x'] = neighbor[idx2]['x'], neighbor[idx1]['x']
            neighbor[idx1]['y'], neighbor[idx2]['y'] = neighbor[idx2]['y'], neighbor[idx1]['y']
        
        return neighbor
    
    def _accept_move(self, current_cost: float, neighbor_cost: float, temperature: float) -> bool:
        """Decide whether to accept a move based on simulated annealing criteria."""
        if neighbor_cost < current_cost:
            return True
        
        if temperature <= 0:
            return False
        
        # Probability of accepting worse solution
        probability = math.exp(-(neighbor_cost - current_cost) / temperature)
        return random.random() < probability
    
    def _calculate_space_utilization(self, problem, solution: List[Dict]) -> float:
        """Calculate space utilization percentage."""
        if not solution:
            return 0.0
        
        total_furniture_area = 0
        for item in solution:
            furniture_id = item['furniture_id']
            piece = problem.furniture_pieces[furniture_id]
            width, depth = problem._get_rotated_dimensions(piece, item['rotation'])
            total_furniture_area += width * depth
        
        room_area = problem.room.width * problem.room.depth
        return (total_furniture_area / room_area) * 100

    def _create_initial_plot(self, problem, solution: List[Dict]):
        """Create initial room layout visualization."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        self._plot_room_layout(ax, problem, solution, "Initial Random Arrangement")

        plt.tight_layout()
        if self.save_plots:
            plt.savefig('furniture_initial_layout.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _create_progress_plot(self, problem, solution: List[Dict], iteration: int,
                            cost_history: List[float], temperature_history: List[float]):
        """Create progress visualization during optimization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Room layout
        self._plot_room_layout(ax1, problem, solution, f"Current Layout (Iteration {iteration})")

        # Cost history
        ax2.plot(cost_history, 'b-', alpha=0.7, linewidth=1)
        ax2.set_title('Cost Evolution')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Cost')
        ax2.grid(True, alpha=0.3)

        # Temperature history
        ax3.plot(temperature_history, 'r-', alpha=0.7, linewidth=1)
        ax3.set_title('Temperature Schedule')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Temperature')
        ax3.grid(True, alpha=0.3)

        # Cost vs Temperature
        if len(cost_history) > 1 and len(temperature_history) > 1:
            scatter = ax4.scatter(temperature_history, cost_history,
                                c=range(len(cost_history)), cmap='viridis', alpha=0.6, s=1)
            ax4.set_title('Cost vs Temperature')
            ax4.set_xlabel('Temperature')
            ax4.set_ylabel('Cost')
            ax4.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax4, label='Iteration')

        plt.tight_layout()
        if self.save_plots:
            plt.savefig(f'furniture_progress_{iteration}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _create_final_plots(self, problem, best_solution: List[Dict],
                          cost_history: List[float], temperature_history: List[float],
                          acceptance_history: List[float]):
        """Create comprehensive final visualization."""
        fig = plt.figure(figsize=(20, 16))

        # Main room layout (large subplot)
        ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        self._plot_detailed_room_layout(ax_main, problem, best_solution, "Optimized Furniture Arrangement")

        # Cost evolution
        ax_cost = plt.subplot2grid((3, 3), (0, 2))
        ax_cost.plot(cost_history, 'b-', alpha=0.8, linewidth=1.5)
        ax_cost.set_title('Cost Evolution', fontsize=12, fontweight='bold')
        ax_cost.set_xlabel('Iteration')
        ax_cost.set_ylabel('Cost')
        ax_cost.grid(True, alpha=0.3)

        # Temperature schedule
        ax_temp = plt.subplot2grid((3, 3), (1, 2))
        ax_temp.plot(temperature_history, 'r-', alpha=0.8, linewidth=1.5)
        ax_temp.set_title('Temperature Schedule', fontsize=12, fontweight='bold')
        ax_temp.set_xlabel('Iteration')
        ax_temp.set_ylabel('Temperature')
        ax_temp.grid(True, alpha=0.3)

        # Acceptance rate
        ax_accept = plt.subplot2grid((3, 3), (2, 0))
        ax_accept.plot(acceptance_history, 'g-', alpha=0.8, linewidth=1.5)
        ax_accept.set_title('Acceptance Rate', fontsize=12, fontweight='bold')
        ax_accept.set_xlabel('Iteration')
        ax_accept.set_ylabel('Acceptance Rate')
        ax_accept.grid(True, alpha=0.3)

        # Solution metrics
        ax_metrics = plt.subplot2grid((3, 3), (2, 1), colspan=2)
        self._plot_solution_metrics(ax_metrics, problem, best_solution)

        plt.tight_layout()
        if self.save_plots:
            plt.savefig('furniture_final_optimization.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_room_layout(self, ax, problem, solution: List[Dict], title: str):
        """Plot basic room layout with furniture."""
        # Clear axis
        ax.clear()

        # Room boundaries
        room_rect = Rectangle((0, 0), problem.room.width, problem.room.depth,
                            linewidth=3, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_patch(room_rect)

        # Door
        door_rect = Rectangle((problem.room.door_x, problem.room.door_y),
                            problem.room.door_width, 20,
                            linewidth=2, edgecolor='brown', facecolor='brown')
        ax.add_patch(door_rect)

        # Windows
        for wall, start, end in problem.room.windows:
            if pd.notna(wall):
                self._draw_window(ax, problem, wall, start, end)

        # Furniture
        colors = plt.cm.Set3(np.linspace(0, 1, len(solution)))
        for i, item in enumerate(solution):
            self._draw_furniture(ax, problem, item, colors[i])

        # Formatting
        ax.set_xlim(-50, problem.room.width + 50)
        ax.set_ylim(-50, problem.room.depth + 50)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Width (cm)')
        ax.set_ylabel('Depth (cm)')
        ax.grid(True, alpha=0.3)

    def _plot_detailed_room_layout(self, ax, problem, solution: List[Dict], title: str):
        """Plot detailed room layout with clearances and labels."""
        self._plot_room_layout(ax, problem, solution, title)

        # Add clearance zones
        for item in solution:
            self._draw_clearance_zones(ax, problem, item)

        # Add furniture labels
        for item in solution:
            piece = problem.furniture_pieces[item['furniture_id']]
            width, depth = problem._get_rotated_dimensions(piece, item['rotation'])
            center_x = item['x'] + width / 2
            center_y = item['y'] + depth / 2
            ax.text(center_x, center_y, piece.name.split()[0],
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    def _draw_window(self, ax, problem, wall: str, start: float, end: float):
        """Draw window on specified wall."""
        if wall == 'north':
            window_rect = Rectangle((start, -5), end - start, 10,
                                  linewidth=2, edgecolor='lightblue', facecolor='lightblue')
        elif wall == 'south':
            window_rect = Rectangle((start, problem.room.depth - 5), end - start, 10,
                                  linewidth=2, edgecolor='lightblue', facecolor='lightblue')
        elif wall == 'west':
            window_rect = Rectangle((-5, start), 10, end - start,
                                  linewidth=2, edgecolor='lightblue', facecolor='lightblue')
        elif wall == 'east':
            window_rect = Rectangle((problem.room.width - 5, start), 10, end - start,
                                  linewidth=2, edgecolor='lightblue', facecolor='lightblue')
        else:
            return

        ax.add_patch(window_rect)

    def _draw_furniture(self, ax, problem, item: Dict, color):
        """Draw individual furniture piece."""
        piece = problem.furniture_pieces[item['furniture_id']]
        width, depth = problem._get_rotated_dimensions(piece, item['rotation'])

        # Main furniture rectangle
        furniture_rect = Rectangle((item['x'], item['y']), width, depth,
                                 linewidth=2, edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(furniture_rect)

        # Add rotation indicator for rotatable furniture
        if piece.can_rotate and item['rotation'] != 0:
            # Small arrow to show orientation
            center_x = item['x'] + width / 2
            center_y = item['y'] + depth / 2

            # Arrow direction based on rotation
            if item['rotation'] == 90:
                dx, dy = 10, 0
            elif item['rotation'] == 180:
                dx, dy = 0, -10
            elif item['rotation'] == 270:
                dx, dy = -10, 0
            else:
                dx, dy = 0, 10

            ax.arrow(center_x, center_y, dx, dy, head_width=5, head_length=3,
                    fc='red', ec='red', alpha=0.8)

    def _draw_clearance_zones(self, ax, problem, item: Dict):
        """Draw clearance zones around furniture."""
        piece = problem.furniture_pieces[item['furniture_id']]
        width, depth = problem._get_rotated_dimensions(piece, item['rotation'])

        # Front clearance zone
        if piece.clearance_front > 0:
            if item['rotation'] == 0:  # Front faces north
                clear_rect = Rectangle((item['x'], item['y'] - piece.clearance_front),
                                     width, piece.clearance_front,
                                     linewidth=1, edgecolor='red', facecolor='red', alpha=0.1)
            elif item['rotation'] == 90:  # Front faces east
                clear_rect = Rectangle((item['x'] + width, item['y']),
                                     piece.clearance_front, depth,
                                     linewidth=1, edgecolor='red', facecolor='red', alpha=0.1)
            elif item['rotation'] == 180:  # Front faces south
                clear_rect = Rectangle((item['x'], item['y'] + depth),
                                     width, piece.clearance_front,
                                     linewidth=1, edgecolor='red', facecolor='red', alpha=0.1)
            elif item['rotation'] == 270:  # Front faces west
                clear_rect = Rectangle((item['x'] - piece.clearance_front, item['y']),
                                     piece.clearance_front, depth,
                                     linewidth=1, edgecolor='red', facecolor='red', alpha=0.1)

            ax.add_patch(clear_rect)

    def _plot_solution_metrics(self, ax, problem, solution: List[Dict]):
        """Plot solution quality metrics."""
        ax.clear()

        if not solution:
            ax.text(0.5, 0.5, 'No Solution', ha='center', va='center', transform=ax.transAxes)
            return

        # Calculate metrics
        total_cost = problem.evaluate_solution(solution)
        space_util = self._calculate_space_utilization(problem, solution)
        is_valid = problem.is_valid_solution(solution)

        # Create metrics text
        metrics_text = f"""
Solution Quality Metrics:

Total Cost: {total_cost:.2f}
Space Utilization: {space_util:.1f}%
Valid Arrangement: {is_valid}
Room: {problem.room.name}
Furniture Pieces: {len(solution)}

Furniture List:
{chr(10).join([f"• {problem.furniture_pieces[item['furniture_id']].name}" for item in solution[:8]])}
{"..." if len(solution) > 8 else ""}
        """.strip()

        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Solution Summary', fontsize=12, fontweight='bold')
