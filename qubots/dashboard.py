"""
Qubots Dashboard Module
Provides built-in visualization and dashboard capabilities for optimization results.
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import base64
import io

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

@dataclass
class VisualizationData:
    """Standardized visualization data format."""
    chart_type: str
    title: str
    data: Dict[str, Any]
    layout: Dict[str, Any]
    config: Dict[str, Any] = None
    
    def to_dict(self):
        return asdict(self)

@dataclass
class DashboardResult:
    """Standardized dashboard result format for Rastion integration."""
    success: bool
    problem_name: str
    optimizer_name: str
    execution_time: float
    best_value: Optional[float] = None
    best_solution: Optional[Any] = None
    iterations: Optional[int] = None
    convergence_data: Optional[List[Dict]] = None
    visualizations: Optional[List[VisualizationData]] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.visualizations is None:
            self.visualizations = []
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert visualization data to dictionaries
        if self.visualizations:
            result['visualizations'] = [viz.to_dict() for viz in self.visualizations]
        return result
    
    def to_json(self):
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

class QubotsVisualizer:
    """Built-in visualization engine for qubots."""
    
    @staticmethod
    def create_convergence_plot(history: List[Dict], title: str = "Optimization Convergence") -> VisualizationData:
        """Create convergence plot from optimization history."""
        if not history:
            return None
        
        iterations = list(range(len(history)))
        values = [h.get('best_value', h.get('value', 0)) for h in history]
        
        if HAS_PLOTLY:
            # Create Plotly visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=iterations,
                y=values,
                mode='lines+markers',
                name='Best Value',
                line=dict(color='#2563eb', width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Iteration',
                yaxis_title='Objective Value',
                template='plotly_white',
                width=800,
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            return VisualizationData(
                chart_type='plotly',
                title=title,
                data=fig.to_dict()['data'],
                layout=fig.to_dict()['layout'],
                config={'responsive': True, 'displayModeBar': False}
            )
        
        elif HAS_MATPLOTLIB:
            # Fallback to matplotlib
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, values, 'b-o', linewidth=2, markersize=4)
            plt.title(title)
            plt.xlabel('Iteration')
            plt.ylabel('Objective Value')
            plt.grid(True, alpha=0.3)
            
            # Convert to base64 for web display
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return VisualizationData(
                chart_type='image',
                title=title,
                data={'image': f"data:image/png;base64,{image_base64}"},
                layout={}
            )
        
        else:
            # Text-based fallback
            return VisualizationData(
                chart_type='text',
                title=title,
                data={
                    'iterations': iterations,
                    'values': values,
                    'summary': f"Converged from {values[0]:.4f} to {values[-1]:.4f} in {len(values)} iterations"
                },
                layout={}
            )
    
    @staticmethod
    def create_solution_visualization(solution: Any, problem_type: str = "generic") -> Optional[VisualizationData]:
        """Create visualization for solution based on problem type."""
        if not solution:
            return None
        
        # For now, just return a simple representation
        # This can be extended for specific problem types (TSP, scheduling, etc.)
        return VisualizationData(
            chart_type='json',
            title=f'{problem_type.title()} Solution',
            data={'solution': solution},
            layout={}
        )
    
    @staticmethod
    def create_performance_dashboard(result: 'DashboardResult') -> List[VisualizationData]:
        """Create a comprehensive performance dashboard."""
        visualizations = []
        
        # Convergence plot
        if result.convergence_data:
            conv_plot = QubotsVisualizer.create_convergence_plot(
                result.convergence_data, 
                f"{result.optimizer_name} Convergence"
            )
            if conv_plot:
                visualizations.append(conv_plot)
        
        # Solution visualization
        if result.best_solution:
            sol_viz = QubotsVisualizer.create_solution_visualization(
                result.best_solution, 
                result.problem_name
            )
            if sol_viz:
                visualizations.append(sol_viz)
        
        # Performance metrics
        metrics_data = {
            'execution_time': result.execution_time,
            'best_value': result.best_value,
            'iterations': result.iterations,
            'success_rate': 1.0 if result.success else 0.0
        }
        
        visualizations.append(VisualizationData(
            chart_type='metrics',
            title='Performance Metrics',
            data=metrics_data,
            layout={}
        ))
        
        return visualizations

class QubotsAutoDashboard:
    """Automatic dashboard generation for qubots optimization results."""
    
    @staticmethod
    def create_dashboard_result(
        problem_name: str,
        optimizer_name: str,
        optimization_result: Any,
        execution_time: float,
        success: bool = True,
        error_message: str = None
    ) -> DashboardResult:
        """Create a standardized dashboard result from optimization output."""
        
        # Extract common fields from optimization result
        best_value = getattr(optimization_result, 'best_value', None)
        best_solution = getattr(optimization_result, 'best_solution', None)
        history = getattr(optimization_result, 'history', [])
        metadata = getattr(optimization_result, 'metadata', {})
        
        # Create dashboard result
        dashboard_result = DashboardResult(
            success=success,
            problem_name=problem_name,
            optimizer_name=optimizer_name,
            execution_time=execution_time,
            best_value=best_value,
            best_solution=best_solution,
            iterations=len(history) if history else None,
            convergence_data=history,
            metadata=metadata,
            error_message=error_message
        )
        
        # Generate visualizations
        if success and history:
            dashboard_result.visualizations = QubotsVisualizer.create_performance_dashboard(dashboard_result)
        
        return dashboard_result
    
    @staticmethod
    def auto_optimize_with_dashboard(
        problem,
        optimizer,
        problem_name: str = "Unknown Problem",
        optimizer_name: str = "Unknown Optimizer"
    ) -> DashboardResult:
        """Run optimization and automatically generate dashboard results."""
        
        start_time = time.time()
        
        try:
            # Run optimization
            result = optimizer.optimize(problem)
            execution_time = time.time() - start_time
            
            # Create dashboard result
            dashboard_result = QubotsAutoDashboard.create_dashboard_result(
                problem_name=problem_name,
                optimizer_name=optimizer_name,
                optimization_result=result,
                execution_time=execution_time,
                success=True
            )
            
            return dashboard_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Create error dashboard result
            dashboard_result = DashboardResult(
                success=False,
                problem_name=problem_name,
                optimizer_name=optimizer_name,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            return dashboard_result

# Export main classes and functions
__all__ = [
    'DashboardResult',
    'VisualizationData', 
    'QubotsVisualizer',
    'QubotsAutoDashboard'
]
