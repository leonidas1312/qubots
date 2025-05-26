#!/usr/bin/env python3
"""
Real-Time Optimization Logging Demo

This script demonstrates the enhanced qubots framework with real-time logging
capabilities for the Rastion platform playground. It shows how optimization
progress is captured and can be streamed to web interfaces.

Usage:
    python streaming_optimization_demo.py

Features demonstrated:
- Real-time progress logging during optimization
- Detailed iteration-by-iteration progress tracking
- Error handling and logging
- Integration with playground streaming infrastructure
"""

import time
import sys
from typing import List, Dict, Any

# Add qubots to path for local testing
sys.path.insert(0, '..')

from qubots import BaseProblem, BaseOptimizer, OptimizationResult
from qubots.playground_integration import execute_playground_optimization


class StreamingLogCollector:
    """Collects logs for demonstration purposes."""
    
    def __init__(self):
        self.logs = []
        self.start_time = time.time()
    
    def log_callback(self, level: str, message: str, source: str):
        """Callback function to collect logs."""
        timestamp = time.time() - self.start_time
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message,
            'source': source
        }
        self.logs.append(log_entry)
        
        # Print to console with formatting
        print(f"[{timestamp:6.2f}s] [{level.upper():5s}] [{source:10s}] {message}")
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all collected logs."""
        return self.logs.copy()
    
    def print_summary(self):
        """Print a summary of collected logs."""
        print("\n" + "="*80)
        print("LOG SUMMARY")
        print("="*80)
        
        by_level = {}
        by_source = {}
        
        for log in self.logs:
            level = log['level']
            source = log['source']
            
            by_level[level] = by_level.get(level, 0) + 1
            by_source[source] = by_source.get(source, 0) + 1
        
        print(f"Total logs: {len(self.logs)}")
        print(f"Duration: {self.logs[-1]['timestamp']:.2f} seconds" if self.logs else "0 seconds")
        print("\nBy level:")
        for level, count in sorted(by_level.items()):
            print(f"  {level}: {count}")
        
        print("\nBy source:")
        for source, count in sorted(by_source.items()):
            print(f"  {source}: {count}")


def demo_local_optimization():
    """Demonstrate local optimization with real-time logging."""
    print("🎮 Qubots Real-Time Logging Demo")
    print("="*50)
    
    # Create log collector
    log_collector = StreamingLogCollector()
    
    try:
        # Import example models
        from examples.vehicle_routing_problem.qubot import VehicleRoutingProblem
        from examples.genetic_vrp_optimizer.qubot import GeneticVRPOptimizer
        
        print("📊 Creating VRP problem instance...")
        problem = VehicleRoutingProblem(
            n_customers=10,
            n_vehicles=3,
            depot_x=0,
            depot_y=0,
            area_width=100,
            area_height=100,
            min_demand=1,
            max_demand=10,
            vehicle_capacity=30
        )
        
        print("🧬 Creating Genetic Algorithm optimizer...")
        optimizer = GeneticVRPOptimizer(
            population_size=20,
            generations=50,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elite_size=3
        )
        
        print("\n🚀 Starting optimization with real-time logging...")
        print("-" * 50)
        
        # Run optimization with logging
        result = optimizer.optimize(
            problem=problem,
            log_callback=log_collector.log_callback
        )
        
        print("-" * 50)
        print("✅ Optimization completed!")
        print(f"Best solution value: {result.best_value:.6f}")
        print(f"Runtime: {result.runtime_seconds:.3f} seconds")
        print(f"Iterations: {result.iterations}")
        
        # Print log summary
        log_collector.print_summary()
        
        return result, log_collector.get_logs()
        
    except ImportError as e:
        print(f"❌ Error importing example models: {e}")
        print("Make sure you're running from the qubots root directory")
        return None, []
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        return None, log_collector.get_logs()


def demo_playground_integration():
    """Demonstrate playground integration with logging."""
    print("\n🎮 Playground Integration Demo")
    print("="*50)
    
    # Create log collector
    log_collector = StreamingLogCollector()
    
    try:
        print("🔗 Testing playground integration with logging...")
        
        # This would normally be called by the playground service
        result = execute_playground_optimization(
            problem_dir="examples/vehicle_routing_problem",
            optimizer_dir="examples/genetic_vrp_optimizer",
            log_callback=log_collector.log_callback,
            problem_params={
                "n_customers": 8,
                "n_vehicles": 2
            },
            optimizer_params={
                "population_size": 15,
                "generations": 30
            }
        )
        
        print("✅ Playground integration test completed!")
        print(f"Success: {result.get('success', False)}")
        
        if result.get('success'):
            print(f"Best value: {result.get('best_value', 'N/A')}")
            print(f"Execution time: {result.get('execution_time', 'N/A')}")
        
        # Print log summary
        log_collector.print_summary()
        
        return result, log_collector.get_logs()
        
    except Exception as e:
        print(f"❌ Error in playground integration: {e}")
        return None, log_collector.get_logs()


def demo_websocket_simulation():
    """Simulate WebSocket message format for frontend."""
    print("\n🔌 WebSocket Message Simulation")
    print("="*50)
    
    # Create log collector
    log_collector = StreamingLogCollector()
    
    def websocket_log_callback(level: str, message: str, source: str):
        """Simulate WebSocket message sending."""
        # Collect log normally
        log_collector.log_callback(level, message, source)
        
        # Simulate WebSocket message
        websocket_message = {
            "type": "optimization_log",
            "data": {
                "timestamp": time.time(),
                "level": level,
                "message": message,
                "source": source
            }
        }
        
        # In real implementation, this would be sent via WebSocket
        print(f"📡 WebSocket: {websocket_message}")
    
    try:
        # Import and run a quick optimization
        from examples.vehicle_routing_problem.qubot import VehicleRoutingProblem
        from examples.genetic_vrp_optimizer.qubot import GeneticVRPOptimizer
        
        problem = VehicleRoutingProblem(n_customers=5, n_vehicles=2)
        optimizer = GeneticVRPOptimizer(population_size=10, generations=20)
        
        print("🚀 Running optimization with WebSocket simulation...")
        result = optimizer.optimize(problem, log_callback=websocket_log_callback)
        
        print("✅ WebSocket simulation completed!")
        return result, log_collector.get_logs()
        
    except Exception as e:
        print(f"❌ Error in WebSocket simulation: {e}")
        return None, log_collector.get_logs()


def main():
    """Main demo function."""
    print("🎯 Qubots Real-Time Logging Demonstration")
    print("This demo shows the enhanced logging capabilities for the Rastion playground")
    print()
    
    # Demo 1: Local optimization with logging
    result1, logs1 = demo_local_optimization()
    
    # Demo 2: Playground integration
    result2, logs2 = demo_playground_integration()
    
    # Demo 3: WebSocket simulation
    result3, logs3 = demo_websocket_simulation()
    
    # Final summary
    print("\n" + "="*80)
    print("DEMO SUMMARY")
    print("="*80)
    print(f"Local optimization: {'✅ Success' if result1 else '❌ Failed'}")
    print(f"Playground integration: {'✅ Success' if result2 else '❌ Failed'}")
    print(f"WebSocket simulation: {'✅ Success' if result3 else '❌ Failed'}")
    print(f"Total logs collected: {len(logs1) + len(logs2) + len(logs3)}")
    print()
    print("🎉 Real-time logging is ready for the Rastion playground!")
    print("Users will now see detailed optimization progress in the terminal viewer.")


if __name__ == "__main__":
    main()
