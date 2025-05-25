"""
Advanced Fantasy Football Optimization Demo with Benchmarking

This advanced example demonstrates:
1. Comprehensive benchmarking of multiple optimizers
2. Parameter tuning and sensitivity analysis
3. Statistical analysis of optimization results
4. Integration with OR-Tools for exact solutions
5. Performance visualization and reporting

Author: Qubots Community
Version: 1.0.0
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import warnings
from dataclasses import dataclass

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import qubots framework
import qubots
import qubots.rastion as rastion
from qubots import (
    AutoOptimizer, 
    BenchmarkSuite, 
    BenchmarkType,
    BenchmarkMetrics,
    OptimizationResult
)

# Import the main example functions
sys.path.insert(0, os.path.dirname(__file__))
from fantasy_football_optimization_example import (
    setup_authentication,
    load_fantasy_football_problem,
    SimpleRandomSearchOptimizer,
    FantasyFootballGeneticOptimizer
)

@dataclass
class OptimizerConfig:
    """Configuration for an optimizer."""
    name: str
    optimizer_class: Any
    parameters: Dict[str, Any]
    description: str

def create_optimizer_configurations() -> List[OptimizerConfig]:
    """
    Create different optimizer configurations for benchmarking.
    
    Returns:
        List of optimizer configurations
    """
    configs = []
    
    # Random Search variants
    configs.extend([
        OptimizerConfig(
            name="Random Search (Fast)",
            optimizer_class=SimpleRandomSearchOptimizer,
            parameters={"n_trials": 500},
            description="Quick random search with 500 trials"
        ),
        OptimizerConfig(
            name="Random Search (Standard)",
            optimizer_class=SimpleRandomSearchOptimizer,
            parameters={"n_trials": 2000},
            description="Standard random search with 2000 trials"
        ),
        OptimizerConfig(
            name="Random Search (Thorough)",
            optimizer_class=SimpleRandomSearchOptimizer,
            parameters={"n_trials": 5000},
            description="Thorough random search with 5000 trials"
        )
    ])
    
    # Genetic Algorithm variants
    configs.extend([
        OptimizerConfig(
            name="GA (Small Population)",
            optimizer_class=FantasyFootballGeneticOptimizer,
            parameters={
                "population_size": 30,
                "max_generations": 50,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8
            },
            description="Fast GA with small population"
        ),
        OptimizerConfig(
            name="GA (Standard)",
            optimizer_class=FantasyFootballGeneticOptimizer,
            parameters={
                "population_size": 50,
                "max_generations": 100,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8
            },
            description="Standard GA configuration"
        ),
        OptimizerConfig(
            name="GA (Large Population)",
            optimizer_class=FantasyFootballGeneticOptimizer,
            parameters={
                "population_size": 100,
                "max_generations": 150,
                "mutation_rate": 0.05,
                "crossover_rate": 0.9
            },
            description="Thorough GA with large population"
        ),
        OptimizerConfig(
            name="GA (High Mutation)",
            optimizer_class=FantasyFootballGeneticOptimizer,
            parameters={
                "population_size": 50,
                "max_generations": 100,
                "mutation_rate": 0.2,
                "crossover_rate": 0.7
            },
            description="GA with high mutation rate for exploration"
        )
    ])
    
    return configs

def run_comprehensive_benchmark(problem, configs: List[OptimizerConfig], 
                               num_runs: int = 5) -> Dict[str, List[OptimizationResult]]:
    """
    Run comprehensive benchmark across multiple optimizers and configurations.
    
    Args:
        problem: Fantasy football problem instance
        configs: List of optimizer configurations
        num_runs: Number of independent runs per configuration
        
    Returns:
        Dictionary mapping configuration names to lists of results
    """
    print(f"\n🏁 Running Comprehensive Benchmark")
    print("=" * 60)
    print(f"Configurations: {len(configs)}")
    print(f"Runs per configuration: {num_runs}")
    print(f"Total optimization runs: {len(configs) * num_runs}")
    print()
    
    all_results = {}
    
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Testing {config.name}")
        print(f"   Description: {config.description}")
        print(f"   Parameters: {config.parameters}")
        
        config_results = []
        
        for run in range(num_runs):
            print(f"   Run {run + 1}/{num_runs}...", end=" ")
            
            try:
                # Create optimizer instance
                optimizer = config.optimizer_class(**config.parameters)
                
                # Run optimization
                start_time = time.time()
                result = optimizer.optimize(problem)
                runtime = time.time() - start_time
                
                # Update runtime in result
                if hasattr(result, 'runtime_seconds'):
                    result.runtime_seconds = runtime
                
                config_results.append(result)
                
                # Quick result summary
                if result and result.best_solution is not None:
                    feasible = problem.is_feasible(result.best_solution)
                    points = result.best_value if result.best_value != float('inf') else 0
                    print(f"✅ {points:.1f} pts ({'✓' if feasible else '✗'})")
                else:
                    print("❌ Failed")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                config_results.append(None)
        
        all_results[config.name] = config_results
        print()
    
    return all_results

def analyze_benchmark_results(results: Dict[str, List[OptimizationResult]]) -> pd.DataFrame:
    """
    Analyze benchmark results and create summary statistics.
    
    Args:
        results: Dictionary mapping configuration names to lists of results
        
    Returns:
        DataFrame with summary statistics
    """
    print(f"\n📊 Analyzing Benchmark Results")
    print("=" * 50)
    
    analysis_data = []
    
    for config_name, config_results in results.items():
        # Filter valid results
        valid_results = [r for r in config_results if r is not None and r.best_solution is not None]
        
        if not valid_results:
            analysis_data.append({
                'Configuration': config_name,
                'Success Rate': 0.0,
                'Mean Points': 0.0,
                'Std Points': 0.0,
                'Best Points': 0.0,
                'Worst Points': 0.0,
                'Mean Runtime': 0.0,
                'Feasible Solutions': 0,
                'Total Runs': len(config_results)
            })
            continue
        
        # Extract metrics
        points = [r.best_value for r in valid_results if r.best_value != float('inf')]
        runtimes = [r.runtime_seconds for r in valid_results if hasattr(r, 'runtime_seconds')]
        feasible_count = sum(1 for r in valid_results if r.is_feasible)
        
        if points:
            analysis_data.append({
                'Configuration': config_name,
                'Success Rate': len(valid_results) / len(config_results),
                'Mean Points': np.mean(points),
                'Std Points': np.std(points),
                'Best Points': np.max(points),
                'Worst Points': np.min(points),
                'Mean Runtime': np.mean(runtimes) if runtimes else 0.0,
                'Feasible Solutions': feasible_count,
                'Total Runs': len(config_results)
            })
        else:
            analysis_data.append({
                'Configuration': config_name,
                'Success Rate': 0.0,
                'Mean Points': 0.0,
                'Std Points': 0.0,
                'Best Points': 0.0,
                'Worst Points': 0.0,
                'Mean Runtime': np.mean(runtimes) if runtimes else 0.0,
                'Feasible Solutions': feasible_count,
                'Total Runs': len(config_results)
            })
    
    df = pd.DataFrame(analysis_data)
    
    # Sort by mean points (descending)
    df = df.sort_values('Mean Points', ascending=False)
    
    # Display results
    print("Summary Statistics:")
    print(df.round(2).to_string(index=False))
    
    return df

def create_performance_visualization(results: Dict[str, List[OptimizationResult]], 
                                   analysis_df: pd.DataFrame):
    """
    Create performance visualization plots.
    
    Args:
        results: Dictionary mapping configuration names to lists of results
        analysis_df: DataFrame with summary statistics
    """
    print(f"\n📈 Creating Performance Visualizations")
    print("=" * 50)
    
    try:
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fantasy Football Optimization Benchmark Results', fontsize=16)
        
        # 1. Mean Points Comparison (Bar Chart)
        ax1 = axes[0, 0]
        valid_configs = analysis_df[analysis_df['Mean Points'] > 0]
        if not valid_configs.empty:
            bars = ax1.bar(range(len(valid_configs)), valid_configs['Mean Points'])
            ax1.set_xlabel('Configuration')
            ax1.set_ylabel('Mean Points')
            ax1.set_title('Mean Points by Configuration')
            ax1.set_xticks(range(len(valid_configs)))
            ax1.set_xticklabels(valid_configs['Configuration'], rotation=45, ha='right')
            
            # Add error bars
            ax1.errorbar(range(len(valid_configs)), valid_configs['Mean Points'], 
                        yerr=valid_configs['Std Points'], fmt='none', color='black', capsize=3)
        
        # 2. Runtime vs Performance Scatter Plot
        ax2 = axes[0, 1]
        if not valid_configs.empty:
            scatter = ax2.scatter(valid_configs['Mean Runtime'], valid_configs['Mean Points'], 
                                s=100, alpha=0.7, c=range(len(valid_configs)), cmap='viridis')
            ax2.set_xlabel('Mean Runtime (seconds)')
            ax2.set_ylabel('Mean Points')
            ax2.set_title('Performance vs Runtime Trade-off')
            
            # Add configuration labels
            for i, row in valid_configs.iterrows():
                ax2.annotate(row['Configuration'][:10], 
                           (row['Mean Runtime'], row['Mean Points']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 3. Success Rate Comparison
        ax3 = axes[1, 0]
        if not analysis_df.empty:
            bars = ax3.bar(range(len(analysis_df)), analysis_df['Success Rate'])
            ax3.set_xlabel('Configuration')
            ax3.set_ylabel('Success Rate')
            ax3.set_title('Success Rate by Configuration')
            ax3.set_xticks(range(len(analysis_df)))
            ax3.set_xticklabels(analysis_df['Configuration'], rotation=45, ha='right')
            ax3.set_ylim(0, 1.1)
        
        # 4. Points Distribution Box Plot
        ax4 = axes[1, 1]
        points_data = []
        labels = []
        for config_name, config_results in results.items():
            valid_results = [r for r in config_results if r is not None and r.best_solution is not None]
            points = [r.best_value for r in valid_results if r.best_value != float('inf')]
            if points:
                points_data.append(points)
                labels.append(config_name[:10])
        
        if points_data:
            ax4.boxplot(points_data, labels=labels)
            ax4.set_xlabel('Configuration')
            ax4.set_ylabel('Points')
            ax4.set_title('Points Distribution')
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(os.path.dirname(__file__), 'fantasy_football_benchmark_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to: {output_path}")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        print("   Make sure matplotlib is installed: pip install matplotlib seaborn")

def generate_detailed_report(problem, results: Dict[str, List[OptimizationResult]], 
                           analysis_df: pd.DataFrame):
    """
    Generate a detailed text report of the benchmark results.
    
    Args:
        problem: Fantasy football problem instance
        results: Dictionary mapping configuration names to lists of results
        analysis_df: DataFrame with summary statistics
    """
    print(f"\n📝 Generating Detailed Report")
    print("=" * 50)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("FANTASY FOOTBALL OPTIMIZATION BENCHMARK REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Problem Information
    report_lines.append("PROBLEM INFORMATION:")
    report_lines.append(f"  Problem Name: {problem.metadata.name}")
    report_lines.append(f"  Number of Players: {problem.n_players}")
    report_lines.append(f"  Salary Cap: ${problem.max_salary:,}")
    report_lines.append(f"  Problem Type: {problem.metadata.problem_type}")
    report_lines.append(f"  Objective: {problem.metadata.objective_type}")
    report_lines.append("")
    
    # Benchmark Summary
    total_runs = sum(len(config_results) for config_results in results.values())
    successful_runs = sum(len([r for r in config_results if r is not None and r.best_solution is not None]) 
                         for config_results in results.values())
    
    report_lines.append("BENCHMARK SUMMARY:")
    report_lines.append(f"  Total Configurations: {len(results)}")
    report_lines.append(f"  Total Runs: {total_runs}")
    report_lines.append(f"  Successful Runs: {successful_runs}")
    report_lines.append(f"  Overall Success Rate: {successful_runs/total_runs:.2%}")
    report_lines.append("")
    
    # Top Performers
    if not analysis_df.empty:
        top_3 = analysis_df.head(3)
        report_lines.append("TOP PERFORMING CONFIGURATIONS:")
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            report_lines.append(f"  {i}. {row['Configuration']}")
            report_lines.append(f"     Mean Points: {row['Mean Points']:.2f} ± {row['Std Points']:.2f}")
            report_lines.append(f"     Best Points: {row['Best Points']:.2f}")
            report_lines.append(f"     Success Rate: {row['Success Rate']:.2%}")
            report_lines.append(f"     Mean Runtime: {row['Mean Runtime']:.2f}s")
            report_lines.append("")
    
    # Detailed Results
    report_lines.append("DETAILED RESULTS BY CONFIGURATION:")
    report_lines.append("-" * 50)
    
    for config_name, config_results in results.items():
        report_lines.append(f"\n{config_name}:")
        valid_results = [r for r in config_results if r is not None and r.best_solution is not None]
        
        if valid_results:
            points = [r.best_value for r in valid_results if r.best_value != float('inf')]
            feasible_count = sum(1 for r in valid_results if r.is_feasible)
            
            report_lines.append(f"  Runs: {len(config_results)}")
            report_lines.append(f"  Successful: {len(valid_results)}")
            report_lines.append(f"  Feasible Solutions: {feasible_count}")
            
            if points:
                report_lines.append(f"  Points - Mean: {np.mean(points):.2f}, Std: {np.std(points):.2f}")
                report_lines.append(f"  Points - Best: {np.max(points):.2f}, Worst: {np.min(points):.2f}")
        else:
            report_lines.append(f"  No successful runs")
    
    # Save report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(os.path.dirname(__file__), 'fantasy_football_benchmark_report.txt')
    
    try:
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"✅ Detailed report saved to: {report_path}")
    except Exception as e:
        print(f"❌ Error saving report: {e}")
    
    # Display summary
    print("\nREPORT SUMMARY:")
    print(f"  Total configurations tested: {len(results)}")
    print(f"  Total optimization runs: {total_runs}")
    print(f"  Overall success rate: {successful_runs/total_runs:.2%}")
    
    if not analysis_df.empty and analysis_df['Mean Points'].max() > 0:
        best_config = analysis_df.iloc[0]
        print(f"  Best configuration: {best_config['Configuration']}")
        print(f"  Best mean points: {best_config['Mean Points']:.2f}")

def main():
    """Main function for advanced fantasy football optimization demo."""
    print("🏈 Advanced Fantasy Football Optimization Demo")
    print("=" * 70)
    print("This demo runs comprehensive benchmarks across multiple optimizers")
    print("and configurations to find the best approach for fantasy football.")
    print()
    
    # Setup
    setup_authentication()
    
    # Load problem
    problem = load_fantasy_football_problem()
    if problem is None:
        print("❌ Failed to load fantasy football problem. Exiting.")
        return
    
    # Create optimizer configurations
    configs = create_optimizer_configurations()
    print(f"✅ Created {len(configs)} optimizer configurations")
    
    # Run benchmark
    num_runs = 3  # Adjust based on time constraints
    results = run_comprehensive_benchmark(problem, configs, num_runs)
    
    # Analyze results
    analysis_df = analyze_benchmark_results(results)
    
    # Create visualizations
    create_performance_visualization(results, analysis_df)
    
    # Generate report
    generate_detailed_report(problem, results, analysis_df)
    
    print(f"\n🎯 Advanced Demo Completed!")
    print("Check the generated files for detailed analysis and visualizations.")

if __name__ == "__main__":
    main()
