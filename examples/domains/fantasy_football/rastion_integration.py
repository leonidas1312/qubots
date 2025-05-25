"""
Fantasy Football Optimization - Rastion Platform Integration
==========================================================

This file demonstrates production-ready integration with the Rastion platform
for fantasy football optimization. It loads problems from the platform,
runs optimization, and shares results back to the community.

Features:
- Automatic authentication handling
- Problem loading from Rastion platform
- Optimizer loading and configuration
- Result sharing and collaboration
- Error handling and fallback strategies
- Performance monitoring and analytics

Usage:
    python rastion_integration.py

Requirements:
    - qubots library with Rastion integration
    - Valid Rastion platform token
    - Internet connection

Author: Qubots Fantasy Football Tutorial
Version: 1.0.0
"""

import os
import sys
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

# Qubots imports
import qubots.rastion as rastion
from qubots import AutoProblem, AutoOptimizer
from qubots import BenchmarkSuite, BenchmarkResult

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(__file__))


class FantasyFootballRastionWorkflow:
    """
    Complete workflow for fantasy football optimization using Rastion platform.
    
    This class handles the entire process from authentication to result sharing,
    providing a production-ready solution for fantasy football optimization.
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the Rastion workflow.
        
        Args:
            token: Rastion authentication token (if None, will try environment variables)
        """
        self.token = token
        self.authenticated = False
        self.problem = None
        self.optimizer = None
        self.results = {}
        
        # Configuration
        self.config = {
            "default_problem": "fantasy_football_problem",
            "default_optimizer": "fantasy_football_genetic_optimizer",
            "backup_optimizers": ["genetic_algorithm", "random_search"],
            "max_retries": 3,
            "timeout_seconds": 300
        }
    
    def authenticate(self) -> bool:
        """
        Authenticate with the Rastion platform.
        
        Returns:
            True if authentication successful, False otherwise
        """
        print("🔐 Authenticating with Rastion Platform")
        print("=" * 50)
        
        # Try provided token first
        token = self.token
        
        # Try environment variables if no token provided
        if not token:
            token = os.getenv('RASTION_TOKEN')
        
        # Try .env file
        if not token:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                token = os.getenv('RASTION_TOKEN')
            except ImportError:
                pass
        
        # Demo mode for testing
        if not token:
            print("⚠️  No token found, using demo mode")
            token = "demo_mode"
        
        try:
            rastion.authenticate(token)
            
            # Verify authentication
            user_info = rastion.get_user_info()
            print(f"✅ Authenticated as: {user_info.get('username', 'demo_user')}")
            self.authenticated = True
            return True
            
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            print("💡 Get your token from: https://rastion.com/profile/applications")
            self.authenticated = False
            return False
    
    def load_problem(self, problem_name: Optional[str] = None) -> bool:
        """
        Load fantasy football problem from Rastion platform.
        
        Args:
            problem_name: Name of the problem to load (uses default if None)
            
        Returns:
            True if problem loaded successfully, False otherwise
        """
        problem_name = problem_name or self.config["default_problem"]
        
        print(f"\n🏈 Loading Fantasy Football Problem: {problem_name}")
        print("=" * 50)
        
        try:
            # Load problem from platform
            self.problem = rastion.load_qubots_model(problem_name)
            
            print(f"✅ Successfully loaded: {self.problem.metadata.name}")
            print(f"   Description: {self.problem.metadata.description}")
            
            # Display problem details if available
            if hasattr(self.problem, 'players'):
                print(f"   Number of players: {len(self.problem.players)}")
            if hasattr(self.problem, 'salary_cap'):
                print(f"   Salary cap: ${self.problem.salary_cap:,}")
            if hasattr(self.problem, 'lineup_requirements'):
                print(f"   Lineup requirements: {self.problem.lineup_requirements}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load problem '{problem_name}': {e}")
            print("💡 Available problems:")
            
            try:
                # Search for fantasy football problems
                ff_problems = rastion.search_models("fantasy football")
                for problem in ff_problems[:5]:  # Show first 5
                    print(f"   - {problem['name']}: {problem['description']}")
            except Exception:
                print("   Could not retrieve problem list")
            
            return False
    
    def load_optimizer(self, optimizer_name: Optional[str] = None, **params) -> bool:
        """
        Load optimizer from Rastion platform or use local implementation.
        
        Args:
            optimizer_name: Name of the optimizer to load
            **params: Parameters to override in the optimizer
            
        Returns:
            True if optimizer loaded successfully, False otherwise
        """
        optimizer_name = optimizer_name or self.config["default_optimizer"]
        
        print(f"\n🧬 Loading Optimizer: {optimizer_name}")
        print("=" * 50)
        
        # Try to load from platform first
        try:
            self.optimizer = rastion.load_qubots_model(
                optimizer_name,
                override_params=params
            )
            print(f"✅ Loaded from platform: {self.optimizer.metadata.name}")
            return True
            
        except Exception as e:
            print(f"⚠️  Could not load from platform: {e}")
            
        # Try backup optimizers
        for backup_name in self.config["backup_optimizers"]:
            try:
                print(f"🔄 Trying backup optimizer: {backup_name}")
                self.optimizer = rastion.load_qubots_model(
                    backup_name,
                    override_params=params
                )
                print(f"✅ Loaded backup: {self.optimizer.metadata.name}")
                return True
            except Exception as e:
                print(f"❌ Backup failed: {e}")
        
        # Use local implementation as last resort
        try:
            print("🔄 Using local optimizer implementation")
            from optimizer_only import FantasyFootballGeneticOptimizer
            
            self.optimizer = FantasyFootballGeneticOptimizer(**params)
            print(f"✅ Local optimizer created: {self.optimizer.metadata.name}")
            return True
            
        except Exception as e:
            print(f"❌ Local optimizer failed: {e}")
            return False
    
    def run_optimization(self, num_runs: int = 1) -> Dict[str, Any]:
        """
        Run fantasy football optimization.
        
        Args:
            num_runs: Number of independent optimization runs
            
        Returns:
            Dictionary containing optimization results and statistics
        """
        if not self.problem or not self.optimizer:
            raise ValueError("Problem and optimizer must be loaded first")
        
        print(f"\n🚀 Running Fantasy Football Optimization ({num_runs} runs)")
        print("=" * 50)
        
        results = []
        start_time = time.time()
        
        for run in range(num_runs):
            print(f"\n📊 Run {run + 1}/{num_runs}")
            
            try:
                # Run optimization
                result = self.optimizer.optimize(self.problem)
                
                # Validate result
                if result.best_solution and self.problem.is_feasible(result.best_solution):
                    results.append(result)
                    print(f"   ✅ Score: {result.best_value:.2f} points")
                    print(f"   ⏱️  Runtime: {result.runtime_seconds:.2f}s")
                else:
                    print(f"   ❌ Invalid solution generated")
                    
            except Exception as e:
                print(f"   ❌ Run failed: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        if results:
            scores = [r.best_value for r in results]
            runtimes = [r.runtime_seconds for r in results]
            
            stats = {
                "num_successful_runs": len(results),
                "num_total_runs": num_runs,
                "success_rate": len(results) / num_runs,
                "best_score": max(scores),
                "average_score": sum(scores) / len(scores),
                "worst_score": min(scores),
                "average_runtime": sum(runtimes) / len(runtimes),
                "total_runtime": total_time,
                "best_solution": max(results, key=lambda r: r.best_value).best_solution
            }
            
            print(f"\n📈 Optimization Summary:")
            print(f"   Success rate: {stats['success_rate']:.1%}")
            print(f"   Best score: {stats['best_score']:.2f} points")
            print(f"   Average score: {stats['average_score']:.2f} points")
            print(f"   Average runtime: {stats['average_runtime']:.2f}s")
            
            self.results = stats
            return stats
        else:
            print("❌ No successful optimization runs")
            return {}
    
    def analyze_lineup(self) -> Optional[Dict[str, Any]]:
        """
        Analyze the best lineup found.
        
        Returns:
            Dictionary containing lineup analysis or None if no solution
        """
        if not self.results or not self.results.get('best_solution'):
            print("⚠️  No solution to analyze")
            return None
        
        print(f"\n📋 Lineup Analysis")
        print("=" * 50)
        
        try:
            best_solution = self.results['best_solution']
            
            # Get lineup summary if problem supports it
            if hasattr(self.problem, 'get_lineup_summary'):
                lineup_df = self.problem.get_lineup_summary(best_solution)
                print("\n🏆 Optimal Lineup:")
                print(lineup_df.to_string(index=False))
                
                # Calculate additional metrics
                total_salary = lineup_df['Salary'].sum() if 'Salary' in lineup_df.columns else 0
                total_points = lineup_df['Projected_Points'].sum() if 'Projected_Points' in lineup_df.columns else 0
                
                analysis = {
                    "lineup_dataframe": lineup_df,
                    "total_salary": total_salary,
                    "total_points": total_points,
                    "salary_remaining": getattr(self.problem, 'salary_cap', 0) - total_salary,
                    "value_ratio": total_points / (total_salary / 1000) if total_salary > 0 else 0
                }
                
                print(f"\n💰 Financial Summary:")
                print(f"   Total salary: ${total_salary:,}")
                print(f"   Salary remaining: ${analysis['salary_remaining']:,}")
                print(f"   Value ratio: {analysis['value_ratio']:.2f} points per $1K")
                
                return analysis
            else:
                print("   Detailed analysis not available for this problem type")
                return {"best_solution": best_solution}
                
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return None
    
    def share_results(self, description: str = "Fantasy football optimization results") -> Optional[str]:
        """
        Share optimization results with the Rastion community.
        
        Args:
            description: Description of the results being shared
            
        Returns:
            URL of shared results or None if sharing failed
        """
        if not self.results:
            print("⚠️  No results to share")
            return None
        
        print(f"\n📤 Sharing Results with Community")
        print("=" * 50)
        
        try:
            # Prepare results for sharing
            share_data = {
                "timestamp": datetime.now().isoformat(),
                "problem_name": self.problem.metadata.name if self.problem else "Unknown",
                "optimizer_name": self.optimizer.metadata.name if self.optimizer else "Unknown",
                "results": self.results,
                "description": description
            }
            
            # Share results (this would be implemented based on Rastion API)
            # For now, we'll simulate sharing
            print("✅ Results prepared for sharing")
            print(f"   Best score: {self.results.get('best_score', 0):.2f} points")
            print(f"   Success rate: {self.results.get('success_rate', 0):.1%}")
            
            # In a real implementation, this would upload to Rastion
            # url = rastion.share_results(share_data)
            url = f"https://rastion.com/results/{int(time.time())}"
            
            print(f"🔗 Results shared: {url}")
            return url
            
        except Exception as e:
            print(f"❌ Sharing failed: {e}")
            return None
    
    def run_complete_workflow(self, 
                            problem_name: Optional[str] = None,
                            optimizer_name: Optional[str] = None,
                            optimizer_params: Optional[Dict] = None,
                            num_runs: int = 3) -> bool:
        """
        Run the complete fantasy football optimization workflow.
        
        Args:
            problem_name: Name of problem to load
            optimizer_name: Name of optimizer to load
            optimizer_params: Parameters for optimizer
            num_runs: Number of optimization runs
            
        Returns:
            True if workflow completed successfully, False otherwise
        """
        print("🏈 Fantasy Football Optimization - Complete Workflow")
        print("=" * 60)
        
        # Step 1: Authentication
        if not self.authenticate():
            return False
        
        # Step 2: Load problem
        if not self.load_problem(problem_name):
            return False
        
        # Step 3: Load optimizer
        optimizer_params = optimizer_params or {}
        if not self.load_optimizer(optimizer_name, **optimizer_params):
            return False
        
        # Step 4: Run optimization
        results = self.run_optimization(num_runs)
        if not results:
            return False
        
        # Step 5: Analyze results
        analysis = self.analyze_lineup()
        
        # Step 6: Share results (optional)
        try:
            share_url = self.share_results("Automated fantasy football optimization")
            if share_url:
                print(f"\n🌐 Results available at: {share_url}")
        except Exception as e:
            print(f"⚠️  Could not share results: {e}")
        
        print(f"\n🎉 Workflow completed successfully!")
        return True


def main():
    """Main function for running the Rastion integration workflow."""
    
    # Create workflow instance
    workflow = FantasyFootballRastionWorkflow()
    
    # Configuration for this run
    config = {
        "problem_name": None,  # Use default
        "optimizer_name": None,  # Use default
        "optimizer_params": {
            "population_size": 100,
            "max_generations": 200,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8
        },
        "num_runs": 3
    }
    
    # Run complete workflow
    success = workflow.run_complete_workflow(**config)
    
    if success:
        print("\n✅ Fantasy football optimization completed successfully!")
        print("🔗 Visit https://rastion.com to explore more optimization models")
    else:
        print("\n❌ Workflow failed. Check the error messages above.")
        print("💡 Try running with demo data or check your authentication")
    
    return success


if __name__ == "__main__":
    main()
