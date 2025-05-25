"""
Demonstration script for uploading the Fantasy Football Problem to Rastion.

This script shows how to:
1. Create and test the fantasy football problem
2. Upload it to the Rastion platform
3. Load it back from Rastion
4. Run optimization on the loaded model
"""

import sys
import os
import pandas as pd
import numpy as np

# Add current directory to path to import our fantasy football problem
sys.path.insert(0, os.path.dirname(__file__))

# Import qubots and our problem
import qubots
import qubots.rastion as rastion
from fantasy_football import FantasyFootballProblem


def test_fantasy_football_problem():
    """Test the fantasy football problem locally before uploading."""
    print("🏈 Testing Fantasy Football Problem Locally")
    print("=" * 50)
    
    try:
        # Create problem instance
        problem = FantasyFootballProblem()
        
        print(f"✅ Problem created successfully")
        print(f"   Name: {problem.metadata.name}")
        print(f"   Players: {problem.n_players}")
        print(f"   Problem Type: {problem.metadata.problem_type}")
        print(f"   Objective: {problem.metadata.objective_type}")
        
        # Test random solution generation
        print("\n🎲 Testing random solution generation...")
        solution = problem.random_solution()
        print(f"   Solution length: {len(solution)}")
        print(f"   Selected players: {sum(solution)}")
        
        # Test feasibility checking
        print("\n✅ Testing feasibility checking...")
        is_feasible = problem.is_feasible(solution)
        print(f"   Random solution feasible: {is_feasible}")
        
        # Test evaluation
        print("\n📊 Testing solution evaluation...")
        points = problem.evaluate_solution(solution)
        print(f"   Total projected points: {points:.2f}")
        
        # Show lineup summary
        print("\n📋 Lineup Summary:")
        lineup = problem.get_lineup_summary(solution)
        if not lineup.empty:
            print(lineup.to_string(index=False))
        
        return problem
        
    except Exception as e:
        print(f"❌ Error testing problem: {e}")
        return None


def upload_to_rastion(problem):
    """Upload the fantasy football problem to Rastion."""
    print("\n🚀 Uploading to Rastion Platform")
    print("=" * 50)
    #rastion.authenticate('3089deb571cf4878bcc84195f3a67ea38f3e2cb9')
    try:
        # Check if authenticated
        if not rastion.is_authenticated():
            print("⚠️  Not authenticated with Rastion.")
            print("   To upload, you need to authenticate first:")
            print("   1. Get your Gitea token from https://hub.rastion.com")
            print("   2. Run: rastion.authenticate('your_token_here')")
            print("\n   For now, we'll simulate the upload process...")
            
            # Simulate upload by showing what would be uploaded
            from qubots.rastion_client import QubotPackager
            
            package = QubotPackager.package_model(
                problem, 
                "fantasy_football_problem",
                "DraftKings fantasy football lineup optimization problem"
            )
            
            print("✅ Package created successfully!")
            print("   Files that would be uploaded:")
            for filename in package.keys():
                print(f"   📄 {filename}")
                
            print("\n📦 Package contents preview:")
            if 'config.json' in package:
                import json
                config = json.loads(package['config.json'])
                print(f"   Model type: {config.get('type', 'unknown')}")
                print(f"   Entry point: {config.get('entry_point', 'unknown')}")
                print(f"   Class name: {config.get('class_name', 'unknown')}")
            
            return None
        else:
            # Actually upload to Rastion
            print("🔐 Authenticated with Rastion. Uploading...")
            
            url = rastion.upload_model(
                model=problem,
                name="fantasy_football_problem",
                description="DraftKings fantasy football lineup optimization problem that maximizes projected points while satisfying position and salary constraints",
                requirements=["pandas", "numpy", "qubots"]
            )
            
            print(f"✅ Successfully uploaded to: {url}")
            return url
            
    except Exception as e:
        print(f"❌ Error uploading to Rastion: {e}")
        return None


def demonstrate_loading_from_rastion():
    """Demonstrate how to load the model from Rastion."""
    print("\n📥 Loading from Rastion Platform")
    print("=" * 50)
    
    print("After successful upload, users can load the model like this:")
    print()
    print("```python")
    print("import qubots.rastion as rastion")
    print()
    print("# Load the fantasy football problem")
    print("problem = rastion.load_qubots_model('fantasy_football_problem')")
    print()
    print("# Generate a random lineup")
    print("solution = problem.random_solution()")
    print()
    print("# Evaluate the lineup")
    print("points = problem.evaluate_solution(solution)")
    print("print(f'Total projected points: {points:.2f}')")
    print()
    print("# Get lineup details")
    print("lineup = problem.get_lineup_summary(solution)")
    print("print(lineup)")
    print("```")


def demonstrate_with_optimizer():
    """Show how the problem can be used with optimizers."""
    print("\n🔧 Using with Optimizers")
    print("=" * 50)
    
    print("The fantasy football problem can be used with any qubots optimizer:")
    print()
    print("```python")
    print("import qubots.rastion as rastion")
    print("from qubots import AutoOptimizer")
    print()
    print("# Load problem and optimizer")
    print("problem = rastion.load_qubots_model('fantasy_football_problem')")
    print("optimizer = AutoOptimizer.from_repo('Rastion/genetic_algorithm')")
    print()
    print("# Run optimization")
    print("result = optimizer.optimize(problem)")
    print("print(f'Best lineup points: {result.best_value:.2f}')")
    print()
    print("# Show best lineup")
    print("best_lineup = problem.get_lineup_summary(result.best_solution)")
    print("print(best_lineup)")
    print("```")


def main():
    """Main demonstration function."""
    print("🏈 Fantasy Football Problem - Rastion Upload Demo")
    print("=" * 60)
    
    # Step 1: Test the problem locally
    problem = test_fantasy_football_problem()
    
    if problem is None:
        print("❌ Local testing failed. Cannot proceed with upload.")
        return
    
    # Step 2: Upload to Rastion (or simulate)
    upload_url = upload_to_rastion(problem)
    
    # Step 3: Show how to load from Rastion
    demonstrate_loading_from_rastion()
    
    # Step 4: Show integration with optimizers
    demonstrate_with_optimizer()
    
    print("\n✨ Demo completed!")
    print("\n📚 Next Steps:")
    print("1. Get a Gitea token from https://hub.rastion.com")
    print("2. Authenticate: rastion.authenticate('your_token')")
    print("3. Run this script again to actually upload")
    print("4. Share your model with the community!")
    print("5. Try different optimizers on your problem")


if __name__ == "__main__":
    main()
