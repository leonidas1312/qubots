# Fantasy Football Optimization Tutorial

This comprehensive tutorial demonstrates how to use qubots for fantasy football optimization, including the recommended 3-file structure for production deployment and Rastion platform integration.

## 🎯 Learning Objectives

After completing this tutorial, you will be able to:
- Understand fantasy football optimization as a constrained optimization problem
- Implement fantasy football optimizers using qubots
- Use the 3-file structure for production deployment
- Integrate with the Rastion platform for data loading and result sharing
- Handle real-world constraints (salary caps, position requirements, player exclusions)
- Benchmark different optimization strategies

## 📋 Prerequisites

- Completed the [Getting Started Tutorial](getting_started.md)
- Basic understanding of fantasy football rules
- Familiarity with constraint optimization

## 🏈 Fantasy Football as an Optimization Problem

Fantasy football lineup optimization is a classic **knapsack problem with constraints**:

### Objective
Maximize projected points while staying within constraints

### Constraints
- **Salary Cap**: Total salary ≤ budget (e.g., $50,000)
- **Position Requirements**: Exact number of each position (QB, RB, WR, TE, FLEX, DST, K)
- **Player Uniqueness**: Each player selected at most once
- **Team Limits**: Maximum players from same NFL team (optional)

### Mathematical Formulation
```
Maximize: Σ(points_i × x_i)
Subject to:
- Σ(salary_i × x_i) ≤ salary_cap
- Σ(x_i for i in position_j) = required_count_j  ∀ positions j
- x_i ∈ {0, 1}  ∀ players i
```

## 📁 The 3-File Structure

For production fantasy football optimizers, we recommend this structure:

### 1. `local_testing.py` - Local Development and Testing
- Complete standalone implementation
- Sample data generation
- Testing and validation
- Development and debugging

### 2. `optimizer_only.py` - Pure Optimizer Class
- Only the optimizer class definition
- No data loading or external dependencies
- Ready for upload to Rastion platform
- Reusable across different problems

### 3. `rastion_integration.py` - Platform Integration
- Loads problems from Rastion platform
- Handles authentication and data fetching
- Runs optimization and shares results
- Production deployment ready

Let's implement each file:

## 📄 File 1: Local Testing (`local_testing.py`)

```python
"""
Fantasy Football Optimization - Local Testing
============================================

Complete standalone implementation for local development and testing.
Includes sample data generation and comprehensive testing.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

# Qubots imports
from qubots import (
    BaseProblem, BaseOptimizer,
    ProblemMetadata, OptimizerMetadata,
    ProblemType, ObjectiveType, DifficultyLevel,
    OptimizerType, OptimizerFamily,
    OptimizationResult
)

@dataclass
class Player:
    """Represents a fantasy football player."""
    name: str
    position: str
    team: str
    salary: int
    projected_points: float
    ownership_pct: float = 0.0
    
class FantasyFootballProblem(BaseProblem):
    """
    Fantasy Football lineup optimization problem.
    
    Maximize projected points while satisfying salary cap and position constraints.
    """
    
    def __init__(self, 
                 players: List[Player],
                 salary_cap: int = 50000,
                 lineup_requirements: Dict[str, int] = None):
        """
        Initialize fantasy football problem.
        
        Args:
            players: List of available players
            salary_cap: Maximum total salary
            lineup_requirements: Required number of each position
        """
        self.players = players
        self.salary_cap = salary_cap
        self.lineup_requirements = lineup_requirements or {
            'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1, 'DST': 1, 'K': 1
        }
        
        # Create position mappings
        self.position_players = {}
        for pos in self.lineup_requirements:
            if pos == 'FLEX':
                # FLEX can be RB, WR, or TE
                self.position_players[pos] = [
                    i for i, p in enumerate(players) 
                    if p.position in ['RB', 'WR', 'TE']
                ]
            else:
                self.position_players[pos] = [
                    i for i, p in enumerate(players) 
                    if p.position == pos
                ]
        
        metadata = ProblemMetadata(
            name="Fantasy Football Lineup Optimization",
            description=f"Optimize DraftKings lineup with {len(players)} players",
            problem_type=ProblemType.COMBINATORIAL,
            objective_type=ObjectiveType.MAXIMIZE,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            domain="fantasy_sports",
            tags={"fantasy_football", "knapsack", "combinatorial", "sports"},
            author="Fantasy Football Tutorial",
            version="1.0.0",
            dimension=len(players)
        )
        
        super().__init__(metadata)
    
    def evaluate_solution(self, solution: List[int]) -> float:
        """
        Evaluate a fantasy football lineup.
        
        Args:
            solution: List of player indices representing the lineup
            
        Returns:
            Total projected points (negative if infeasible for minimization algorithms)
        """
        if not self.is_feasible(solution):
            return -float('inf')  # Infeasible solutions get very low score
        
        total_points = sum(self.players[i].projected_points for i in solution)
        return total_points
    
    def is_feasible(self, solution: List[int]) -> bool:
        """Check if lineup satisfies all constraints."""
        if not solution:
            return False
        
        # Check for duplicates
        if len(solution) != len(set(solution)):
            return False
        
        # Check salary cap
        total_salary = sum(self.players[i].salary for i in solution)
        if total_salary > self.salary_cap:
            return False
        
        # Check position requirements
        position_counts = {pos: 0 for pos in self.lineup_requirements}
        used_flex_positions = []
        
        for player_idx in solution:
            player = self.players[player_idx]
            pos = player.position
            
            if pos in position_counts and position_counts[pos] < self.lineup_requirements[pos]:
                position_counts[pos] += 1
            elif pos in ['RB', 'WR', 'TE'] and position_counts['FLEX'] < self.lineup_requirements['FLEX']:
                position_counts['FLEX'] += 1
                used_flex_positions.append(pos)
            else:
                return False  # Can't place this player
        
        # Check all positions are filled
        for pos, required in self.lineup_requirements.items():
            if position_counts[pos] != required:
                return False
        
        return True
    
    def get_random_solution(self) -> List[int]:
        """Generate a random feasible lineup."""
        lineup = []
        
        # Fill required positions
        for pos, count in self.lineup_requirements.items():
            available_players = [
                i for i in self.position_players[pos] 
                if i not in lineup
            ]
            
            if len(available_players) < count:
                # Not enough players for this position, try again
                return self.get_random_solution()
            
            selected = np.random.choice(available_players, size=count, replace=False)
            lineup.extend(selected)
        
        # Validate salary constraint
        total_salary = sum(self.players[i].salary for i in lineup)
        if total_salary <= self.salary_cap:
            return lineup
        else:
            # Try again if over salary cap
            return self.get_random_solution()
    
    def get_lineup_summary(self, solution: List[int]) -> pd.DataFrame:
        """Get detailed lineup summary as DataFrame."""
        lineup_data = []
        for player_idx in solution:
            player = self.players[player_idx]
            lineup_data.append({
                'Name': player.name,
                'Position': player.position,
                'Team': player.team,
                'Salary': player.salary,
                'Projected_Points': player.projected_points,
                'Value': player.projected_points / (player.salary / 1000)  # Points per $1K
            })
        
        df = pd.DataFrame(lineup_data)
        return df.sort_values('Position')

class FantasyFootballGeneticOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimized for fantasy football lineup construction.
    """
    
    def __init__(self,
                 population_size: int = 100,
                 max_generations: int = 200,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elitism_count: int = 5):
        
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        
        metadata = OptimizerMetadata(
            name="Fantasy Football Genetic Algorithm",
            description="Genetic algorithm specialized for fantasy football lineup optimization",
            optimizer_type=OptimizerType.METAHEURISTIC,
            optimizer_family=OptimizerFamily.EVOLUTIONARY,
            author="Fantasy Football Tutorial",
            version="1.0.0",
            supports_constraints=True,
            typical_problems=["fantasy_football", "knapsack", "combinatorial"]
        )
        
        super().__init__(metadata, **locals())
    
    def _optimize_implementation(self, problem: FantasyFootballProblem, initial_solution=None):
        """Run genetic algorithm optimization for fantasy football."""
        start_time = time.time()
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            lineup = problem.get_random_solution()
            population.append(lineup)
        
        best_solution = None
        best_value = -float('inf')
        evaluations = 0
        
        for generation in range(self.max_generations):
            # Evaluate population
            fitness_values = []
            for individual in population:
                fitness = problem.evaluate_solution(individual)
                fitness_values.append(fitness)
                evaluations += 1
                
                if fitness > best_value:
                    best_solution = individual.copy()
                    best_value = fitness
            
            # Selection and reproduction
            new_population = []
            
            # Elitism
            elite_indices = np.argsort(fitness_values)[-self.elitism_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_values)
                parent2 = self._tournament_selection(population, fitness_values)
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._fantasy_crossover(parent1, parent2, problem)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if np.random.random() < self.mutation_rate:
                    child1 = self._fantasy_mutation(child1, problem)
                if np.random.random() < self.mutation_rate:
                    child2 = self._fantasy_mutation(child2, problem)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        end_time = time.time()
        
        return OptimizationResult(
            best_solution=best_solution,
            best_value=best_value,
            iterations=self.max_generations,
            evaluations=evaluations,
            runtime_seconds=end_time - start_time,
            convergence_achieved=True,
            termination_reason="Maximum generations reached"
        )
    
    def _tournament_selection(self, population, fitness_values, tournament_size=3):
        """Tournament selection for genetic algorithm."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _fantasy_crossover(self, parent1, parent2, problem):
        """Fantasy football specific crossover operation."""
        # Position-based crossover: preserve position structure
        child1, child2 = [], []
        
        for pos, count in problem.lineup_requirements.items():
            pos_players1 = [p for p in parent1 if problem.players[p].position == pos or 
                           (pos == 'FLEX' and problem.players[p].position in ['RB', 'WR', 'TE'])]
            pos_players2 = [p for p in parent2 if problem.players[p].position == pos or 
                           (pos == 'FLEX' and problem.players[p].position in ['RB', 'WR', 'TE'])]
            
            # Randomly choose from both parents
            combined = list(set(pos_players1 + pos_players2))
            if len(combined) >= count:
                selected = np.random.choice(combined, count, replace=False)
                child1.extend(selected[:count//2 + count%2])
                child2.extend(selected[count//2 + count%2:])
        
        # Ensure feasibility
        if not problem.is_feasible(child1):
            child1 = problem.get_random_solution()
        if not problem.is_feasible(child2):
            child2 = problem.get_random_solution()
        
        return child1, child2
    
    def _fantasy_mutation(self, individual, problem):
        """Fantasy football specific mutation operation."""
        mutated = individual.copy()
        
        # Randomly replace one player with another from same position
        if len(mutated) > 0:
            # Choose random player to replace
            replace_idx = np.random.randint(len(mutated))
            old_player_idx = mutated[replace_idx]
            old_player = problem.players[old_player_idx]
            
            # Find replacement candidates
            candidates = [
                i for i, p in enumerate(problem.players)
                if p.position == old_player.position and i not in mutated
            ]
            
            if candidates:
                new_player_idx = np.random.choice(candidates)
                mutated[replace_idx] = new_player_idx
                
                # Check if still feasible
                if not problem.is_feasible(mutated):
                    mutated[replace_idx] = old_player_idx  # Revert if infeasible
        
        return mutated

def generate_sample_data(num_players: int = 400) -> List[Player]:
    """Generate sample fantasy football player data for testing."""
    positions = ['QB', 'RB', 'WR', 'TE', 'DST', 'K']
    teams = ['KC', 'BUF', 'CIN', 'LAC', 'SF', 'PHI', 'DAL', 'MIA', 'BAL', 'MIN']
    
    players = []
    
    for i in range(num_players):
        pos = np.random.choice(positions, p=[0.15, 0.25, 0.35, 0.15, 0.05, 0.05])
        team = np.random.choice(teams)
        
        # Position-based salary and points
        if pos == 'QB':
            salary = np.random.randint(5500, 8500)
            points = np.random.normal(18, 4)
        elif pos == 'RB':
            salary = np.random.randint(4000, 9000)
            points = np.random.normal(12, 5)
        elif pos == 'WR':
            salary = np.random.randint(3500, 8500)
            points = np.random.normal(10, 4)
        elif pos == 'TE':
            salary = np.random.randint(3000, 7000)
            points = np.random.normal(8, 3)
        elif pos == 'DST':
            salary = np.random.randint(2000, 3500)
            points = np.random.normal(7, 3)
        else:  # K
            salary = np.random.randint(4000, 5500)
            points = np.random.normal(8, 2)
        
        points = max(0, points)  # No negative points
        
        player = Player(
            name=f"{pos}_{team}_{i}",
            position=pos,
            team=team,
            salary=int(salary),
            projected_points=round(points, 1)
        )
        players.append(player)
    
    return players

def main():
    """Main function for local testing."""
    print("🏈 Fantasy Football Optimization - Local Testing")
    print("=" * 60)
    
    # Generate sample data
    print("\n📊 Generating sample player data...")
    players = generate_sample_data(400)
    print(f"Generated {len(players)} players")
    
    # Create problem
    print("\n🎯 Creating fantasy football problem...")
    problem = FantasyFootballProblem(players, salary_cap=50000)
    print(f"Salary cap: ${problem.salary_cap:,}")
    print(f"Lineup requirements: {problem.lineup_requirements}")
    
    # Test random solution
    print("\n🎲 Testing random solution generation...")
    random_lineup = problem.get_random_solution()
    random_score = problem.evaluate_solution(random_lineup)
    print(f"Random lineup score: {random_score:.2f} points")
    print(f"Random lineup feasible: {problem.is_feasible(random_lineup)}")
    
    # Create and run optimizer
    print("\n🧬 Running genetic algorithm optimization...")
    optimizer = FantasyFootballGeneticOptimizer(
        population_size=50,
        max_generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    result = optimizer.optimize(problem)
    
    # Display results
    print(f"\n🏆 Optimization Results:")
    print(f"Best score: {result.best_value:.2f} points")
    print(f"Runtime: {result.runtime_seconds:.2f} seconds")
    print(f"Evaluations: {result.evaluations}")
    
    # Show lineup details
    if result.best_solution:
        print(f"\n📋 Optimal Lineup:")
        lineup_df = problem.get_lineup_summary(result.best_solution)
        print(lineup_df.to_string(index=False))
        
        total_salary = lineup_df['Salary'].sum()
        total_points = lineup_df['Projected_Points'].sum()
        print(f"\nTotal Salary: ${total_salary:,} (${problem.salary_cap - total_salary:,} remaining)")
        print(f"Total Points: {total_points:.2f}")
        print(f"Average Value: {total_points / (total_salary / 1000):.2f} points per $1K")

if __name__ == "__main__":
    main()
```

This completes the local testing file. The file includes:

1. **Complete Problem Definition**: `FantasyFootballProblem` class with all constraints
2. **Specialized Optimizer**: `FantasyFootballGeneticOptimizer` with fantasy-specific operations
3. **Sample Data Generation**: `generate_sample_data()` for testing
4. **Comprehensive Testing**: Full workflow demonstration
5. **Detailed Analysis**: Lineup summary and performance metrics

## 🎯 Key Features Demonstrated

### Problem Modeling
- **Constraint Handling**: Salary cap, position requirements, player uniqueness
- **Feasibility Checking**: Comprehensive validation of lineup constraints
- **Objective Function**: Maximizing projected points

### Optimization Strategy
- **Position-Aware Crossover**: Maintains position structure during breeding
- **Smart Mutation**: Replaces players within same position
- **Constraint Preservation**: Ensures all offspring remain feasible

### Testing and Validation
- **Sample Data**: Realistic player data generation
- **Performance Metrics**: Runtime, evaluations, convergence tracking
- **Result Analysis**: Detailed lineup breakdown and value analysis

## 📚 Next Steps

In the next parts of this tutorial, we'll create:

1. **`optimizer_only.py`**: Clean optimizer class for Rastion upload
2. **`rastion_integration.py`**: Platform integration and production deployment

This 3-file structure provides:
- **Development Flexibility**: Full local testing capabilities
- **Production Readiness**: Clean, deployable optimizer
- **Platform Integration**: Seamless Rastion platform usage

Continue to the next sections to complete the full fantasy football optimization workflow! 🚀
