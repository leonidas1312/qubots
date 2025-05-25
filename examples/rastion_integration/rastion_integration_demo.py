"""
Comprehensive demonstration of the Qubots-Rastion integration.
Shows the complete workflow from model creation to upload and loading.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import qubots
import qubots.rastion as rastion
from qubots import (
    BaseProblem, BaseOptimizer, 
    ProblemMetadata, OptimizerMetadata,
    ProblemType, OptimizerType, OptimizerFamily,
    ObjectiveType, DifficultyLevel
)
import numpy as np
from typing import List, Tuple, Any, Dict


# Example 1: Create a simple optimization problem
class SimpleTSPProblem(BaseProblem):
    """A simple Traveling Salesman Problem implementation."""
    
    def __init__(self, cities: List[Tuple[float, float]] = None):
        if cities is None:
            # Default 5-city problem
            cities = [(0, 0), (1, 2), (3, 1), (2, 3), (0, 1)]
        
        self.cities = cities
        self.n_cities = len(cities)
        
        # Calculate distance matrix
        self.distances = self._calculate_distances()
        
        # Initialize metadata
        super().__init__(
            metadata=ProblemMetadata(
                name="Simple TSP Problem",
                description="A simple Traveling Salesman Problem with configurable cities",
                author="Qubots Demo",
                version="1.0.0",
                problem_type=ProblemType.COMBINATORIAL,
                objective_type=ObjectiveType.MINIMIZATION,
                difficulty_level=DifficultyLevel.MEDIUM,
                tags={"tsp", "routing", "combinatorial", "demo"}
            )
        )
    
    def _calculate_distances(self) -> np.ndarray:
        """Calculate distance matrix between cities."""
        n = len(self.cities)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = self.cities[i]
                    x2, y2 = self.cities[j]
                    distances[i][j] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        return distances
    
    def evaluate(self, solution: List[int]) -> float:
        """
        Evaluate a TSP solution (tour).
        
        Args:
            solution: List of city indices representing the tour
            
        Returns:
            Total tour distance
        """
        if len(solution) != self.n_cities:
            raise ValueError(f"Solution must visit all {self.n_cities} cities")
        
        total_distance = 0.0
        for i in range(len(solution)):
            current_city = solution[i]
            next_city = solution[(i + 1) % len(solution)]
            total_distance += self.distances[current_city][next_city]
        
        return total_distance
    
    def is_valid(self, solution: List[int]) -> bool:
        """Check if a solution is valid."""
        return (len(solution) == self.n_cities and 
                set(solution) == set(range(self.n_cities)))
    
    def get_random_solution(self) -> List[int]:
        """Generate a random valid solution."""
        solution = list(range(self.n_cities))
        np.random.shuffle(solution)
        return solution


# Example 2: Create a simple genetic algorithm optimizer
class SimpleGeneticAlgorithm(BaseOptimizer):
    """A simple genetic algorithm for combinatorial optimization."""
    
    def __init__(self, population_size: int = 50, generations: int = 100, 
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Initialize metadata
        super().__init__(
            metadata=OptimizerMetadata(
                name="Simple Genetic Algorithm",
                description="A basic genetic algorithm for combinatorial optimization problems",
                author="Qubots Demo",
                version="1.0.0",
                optimizer_type=OptimizerType.METAHEURISTIC,
                optimizer_family=OptimizerFamily.EVOLUTIONARY,
                tags={"genetic", "evolutionary", "metaheuristic", "demo"}
            )
        )
    
    def optimize(self, problem: BaseProblem) -> Any:
        """
        Optimize the given problem using genetic algorithm.
        
        Args:
            problem: Problem instance to optimize
            
        Returns:
            Best solution found
        """
        # Initialize population
        population = [problem.get_random_solution() for _ in range(self.population_size)]
        
        best_solution = None
        best_fitness = float('inf')
        
        for generation in range(self.generations):
            # Evaluate population
            fitness_scores = []
            for individual in population:
                fitness = problem.evaluate(individual)
                fitness_scores.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = individual.copy()
            
            # Selection, crossover, and mutation
            new_population = []
            
            for _ in range(self.population_size // 2):
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if np.random.random() < self.mutation_rate:
                    child1 = self._mutate(child1)
                if np.random.random() < self.mutation_rate:
                    child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'generations': self.generations
        }
    
    def _tournament_selection(self, population: List[List[int]], 
                            fitness_scores: List[float], tournament_size: int = 3) -> List[int]:
        """Tournament selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_index].copy()
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Order crossover (OX)."""
        size = len(parent1)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        
        child1 = [-1] * size
        child2 = [-1] * size
        
        # Copy segments
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        # Fill remaining positions
        self._fill_child(child1, parent2, start, end)
        self._fill_child(child2, parent1, start, end)
        
        return child1, child2
    
    def _fill_child(self, child: List[int], parent: List[int], start: int, end: int):
        """Fill child with remaining elements from parent."""
        child_set = set(child[start:end])
        parent_filtered = [x for x in parent if x not in child_set]
        
        j = 0
        for i in range(len(child)):
            if child[i] == -1:
                child[i] = parent_filtered[j]
                j += 1
    
    def _mutate(self, individual: List[int]) -> List[int]:
        """Swap mutation."""
        mutated = individual.copy()
        i, j = np.random.choice(len(mutated), 2, replace=False)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated


def demonstrate_workflow():
    """Demonstrate the complete Qubots-Rastion workflow."""
    print("🚀 Qubots-Rastion Integration Demo")
    print("=" * 50)
    
    # Step 1: Create optimization models
    print("\n📦 Step 1: Creating optimization models...")
    
    # Create a TSP problem
    tsp_problem = SimpleTSPProblem()
    print(f"✅ Created TSP problem with {tsp_problem.n_cities} cities")
    
    # Create a genetic algorithm optimizer
    ga_optimizer = SimpleGeneticAlgorithm(population_size=30, generations=50)
    print("✅ Created genetic algorithm optimizer")
    
    # Step 2: Test the models locally
    print("\n🧪 Step 2: Testing models locally...")
    
    # Test the problem
    random_solution = tsp_problem.get_random_solution()
    fitness = tsp_problem.evaluate(random_solution)
    print(f"✅ Random solution fitness: {fitness:.2f}")
    
    # Test the optimizer
    result = ga_optimizer.optimize(tsp_problem)
    print(f"✅ GA optimization result: {result['best_fitness']:.2f}")
    
    # Step 3: Demonstrate upload (simulation)
    print("\n📤 Step 3: Upload simulation...")
    print("Note: Actual upload requires authentication with rastion.authenticate(token)")
    
    # Show what would be uploaded
    from qubots.rastion_client import QubotPackager
    
    problem_package = QubotPackager.package_model(
        tsp_problem, "simple_tsp_problem", 
        "A simple TSP problem for demonstration"
    )
    
    optimizer_package = QubotPackager.package_model(
        ga_optimizer, "simple_genetic_algorithm",
        "A basic genetic algorithm optimizer"
    )
    
    print("✅ Problem package created:")
    for filename in problem_package.keys():
        print(f"   📄 {filename}")
    
    print("✅ Optimizer package created:")
    for filename in optimizer_package.keys():
        print(f"   📄 {filename}")
    
    # Step 4: Demonstrate loading interface
    print("\n📥 Step 4: Loading interface demonstration...")
    print("After upload, users would load models like this:")
    print()
    print("```python")
    print("import qubots.rastion as rastion")
    print()
    print("# One-line model loading")
    print("problem = rastion.load_qubots_model('simple_tsp_problem')")
    print("optimizer = rastion.load_qubots_model('simple_genetic_algorithm')")
    print()
    print("# Run optimization")
    print("result = optimizer.optimize(problem)")
    print("print(f'Best solution: {result[\"best_solution\"]}')") 
    print("print(f'Best fitness: {result[\"best_fitness\"]}')") 
    print("```")
    
    # Step 5: Show discovery features
    print("\n🔍 Step 5: Model discovery features...")
    print("Users can discover models using:")
    print()
    print("```python")
    print("# Search for models")
    print("models = rastion.search_models('genetic algorithm')")
    print()
    print("# List available models")
    print("all_models = rastion.discover_models()")
    print()
    print("# List user's models")
    print("my_models = rastion.list_my_models()")
    print("```")
    
    print("\n✨ Demo completed! The integration provides:")
    print("   🔹 One-line model loading: rastion.load_qubots_model('name')")
    print("   🔹 Easy model upload: rastion.upload_model(model, name, description)")
    print("   🔹 Model discovery: rastion.discover_models(query)")
    print("   🔹 Seamless workflow from creation to sharing")


if __name__ == "__main__":
    demonstrate_workflow()
