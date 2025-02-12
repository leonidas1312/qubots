from qubots.base_optimizer import BaseOptimizer
import random
import numpy as np

class ParticleSwarmOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO).
    """
    def __init__(self, swarm_size=30, max_iters=100, inertia=0.5, cognitive=1.0, social=1.0, verbose=False):
        self.swarm_size = swarm_size
        self.max_iters = max_iters
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.verbose = verbose
    
    def optimize(self, problem, initial_solution=None, bounds=None, **kwargs):
        if initial_solution is None:
            x = np.array(problem.random_solution(), dtype=float)
        else:
            x = np.array(initial_solution, dtype=float)
        dim = len(x)
        if bounds is None:
            bounds = [(0, 1)] * dim
        
        swarm = []
        for _ in range(self.swarm_size):
            pos = np.array([random.uniform(b[0], b[1]) for b in bounds])
            vel = np.array([random.uniform(-1, 1) for _ in range(dim)])
            score = problem.evaluate_solution(pos)
            swarm.append({'pos': pos, 'vel': vel, 'best_pos': pos.copy(), 'best_score': score})
        
        global_best = min(swarm, key=lambda p: p['best_score'])
        global_best_pos = global_best['best_pos'].copy()
        global_best_score = global_best['best_score']
        
        for iter in range(self.max_iters):
            for particle in swarm:
                r1 = random.random()
                r2 = random.random()
                cognitive_vel = self.cognitive * r1 * (particle['best_pos'] - particle['pos'])
                social_vel = self.social * r2 * (global_best_pos - particle['pos'])
                particle['vel'] = self.inertia * particle['vel'] + cognitive_vel + social_vel
                particle['pos'] = particle['pos'] + particle['vel']
                for i, (lb, ub) in enumerate(bounds):
                    particle['pos'][i] = max(lb, min(ub, particle['pos'][i]))
                score = problem.evaluate_solution(particle['pos'])
                if score < particle['best_score']:
                    particle['best_score'] = score
                    particle['best_pos'] = particle['pos'].copy()
                    if score < global_best_score:
                        global_best_score = score
                        global_best_pos = particle['pos'].copy()
            if self.verbose:
                print(f"PSO Iteration {iter}: Best Score = {global_best_score}")
        return global_best_pos, global_best_score
