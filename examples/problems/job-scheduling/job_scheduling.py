from rastion_hub.base_problem import BaseProblem
import random

class JobSchedulingProblem(BaseProblem):
    """
    A simple Job Scheduling Problem.
    Each job has a processing time and a deadline.
    The objective is to minimize total tardiness.
    """
    def __init__(self, processing_times, deadlines):
        self.processing_times = processing_times
        self.deadlines = deadlines
        self.num_jobs = len(processing_times)
    
    def evaluate_solution(self, solution) -> float:
        # 'solution' is a permutation of job indices.
        time = 0
        total_tardiness = 0
        for job in solution:
            time += self.processing_times[job]
            tardiness = max(0, time - self.deadlines[job])
            total_tardiness += tardiness
        return total_tardiness
    
    def random_solution(self):
        jobs = list(range(self.num_jobs))
        random.shuffle(jobs)
        return jobs
