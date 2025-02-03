from rastion_core.base_problem import BaseProblem
import random

class OptimalControlProblem(BaseProblem):
    """
    Optimal Control Problem:
    Control a dynamic system over time to reach a target state.
    A solution is a sequence of control actions.
    """
    def __init__(self, target_state, time_steps, control_bounds):
        self.target_state = target_state
        self.time_steps = time_steps
        self.control_bounds = control_bounds  # list of [min, max] per dimension.
        self.state_dim = len(target_state)
    
    def evaluate_solution(self, controls) -> float:
        # Simple simulation: state_{t+1} = state_t + control_t.
        state = [0] * self.state_dim
        total_error = 0
        for control in controls:
            state = [s + c for s, c in zip(state, control)]
            error = sum(abs(s - t) for s, t in zip(state, self.target_state))
            total_error += error
        return total_error
    
    def random_solution(self):
        return [[random.uniform(b[0], b[1]) for b in self.control_bounds] for _ in range(self.time_steps)]
