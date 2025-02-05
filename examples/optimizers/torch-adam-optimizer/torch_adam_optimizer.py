# File: torch_adam_optimizer.py

import torch
from rastion_core.base_optimizer import BaseOptimizer

class TorchAdamOptimizer(BaseOptimizer):
    """
    A classical optimizer that uses PyTorch's Adam optimizer to update
    the variational parameters.
    """
    def __init__(self, lr=0.01, max_steps=100, verbose=False):
        """
        :param lr: Learning rate.
        :param max_steps: Number of optimization steps.
        :param verbose: If True, print progress.
        """
        self.lr = lr
        self.max_steps = max_steps
        self.verbose = verbose

    def optimize(self, problem, **kwargs):
        """
        Optimize the given problem’s cost function over the quantum-circuit parameters.
        The problem is assumed to be an instance of QuantumParameterProblem.
        Additional keyword arguments (e.g., "cost_function") may be provided.
        
        Returns:
            - optimized parameters as a NumPy array.
            - the final cost (a float).
        """
        # Get an initial solution (θ) from the problem.
        initial_theta = problem.random_solution()  # a list of floats
        theta = torch.tensor(initial_theta, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=self.lr)
        cost_function = kwargs.get("cost_function")
        if cost_function is None:
            raise ValueError("A cost_function must be provided as a keyword argument.")

        for step in range(self.max_steps):
            optimizer.zero_grad()
            # Evaluate the cost using the provided cost_function.
            # (It is assumed cost_function takes a NumPy array as input.)
            cost = cost_function(theta.detach().numpy())
            # Convert the cost to a torch tensor so that we can backpropagate.
            cost_tensor = torch.tensor(cost, dtype=torch.float32, requires_grad=True)
            cost_tensor.backward()
            optimizer.step()
            if self.verbose:
                print(f"Step {step}: cost = {cost}")
        final_cost = cost_function(theta.detach().numpy())
        return theta.detach().numpy(), final_cost
