import numpy as np

class GradientDescent:
    """
    A basic gradient descent solver for continuous optimization.
    Expects:
      - problem.evaluate_solution(x) => float
      - either problem.gradient(x) => ndarray or a user-provided grad_func
    """

    def __init__(self, lr=0.01, max_iters=100, tol=1e-6, grad_func=None, verbose=False):
        """
        :param lr: Learning rate
        :param max_iters: Maximum number of iterations
        :param tol: Tolerance for convergence
        :param grad_func: Optional function grad_func(x) -> gradient vector
        :param verbose: Print progress or not
        """
        self.lr = lr
        self.max_iters = max_iters
        self.tol = tol
        self.grad_func = grad_func
        self.verbose = verbose

    def optimize(self, problem, initial_solution=None):
        """
        If 'initial_solution' is not provided, we try problem.random_solution().
        Returns (best_solution, best_cost).
        """
        if initial_solution is None:
            try:
                x = np.array(problem.random_solution(), dtype=float)
            except NotImplementedError:
                # If problem doesn't define random_solution, pick zeros or random
                x = np.zeros(getattr(problem, "dimension", 1), dtype=float)
        else:
            x = np.array(initial_solution, dtype=float)

        best_solution = x
        best_cost = problem.evaluate_solution(x)

        for iteration in range(self.max_iters):
            # Compute gradient
            if self.grad_func is not None:
                grad = self.grad_func(x)
            elif hasattr(problem, "gradient"):
                grad = problem.gradient(x)
            else:
                raise NotImplementedError("No gradient function supplied or defined in the problem.")

            # Gradient-based update
            new_x = x - self.lr * grad

            # Check feasibility if problem has constraints
            if not problem.is_feasible(new_x):
                # Optionally handle infeasible updates (project back, etc.)
                # For simplicity, we just skip or break
                break

            new_cost = problem.evaluate_solution(new_x)
            if self.verbose:
                print(f"Iter {iteration}: cost={new_cost}, x={new_x}")

            # Check for improvement & convergence
            if abs(new_cost - best_cost) < self.tol:
                return (new_x, new_cost)

            if new_cost < best_cost:
                best_solution, best_cost = new_x, new_cost

            x = new_x

        return best_solution, best_cost

