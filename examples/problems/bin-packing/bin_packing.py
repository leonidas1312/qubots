from rastion_hub.base_problem import BaseProblem
import random

class BinPackingProblem(BaseProblem):
    """
    Bin Packing Problem:
    Pack items with given sizes into bins of fixed capacity.
    A solution is a list where each entry indicates the bin index assigned to an item.
    The goal is to minimize the number of bins used.
    """
    def __init__(self, item_sizes, bin_capacity):
        self.item_sizes = item_sizes
        self.bin_capacity = bin_capacity
        self.num_items = len(item_sizes)
    
    def evaluate_solution(self, solution) -> float:
        bin_usage = {}
        for item, bin_index in enumerate(solution):
            bin_usage.setdefault(bin_index, 0)
            bin_usage[bin_index] += self.item_sizes[item]
        bins_used = len(bin_usage)
        penalty = 0
        for usage in bin_usage.values():
            if usage > self.bin_capacity:
                penalty += (usage - self.bin_capacity) * 1e5
        return bins_used + penalty
    
    def random_solution(self):
        return [random.randint(0, self.num_items - 1) for _ in range(self.num_items)]
