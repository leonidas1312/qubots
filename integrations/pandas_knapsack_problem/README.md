# Pandas Campaign Knapsack Problem

Uses `pandas` to load a campaign-budget CSV and exposes both:

- black-box `evaluate()` / `random_solution()`
- structured `as_milp()`

That lets the same data-backed problem run under black-box optimizers, HiGHS,
CP-SAT, or the PuLP integration.
