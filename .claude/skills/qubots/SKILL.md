---
name: qubots
description: Author, validate, and benchmark qubots optimization components (problems and solvers). Use when the user is creating a new component repo, debugging an existing one, building a dataset, or running a leaderboard.
when: User mentions qubots, "new problem", "new optimizer", a `qubots.yaml` manifest, an `as_milp()` method, a MIPLIB instance, MPS files, or asks to scaffold/validate/benchmark an OR component. Also when editing files under examples/ or any directory containing a `qubots.yaml`.
---

# Qubots authoring skill

You are helping a developer build, ship, and benchmark **pluggable optimization components** in the qubots framework. Components are small directories with a manifest and an entrypoint module — no global registry, no plugin install. The hub primitive is a git URL.

**This skill is for scaffolding and iteration, not for translating vague business descriptions into correct optimization models.** Generating a correct MILP formulation from an English problem statement is unreliable — the developer must validate the math. Your job: scaffold the boilerplate, fill in well-specified formulations, run the solver, react to the output.

## Mental model

A qubots **component repo** is a directory containing:

```
my_component/
  qubots.yaml      # manifest: type, name, entrypoint, parameters
  qubot.py         # Python module with the entrypoint class
  README.md        # what this component does, how to run
```

Two component types: `problem` and `optimizer`. They compose via the
`AutoProblem` / `AutoOptimizer` loaders or the CLI.

**Problem flavors:**

| Flavor | Implements | Pick when |
|---|---|---|
| `blackbox` | `evaluate(solution) -> float` (lower is better) + `random_solution()` | The fitness is easy to compute but hard to express linearly |
| `milp` | `as_milp() -> MILPModel` | The problem is naturally LP/MILP/CP-SAT-shaped |
| `dual` | both | You want the same instance to run under metaheuristics AND structured solvers (best for benchmarks) |

**Optimizer flavors:**

| Flavor | Consumes | Examples in repo |
|---|---|---|
| `blackbox` | `problem.random_solution()` + `problem.evaluate()` | `examples/random_search_optimizer`, `examples/hill_climb_optimizer`, `examples/simulated_annealing_optimizer` |
| `milp` | `problem.as_milp() -> MILPModel` | `examples/highs_optimizer`, `examples/cpsat_optimizer` |

## Your default workflow

1. **Scaffold**, don't hand-write boilerplate. The CLI guarantees a valid component:
   ```bash
   qubots new problem   --name shift_scheduler   --flavor dual
   qubots new optimizer --name my_milp_solver    --flavor milp
   ```
   Validates with `qubots validate <path>` immediately.

2. **Specify before formulating.** Before touching `as_milp()` or `evaluate`, write down in plain language:
   - decision variables (what's chosen)
   - objective (min or max, what)
   - constraints (each one as a clean inequality/equality)
   - parameters (data inputs and their types)

   If the user hands you a vague business description, your first message back should pin these four things down. Do **not** invent constraints to fill gaps — ask.

3. **Implement against the contract.** See "Reference" below. Use the existing `examples/knapsack_milp_problem` as a clean template for `dual` problems.

4. **Run it.** Always validate, then run on a tiny instance, then read solver output before scaling up:
   ```bash
   qubots validate <path>
   qubots run --problem <path> --optimizer examples/highs_optimizer --problem-params '{"size": 5}'
   ```
   Cross-check with another solver (`examples/cpsat_optimizer` or a metaheuristic) to catch sign errors and missing constraints.

5. **React to solver output.**
   - `status: "infeasible"` → some constraint can't be satisfied. Inspect bounds + RHS for the smallest case where you expect a feasible solution.
   - `best_value` has the wrong sign → check `sense` ("min"/"max") and remember the qubots convention: `best_value` is what the solver internally minimizes, so `max` problems return a negated objective.
   - `n_constraints` is zero → you forgot to populate `A_ub`/`A_eq`.
   - CP-SAT errors `requires every variable to be integer` or `not integer-valued` → use HiGHS instead, or scale your coefficients.

## Reference: the contracts

### `MILPModel` (qubots.core.milp)

```python
MILPModel(
    sense="min" | "max",
    c=[c0, c1, ...],            # objective coefficients (n vars)
    var_names=["x0", "x1", ...], # optional, defaults to x{i}
    integrality=[True, False, ...],  # True = integer/binary
    lb=[0.0, ...], ub=[1.0, ...],     # use math.inf for unbounded
    A_ub=[[a00, a01, ...], ...], b_ub=[b0, ...],   # A_ub @ x <= b_ub
    A_eq=[[...], ...], b_eq=[...],                  # A_eq @ x == b_eq
    constraint_names=[...],  # optional
)
```

- For binary variables: `integrality=[True]*n, lb=[0]*n, ub=[1]*n`.
- For >= constraints: rewrite as `-row · x <= -rhs` and add to `A_ub`.
- All shapes are validated in `__post_init__`; mismatches raise immediately.

### `BaseProblem` (qubots.core.problem)

If you implement only `as_milp()`, `BaseProblem` provides a default
`evaluate()` (signed objective + 100 × constraint-violation penalty) and
`random_solution()` (uniform sample within bounds). Override `evaluate` for
tighter problem-specific penalties.

If you implement `evaluate(solution) -> float`, **lower is better.** Penalize
infeasibility with a large positive number so feasible solutions always score
below infeasible ones.

### `BaseOptimizer` (qubots.core.optimizer)

Implement `optimize(problem) -> Result`. The `Result` dataclass:

```python
Result(
    best_value=...,         # what was minimized (negate for max-sense problems)
    best_solution=...,      # whatever shape your problem expects
    runtime_seconds=...,
    status="ok" | "feasible" | "infeasible" | "error" | ...,
    trace=[...],            # optional: per-iteration best values
    metadata={...},         # solver-specific stats
)
```

### `qubots.yaml` manifest

```yaml
qubots_schema_version: 1
type: problem      # or "optimizer"
name: my_component  # snake_case, [a-z][a-z0-9_]*
entrypoint: qubot.py:MyClass

parameters:        # exposed via set_parameters() and dataset YAMLs
  n_items:
    default: 30
  seed:
    default: 0

tunable_parameters: # optional: search spaces for qubots finetune
  iterations:
    type: int
    min: 50
    max: 5000
```

## Common pitfalls (do not repeat)

- **Wrong sign on objective.** If a `max` problem returns negative `best_value`, that's correct (it's the negated objective). If a `min` problem returns positive when the true optimum is negative, that's also correct. Check `metadata["sense"]` and `metadata["objective"]`.
- **CP-SAT with continuous vars.** CP-SAT requires every variable integer-valued. Use HiGHS for mixed/continuous models.
- **Path-traversal manifest.** `entrypoint` must be relative and inside the repo. `../foo.py:Bar` is rejected at load time.
- **Remote loads need opt-in.** `github:owner/repo@sha:subdir` requires `QUBOTS_TRUST_REMOTE_CODE=1` (or `--trust-remote-code`). Do not silently set this for the user — it executes arbitrary third-party Python.
- **Dense MPS reader.** `MPSProblem` builds a dense matrix. Don't try to load million-nonzero MIPLIB instances yet; start with `gen-ip002` / `flugpl`-class instances.
- **Don't invent constraints.** If the user's spec is incomplete, ask. The previous attempt at LLM-generated formulations was removed because it produced plausible-looking but wrong models — do not bring that back.

## Checklist before declaring a component done

- [ ] `qubots validate <path>` returns `[OK]`.
- [ ] On the smallest instance, the structured solver returns `status: "ok"` and a feasible solution (verify with `milp.is_feasible(x)`).
- [ ] On the same instance, a metaheuristic and a structured solver agree on the optimum to within tolerance (or the metaheuristic is at least feasible).
- [ ] `qubots benchmark` runs against a one-instance dataset YAML without errors.
- [ ] README explains the decision variables, objective, and constraints in plain language.
