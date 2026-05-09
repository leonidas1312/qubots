# Changelog

All notable changes to qubots are tracked here.

## 2.0.0 — 2026-05-09

This is a clean break from the 1.x line, which had drifted toward
LLM-mediated optimization onboarding. 2.0 refocuses qubots on **pluggable
problem and solver components** with a community leaderboard.

### Removed (breaking)

- `qubots.formalize` (the spreadsheet → LLM → solution pipeline) and the
  `from-spreadsheet` CLI command.
- `qubots.demo` Streamlit UI and the `demo` CLI command.
- `[xlsx]` and `[demo]` install extras.
- Any code paths that depended on Ollama / template-based model
  generation are gone. If you relied on these, pin `qubots<2.0`.

### Added

- **MILPModel + `as_milp()` protocol.** Problems can expose structured
  LP/MILP form to solvers like HiGHS and CP-SAT while staying compatible
  with blackbox optimizers.
- **HiGHS** (`examples/highs_optimizer`) and **OR-Tools CP-SAT**
  (`examples/cpsat_optimizer`) ship as first-class optimizer components.
- **MPS reader and on-demand MIPLIB fetcher** (`qubots.contrib.mps`,
  `qubots.contrib.miplib`). Industry-standard `.mps` / `.lp` files load
  directly as `MILPModel`s; `fetch_miplib("flugpl")` downloads + caches
  benchmark instances under `~/.cache/qubots/miplib/`.
- **`qubots new` scaffolder.** Generates a fresh problem or optimizer
  component repo (manifest + entrypoint + README) that passes
  validation immediately. Flavors: `blackbox`, `milp`, `dual` for
  problems; `blackbox`, `milp` for optimizers.
- **`qubots leaderboard` CLI.** Reads a benchmark suite YAML and a
  directory of submission YAMLs, runs every submission against every
  benchmark, and emits a ranked `LEADERBOARD.md` + `leaderboard.json`.
  Used by [qubots-leaderboard](https://github.com/leonidas1312/qubots-leaderboard).
- **Per-optimizer parameter overrides** in `benchmark()` and the
  leaderboard runner: each submission's `parameters:` block is applied
  before `optimize()`.
- **Manifest schema v2** with declared `requirements:` (a list of pip
  specs the component needs to import). v1 manifests still load.
- **Hub safety**: `QUBOTS_TRUST_REMOTE_CODE=1` gate for `github:` specs,
  path-traversal protection on entrypoints, schema-version refusal for
  unknown manifests.
- **`.claude/skills/qubots`** — agent guidance for component authoring,
  validation, and leaderboard contribution.

### Notes

- Tests: 99 passing covering schema, security, scaffolder, leaderboard,
  HiGHS/CP-SAT, MIPLIB, examples.
- The default branch is `master` (rename to `main` is a future step).
