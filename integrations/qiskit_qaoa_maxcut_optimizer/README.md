# Qiskit QAOA MaxCut Optimizer

Small local QAOA-style MaxCut optimizer using Qiskit `QuantumCircuit` and
`Statevector`. It grid-searches one shared `(gamma, beta)` pair and samples the
best statevector locally.

This is intentionally a small-graph integration demo, not a production quantum
workflow. It does not require Aer or cloud hardware.
