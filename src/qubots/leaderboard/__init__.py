"""Leaderboard primitive: run a suite of benchmarks against a set of submissions
and emit a leaderboard.json + LEADERBOARD.md.

Used both locally (any group can run their own internal leaderboard) and by the
community ``qubots-leaderboard`` repo's GitHub Actions runner.
"""

from qubots.leaderboard.leaderboard import (
    BenchmarkResult,
    LeaderboardReport,
    Submission,
    SuiteSpec,
    load_submission,
    load_submissions_from_dir,
    load_suite,
    report_to_markdown,
    run_leaderboard,
    write_report,
)

__all__ = [
    "BenchmarkResult",
    "LeaderboardReport",
    "Submission",
    "SuiteSpec",
    "load_submission",
    "load_submissions_from_dir",
    "load_suite",
    "report_to_markdown",
    "run_leaderboard",
    "write_report",
]
