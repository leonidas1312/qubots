"""Leaderboard runner: suite YAML + submission YAMLs -> leaderboard report.

Suite YAML schema (v1):

    qubots_leaderboard_schema_version: 1
    name: qubots-flagship
    description: "Canonical OR business-ops benchmarks"
    benchmarks:
      - name: knapsack_small
        dataset: ../examples/datasets/knapsack_small.yaml

Submission YAML schema (v1):

    qubots_submission_schema_version: 1
    spec: examples/highs_optimizer            # or github:owner/repo@sha:subdir
    submitter: leonidas1312
    display_name: "HiGHS"
    parameters:
      time_limit_seconds: 30

The runner does **not** sandbox dependencies in-process. The community
``qubots-leaderboard`` repo's GitHub Actions workflow installs each
submission's declared ``requirements:`` (from its component manifest) into
an isolated venv and shells out to ``qubots leaderboard`` once per submission;
this primitive just runs whatever's importable in the current env.
"""

from __future__ import annotations

import datetime as _dt
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from qubots.benchmark.benchmark import benchmark as run_benchmark
from qubots.hub.resolver import derive_repo_name


SUITE_SCHEMA_VERSION = 1
SUBMISSION_SCHEMA_VERSION = 1


# ---------- data classes -----------------------------------------------------


@dataclass
class SuiteSpec:
    name: str
    description: str
    benchmarks: list[dict[str, Any]]
    source_path: Path


@dataclass
class Submission:
    spec: str
    submitter: str
    display_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    @property
    def submission_id(self) -> str:
        return f"{self.submitter}/{self.display_name}"


@dataclass
class BenchmarkResult:
    benchmark_name: str
    submission_id: str
    submitter: str
    display_name: str
    spec: str
    num_runs: int
    mean_best_value: float
    mean_runtime_seconds: float
    success_rate: float
    runs: list[dict[str, Any]]


@dataclass
class LeaderboardReport:
    suite_name: str
    suite_description: str
    generated_at: str
    benchmarks: list[str]
    submissions: list[dict[str, Any]]
    results: list[BenchmarkResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "suite_description": self.suite_description,
            "generated_at": self.generated_at,
            "benchmarks": list(self.benchmarks),
            "submissions": list(self.submissions),
            "results": [asdict(r) for r in self.results],
        }


# ---------- loaders ----------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return raw


def _check_schema_version(raw: dict[str, Any], key: str, expected: int, where: Path) -> None:
    version = raw.get(key, expected)
    try:
        version = int(version)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer in {where}") from exc
    if version != expected:
        raise ValueError(
            f"Unsupported {key}={version} in {where}. This qubots installation "
            f"supports {key}={expected}."
        )


def load_suite(path: str | Path) -> SuiteSpec:
    suite_path = Path(path).expanduser().resolve()
    if not suite_path.exists():
        raise FileNotFoundError(f"Suite file not found: {suite_path}")
    raw = _load_yaml(suite_path)
    _check_schema_version(raw, "qubots_leaderboard_schema_version", SUITE_SCHEMA_VERSION, suite_path)

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Suite 'name' must be a non-empty string: {suite_path}")

    description = raw.get("description") or ""
    if not isinstance(description, str):
        raise ValueError(f"Suite 'description' must be a string: {suite_path}")

    benchmarks_raw = raw.get("benchmarks")
    if not isinstance(benchmarks_raw, list) or not benchmarks_raw:
        raise ValueError(
            f"Suite 'benchmarks' must be a non-empty list of "
            f"{{name, dataset}} entries: {suite_path}"
        )

    benchmarks: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, entry in enumerate(benchmarks_raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Suite benchmarks[{index}] must be a mapping: {suite_path}"
            )
        bench_name = entry.get("name")
        bench_dataset = entry.get("dataset")
        if not isinstance(bench_name, str) or not bench_name.strip():
            raise ValueError(
                f"Suite benchmarks[{index}].name must be a non-empty string: {suite_path}"
            )
        if bench_name in seen_names:
            raise ValueError(
                f"Duplicate benchmark name {bench_name!r} in suite: {suite_path}"
            )
        seen_names.add(bench_name)
        if not isinstance(bench_dataset, str) or not bench_dataset.strip():
            raise ValueError(
                f"Suite benchmarks[{index}].dataset must be a string path: {suite_path}"
            )

        dataset_path = Path(bench_dataset)
        if not dataset_path.is_absolute():
            dataset_path = (suite_path.parent / dataset_path).resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset for benchmark {bench_name!r} not found: {dataset_path}"
            )

        benchmarks.append(
            {
                "name": bench_name,
                "dataset_path": dataset_path,
            }
        )

    return SuiteSpec(
        name=name.strip(),
        description=description.strip(),
        benchmarks=benchmarks,
        source_path=suite_path,
    )


def load_submission(path: str | Path) -> Submission:
    sub_path = Path(path).expanduser().resolve()
    if not sub_path.exists():
        raise FileNotFoundError(f"Submission file not found: {sub_path}")
    raw = _load_yaml(sub_path)
    _check_schema_version(raw, "qubots_submission_schema_version", SUBMISSION_SCHEMA_VERSION, sub_path)

    spec = raw.get("spec")
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(f"Submission 'spec' must be a non-empty string: {sub_path}")

    submitter = raw.get("submitter")
    if not isinstance(submitter, str) or not submitter.strip():
        raise ValueError(f"Submission 'submitter' must be a non-empty string: {sub_path}")

    display_name = raw.get("display_name") or submitter
    if not isinstance(display_name, str) or not display_name.strip():
        raise ValueError(
            f"Submission 'display_name' must be a non-empty string: {sub_path}"
        )

    parameters = raw.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError(f"Submission 'parameters' must be a mapping: {sub_path}")

    # Resolve a relative spec against the submission file's directory
    # (so submissions/foo.yaml can say spec: ../my_optimizer).
    resolved_spec = spec.strip()
    if not resolved_spec.startswith("github:"):
        candidate = Path(resolved_spec)
        if not candidate.is_absolute():
            absolute = (sub_path.parent / candidate).resolve()
            if absolute.exists():
                resolved_spec = str(absolute)

    return Submission(
        spec=resolved_spec,
        submitter=submitter.strip(),
        display_name=display_name.strip(),
        parameters=dict(parameters),
        source_path=sub_path,
    )


def load_submissions_from_dir(path: str | Path) -> list[Submission]:
    root = Path(path).expanduser().resolve()
    if root.is_file():
        return [load_submission(root)]
    if not root.is_dir():
        raise FileNotFoundError(f"Submissions path not found: {root}")
    submissions: list[Submission] = []
    for yaml_path in sorted(root.rglob("*.yaml")):
        submissions.append(load_submission(yaml_path))
    return submissions


# ---------- runner -----------------------------------------------------------


def _empty_stats(spec: str, submission: Submission) -> dict[str, Any]:
    return {
        "submission_id": submission.submission_id,
        "submitter": submission.submitter,
        "display_name": submission.display_name,
        "spec": spec,
    }


def run_leaderboard(
    suite: SuiteSpec | str | Path,
    submissions: list[Submission] | str | Path,
    *,
    repeats: int = 1,
    seed: int | None = None,
) -> LeaderboardReport:
    """Run every submission against every benchmark in the suite."""
    if not isinstance(suite, SuiteSpec):
        suite = load_suite(suite)

    if isinstance(submissions, (str, Path)):
        submissions = load_submissions_from_dir(submissions)
    if not submissions:
        raise ValueError("No submissions to run.")

    submission_payloads: list[dict[str, Any]] = []
    for sub in submissions:
        submission_payloads.append(
            {
                "submission_id": sub.submission_id,
                "submitter": sub.submitter,
                "display_name": sub.display_name,
                "spec": sub.spec,
                "parameters": dict(sub.parameters),
            }
        )

    results: list[BenchmarkResult] = []
    for bench in suite.benchmarks:
        bench_name = bench["name"]
        dataset_path = bench["dataset_path"]
        for sub in submissions:
            report = run_benchmark(
                problem_repo=None,
                dataset_path=dataset_path,
                optimizers=[sub.spec],
                optimizer_params=[dict(sub.parameters)] if sub.parameters else None,
                repeats=repeats,
                seed=seed,
            )
            row = report["results"][0]
            results.append(
                BenchmarkResult(
                    benchmark_name=bench_name,
                    submission_id=sub.submission_id,
                    submitter=sub.submitter,
                    display_name=sub.display_name,
                    spec=sub.spec,
                    num_runs=int(row["num_runs"]),
                    mean_best_value=float(row["mean_best_value"]),
                    mean_runtime_seconds=float(row["mean_runtime_seconds"]),
                    success_rate=float(row["success_rate"]),
                    runs=list(row["runs"]),
                )
            )

    return LeaderboardReport(
        suite_name=suite.name,
        suite_description=suite.description,
        generated_at=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds"),
        benchmarks=[b["name"] for b in suite.benchmarks],
        submissions=submission_payloads,
        results=results,
    )


# ---------- output formatting ------------------------------------------------


def _rank_results(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    """Sort by mean_best_value ascending (qubots convention: lower is better)."""
    return sorted(
        results,
        key=lambda r: (r.mean_best_value, r.mean_runtime_seconds, r.submission_id),
    )


def _wins_summary(report: LeaderboardReport) -> dict[str, dict[str, int]]:
    """For each submission, count how many benchmarks it ranks #1 / #2 / #3 in."""
    counts: dict[str, dict[str, int]] = {}
    for sub in report.submissions:
        counts[sub["submission_id"]] = {"rank1": 0, "rank2": 0, "rank3": 0}

    for bench_name in report.benchmarks:
        rows = [r for r in report.results if r.benchmark_name == bench_name]
        ranked = _rank_results(rows)
        for index, row in enumerate(ranked):
            if index == 0:
                counts[row.submission_id]["rank1"] += 1
            elif index == 1:
                counts[row.submission_id]["rank2"] += 1
            elif index == 2:
                counts[row.submission_id]["rank3"] += 1
    return counts


def report_to_markdown(report: LeaderboardReport) -> str:
    lines: list[str] = []
    lines.append(f"# {report.suite_name}")
    lines.append("")
    if report.suite_description:
        lines.append(f"> {report.suite_description}")
        lines.append("")
    lines.append(f"_Generated: {report.generated_at}_")
    lines.append("")
    lines.append(
        f"_{len(report.submissions)} submission(s) × {len(report.benchmarks)} benchmark(s)_"
    )
    lines.append("")

    # Per-benchmark tables
    for bench_name in report.benchmarks:
        rows = [r for r in report.results if r.benchmark_name == bench_name]
        if not rows:
            continue
        ranked = _rank_results(rows)
        lines.append(f"## {bench_name}")
        lines.append("")
        lines.append(
            "| rank | submission | submitter | mean_best_value | mean_runtime_s | success |"
        )
        lines.append("|---:|---|---|---:|---:|---:|")
        for index, row in enumerate(ranked, start=1):
            lines.append(
                "| {rank} | {name} | {who} | {val:.6f} | {rt:.6f} | {sr:.0%} |".format(
                    rank=index,
                    name=row.display_name,
                    who=row.submitter,
                    val=row.mean_best_value,
                    rt=row.mean_runtime_seconds,
                    sr=row.success_rate,
                )
            )
        lines.append("")

    # Aggregate wins
    wins = _wins_summary(report)
    if wins:
        lines.append("## Summary")
        lines.append("")
        lines.append("| submission | submitter | #1 | #2 | #3 |")
        lines.append("|---|---|---:|---:|---:|")
        sorted_subs = sorted(
            report.submissions,
            key=lambda s: (
                -wins[s["submission_id"]]["rank1"],
                -wins[s["submission_id"]]["rank2"],
                -wins[s["submission_id"]]["rank3"],
                s["submission_id"],
            ),
        )
        for sub in sorted_subs:
            counts = wins[sub["submission_id"]]
            lines.append(
                f"| {sub['display_name']} | {sub['submitter']} | "
                f"{counts['rank1']} | {counts['rank2']} | {counts['rank3']} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(
    report: LeaderboardReport,
    *,
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> tuple[Path | None, Path | None]:
    """Write the leaderboard report to disk. Returns (json_path, markdown_path)."""
    written_json: Path | None = None
    written_md: Path | None = None

    if json_path is not None:
        jp = Path(json_path)
        jp.parent.mkdir(parents=True, exist_ok=True)
        with jp.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=_json_default)
        written_json = jp

    if markdown_path is not None:
        mp = Path(markdown_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(report_to_markdown(report), encoding="utf-8")
        written_md = mp

    return written_json, written_md


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON-serializable")


def derive_default_display_name(spec: str) -> str:
    return derive_repo_name(spec)
