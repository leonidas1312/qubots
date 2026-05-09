"""Typer CLI for Qubots."""

from importlib.metadata import PackageNotFoundError, version as pkg_version
import json
import os
from pathlib import Path
from typing import Any

import typer

from qubots.auto.auto_optimizer import AutoOptimizer
from qubots.benchmark.benchmark import (
    benchmark as run_benchmark,
    report_to_markdown,
    write_report,
)
from qubots.export.trained import export_trained_optimizer
from qubots.hub.resolver import RemoteRepoNotAllowedError
from qubots.leaderboard import (
    load_submissions_from_dir,
    load_suite,
    report_to_markdown as leaderboard_to_markdown,
    run_leaderboard,
    write_report as write_leaderboard_report,
)
from qubots.pipeline.pipeline import pipeline as build_pipeline
from qubots.scaffold import (
    OPTIMIZER_FLAVORS,
    PROBLEM_FLAVORS,
    scaffold_optimizer,
    scaffold_problem,
)
from qubots.tune.dataset import load_dataset_spec
from qubots.validate.validate import validate_repo, validate_tree

app = typer.Typer(help="Qubots CLI")


def _parse_json_mapping(raw: str | None, option_name: str) -> dict[str, Any]:
    if raw is None or raw.strip() == "":
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"Invalid JSON for {option_name}: {exc.msg}"
        ) from exc

    if not isinstance(parsed, dict):
        raise typer.BadParameter(f"{option_name} must decode to a JSON object")

    return parsed


def _enable_remote_if_requested(allow_remote: bool, trust_remote_code: bool = False) -> None:
    if trust_remote_code:
        os.environ["QUBOTS_TRUST_REMOTE_CODE"] = "1"
    if allow_remote:
        # Legacy alias; both flags enable remote loading.
        os.environ["QUBOTS_ALLOW_REMOTE"] = "1"


@app.command()
def version() -> None:
    """Print installed package version."""
    try:
        typer.echo(pkg_version("qubots"))
    except PackageNotFoundError:
        typer.echo("0.0.0")


@app.command()
def finetune(
    problem: str | None = typer.Option(
        None,
        "--problem",
        help="Problem repo spec (local path or github:...). Optional if dataset defines 'problem'.",
    ),
    optimizer: str = typer.Option(
        ..., "--optimizer", help="Optimizer repo spec (local path or github:...)."
    ),
    dataset: Path = typer.Option(..., "--dataset", help="Path to dataset YAML."),
    budget: int = typer.Option(20, "--budget", min=1, help="Number of tuning trials."),
    metric: str = typer.Option(
        "mean_best_value",
        "--metric",
        help="Aggregation metric. Supported: mean_best_value.",
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Output directory for trained.json artifact."
    ),
    seed: int | None = typer.Option(None, "--seed", help="Random seed."),
    allow_remote: bool = typer.Option(
        False,
        "--allow-remote",
        help="Legacy alias for --trust-remote-code.",
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help=(
            "Allow loading and executing remote github: component repos. "
            "This runs arbitrary third-party Python on your machine; only "
            "enable for sources you trust."
        ),
    ),
) -> None:
    """Fine-tune optimizer hyperparameters against a dataset of problem instances."""
    _enable_remote_if_requested(allow_remote, trust_remote_code)
    try:
        dataset_spec = load_dataset_spec(dataset)
        resolved_problem = problem
        if resolved_problem is None and dataset_spec.problem is not None:
            if dataset_spec.problem.startswith("github:"):
                resolved_problem = dataset_spec.problem
            else:
                candidate = Path(dataset_spec.problem)
                if not candidate.is_absolute():
                    candidate = (Path(dataset).resolve().parent / candidate).resolve()
                resolved_problem = str(candidate)

        if resolved_problem is None:
            raise typer.BadParameter(
                "Either provide --problem or include 'problem' in dataset YAML."
            )

        opt = AutoOptimizer.from_repo(optimizer)
        run = opt.finetune(
            problem_repo=resolved_problem,
            dataset_path=dataset,
            budget=budget,
            metric=metric,
            out_dir=out,
            seed=seed,
            optimizer_repo=optimizer,
        )
    except RemoteRepoNotAllowedError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    typer.echo(f"Best score: {run.score}")
    typer.echo(f"Artifact saved: {run.artifact_path}")


@app.command()
def run(
    problem: str = typer.Option(
        ..., "--problem", help="Problem repo spec (local path or github:...)."
    ),
    optimizer: str = typer.Option(
        ..., "--optimizer", help="Optimizer repo spec (local path or github:...)."
    ),
    trained: Path | None = typer.Option(
        None, "--trained", help="Optional path to trained.json artifact."
    ),
    problem_params: str | None = typer.Option(
        None, "--problem-params", help='JSON object, e.g. {"n_bits": 64}'
    ),
    optimizer_params: str | None = typer.Option(
        None, "--optimizer-params", help='JSON object, e.g. {"iterations": 500}'
    ),
    allow_remote: bool = typer.Option(
        False,
        "--allow-remote",
        help="Legacy alias for --trust-remote-code.",
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help=(
            "Allow loading and executing remote github: component repos. "
            "This runs arbitrary third-party Python on your machine; only "
            "enable for sources you trust."
        ),
    ),
) -> None:
    """Run a single optimization via the pipeline API."""
    _enable_remote_if_requested(allow_remote, trust_remote_code)
    try:
        parsed_problem_params = _parse_json_mapping(problem_params, "--problem-params")
        parsed_optimizer_params = _parse_json_mapping(
            optimizer_params, "--optimizer-params"
        )

        runner = build_pipeline(problem=problem, optimizer=optimizer, trained=trained)
        result = runner(
            problem_params=parsed_problem_params,
            optimizer_params=parsed_optimizer_params,
        )
    except RemoteRepoNotAllowedError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    typer.echo(f"Best value: {result.best_value}")
    typer.echo(f"Status: {result.status}")


@app.command()
def benchmark(
    dataset: Path = typer.Option(..., "--dataset", help="Path to dataset YAML."),
    problem: str | None = typer.Option(
        None,
        "--problem",
        help="Problem repo spec (local path or github:...). Optional if dataset defines 'problem'.",
    ),
    optimizers: list[str] = typer.Option(
        ...,
        "--optimizers",
        help="Optimizer repo specs (local path or github:...) and/or trained.json paths.",
    ),
    repeats: int = typer.Option(1, "--repeats", min=1, help="Runs per dataset instance."),
    seed: int | None = typer.Option(None, "--seed", help="Random seed."),
    out: Path | None = typer.Option(None, "--out", help="Optional JSON report output path."),
    allow_remote: bool = typer.Option(
        False,
        "--allow-remote",
        help="Legacy alias for --trust-remote-code.",
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help=(
            "Allow loading and executing remote github: component repos. "
            "This runs arbitrary third-party Python on your machine; only "
            "enable for sources you trust."
        ),
    ),
) -> None:
    """Benchmark one or more optimizers across a dataset."""
    _enable_remote_if_requested(allow_remote, trust_remote_code)
    try:
        report = run_benchmark(
            problem_repo=problem,
            dataset_path=dataset,
            optimizers=optimizers,
            repeats=repeats,
            seed=seed,
        )
    except RemoteRepoNotAllowedError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    typer.echo(report_to_markdown(report))
    if out is not None:
        path = write_report(report, out)
        typer.echo(f"Report saved: {path}")


@app.command(name="export-trained")
def export_trained(
    trained: Path = typer.Option(..., "--trained", help="Path to trained.json artifact."),
    out: Path = typer.Option(..., "--out", help="Output folder for exported optimizer."),
    name: str | None = typer.Option(None, "--name", help="Optional optimizer component name."),
) -> None:
    """Export a trained artifact as a reusable optimizer component repo."""
    exported = export_trained_optimizer(trained_path=trained, out_dir=out, name=name)
    typer.echo(f"Exported optimizer repo: {exported}")


@app.command(name="validate")
def validate_command(
    path: str = typer.Argument(..., metavar="PATH"),
    recursive: bool = typer.Option(
        False, "--recursive", help="Validate all subfolders containing qubots.yaml."
    ),
    allow_remote: bool = typer.Option(
        False,
        "--allow-remote",
        help="Legacy alias for --trust-remote-code.",
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help=(
            "Allow loading and executing remote github: component repos. "
            "This runs arbitrary third-party Python on your machine; only "
            "enable for sources you trust."
        ),
    ),
) -> None:
    """Validate one component repo or a tree of component repos."""
    _enable_remote_if_requested(allow_remote, trust_remote_code)
    if recursive:
        report = validate_tree(path)
        if not report:
            typer.echo(f"[FAIL] No component repos found under: {path}")
            raise typer.Exit(code=1)
    else:
        report = {str(path): validate_repo(path)}

    has_issues = False
    for repo_path, issues in report.items():
        if issues:
            has_issues = True
            typer.echo(f"[FAIL] {repo_path}")
            for issue in issues:
                typer.echo(f"  - {issue}")
        else:
            typer.echo(f"[OK] {repo_path}")

    raise typer.Exit(code=1 if has_issues else 0)


@app.command(name="new")
def new_component(
    kind: str = typer.Argument(
        ..., metavar="KIND", help="Component kind: 'problem' or 'optimizer'."
    ),
    name: str = typer.Option(..., "--name", help="Component name (slug-friendly, e.g. 'shift_scheduler')."),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output directory for the generated repo. Defaults to ./<name>.",
    ),
    flavor: str | None = typer.Option(
        None,
        "--flavor",
        help=(
            "Component flavor. Problems: 'blackbox', 'milp', 'dual' (default 'dual'). "
            "Optimizers: 'blackbox' (default), 'milp'."
        ),
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite files in the output directory if it already exists.",
    ),
) -> None:
    """Scaffold a new qubots problem or optimizer component repo."""
    kind_normalized = kind.strip().lower()
    if kind_normalized not in {"problem", "optimizer"}:
        raise typer.BadParameter("KIND must be 'problem' or 'optimizer'")

    target_dir = out if out is not None else Path(name)

    try:
        if kind_normalized == "problem":
            chosen_flavor = (flavor or "dual").strip().lower()
            if chosen_flavor not in PROBLEM_FLAVORS:
                raise typer.BadParameter(
                    f"Problem --flavor must be one of: {', '.join(PROBLEM_FLAVORS)}"
                )
            result = scaffold_problem(name, target_dir, flavor=chosen_flavor, force=force)
        else:
            chosen_flavor = (flavor or "blackbox").strip().lower()
            if chosen_flavor not in OPTIMIZER_FLAVORS:
                raise typer.BadParameter(
                    f"Optimizer --flavor must be one of: {', '.join(OPTIMIZER_FLAVORS)}"
                )
            result = scaffold_optimizer(name, target_dir, flavor=chosen_flavor, force=force)
    except (FileExistsError, ValueError) as exc:
        typer.echo(f"[FAIL] {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"[OK] Created {kind_normalized} '{name}' ({chosen_flavor}) at {result.path}")
    for path in result.files:
        typer.echo(f"  + {path.relative_to(result.path)}")
    typer.echo("")
    typer.echo(f"Validate with: qubots validate {result.path}")


@app.command(name="leaderboard")
def leaderboard_command(
    suite: Path = typer.Option(..., "--suite", help="Path to a leaderboard suite YAML."),
    submissions: Path = typer.Option(
        ...,
        "--submissions",
        help="Path to a submission YAML file or a directory of submission YAMLs.",
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Optional output path for LEADERBOARD.md."
    ),
    json_out: Path | None = typer.Option(
        None, "--json", help="Optional output path for leaderboard.json."
    ),
    repeats: int = typer.Option(1, "--repeats", min=1, help="Runs per dataset instance."),
    seed: int | None = typer.Option(None, "--seed", help="Random seed."),
    allow_remote: bool = typer.Option(
        False,
        "--allow-remote",
        help="Legacy alias for --trust-remote-code.",
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help=(
            "Allow loading and executing remote github: component repos. "
            "This runs arbitrary third-party Python on your machine; only "
            "enable for sources you trust."
        ),
    ),
) -> None:
    """Run a leaderboard: every submission against every benchmark in the suite."""
    _enable_remote_if_requested(allow_remote, trust_remote_code)
    try:
        suite_spec = load_suite(suite)
        submission_list = load_submissions_from_dir(submissions)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"[FAIL] {exc}")
        raise typer.Exit(code=1) from exc

    if not submission_list:
        typer.echo("[FAIL] No submissions found at the given path.")
        raise typer.Exit(code=1)

    try:
        report = run_leaderboard(
            suite_spec,
            submission_list,
            repeats=repeats,
            seed=seed,
        )
    except RemoteRepoNotAllowedError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    typer.echo(leaderboard_to_markdown(report))

    if out is not None or json_out is not None:
        json_path, md_path = write_leaderboard_report(
            report,
            json_path=json_out,
            markdown_path=out,
        )
        if md_path is not None:
            typer.echo(f"Markdown written: {md_path}")
        if json_path is not None:
            typer.echo(f"JSON written: {json_path}")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    show_version: bool = typer.Option(
        False,
        "--version",
        help="Show the installed version and exit.",
    ),
) -> None:
    """Root command group."""
    if show_version:
        version()
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        # Let Typer handle --help output naturally.
        return
