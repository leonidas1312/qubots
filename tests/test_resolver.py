from pathlib import Path
import subprocess

import pytest

from qubots.hub.resolver import (
    RemoteRepoNotAllowedError,
    parse_github_spec,
    resolve_repo,
    resolve_repo_info,
)
from qubots.cli.app import app
from qubots.validate.validate import validate_repo
from typer.testing import CliRunner


ROOT = Path(__file__).resolve().parents[1]


def test_parse_github_spec_full_and_default_ref() -> None:
    parsed = parse_github_spec("github:alice/my-solver@f00dbabe:solver_component")
    assert parsed.owner == "alice"
    assert parsed.repo == "my-solver"
    assert parsed.ref == "f00dbabe"
    assert parsed.subdir == "solver_component"

    parsed_default = parse_github_spec("github:qubots-ai/problems")
    assert parsed_default.owner == "qubots-ai"
    assert parsed_default.repo == "problems"
    assert parsed_default.ref == "main"
    assert parsed_default.subdir is None


def test_resolver_rejects_remote_when_not_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QUBOTS_ALLOW_REMOTE", raising=False)

    with pytest.raises(RemoteRepoNotAllowedError):
        resolve_repo("github:alice/my-solver@main")

    issues = validate_repo("github:alice/my-solver@main")
    assert issues
    assert "QUBOTS_ALLOW_REMOTE=1" in issues[0]


def test_resolver_clones_and_reuses_cache_with_mocked_git(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("QUBOTS_ALLOW_REMOTE", "1")
    monkeypatch.setenv("QUBOTS_CACHE_DIR", str(tmp_path / "cache"))

    calls: list[list[str]] = []

    def fake_run(
        args: list[str],
        cwd: str | None = None,
        check: bool = True,
        capture_output: bool = True,
        text: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))

        if args[:2] == ["git", "clone"]:
            target = Path(args[3])
            target.mkdir(parents=True, exist_ok=True)
            (target / ".git").mkdir(parents=True, exist_ok=True)
            (target / "component").mkdir(parents=True, exist_ok=True)

        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("qubots.hub.resolver.subprocess.run", fake_run)

    info = resolve_repo_info("github:alice/my-solver@abc123:component")
    assert info.is_remote is True
    assert info.ref == "abc123"
    assert info.resolved_path.exists()
    assert info.resolved_path.name == "component"

    assert any(cmd[:2] == ["git", "clone"] for cmd in calls)
    assert any(cmd[:2] == ["git", "checkout"] for cmd in calls)

    calls.clear()
    info_again = resolve_repo_info("github:alice/my-solver@abc123:component")
    assert info_again.resolved_path == info.resolved_path
    assert calls == []


def test_run_cli_remote_disabled_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QUBOTS_ALLOW_REMOTE", raising=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--problem",
            "github:alice/problem-repo@main:component",
            "--optimizer",
            str(ROOT / "examples" / "random_search_optimizer"),
        ],
    )
    assert result.exit_code == 1
    assert "QUBOTS_ALLOW_REMOTE=1" in result.output
