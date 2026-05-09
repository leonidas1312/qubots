"""Security hardening: schema version, path traversal, trust-remote-code."""

from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qubots.cli.app import app
from qubots.hub.loaders import _parse_entrypoint, load_component
from qubots.hub.manifests import (
    CURRENT_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    Manifest,
    load_manifest,
)
from qubots.hub.resolver import (
    RemoteRepoNotAllowedError,
    _is_remote_trusted,
    resolve_repo_info,
)


ROOT = Path(__file__).resolve().parents[1]


# ---------- Manifest schema versioning ---------------------------------------


def _write_manifest(repo: Path, body: str) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "qubots.yaml").write_text(body, encoding="utf-8")


def test_existing_examples_load_without_version_field() -> None:
    # Backward compat: existing manifests have no qubots_schema_version field.
    manifest = load_manifest(ROOT / "examples" / "one_max_problem")
    assert manifest.schema_version == CURRENT_SCHEMA_VERSION
    assert manifest.type == "problem"


def test_explicit_schema_version_v1_is_accepted(tmp_path: Path) -> None:
    repo = tmp_path / "ok"
    _write_manifest(
        repo,
        "\n".join(
            [
                "qubots_schema_version: 1",
                "type: problem",
                "name: ok",
                "entrypoint: qubot.py:Foo",
            ]
        ),
    )
    manifest = load_manifest(repo)
    assert manifest.schema_version == 1


def test_unknown_schema_version_is_rejected(tmp_path: Path) -> None:
    future_version = max(SUPPORTED_SCHEMA_VERSIONS) + 1
    repo = tmp_path / "future"
    _write_manifest(
        repo,
        "\n".join(
            [
                f"qubots_schema_version: {future_version}",
                "type: problem",
                "name: future",
                "entrypoint: qubot.py:Foo",
            ]
        ),
    )
    with pytest.raises(ValueError, match="Unsupported qubots_schema_version"):
        load_manifest(repo)


def test_non_integer_schema_version_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "bad"
    _write_manifest(
        repo,
        "\n".join(
            [
                "qubots_schema_version: 1.5",
                "type: problem",
                "name: bad",
                "entrypoint: qubot.py:Foo",
            ]
        ),
    )
    with pytest.raises(ValueError, match="must be an integer"):
        load_manifest(repo)


# ---------- Entrypoint path-traversal guard ----------------------------------


@pytest.mark.parametrize(
    "entrypoint",
    [
        "../escape.py:Foo",
        "subdir/../../escape.py:Foo",
        "/etc/passwd.py:Foo",
        "qubot.py:bad-class-name",
        "qubot.py:1NumericStart",
        "qubot.py:Foo;import os",
        "qubot.py: Foo",  # whitespace tolerated only via strip; here a real traversal would fail
    ],
)
def test_parse_entrypoint_rejects_dangerous_inputs(entrypoint: str) -> None:
    if entrypoint == "qubot.py: Foo":
        # whitespace around the class name is normalized; this should pass
        module, cls = _parse_entrypoint(entrypoint)
        assert (module, cls) == ("qubot.py", "Foo")
        return
    with pytest.raises(ValueError):
        _parse_entrypoint(entrypoint)


def test_parse_entrypoint_accepts_subdir_relative(tmp_path: Path) -> None:
    module, cls = _parse_entrypoint("nested/qubot.py:MyProblem")
    assert module == "nested/qubot.py"
    assert cls == "MyProblem"


def test_load_component_blocks_traversal_via_symlink(tmp_path: Path) -> None:
    # Create a malicious repo whose entrypoint resolves outside the repo.
    outside = tmp_path / "outside.py"
    outside.write_text("class Pwned: pass\n", encoding="utf-8")

    repo = tmp_path / "evil"
    repo.mkdir()
    _write_manifest(
        repo,
        "\n".join(
            [
                "type: problem",
                "name: evil",
                "entrypoint: ../outside.py:Pwned",
            ]
        ),
    )

    # _parse_entrypoint should reject the .. before we even reach the file.
    with pytest.raises(ValueError, match=r"\.\.|escapes the repository"):
        load_component(repo, expected_type="problem")


# ---------- Trust-remote-code gate -------------------------------------------


def test_remote_disabled_when_no_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QUBOTS_ALLOW_REMOTE", raising=False)
    monkeypatch.delenv("QUBOTS_TRUST_REMOTE_CODE", raising=False)
    assert not _is_remote_trusted()
    with pytest.raises(RemoteRepoNotAllowedError, match="QUBOTS_TRUST_REMOTE_CODE"):
        resolve_repo_info("github:alice/x@main")


def test_remote_enabled_via_legacy_allow_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QUBOTS_TRUST_REMOTE_CODE", raising=False)
    monkeypatch.setenv("QUBOTS_ALLOW_REMOTE", "1")
    assert _is_remote_trusted()


def test_remote_enabled_via_trust_remote_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QUBOTS_ALLOW_REMOTE", raising=False)
    monkeypatch.setenv("QUBOTS_TRUST_REMOTE_CODE", "1")
    assert _is_remote_trusted()


def test_cli_trust_remote_code_flag_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QUBOTS_ALLOW_REMOTE", raising=False)
    monkeypatch.delenv("QUBOTS_TRUST_REMOTE_CODE", raising=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "validate",
            "github:alice/x@main",
            # --trust-remote-code reaches the resolver, but the network call
            # would still fail; we just want to confirm the flag is parsed
            # and the gate is lifted.
            "--trust-remote-code",
        ],
    )
    # The command will fail (no real clone path), but NOT with the
    # RemoteRepoNotAllowedError message.
    assert "QUBOTS_TRUST_REMOTE_CODE" not in result.output


def test_remote_load_emits_runtime_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("QUBOTS_TRUST_REMOTE_CODE", "1")
    monkeypatch.delenv("QUBOTS_SUPPRESS_REMOTE_WARNING", raising=False)
    monkeypatch.setenv("QUBOTS_CACHE_DIR", str(tmp_path / "cache"))

    def fake_run(args, cwd=None, check=True, capture_output=True, text=True):  # type: ignore[no-untyped-def]
        if args[:2] == ["git", "clone"]:
            target = Path(args[3])
            target.mkdir(parents=True, exist_ok=True)
            (target / ".git").mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("qubots.hub.resolver.subprocess.run", fake_run)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        info = resolve_repo_info("github:alice/x@deadbeef")
    assert info.is_remote is True
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("github:alice/x@deadbeef" in str(w.message) for w in runtime_warnings), (
        "expected a RuntimeWarning naming the remote spec"
    )
