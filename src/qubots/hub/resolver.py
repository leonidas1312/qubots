"""Repository resolver for local paths and GitHub specs.

Security model:

- Local paths are trusted (the user has the code on disk and can read it).
- GitHub specs are untrusted by default. Loading them executes arbitrary
  Python code from a third party, so two opt-ins are needed:

  - ``QUBOTS_TRUST_REMOTE_CODE=1`` (preferred) or ``--trust-remote-code``
    on the CLI, OR
  - ``QUBOTS_ALLOW_REMOTE=1`` (legacy alias kept for backward compatibility).

  Either flag enables remote loading; the explicit
  ``trust-remote-code`` name is preferred because it makes the security
  trade-off visible. Each remote load also emits a ``RuntimeWarning``
  showing the resolved repo + ref so users can see what they ran.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_GITHUB_SPEC_RE = re.compile(
    r"^github:(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)"
    r"(?:@(?P<ref>[^:]+))?"
    r"(?::(?P<subdir>.+))?$"
)


@dataclass(frozen=True)
class GitHubSpec:
    owner: str
    repo: str
    ref: str
    subdir: str | None


@dataclass(frozen=True)
class ResolvedRepo:
    spec: str
    resolved_path: Path
    is_remote: bool
    ref: str | None
    owner: str | None
    repo: str | None
    subdir: str | None


class RemoteRepoNotAllowedError(PermissionError):
    pass


def is_github_spec(spec: str | Path) -> bool:
    return str(spec).startswith("github:")


def parse_github_spec(spec: str) -> GitHubSpec:
    match = _GITHUB_SPEC_RE.match(spec.strip())
    if not match:
        raise ValueError(
            "Invalid GitHub spec. Expected format: "
            "github:<owner>/<repo>@<ref>[:<subdir>]"
        )

    owner = match.group("owner")
    repo = match.group("repo")
    ref = match.group("ref") or "main"
    subdir = match.group("subdir")

    return GitHubSpec(owner=owner, repo=repo, ref=ref, subdir=subdir)


def _cache_root() -> Path:
    custom = os.environ.get("QUBOTS_CACHE_DIR")
    if custom:
        return Path(custom).expanduser().resolve()
    return Path.home() / ".cache" / "qubots" / "repos"


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _is_remote_trusted() -> bool:
    """Either the modern flag or the legacy alias enables remote loading."""
    return _truthy_env("QUBOTS_TRUST_REMOTE_CODE") or _truthy_env("QUBOTS_ALLOW_REMOTE")


def _assert_remote_allowed() -> None:
    if _is_remote_trusted():
        return
    raise RemoteRepoNotAllowedError(
        "Loading a remote GitHub spec executes arbitrary third-party Python "
        "code on your machine. To opt in, set QUBOTS_TRUST_REMOTE_CODE=1 "
        "(or pass --trust-remote-code on the CLI). "
        "QUBOTS_ALLOW_REMOTE=1 / --allow-remote remain as legacy aliases."
    )


def _warn_on_remote_load(spec: GitHubSpec, resolved_path: Path) -> None:
    if _truthy_env("QUBOTS_SUPPRESS_REMOTE_WARNING"):
        return
    suffix = f":{spec.subdir}" if spec.subdir else ""
    location = f"github:{spec.owner}/{spec.repo}{suffix}@{spec.ref}"
    warnings.warn(
        f"Executing remote qubots component {location} "
        f"(resolved to {resolved_path}). Trust the source.",
        RuntimeWarning,
        stacklevel=3,
    )


def _run_git(args: list[str], cwd: Path | None = None) -> None:
    try:
        subprocess.run(
            args,
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or str(exc)
        raise RuntimeError(f"Git command failed: {' '.join(args)}\n{details}") from exc


def _clone_or_reuse(github: GitHubSpec) -> Path:
    base_dir = _cache_root() / f"{github.owner}__{github.repo}" / github.ref
    git_dir = base_dir / ".git"

    if git_dir.exists():
        return base_dir

    base_dir.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = base_dir.with_name(base_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    remote_url = f"https://github.com/{github.owner}/{github.repo}.git"

    _run_git(["git", "clone", remote_url, str(tmp_dir)])
    _run_git(["git", "checkout", github.ref], cwd=tmp_dir)

    try:
        tmp_dir.rename(base_dir)
    except FileExistsError:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return base_dir


def _manifest_name(repo_path: Path) -> str | None:
    manifest_path = repo_path / "qubots.yaml"
    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        return None

    name = raw.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def derive_repo_name(spec: str | Path, resolved_path: str | Path | None = None) -> str:
    spec_str = str(spec)
    resolved = Path(resolved_path).resolve() if resolved_path is not None else None

    if resolved is not None:
        from_manifest = _manifest_name(resolved)
        if from_manifest:
            return from_manifest

    if is_github_spec(spec_str):
        parsed = parse_github_spec(spec_str)
        if resolved is not None:
            from_manifest = _manifest_name(resolved)
            if from_manifest:
                return from_manifest
        suffix = f":{parsed.subdir}" if parsed.subdir else ""
        return f"{parsed.owner}/{parsed.repo}{suffix}@{parsed.ref}"

    path = Path(spec_str)
    if resolved is None:
        resolved = path.resolve()

    from_manifest = _manifest_name(resolved)
    if from_manifest:
        return from_manifest

    return path.name or str(path)


def resolve_repo_info(spec: str | Path) -> ResolvedRepo:
    spec_str = str(spec)
    as_path = Path(spec_str).expanduser()

    if as_path.exists():
        resolved = as_path.resolve()
        return ResolvedRepo(
            spec=spec_str,
            resolved_path=resolved,
            is_remote=False,
            ref=None,
            owner=None,
            repo=None,
            subdir=None,
        )

    if not is_github_spec(spec_str):
        resolved = as_path.resolve()
        return ResolvedRepo(
            spec=spec_str,
            resolved_path=resolved,
            is_remote=False,
            ref=None,
            owner=None,
            repo=None,
            subdir=None,
        )

    _assert_remote_allowed()
    parsed = parse_github_spec(spec_str)
    repo_root = _clone_or_reuse(parsed)

    resolved_path = repo_root
    if parsed.subdir:
        resolved_path = repo_root / parsed.subdir
        if not resolved_path.exists() or not resolved_path.is_dir():
            raise FileNotFoundError(
                f"GitHub spec subdir not found in cloned repo: {parsed.subdir}"
            )

    final_path = resolved_path.resolve()
    _warn_on_remote_load(parsed, final_path)

    return ResolvedRepo(
        spec=spec_str,
        resolved_path=final_path,
        is_remote=True,
        ref=parsed.ref,
        owner=parsed.owner,
        repo=parsed.repo,
        subdir=parsed.subdir,
    )


def resolve_repo(spec: str | Path) -> Path:
    return resolve_repo_info(spec).resolved_path
