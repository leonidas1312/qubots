"""Repository resolver for local paths and GitHub specs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
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


def _assert_remote_allowed() -> None:
    if os.environ.get("QUBOTS_ALLOW_REMOTE") == "1":
        return
    raise RemoteRepoNotAllowedError(
        "Remote GitHub specs are disabled. Set QUBOTS_ALLOW_REMOTE=1 "
        "or pass --allow-remote on CLI commands."
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

    return ResolvedRepo(
        spec=spec_str,
        resolved_path=resolved_path.resolve(),
        is_remote=True,
        ref=parsed.ref,
        owner=parsed.owner,
        repo=parsed.repo,
        subdir=parsed.subdir,
    )


def resolve_repo(spec: str | Path) -> Path:
    return resolve_repo_info(spec).resolved_path
