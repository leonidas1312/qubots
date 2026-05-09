"""Schema v2 introduces a top-level ``requirements:`` field on component manifests.

v1 manifests must continue to load (no ``requirements`` field, treated as []).
v2 manifests are accepted and parsed. Malformed ``requirements`` entries are
rejected by both ``load_manifest`` and ``validate_repo``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qubots.hub.manifests import (
    CURRENT_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    load_manifest,
)
from qubots.validate.validate import validate_repo


ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(repo: Path, body: str) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "qubots.yaml").write_text(body, encoding="utf-8")


def test_current_schema_is_v2() -> None:
    assert CURRENT_SCHEMA_VERSION == 2
    assert 1 in SUPPORTED_SCHEMA_VERSIONS  # v1 backward compat
    assert 2 in SUPPORTED_SCHEMA_VERSIONS


def test_v1_manifest_still_loads_with_empty_requirements(tmp_path: Path) -> None:
    repo = tmp_path / "v1_repo"
    _write_manifest(
        repo,
        "\n".join(
            [
                "qubots_schema_version: 1",
                "type: problem",
                "name: legacy",
                "entrypoint: qubot.py:Foo",
            ]
        ),
    )
    manifest = load_manifest(repo)
    assert manifest.schema_version == 1
    assert manifest.requirements == []


def test_v2_manifest_with_requirements_loads(tmp_path: Path) -> None:
    repo = tmp_path / "v2_repo"
    _write_manifest(
        repo,
        "\n".join(
            [
                "qubots_schema_version: 2",
                "type: optimizer",
                "name: highs_demo",
                "entrypoint: qubot.py:HighsDemo",
                "requirements:",
                '  - "highspy>=1.7"',
                '  - "certifi>=2024.0"',
            ]
        ),
    )
    manifest = load_manifest(repo)
    assert manifest.schema_version == 2
    assert manifest.requirements == ["highspy>=1.7", "certifi>=2024.0"]


def test_v2_manifest_rejects_non_list_requirements(tmp_path: Path) -> None:
    repo = tmp_path / "bad_req"
    _write_manifest(
        repo,
        "\n".join(
            [
                "qubots_schema_version: 2",
                "type: optimizer",
                "name: bad",
                "entrypoint: qubot.py:Foo",
                "requirements: not_a_list",
            ]
        ),
    )
    with pytest.raises(ValueError, match="requirements"):
        load_manifest(repo)


def test_v2_manifest_rejects_non_string_requirement(tmp_path: Path) -> None:
    repo = tmp_path / "bad_req"
    _write_manifest(
        repo,
        "\n".join(
            [
                "qubots_schema_version: 2",
                "type: optimizer",
                "name: bad",
                "entrypoint: qubot.py:Foo",
                "requirements:",
                "  - 42",
            ]
        ),
    )
    with pytest.raises(ValueError, match="requirements"):
        load_manifest(repo)


def test_validate_repo_flags_invalid_requirement_spec(tmp_path: Path) -> None:
    repo = tmp_path / "bad_spec"
    repo.mkdir(parents=True)
    (repo / "qubots.yaml").write_text(
        "\n".join(
            [
                "qubots_schema_version: 2",
                "type: problem",
                "name: bad_spec",
                "entrypoint: qubot.py:Foo",
                "requirements:",
                "  - '!! not a pip spec !!'",
            ]
        ),
        encoding="utf-8",
    )
    (repo / "qubot.py").write_text(
        "from qubots.core.problem import BaseProblem\n"
        "class Foo(BaseProblem):\n"
        "    def evaluate(self, s): return 0.0\n"
        "    def random_solution(self): return 0\n",
        encoding="utf-8",
    )
    issues = validate_repo(repo)
    assert any("requirements" in issue for issue in issues)


def test_existing_examples_still_validate_after_v2_bump() -> None:
    """Real example components on disk continue to pass validation."""
    for repo_name in (
        "knapsack_milp_problem",
        "highs_optimizer",
        "cpsat_optimizer",
        "mps_problem",
        "one_max_problem",
        "random_search_optimizer",
    ):
        issues = validate_repo(ROOT / "examples" / repo_name)
        assert issues == [], f"{repo_name} regressed: {issues}"
