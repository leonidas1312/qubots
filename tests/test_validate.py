from pathlib import Path

from qubots.cli.app import app
from qubots.validate.validate import validate_repo, validate_tree
from typer.testing import CliRunner


ROOT = Path(__file__).resolve().parents[1]


def test_validate_good_example_repos() -> None:
    assert validate_repo(ROOT / "examples" / "one_max_problem") == []
    assert validate_repo(ROOT / "examples" / "random_search_optimizer") == []


def test_validate_bad_repo_reports_issues(tmp_path: Path) -> None:
    bad_repo = tmp_path / "bad_repo"
    bad_repo.mkdir(parents=True, exist_ok=True)
    (bad_repo / "qubots.yaml").write_text(
        "\n".join(
            [
                "type: optimizer",
                "entrypoint: not_a_valid_entrypoint",
            ]
        ),
        encoding="utf-8",
    )

    issues = validate_repo(bad_repo)
    assert issues
    assert any("missing required keys" in issue.lower() for issue in issues)
    assert any("entrypoint" in issue.lower() for issue in issues)


def test_validate_tree_recursive_finds_multiple(tmp_path: Path) -> None:
    good_repo = tmp_path / "good_problem"
    good_repo.mkdir(parents=True, exist_ok=True)
    (good_repo / "qubots.yaml").write_text(
        "\n".join(
            [
                "type: problem",
                "name: good_problem",
                "entrypoint: qubot.py:GoodProblem",
            ]
        ),
        encoding="utf-8",
    )
    (good_repo / "qubot.py").write_text(
        "\n".join(
            [
                "from qubots.core.problem import BaseProblem",
                "",
                "class GoodProblem(BaseProblem):",
                "    def evaluate(self, solution):",
                "        return 0.0",
                "",
                "    def random_solution(self):",
                "        return 0",
            ]
        ),
        encoding="utf-8",
    )

    bad_repo = tmp_path / "bad_optimizer"
    bad_repo.mkdir(parents=True, exist_ok=True)
    (bad_repo / "qubots.yaml").write_text(
        "\n".join(
            [
                "type: wrong_type",
                "name: bad_optimizer",
                "entrypoint: qubot.py:MissingClass",
            ]
        ),
        encoding="utf-8",
    )
    (bad_repo / "qubot.py").write_text("class SomethingElse:\n    pass\n", encoding="utf-8")

    report = validate_tree(tmp_path)
    assert str(good_repo.resolve()) in report
    assert str(bad_repo.resolve()) in report
    assert report[str(good_repo.resolve())] == []
    assert report[str(bad_repo.resolve())]


def test_validate_cli_exit_codes(tmp_path: Path) -> None:
    runner = CliRunner()

    ok_result = runner.invoke(app, ["validate", str(ROOT / "examples" / "one_max_problem")])
    assert ok_result.exit_code == 0
    assert "[OK]" in ok_result.output

    bad_repo = tmp_path / "bad_repo_cli"
    bad_repo.mkdir(parents=True, exist_ok=True)
    (bad_repo / "qubots.yaml").write_text("type: optimizer\n", encoding="utf-8")

    bad_result = runner.invoke(app, ["validate", str(bad_repo)])
    assert bad_result.exit_code == 1
    assert "[FAIL]" in bad_result.output
