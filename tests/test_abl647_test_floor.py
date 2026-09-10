"""ABL-647: the CI gate that answers "did anything RUN?".

pytest's exit code answers "did anything fail?" and nothing else. A narrowed
`testpaths`, a collection error, or a deleted file all leave the command exiting
0 having run less -- and CI is the one reader that never notices, because nobody
reads a green check. `scripts/test_floor.py` reads the junit report and asserts
the counts; this file asserts the gate.

The skip allowance is the half that is easy to get wrong. A skipped test still
counts in the junit `tests` attribute, so a count floor alone cannot see one --
and this repo has a real gated set, since many tests need the replica database
or `models/` artifacts that no runner has. The allowance is therefore a ceiling
that is measured, not zero: what it buys is that answering a red run with
`@pytest.mark.skip` has to move a number in a diff.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "test_floor.py"
_spec = importlib.util.spec_from_file_location("abl647_test_floor", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
test_floor = importlib.util.module_from_spec(_spec)
sys.modules["abl647_test_floor"] = test_floor
_spec.loader.exec_module(test_floor)


FLOOR = {"tests": 100, "max_skipped": 4}


def _counts(**over: int) -> dict[str, int]:
    base = {"tests": 100, "failures": 0, "errors": 0, "skipped": 0}
    base.update(over)
    return base


def _write_report(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "report.xml"
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# read_junit_counts
# ---------------------------------------------------------------------------


def test_reads_a_testsuites_wrapped_report(tmp_path: Path) -> None:
    """What pytest actually writes: one `<testsuite>` inside `<testsuites>`."""
    report = _write_report(
        tmp_path,
        '<?xml version="1.0" encoding="utf-8"?><testsuites>'
        '<testsuite name="pytest" errors="1" failures="2" skipped="3" tests="40" />'
        "</testsuites>",
    )
    assert test_floor.read_junit_counts(report) == {
        "tests": 40,
        "failures": 2,
        "errors": 1,
        "skipped": 3,
    }


def test_reads_a_bare_testsuite_root(tmp_path: Path) -> None:
    """Also valid junit, and emitted by other tools. Not a shape to fall over on."""
    report = _write_report(
        tmp_path,
        '<testsuite name="pytest" errors="0" failures="0" skipped="1" tests="9" />',
    )
    assert test_floor.read_junit_counts(report)["tests"] == 9


def test_sums_across_multiple_suites(tmp_path: Path) -> None:
    report = _write_report(
        tmp_path,
        "<testsuites>"
        '<testsuite tests="10" failures="1" errors="0" skipped="2" />'
        '<testsuite tests="5" failures="0" errors="1" skipped="0" />'
        "</testsuites>",
    )
    counts = test_floor.read_junit_counts(report)
    assert counts == {"tests": 15, "failures": 1, "errors": 1, "skipped": 2}


def test_missing_attributes_read_as_zero_rather_than_crashing(tmp_path: Path) -> None:
    report = _write_report(tmp_path, '<testsuite tests="3" />')
    assert test_floor.read_junit_counts(report) == {
        "tests": 3,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_accepts_a_run_that_meets_the_floor_exactly() -> None:
    assert test_floor.evaluate(_counts(), FLOOR) == []


def test_accepts_a_run_above_the_floor() -> None:
    """Adding tests must not require bumping the floor first."""
    assert test_floor.evaluate(_counts(tests=140), FLOOR) == []


def test_rejects_a_green_run_that_lost_tests() -> None:
    """The ABL-647 case: nothing failed, so pytest exits 0, and 40 tests are gone."""
    problems = test_floor.evaluate(_counts(tests=60), FLOOR)
    assert len(problems) == 1
    assert "Ran 60 tests; the floor is 100" in problems[0]


def test_rejects_a_run_with_failures_even_when_the_count_is_met() -> None:
    assert "the run failed" in test_floor.evaluate(_counts(failures=1), FLOOR)[0]


def test_rejects_a_collection_error_which_is_an_error_and_not_a_failure() -> None:
    """A file that raises at import reports zero failures and one error."""
    assert "1 error(s)" in test_floor.evaluate(_counts(errors=1), FLOOR)[0]


def test_rejects_a_skip_past_the_allowance() -> None:
    problems = test_floor.evaluate(_counts(skipped=5), FLOOR)
    assert len(problems) == 1
    assert "the allowance is 4" in problems[0]


def test_allows_skips_up_to_the_allowance() -> None:
    """Tests gated on the replica DB and on `models/` are expected to skip in CI."""
    assert test_floor.evaluate(_counts(skipped=4), FLOOR) == []


@pytest.mark.parametrize("field", ["tests", "max_skipped"])
def test_refuses_an_unmeasured_floor_instead_of_passing_by_default(field: str) -> None:
    """`None` must read as "nobody measured this", never as "no limit, so fine"."""
    floor = dict(FLOOR)
    floor[field] = None
    problems = test_floor.evaluate(_counts(), floor)
    assert len(problems) == 1
    assert "scripts/test_floor.py" in problems[0]


# ---------------------------------------------------------------------------
# slack -- the direction the gate could not see (ABL-742)
# ---------------------------------------------------------------------------


def test_reports_the_surplus_when_a_run_is_above_its_floor() -> None:
    """The ABL-742 case: main ran 1833 against a floor of 1829 and stayed green."""
    assert test_floor.slack(_counts(tests=1833), {"tests": 1829, "max_skipped": 4}) == 4


def test_slack_is_none_when_the_run_meets_the_floor_exactly() -> None:
    assert test_floor.slack(_counts(), FLOOR) is None


def test_slack_is_none_when_the_run_came_in_short() -> None:
    """A drop is `evaluate`'s job and a failure. Slack must not also claim it."""
    assert test_floor.slack(_counts(tests=60), FLOOR) is None


def test_slack_is_none_against_an_unmeasured_floor() -> None:
    """`None` there already fails the build; there is no surplus to compute."""
    assert test_floor.slack(_counts(tests=999), {"tests": None}) is None


def test_slack_does_not_make_the_run_fail() -> None:
    """The asymmetry is the design: a pull_request build runs the MERGE of head
    into base, so a branch whose base gained tests legitimately runs more than
    its own floor. Reporting slack must not turn that into a red PR."""
    assert test_floor.evaluate(_counts(tests=1833), {"tests": 1829, "max_skipped": 4}) == []


def test_the_notice_names_the_number_to_record_not_just_the_surplus() -> None:
    """Naming the surplus leaves arithmetic to the reader, which is the step
    every stale floor in this repo got wrong."""
    notice = "\n".join(test_floor.describe_slack(1833, 4))
    assert "FLOOR['tests'] = 1833" in notice


def test_the_notice_carries_the_entry_point_rule() -> None:
    """"I added no tests, so the floor does not move" is false in this repo:
    a `scripts/*.py` entry point is +2 through two glob-parametrized families."""
    notice = "\n".join(test_floor.describe_slack(1833, 4))
    assert "+2" in notice and "scripts/*.py" in notice


def test_the_docstring_names_the_tests_that_make_an_entry_point_plus_two() -> None:
    """Measured on a checkout of main: adding one throwaway script took a
    collect of these two files 292 -> 294, and a throwaway `src/` module took it
    to 295. The docstring is the prominent copy of that rule, so it must keep
    naming the tests responsible rather than just asserting the number."""
    doc = test_floor.__doc__ or ""
    assert "test_help_text_is_ascii" in doc
    assert "test_script_import_preamble" in doc
    assert "test_no_flat_intra_src_imports" in doc


# ---------------------------------------------------------------------------
# main -- what the CI log and the PR checks tab actually show
# ---------------------------------------------------------------------------


def _report_running(tmp_path: Path, tests: int, skipped: int = 0) -> Path:
    return _write_report(
        tmp_path,
        f'<testsuites><testsuite tests="{tests}" failures="0" errors="0" '
        f'skipped="{skipped}" /></testsuites>',
    )


def test_the_headline_shows_the_surplus_and_the_run_still_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduces run 34504649220 on main at `999cc0d`, whose log read
    `1833 tests (floor 1829), 0 failed, 0 errored, 4 skipped` with no hint that
    four tests were outside the floor's view."""
    monkeypatch.setattr(test_floor, "FLOOR", {"tests": 1829, "max_skipped": 4})
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    assert test_floor.main([str(_report_running(tmp_path, 1833, skipped=4))]) == 0

    out = capsys.readouterr().out
    assert "1833 tests (floor 1829, SLACK +4 -- raise it)" in out
    assert "FLOOR['tests'] = 1833" in out


def test_a_run_at_its_floor_says_nothing_about_slack(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The quiet case has to stay quiet, or the notice is noise and gets ignored."""
    monkeypatch.setattr(test_floor, "FLOOR", {"tests": 1829, "max_skipped": 4})
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    assert test_floor.main([str(_report_running(tmp_path, 1829))]) == 0

    out = capsys.readouterr().out
    assert "1829 tests (floor 1829)," in out
    assert "SLACK" not in out


def test_slack_is_annotated_on_a_github_runner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A log line does not fix the visibility problem, because the problem is
    that nobody opens a green job's log. The annotation renders on the run
    summary and the checks tab without expanding a step."""
    monkeypatch.setattr(test_floor, "FLOOR", {"tests": 1829, "max_skipped": 4})
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    assert test_floor.main([str(_report_running(tmp_path, 1833, skipped=4))]) == 0

    annotations = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("::")
    ]
    assert len(annotations) == 1
    assert annotations[0].startswith("::warning file=scripts/test_floor.py::")
    assert "FLOOR['tests'] = 1833" in annotations[0]


def test_no_annotation_off_a_github_runner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`::warning` is literal text anywhere else -- a workstation run of the
    gate should not print a line that only means something to Actions."""
    monkeypatch.setattr(test_floor, "FLOOR", {"tests": 1829, "max_skipped": 4})
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    test_floor.main([str(_report_running(tmp_path, 1833, skipped=4))])

    assert "::warning" not in capsys.readouterr().out


def test_the_annotation_is_a_single_line() -> None:
    """A workflow command is line-delimited: a raw newline truncates the
    annotation at the first one and leaks the rest as log noise."""
    rendered = test_floor.github_warning("first\nsecond\r\nthird")
    assert "\n" not in rendered and "\r" not in rendered
    assert "%0A" in rendered and "%0D" in rendered


def test_the_annotation_escapes_percent_before_anything_else() -> None:
    """Escaping `%` after inserting `%0A` would re-escape the escapes."""
    assert test_floor.github_warning("100%\n2") == (
        "::warning file=scripts/test_floor.py::100%25%0A2"
    )


def test_the_recorded_floor_is_measured_not_left_unset() -> None:
    """The shipped FLOOR must carry real numbers.

    `None` is the bootstrap state: the gate reports what it saw and fails, so
    the first CI run measures the floor and the second commit records it. This
    test is what stops that bootstrap state from becoming permanent.
    """
    assert test_floor.FLOOR["tests"] is not None, "run CI once, then record the count"
    assert test_floor.FLOOR["max_skipped"] is not None
    assert test_floor.FLOOR["tests"] > 0
