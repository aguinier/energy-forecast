#!/usr/bin/env python3
"""CI tripwire (ABL-647): a green suite that ran fewer tests than it used to is
not a green suite.

    python scripts/test_floor.py <junit-xml-report>

pytest's exit code answers "did anything fail?". It does not answer "did
anything run?" -- a narrowed `testpaths`, a collection error swallowed by a
`-p no:cacheprovider` style flag, or a deleted file can leave the command
exiting 0 having run less. `pytest.ini` pins `testpaths = tests` precisely
because collection scope has bitten this repo before (ABL-336), and CI is the
one reader that never notices a quiet drop, because nobody reads a green check.

The count check fires on a DROP only, so the gate also reports SLACK -- a run
that collected MORE than the recorded floor (ABL-742). Slack is not a build
failure; see the note above `slack()` for why, and for what it prints instead.

**Adding no test file does not mean the floor does not move.** Two families are
glob-parametrized over the entry points, so a new `scripts/*.py` script is +2
collected tests on its own::

    tests/test_help_text_encoding.py::test_help_text_is_ascii[scripts/<name>.py]
    tests/test_script_imports.py::test_script_import_preamble[<name>.py]

and a new top-level `src/` module is +1 more::

    tests/test_script_imports.py::test_no_flat_intra_src_imports[src/<name>.py]

Measured, not read off the source: adding one throwaway script and one
throwaway `src/` module to a checkout of main took a collect of those two files
from 292 to 294 to 295. That is how main's floor went 4 stale -- the ABL-739
train merged two PRs that added two entry points and no test file, so neither
author had a reason to touch `FLOOR`, and CI ran 1833 against 1829 and stayed
green.

Standard library only, on purpose: it has to run in a job whose `pip install`
step is the thing under suspicion.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# The measured floor, set from a full run ON THE CI RUNNER -- not from a
# workstation, and not guessed.
#
# Raising it is routine: do it in the commit that adds the tests. LOWERING it is
# the interesting case, because it means test coverage left the repo -- say in
# the commit message which tests went and why. A floor lowered to make CI green
# again, with no explanation, is the failure this file exists to prevent.
#
# `max_skipped` is separate because a skipped test still counts in the junit
# `tests` attribute, so the count floor cannot see one. A skip past it fails the
# build, so answering a red run with `@pytest.mark.skip` takes a diff and a
# reason.
#
# The allowance was 1, which was lower than expected and is itself a result:
# tests gated on the replica database and on `models/` artifacts were assumed to
# be a sizeable set, and the runner has neither. It ran 1766 and skipped one.
# Those tests do not skip on a bare runner — they assert against stored
# documents. The pytest step passes `-rs`, so the log always names what skipped
# and why; read that before changing this number.
#
# It is 4 as of ABL-715, which declared three of them. All three are the same
# gate: the chronos-2 entry points (`scripts/forecast_chronos2.py`,
# `scripts/train_chronos2.py`, and the `chronos-2` runner launch) import torch,
# and no requirements file in this repo declares torch, so they cannot be
# imported from a clean checkout. The gate is keyed on torch being absent, never
# on the runner's name, so all three RUN on any box that has it —
# `tests/chronos2_env.py` carries the reasoning and the cost argument for not
# installing torch on every PR.
#
# ABL-732 adds 10 (tests/test_abl732_transcript_wrap.py), so the count goes to
# 1828. Four of them execute a PowerShell block and are gated on a `pwsh` or
# `powershell` on PATH. PowerShell Core is preinstalled on `ubuntu-latest`, so
# they are expected to RUN and `max_skipped` is left at 4; if the runner's image
# ever drops it they skip instead, and this gate is the thing that says so
# rather than the four quietly stopping.
#
# ABL-735 adds 1 to the same file -- the non-vacuity control on that file's own
# fixture-rebinding guard -- so the count goes to 1829. It is deliberately NOT
# gated on PowerShell (it asserts on a guard that raises before any subprocess),
# so the allowance stays at 4.
#
# ABL-742 records what main was ALREADY running but this number could not see.
# The ABL-739 train (`999cc0d`) merged two PRs that added no test file and two
# `scripts/*.py` entry points, and CI measured 1833 against this floor of 1829
# and stayed green. 1833 is the CI number, from run 34504649220 on the runner --
# the workstation venv is on a different Python minor to `.python-version`, so a
# workstation count is not admissible here. ABL-742 then adds 14 to
# `tests/test_abl647_test_floor.py` (purely additive: 14 collected -> 28), so
# the count goes 1833 + 14 -> 1847. None of the 14 are gated on anything -- they
# are stdlib, `tmp_path` and `monkeypatch` -- so `max_skipped` stays at 4.
#
# Note for whoever merges this against another branch that also moves this
# number: those increments are ADDITIVE. Sum them; do not take the higher side.
# That resolution is exactly what this commit makes visible, because taking the
# higher side leaves the floor BELOW what the merged tree runs, and a floor
# below the run is green.
#
# `None` means "not yet measured": the gate then reports what it saw and fails,
# so a floor cannot be quietly left unset.
FLOOR: dict[str, int | None] = {
    "tests": 1847,
    "max_skipped": 4,
}


def read_junit_counts(path: Path) -> dict[str, int]:
    """Flatten a junit XML report into the numbers this gate cares about.

    pytest writes a `<testsuites>` root wrapping one `<testsuite>`, but a bare
    `<testsuite>` root is also valid junit and other tools emit it. Summing over
    every `testsuite` element handles both without caring which one it got.
    """
    root = ET.parse(path).getroot()
    suites = root.iter("testsuite")

    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.get(key, 0) or 0)
    return counts


def evaluate(counts: dict[str, int], floor: dict[str, int | None]) -> list[str]:
    """Return a list of human-readable problems; empty means the run is fine."""
    problems: list[str] = []

    if counts["failures"] or counts["errors"]:
        problems.append(
            f"The report says the run failed ({counts['failures']} failure(s), "
            f"{counts['errors']} error(s))."
        )

    floor_tests = floor.get("tests")
    if floor_tests is None:
        problems.append(
            "No test floor is recorded. Set FLOOR['tests'] in scripts/test_floor.py "
            f"to the number this run measured ({counts['tests']}), in a commit."
        )
    elif counts["tests"] < floor_tests:
        problems.append(
            f"Ran {counts['tests']} tests; the floor is {floor_tests}. "
            f"{floor_tests - counts['tests']} test(s) that used to run did not."
        )

    max_skipped = floor.get("max_skipped")
    if max_skipped is None:
        problems.append(
            "No skip allowance is recorded. Set FLOOR['max_skipped'] in "
            f"scripts/test_floor.py to what this run measured ({counts['skipped']}), "
            "and say in the commit which tests are gated and on what."
        )
    elif counts["skipped"] > max_skipped:
        problems.append(
            f"{counts['skipped']} test(s) skipped; the allowance is {max_skipped}. "
            "A skipped test still counts in the junit total, so the floor above "
            "cannot see it -- it is asserted separately."
        )

    return problems


# ---------------------------------------------------------------------------
# Slack: the direction this gate could not see (ABL-742)
# ---------------------------------------------------------------------------
#
# `evaluate` above fires on a DROP only. A floor set too LOW is therefore never
# red -- and green is exactly what slack looks like, so nothing says so. Main at
# `999cc0d` ran 1833 tests against a floor of 1829 and passed, blind to those 4
# for as long as the number stayed stale.
#
# Slack is reported, not failed, and the asymmetry is deliberate. A
# `pull_request` build runs the MERGE of head into base, so a branch whose base
# gained tests while it was out legitimately collects more than its own floor;
# failing on that would turn every open PR red the moment main added a test, and
# the fix would be a rebase rather than anything about the branch. Exactness is
# only attainable on a `push` to main, which catches the drift one commit AFTER
# the merge that caused it -- a red main blocks everyone, so that trade is a CI
# ergonomics decision and not this file's to make unilaterally.
#
# What it does instead is remove the excuse: it prints the surplus, the number
# to record, and the rule that explains where the surplus came from.


def slack(counts: dict[str, int], floor: dict[str, int | None]) -> int | None:
    """How many tests this run collected ABOVE the recorded floor.

    `None` means there is nothing to report: the floor is unmeasured (already a
    hard failure in `evaluate`), the run met it exactly, or the run came in
    short (`evaluate`'s job, and a failure rather than slack). Only a strictly
    positive surplus comes back, so callers can treat the result as a flag.
    """
    floor_tests = floor.get("tests")
    if floor_tests is None:
        return None
    surplus = counts["tests"] - floor_tests
    return surplus if surplus > 0 else None


def describe_slack(ran: int, surplus: int) -> list[str]:
    """The lines to print when a run came in above its floor.

    It names the number to WRITE, not just the number it saw. The reader
    already has both counts in front of them and still has to work out what to
    record; every stale floor this repo has had came from someone not doing
    that arithmetic, or not knowing the floor had moved at all.
    """
    return [
        f"SLACK: this run collected {surplus} more test(s) than the recorded floor.",
        f"Record it: set FLOOR['tests'] = {ran} in scripts/test_floor.py, in the "
        "commit that moved it.",
        "A floor below what the suite runs is not a safe margin. It is blind to "
        "exactly those tests disappearing again.",
        "Adding no test FILE does not mean the floor does not move: a new "
        "scripts/*.py entry point is +2 collected tests by itself, and a new "
        "top-level src/ module is +1 more. This file's docstring names the three "
        "glob-parametrized tests responsible.",
        "If the base branch gained tests while this branch was out, this is the "
        "expected shape on a pull_request build. Resolve a conflict on FLOOR by "
        "SUMMING the increments, never by taking the higher of the two sides.",
    ]


def github_warning(message: str) -> str:
    """Render a message as a GitHub Actions warning annotation.

    A log line alone does not fix the visibility problem, because the problem IS
    that nobody opens a green job's log. An annotation renders on the run
    summary page and on the PR's checks tab without anyone expanding a step, and
    it leaves the exit status alone. A workflow command is one line, so the
    newlines have to be escaped rather than emitted.
    """
    escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return f"::warning file=scripts/test_floor.py::{escaped}"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python scripts/test_floor.py <junit-xml-report>", file=sys.stderr)
        return 2

    report = Path(argv[0])
    if not report.exists():
        print(f"test_floor: no report at {report}.", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "pytest was asked for a junit report and did not write one. That is the",
            file=sys.stderr,
        )
        print(
            '"exited 0 having run nothing" shape, not a missing-file nuisance -- treat',
            file=sys.stderr,
        )
        print("the suite as UNRUN, not as passed.", file=sys.stderr)
        return 1

    try:
        counts = read_junit_counts(report)
    except ET.ParseError as exc:
        print(f"test_floor: {report} is not readable XML: {exc}", file=sys.stderr)
        return 1

    floor_tests = FLOOR.get("tests")
    max_skipped = FLOOR.get("max_skipped")
    surplus = slack(counts, FLOOR)

    floor_label = f"floor {floor_tests if floor_tests is not None else 'UNMEASURED'}"
    if surplus is not None:
        floor_label += f", SLACK +{surplus} -- raise it"
    print(
        f"test_floor: {counts['tests']} tests "
        f"({floor_label}), "
        f"{counts['failures']} failed, {counts['errors']} errored, "
        f"{counts['skipped']} skipped "
        f"(allowance {max_skipped if max_skipped is not None else 'UNMEASURED'})"
    )

    # Not stderr: in this file stderr means the build is failing, and slack is
    # not a failure. It goes out on stdout beside the headline it qualifies.
    if surplus is not None:
        advice = describe_slack(counts["tests"], surplus)
        print("")
        for line in advice:
            print(f"  - {line}")
        print("")
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print(github_warning("\n".join(advice)))

    problems = evaluate(counts, FLOOR)
    if not problems:
        return 0

    print("", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "If the drop is deliberate, lower FLOOR in scripts/test_floor.py in the same",
        file=sys.stderr,
    )
    print("commit, and say which tests went and why.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
