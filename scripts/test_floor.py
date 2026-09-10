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

Standard library only, on purpose: it has to run in a job whose `pip install`
step is the thing under suspicion.
"""

from __future__ import annotations

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
# The ABL-739 release train then merged two PRs that added NO test file and did
# not touch this number, and the count still moved 1829 -> 1833. Two new
# `scripts/*.py` entry points (`abl648_weather_retention_model.py`,
# `abl692_calibration_witness.py`) are each picked up by two glob-parametrized
# families -- `test_help_text_encoding.py::test_help_text_is_ascii` and
# `test_script_imports.py::test_script_import_preamble` -- so a script is worth
# +2 on its own. CI measured 1833 against a floor of 1829 on main at 999cc0d and
# stayed green, because slack is invisible here: this gate only fires on a DROP.
# Adding an entry point is a floor raise even when you add no tests.
#
# ABL-733 adds 12 (tests/test_abl733_transcript_buffer_width.py) -> 1845, which
# is 1833 + 12. Confirmed from BOTH sides: the ubuntu-latest runner reported
# `1845 tests (floor ...), 0 failed, 0 errored, 6 skipped` on the PR's merge
# commit, and a workstation collect on the same merge reports 1845 too, so the
# two agree exactly and neither is a platform artifact.
#
# The arithmetic matters more than it looks. ABL-733 branched at 1828; ABL-735
# (+1) and the train (+4) both landed on main while it was out. Those raises are
# ADDITIVE -- resolve a conflict on this file by SUMMING the increments, never
# by taking the higher of the two conflicting floors. Taking the higher side
# here would have given 1841, four BELOW what the merged tree runs, and it would
# have passed: a floor set too low is never red, just silently blind to exactly
# the tests it can no longer see.
#
# `max_skipped` goes 4 -> 6 for two of those twelve:
# `test_the_control_reproduces_the_120_column_wrap` and
# `test_the_shipped_widening_stops_native_output_wrapping`. Both spawn a hidden
# Windows console and read the transcript it produced, because the property
# under test IS the width of the console screen buffer Start-Transcript records.
# `pwsh` on ubuntu-latest has no console screen buffer, so unlike ABL-732's four
# these genuinely cannot run there; they run on the workstation, which is where
# the launcher they cover runs in production. The other ten -- eight structural
# and two that execute a block through plain redirected `pwsh` -- run on CI.
# ABL-735's one and the train's four are not gated, so they do not move the
# allowance. Unlike `tests`, `max_skipped` genuinely IS a max on merge -- it is a
# ceiling, so take the higher side, and only raise it for increments whose tests
# are actually gated. CI measured exactly 6, at the ceiling rather than over it.
#
# `None` means "not yet measured": the gate then reports what it saw and fails,
# so a floor cannot be quietly left unset.
FLOOR: dict[str, int | None] = {
    "tests": 1845,
    "max_skipped": 6,
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
    print(
        f"test_floor: {counts['tests']} tests "
        f"(floor {floor_tests if floor_tests is not None else 'UNMEASURED'}), "
        f"{counts['failures']} failed, {counts['errors']} errored, "
        f"{counts['skipped']} skipped "
        f"(allowance {max_skipped if max_skipped is not None else 'UNMEASURED'})"
    )

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
