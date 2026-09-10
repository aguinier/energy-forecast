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
# The allowance is 1, which is lower than expected and is itself a result: tests
# gated on the replica database and on `models/` artifacts were assumed to be a
# sizeable set, and the runner has neither. It ran 1766 and skipped one. Those
# tests do not skip on a bare runner — they assert against stored documents. The
# pytest step passes `-rs`, so the log always names what skipped and why; read
# that before changing this number.
#
# `None` means "not yet measured": the gate then reports what it saw and fails,
# so a floor cannot be quietly left unset.
FLOOR: dict[str, int | None] = {
    "tests": 1766,
    "max_skipped": 1,
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
