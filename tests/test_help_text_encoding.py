"""Every entry point's `--help` text stays ASCII (ABL-364).

`scripts/train.py --help` exited **1** whenever stdout was a pipe or a file:

    $ python scripts/train.py --help > /dev/null
    UnicodeEncodeError: 'charmap' codec can't encode character '\\u2192'

The mechanism is that CPython picks stdout's encoding from what stdout *is*.
Attached to a Windows console it writes through `WriteConsoleW` and any
character survives, so `--help` looks fine interactively. Redirected, the
encoding falls back to the locale codepage -- cp1252 on this box -- and
`argparse._print_message` raises on the first character that codepage cannot
represent. `print_help()` is called from the `--help` action itself, so the
process dies with a traceback instead of printing usage. Every harness, CI step
and agent captures stdout, so they all saw the crash and only an interactive
operator saw the help.

This is the ABL-340 failure class again (a documented CLI that cannot print its
own usage), and it had already reached
`tests/test_train_sidecar_guard.py::test_help_still_exits_zero`, which runs the
real CLI through `subprocess` and was red on `origin/main` for this reason.

**The rule this file holds: help text is ASCII.** The alternative -- have every
entry point force UTF-8 on stdout -- was rejected: it is a runtime change in 38
scripts, each one forgettable in the 39th, and it would silently re-encode the
log files the scheduled `scripts/workstation/*.ps1` jobs capture. Usage text is
not a report; it has nothing to say that `->` and `--` cannot.

Report *bodies* are the other half of that split and keep their typography. They
are printed from one known place, so they re-encode the stream there instead
(`evaluate_net_position.py:125-132`, `compare_challenger.py:127-133`) -- which
`--help` cannot do, since argparse prints before any line of `main()` runs.

The set is every `scripts/*.py`, every repo-root runner, and every
`config.MODEL_RUNNERS` script. That last group matters because
`test_script_imports.py::test_model_runner_launches` (ABL-354) starts each
runner with `--help` through a pipe to prove it can start, and two of them live
under `src/`, outside the `scripts/` glob.

Static, not behavioural, for the set: `--help` on an arbitrary script means
importing and executing it to module scope, and several of these open databases
or write reports. So the sweep reads the source, and one script whose `--help`
is known safe to run is checked end to end below, under a forced non-UTF-8
stdout so the assertion does not depend on the codepage of the box it runs on.

Known blind spot: a help string built by a call or a `+` concatenation is not
read. f-strings are read as far as their literal parts (3 sites use one, all
interpolating identifiers). A dynamic non-ASCII help string would reach only the
behavioural test.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402

# The same entry-point set as tests/test_script_imports.py: scripts/ and the
# repo-root runners, plus the config.MODEL_RUNNERS scripts. Two of those live
# inside src/ and are launched as `-m src.<module>` (ABL-354), so the scripts/
# glob does not reach them -- see test_model_runners_are_covered.
_ENTRY_POINTS = (
    sorted((REPO_ROOT / "scripts").glob("*.py"))
    + sorted(REPO_ROOT.glob("*.py"))
    + [REPO_ROOT / r["script"] for r in config.MODEL_RUNNERS if r.get("script")]
)
SCRIPTS = sorted({p.resolve() for p in _ENTRY_POINTS if p.is_file()})
#: Repo-relative paths, not bare names: two entry points could share a file name
#: across scripts/ and src/, and a name lookup would then sweep the wrong file.
SCRIPT_IDS = [p.relative_to(REPO_ROOT).as_posix() for p in SCRIPTS]

#: Calls that build the parser. Everything user-visible in `--help` output is a
#: text keyword on one of these.
PARSER_CALLS = {
    "ArgumentParser",
    "add_argument",
    "add_argument_group",
    "add_mutually_exclusive_group",
    "add_parser",
    "add_subparsers",
}

#: Keywords argparse renders into the help output.
TEXT_KEYWORDS = {"help", "description", "epilog", "metavar", "title", "prog", "usage"}


def _literal_strings(node):
    """The string literals inside a help-text expression.

    Handles the two forms that carry prose in this repo: a plain constant, and
    an f-string (whose literal segments are prose even though the interpolated
    parts are not knowable here).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return [v.value for v in node.values
                if isinstance(v, ast.Constant) and isinstance(v.value, str)]
    return []


def help_text_sites(path):
    """[(lineno, what, text)] for everything `path --help` would print.

    `description=__doc__` is the form that matters most here: 18 of the 38 entry
    points pass their module docstring straight into the parser, so a docstring
    is help text and an em dash in one is a crash on a cp850 console -- which is
    how this sweep found nine offenders beyond the one on the issue.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree, clean=False)

    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in PARSER_CALLS:
            continue
        for keyword in node.keywords:
            if keyword.arg not in TEXT_KEYWORDS:
                continue
            value = keyword.value
            if isinstance(value, ast.Name) and value.id == "__doc__":
                if docstring is not None:
                    sites.append((1, f"{keyword.arg}=__doc__ (module docstring)", docstring))
                continue
            for text in _literal_strings(value):
                sites.append((value.lineno, f"{keyword.arg}=", text))
    return sites


def _offenders(text):
    return sorted({c for c in text if ord(c) > 127})


def test_entry_points_are_discovered():
    # Guards the parametrisation below against silently covering nothing.
    assert len(SCRIPTS) >= 30, f"only found {len(SCRIPTS)} entry points"


def test_model_runners_are_covered():
    """The `config.MODEL_RUNNERS` scripts are held to the rule too (ABL-354).

    `tests/test_script_imports.py::test_model_runner_launches` starts each of
    these with `--help` through `subprocess(..., capture_output=True)` -- a
    pipe, which is exactly the stream this file's rule exists for. Two of them
    (`src/chronos_forecaster.py`, `src/tso_correction_forecaster.py`) sit
    outside the `scripts/` glob, so without this they would be *launched* that
    way and swept by nothing.

    The stakes there are higher than a broken `--help`: `forecast_daily` runs a
    runner as a subprocess and records one that cannot start as a failed
    *result*, so the job still reports `[DONE]` with those forecasts missing.
    """
    runners = [r["script"] for r in config.MODEL_RUNNERS if r.get("script")]
    assert runners, "config.MODEL_RUNNERS declares no script entry points"
    missing = [s for s in runners if (REPO_ROOT / s).resolve() not in set(SCRIPTS)]
    assert not missing, (
        "MODEL_RUNNERS entry points not swept for non-ASCII help text:\n  "
        + "\n  ".join(missing)
        + "\n(If one is simply gone, test_model_runner_script_exists says so.)"
    )


def test_a_parser_is_actually_found():
    """The sweep is only worth anything if it reads real parsers. Pins that the
    call/keyword names above still match how this repo builds its CLIs -- rename
    `add_argument` upstream and every test below would pass vacuously."""
    with_parsers = [s for s in SCRIPT_IDS if help_text_sites(REPO_ROOT / s)]
    assert len(with_parsers) >= 20, f"only {len(with_parsers)} entry points parsed a parser"
    assert "scripts/train.py" in with_parsers


@pytest.mark.parametrize("script", SCRIPT_IDS)
def test_help_text_is_ascii(script):
    path = REPO_ROOT / script
    bad = [(lineno, what, _offenders(text))
           for lineno, what, text in help_text_sites(path) if _offenders(text)]
    assert not bad, (
        f"{script} renders non-ASCII in --help:\n  "
        + "\n  ".join(
            f"line {lineno}: {what} contains "
            + ", ".join(f"{c!r} (U+{ord(c):04X})" for c in chars)
            for lineno, what, chars in bad
        )
        + "\nRedirected stdout encodes with the locale codepage, and argparse "
          "raises UnicodeEncodeError there rather than printing usage (ABL-364). "
          "Write it in ASCII: '->' for an arrow, '--' for an em dash. Report "
          "bodies are the exception and re-encode the stream at the print site."
    )


def test_train_help_survives_a_non_utf8_stdout():
    """The behavioural half, on the script the issue was filed against.

    `PYTHONIOENCODING=ascii` is what makes this deterministic. Without it the
    test measures the box: the em dashes this sweep removed encode fine in
    cp1252 and would only have failed on cp850 or in a container, and a
    UTF-8-locale CI would have gone green on the arrow itself.

    `scripts/train.py` is the one run end to end because `--help` is executed
    for real here: argparse exits inside `parse_args()`, before the run touches
    a database, and `tests/test_train_sidecar_guard.py` already imports this
    module at collection time.
    """
    env = dict(os.environ, PYTHONIOENCODING="ascii")
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--help"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=600,
    )

    assert proc.returncode == 0, (
        "train.py --help could not print its own usage to a pipe:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "--cascade" in proc.stdout
