"""The ABL-692 runbook must grep the serving log through a stitched view (ABL-732).

`scripts/workstation/run-net-position-serving.ps1` writes the log with
`Start-Transcript`. A transcript records **native** (Python) output as the
console rendered it -- hard-wrapped at the buffer width, 120 columns for a
non-interactive `powershell.exe`. PowerShell's own `Write-Host` output is not
wrapped.

That splits the runbook's two confirmation checks:

  1. the witness line is `Write-Host`, 181 chars, intact -- check 1 was fine;
  2. the calibrated save line is native output, 133 chars, so the raw file holds
     `... (calibrated s_lo=1.0722` with no `s_hi` and no closing paren.

A regex for `calibrated s_lo=... s_hi=...` over the raw file therefore matched
**zero** lines while the system was working perfectly -- and zero is the exact
state the runbook defines as "the calibration has a second, separate problem".
The documented procedure manufactured a false incident on a correct system.

So the property held here is not "the doc mentions wrapping". It is that the
command printed in the runbook, **executed**, finds the full suffix on a log
that is wrapped the way the real one is. The behavioural tests below extract the
PowerShell block from the report and run it; the control test proves the fixture
is discriminating by running the pre-fix command against the same bytes.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = REPO_ROOT / "reports" / "abl_692_serving_checkout_pin.md"
CHRONOS = REPO_ROOT / "scripts" / "forecast_chronos2.py"

#: The transcript's wrap column. Not a tunable -- it is the default
#: `$Host.UI.RawUI.BufferSize.Width` of a non-interactive powershell.exe, and
#: the measured wrap point of every native line in the live log.
WRAP = 120

POWERSHELL = shutil.which("powershell") or shutil.which("powershell.exe")
needs_powershell = pytest.mark.skipif(
    POWERSHELL is None or sys.platform != "win32",
    reason="the runbook command is Windows PowerShell; nothing to execute here",
)


# ---------------------------------------------------------------------------
# The fixture: a log wrapped the way Start-Transcript wraps the real one.
# ---------------------------------------------------------------------------

def _wrap_native(line: str) -> list[str]:
    """Split `line` into console rows of at most WRAP chars, as a transcript does."""
    return [line[i:i + WRAP] for i in range(0, len(line), WRAP)] or [""]


def _record(stamp: str, msg: str) -> str:
    return f"2026-09-10 {stamp} - energy_forecast.chronos2 - INFO - {msg}"


#: 133 chars: this is the line the runbook's check 2 has to see whole.
CALIBRATED = _record(
    "12:09:49,816",
    "  Saved 216 quantile forecasts to DB (calibrated s_lo=1.0722 s_hi=1.0091)",
)

#: 181 chars, and emitted by Write-Host, so the transcript does not wrap it.
WITNESS = (
    "net-position serving commit: 0cd9ec2640e0c7f887c770f020fe6f39ae1787ae "
    "(origin/main, committed 2026-09-10T11:02:47+02:00) "
    "Merge pull request #116 from aguinier/ABL-669-derived-counts"
)


def _padded_to_exactly_wrap() -> str:
    """A genuine, unwrapped record that happens to be exactly WRAP chars long.

    The naive stitch rule ("join whenever the line is not shorter than WRAP")
    swallows the record that follows this one. Nothing in the live log needed
    the guard against that -- 0 of 189 exactly-120 lines were followed by a new
    record -- so without this fixture line the guard clause would be untested.
    """
    head = _record("08:01:06,638", "Built inference input for NL/net_position ")
    return head + "x" * (WRAP - len(head))


@pytest.fixture(scope="module")
def wrapped_log(tmp_path_factory) -> Path:
    rows: list[str] = [
        "**********************",
        "Windows PowerShell transcript start",
        # >WRAP and NOT continued: transcript header, PowerShell-origin.
        "Host Application: powershell.exe -WindowStyle Hidden -NoProfile "
        "-ExecutionPolicy Bypass -File C:\\Code\\able\\energy-forecast-serving"
        "\\scripts\\workstation\\run-net-position.ps1",
        WITNESS,
    ]
    exactly = _padded_to_exactly_wrap()
    assert len(exactly) == WRAP
    rows.append(exactly)
    rows.append(_record("08:01:06,700", "  Saved 216 quantile forecasts to DB"))
    rows.extend(_wrap_native(CALIBRATED))
    rows.append(_record("12:09:50,000", "Total: 432 forecast points generated"))

    path = tmp_path_factory.mktemp("abl732") / "net-position-forecast.log"
    path.write_text("\n".join(rows) + "\n", encoding="ascii")
    return path


def test_the_fixture_reproduces_the_defect(wrapped_log):
    """Vacuity guard. Every test below is a search; if the fixture were not
    actually wrapped, the fixed command would pass for the wrong reason.
    """
    lines = wrapped_log.read_text(encoding="ascii").splitlines()
    assert len(CALIBRATED) > WRAP, "the calibrated line no longer wraps at all"
    assert not any(re.search(r"calibrated s_lo=[\d.]+ s_hi=[\d.]+\)", ln) for ln in lines), (
        "the fixture is not wrapped: the suffix is already whole in the raw file"
    )
    assert any(ln.startswith(" s_hi=") for ln in lines), "no continuation row"
    assert WITNESS in lines, "the 181-char Write-Host witness line must stay intact"


# ---------------------------------------------------------------------------
# The command the runbook prints, executed.
# ---------------------------------------------------------------------------

def _confirm_block() -> str:
    """The powershell block under 'Confirm at the next 08:00 run'."""
    text = RUNBOOK.read_text(encoding="utf-8")
    assert len(text) > 2000, "the runbook is too short to be the real file"
    anchor = "Confirm at the next 08:00 run"
    assert anchor in text, f"the runbook no longer says {anchor!r}"
    tail = text[text.index(anchor):]
    m = re.search(r"```powershell\n(.*?)```", tail, re.DOTALL)
    assert m, "no powershell block follows the confirm heading"
    return m.group(1)


def test_the_confirm_block_is_ascii():
    """PowerShell 5.1 reads a BOM-less .ps1 in the ANSI codepage. A stray
    en-dash pasted into this block turns into a parse error at 08:00, on the one
    command an operator runs when they already suspect something is wrong.
    """
    block = _confirm_block()
    bad = sorted({c for c in block if ord(c) > 127})
    assert not bad, f"non-ASCII in the confirm block: {bad}"


def test_the_runbook_does_not_grep_the_raw_file(wrapped_log):
    """The pre-fix form, pinned as forbidden: `Select-String <pattern> <path>`
    reads the file itself and therefore reads wrapped rows.
    """
    block = _confirm_block()
    naive = re.findall(r"Select-String\s+\"[^\"]+\"\s+\S*net-position-forecast\.log", block)
    assert not naive, (
        f"the confirm block greps the raw log file: {naive}. That is the ABL-732 "
        "defect -- a wrapped line cannot match a pattern that spans the wrap."
    )
    assert "Get-Content" in block, "nothing reads the log into a stitchable list"


def test_the_stitch_rule_keeps_both_of_its_clauses():
    """Both clauses are load-bearing and each has a distinct failure mode.

    `-eq 120` rather than `>=`: the 53 long lines in the live log are Write-Host
    output and are not continued, so a `>=` rule glues the witness line -- one of
    the two things being read -- onto whatever follows it.

    The new-record clause: a genuine 120-char record is not a wrapped one. The
    transcript is opened -Append, so this file keeps its wrapped history even if
    the launcher's buffer width is ever widened.
    """
    block = _confirm_block()
    assert re.search(r"\.Length\s*-eq\s*120", block), (
        "the stitch must trigger on exactly 120 chars; -ge/-not -lt also joins "
        "the unwrapped Write-Host lines"
    )
    assert re.search(r"-notmatch", block), "no guard against joining two real records"
    assert re.search(r"\\d\{4\}-\\d\{2\}-\\d\{2\}", block), (
        "the guard does not anchor on a log-record timestamp"
    )


def _run_block(block: str, log: Path, tmp_path: Path) -> str:
    script = tmp_path / "confirm.ps1"
    rebound = re.sub(
        r'^\$log\s*=\s*".*"$',
        '$log = "{}"'.format(str(log).replace("\\", "\\\\")),
        block,
        count=1,
        flags=re.MULTILINE,
    )
    assert str(log.name) in rebound, "the $log assignment was not rebound"
    script.write_text(rebound, encoding="ascii")
    proc = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"the documented command failed:\n{proc.stderr}"
    return proc.stdout


@needs_powershell
def test_the_documented_command_finds_the_whole_calibrated_suffix(wrapped_log, tmp_path):
    """The check the ABL-692 unblock actually turns on."""
    out = _run_block(_confirm_block(), wrapped_log, tmp_path)
    assert "band   : calibrated s_lo=1.0722 s_hi=1.0091" in out, (
        "check 2 still cannot see the suffix it tests for; output was:\n" + out
    )
    assert "UNCALIBRATED" not in out, (
        "a correct fixture was reported as the defect state:\n" + out
    )


@needs_powershell
def test_the_documented_command_still_reports_the_witness_sha(wrapped_log, tmp_path):
    """Check 1 was never broken, and the fix must not break it: a stitch rule
    that joined the 181-char Write-Host line would corrupt the SHA read.
    """
    out = _run_block(_confirm_block(), wrapped_log, tmp_path)
    assert "commit : 0cd9ec2640e0c7f887c770f020fe6f39ae1787ae" in out, (
        "the witness SHA did not come back whole:\n" + out
    )
    assert "MISSING" not in out


@needs_powershell
def test_the_commands_own_output_cannot_wrap(wrapped_log, tmp_path):
    """The same defect, one level up -- and the reason this file reports fields
    rather than matched lines.

    The console wraps *its* output at 120 columns too. The first draft of the
    fix echoed `$wit.Line`, so the corrected command handed back the 181-char
    witness line as two physical rows: piped into anything, the SHA read comes
    back truncated at `(origin/main, committed`. Reading a whole line out of
    this log is not something a caller can do; extracting a field is.
    """
    out = _run_block(_confirm_block(), wrapped_log, tmp_path)
    emitted = [ln for ln in out.splitlines() if ln.strip()]
    assert emitted, "the command printed nothing"
    too_long = [ln for ln in emitted if len(ln) >= WRAP]
    assert not too_long, (
        f"output lines at or past the {WRAP}-column wrap: {too_long}"
    )
    assert WITNESS not in out, (
        "the command echoes the 181-char witness line, which the console then "
        "wraps -- report the extracted SHA instead"
    )


@needs_powershell
def test_a_genuine_120_char_record_is_not_swallowed(wrapped_log, tmp_path):
    """The guard clause, exercised. Without it the exactly-120 record joins the
    plain `Saved 216 quantile forecasts to DB` line that follows it, and check 2
    then reports the uncalibrated line as its last match.
    """
    out = _run_block(_confirm_block(), wrapped_log, tmp_path)
    # Joining them makes the plain (uncalibrated) save line the last match for
    # `quantile forecasts to DB`, so check 2 reports the defect state.
    assert "UNCALIBRATED" not in out, (
        "two separate records were stitched together:\n" + out
    )
    assert "band   : calibrated" in out


@needs_powershell
def test_the_pre_fix_command_fails_on_the_same_bytes(wrapped_log, tmp_path):
    """Control. Proves the fixture discriminates rather than the new command
    being trivially satisfiable -- the reason this test file exists is that the
    old command was never run against a wrapped log.
    """
    script = tmp_path / "naive.ps1"
    script.write_text(
        'Select-String "quantile forecasts to DB" "{}" | Select-Object -Last 1\n'.format(
            str(wrapped_log).replace("\\", "\\\\")
        ),
        encoding="ascii",
    )
    proc = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0
    assert "calibrated s_lo=1.0722" in proc.stdout, "the naive form matched nothing at all"
    assert "s_hi" not in proc.stdout, (
        "the pre-fix command sees the suffix, so the fixture does not reproduce "
        "ABL-732 and the tests above prove nothing"
    )


# ---------------------------------------------------------------------------
# The doc/code coupling that made the check silently wrong in the first place.
# ---------------------------------------------------------------------------

def test_the_emitted_suffix_is_the_one_the_runbook_greps_for():
    """The runbook greps a string that `forecast_chronos2.py` formats. Nothing
    connected the two, which is how the check could be wrong for a week without
    anything going red. Renaming the message must land here.
    """
    src = CHRONOS.read_text(encoding="utf-8")
    assert re.search(r"calibrated s_lo=\{[^}]*\} s_hi=\{[^}]*\}", src), (
        "forecast_chronos2.py no longer emits `calibrated s_lo=... s_hi=...`; "
        "the ABL-692 runbook's check 2 greps for exactly that"
    )
    assert "quantile forecasts to DB" in src, (
        "the message the runbook anchors on was renamed"
    )
    runbook = RUNBOOK.read_text(encoding="utf-8")
    assert "quantile forecasts to DB" in runbook
