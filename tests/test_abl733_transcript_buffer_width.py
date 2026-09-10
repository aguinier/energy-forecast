"""The serving launcher must record its transcript wide enough not to wrap (ABL-733).

`Start-Transcript` records the console **screen buffer**, so native (Python)
output lands in `net-position-forecast.log` hard-wrapped at
`$Host.UI.RawUI.BufferSize.Width` -- 120 for the console the scheduled task's
`wscript.exe` wrapper hands `powershell.exe`. PowerShell's own `Write-Host` is
not wrapped, which is why the 181-char witness line survives whole and the
133-char calibrated save line does not (ABL-732).

ABL-732 fixed the *reader* (the runbook stitches continuations). This file holds
the *writer*: `run-net-position-serving.ps1` widens the buffer before anything
native runs, so new output stops wrapping at all.

Truncation was never confined to the one line that made it visible. Re-measured
on the live log 2026-09-10, over lines opening with a Python log timestamp:
**171 of 587 records truncated, across 58 distinct messages** -- including
36 `data stops NNh short of the nominal cutoff` and every
`Failed to forecast XX/net_position: <reason>`, cut at exactly the reason. That
is why the launcher widens rather than shortening the one save line, which is
the cheaper alternative the issue offered.

Three properties, in the order they matter:

  1. **It can never cost a forecast.** A host with no console throws on a RawUI
     set, and ABL-692 already settled that a missing vintage is worse than a
     wrapped one. The set is best-effort and its outcome is logged.
  2. **It actually stops the wrap** under the task's real console conditions --
     asserted by executing the shipped block in a hidden console, against a
     control that reproduces the 120-column wrap on the same bytes.
  3. **It does not retire the stitched read.** The transcript is opened
     `-Append`, so the log now holds wrapped history *and* unwrapped new
     records. The runbook's command is run over exactly that mixture here.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "workstation" / "run-net-position-serving.ps1"
RUNBOOK = REPO_ROOT / "reports" / "abl_692_serving_checkout_pin.md"

#: The wrap column being removed: the default BufferSize.Width of the console a
#: non-interactive powershell.exe is given. Not a tunable -- a measurement.
WRAP = 120

#: Longest record the live log has ever held, measured 2026-09-10 over a
#: stitched view: 221 chars Python-origin, 383 chars including `Write-Host`.
#: A width at or below this would leave the log still wrapping, just less often.
LONGEST_RECORD_SEEN = 383

#: The line ABL-732 could not grep. 133 chars, emitted by Python `logging`,
#: i.e. on stderr -- the stream the real message uses.
CALIBRATED = (
    "2026-09-10 12:09:49,816 - energy_forecast.chronos2 - INFO -   "
    "Saved 216 quantile forecasts to DB (calibrated s_lo=1.0722 s_hi=1.0091)"
)

#: 181 chars and `Write-Host`-origin, so it was never wrapped and must not start
#: being mangled by a reader that assumes everything long is a continuation.
WITNESS = (
    "net-position serving commit: 0cd9ec2640e0c7f887c770f020fe6f39ae1787ae "
    "(origin/main, committed 2026-09-10T11:02:47+02:00) "
    "Merge pull request #116 from aguinier/ABL-669-derived-counts"
)

POWERSHELL = (
    shutil.which("pwsh")
    or shutil.which("powershell")
    or shutil.which("powershell.exe")
)
needs_powershell = pytest.mark.skipif(
    POWERSHELL is None,
    reason="no pwsh/powershell on PATH; the shipped block cannot be executed here",
)

#: The console tests need a Windows console screen buffer, which is the thing
#: being resized -- `pwsh` on the ubuntu-latest CI runner has none, and its
#: transcript does not capture native output through a screen buffer at all. So
#: these two skip on CI and run on the workstation, where production runs.
#: scripts/test_floor.py's skip allowance is what says so out loud.
WINDOWS_POWERSHELL = shutil.which("powershell.exe") if sys.platform == "win32" else None
needs_windows_console = pytest.mark.skipif(
    WINDOWS_POWERSHELL is None,
    reason=(
        "needs a Windows console: the property under test is the width of the "
        "screen buffer Start-Transcript records"
    ),
)


def _read(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    # Vacuity guard: most assertions below are searches, and a search over an
    # empty or truncated file passes by not matching.
    assert len(text) > 500, f"{path.name} is too short to be the real script"
    return text


@pytest.fixture(scope="module")
def launcher() -> str:
    return _read(LAUNCHER)


def _param_default(text: str, name: str) -> str:
    m = re.search(r"\[\w+\]\s*\$" + re.escape(name) + r"\s*=\s*\"?([^\",\r\n)]+)", text)
    assert m, f"no $Name param with a literal default for {name}"
    return m.group(1).strip()


def _widen_block(text: str) -> str:
    """The shipped widening, lifted verbatim from the launcher.

    Extracted rather than paraphrased so that the executed tests below run the
    code that ships. ABL-732's lesson was that a block which is only ever read
    is a block that has not been tested.
    """
    start = re.search(r"^\s*\$widthResult\s*=\s*\"unchanged\"\s*$", text, re.MULTILINE)
    assert start, "no `$widthResult = \"unchanged\"` line to anchor the block on"
    end = re.search(r'^\s*Write-Host "\$WidthPrefix \$widthResult"\s*$', text, re.MULTILINE)
    assert end, "the block no longer ends by reporting what it did"
    assert end.start() > start.start(), "the report precedes the widening"
    block = text[start.start():end.end()]
    assert "BufferSize" in block, "the extracted block does not touch BufferSize"
    return block


# ---------------------------------------------------------------------------
# 1. It can never cost a forecast.
# ---------------------------------------------------------------------------

def test_the_widening_is_wrapped_in_its_own_try_catch(launcher):
    """The whole reason this is safe to put in a production launcher.

    Under `wscript.exe //B` there may be no console at all, and a `set` on
    `BufferSize` then throws. `$ErrorActionPreference = "Stop"` is set at the
    top of this file, so an uncaught throw here aborts before `& $job` -- the
    forecast is lost to a logging nicety. ABL-692 settled that trade the other
    way round: a missing vintage is worse than a wrapped one.
    """
    block = _widen_block(launcher)
    assert re.search(r"\btry\s*\{", block), "the RawUI set is not guarded by try"
    assert re.search(r"\bcatch\s*\{", block), "no catch: a throw here loses the forecast"
    catch = block[block.index("catch"):]
    assert "throw" not in catch, "the catch rethrows, which defeats the guard"
    assert "Write-Error" not in catch, "the catch raises an error record"


def test_a_failure_to_widen_is_reported_not_hidden(launcher):
    """A silent no-op would leave the log wrapped with nothing saying why -- the
    same class of invisibility ABL-692 exists to remove.
    """
    block = _widen_block(launcher)
    catch = block[block.index("catch"):]
    assert "$_.Exception.Message" in catch, "the catch discards the reason"
    assert "$widthResult" in catch, "the catch does not feed the reported outcome"
    assert '$WidthPrefix = "net-position serving transcript width:"' in launcher, (
        "the reported prefix must stay a named constant: this test and any "
        "operator grep of the log both anchor on it"
    )


def test_the_widening_never_narrows(launcher):
    """`BufferSize.Width` below `WindowSize.Width` throws. A host that already
    reports a wide buffer must be left alone, not shrunk to the constant.
    """
    block = _widen_block(launcher)
    assert re.search(r"-lt\s+\$BufferWidth", block), (
        "no guard restricting the set to a widening; an unconditional assignment "
        "turns an already-wide host into a caught exception for no gain"
    )


# ---------------------------------------------------------------------------
# 2. It is wide enough, early enough, and it says what it did.
# ---------------------------------------------------------------------------

def test_the_width_clears_the_widest_record_the_log_has_held(launcher):
    width = int(_param_default(launcher, "BufferWidth"))
    assert width > LONGEST_RECORD_SEEN, (
        f"a {width}-column buffer still wraps the {LONGEST_RECORD_SEEN}-char "
        "records this log already contains"
    )


def test_the_widening_runs_before_the_job(launcher):
    """Output emitted before the resize is wrapped. The job is where every
    native line comes from, so the resize has to precede it.
    """
    block_at = launcher.index(_widen_block(launcher))
    job_at = launcher.index("& $job")
    assert block_at < job_at, "the buffer is widened after the job has already run"


def test_the_outcome_lands_inside_the_transcript(launcher):
    """Deliberately after `Start-Transcript`, not before it.

    Both placements were measured to unwrap native output identically
    (reports/abl_733_transcript_buffer_width.md), so the tie is broken by which
    one leaves evidence: inside the transcript, the log records the width it was
    written at, and a `catch` on a console-less host is readable rather than
    lost to a stream nobody keeps.
    """
    transcript_at = launcher.index("Start-Transcript")
    block_at = launcher.index(_widen_block(launcher))
    assert transcript_at < block_at, (
        "the widening precedes Start-Transcript, so neither its success nor its "
        "failure is recorded anywhere"
    )


def test_the_launcher_is_ascii():
    """PowerShell 5.1 reads a BOM-less `.ps1` in the ANSI codepage. The reported
    outcome contains `->`; an en-dash pasted in its place is a parse error in
    the launcher that runs production, discovered at 08:00.
    """
    raw = LAUNCHER.read_bytes()
    bad = sorted({b for b in raw if b > 127})
    assert not bad, f"non-ASCII bytes in the launcher: {bad}"


# ---------------------------------------------------------------------------
# 3. Executed: the block cannot abort its caller.
# ---------------------------------------------------------------------------

_SENTINEL = "REACHED-THE-LINE-AFTER"


def _harness(tmp_path: Path, launcher_text: str, *, widen: bool, log: Path) -> Path:
    """A minimal launcher: transcript, the shipped block (or not), then output.

    The control differs from the treatment by the presence of that block and
    nothing else -- same harness, same fixture, same bytes written.
    """
    payload = tmp_path / "payload.py"
    payload.write_text(
        "import sys\n"
        f"sys.stderr.write({CALIBRATED!r} + '\\n')\n"
        "sys.stderr.flush()\n"
        f"sys.stdout.write('STDOUT ' + 'y' * 160 + ' END')\n"
        "sys.stdout.write('\\n')\n",
        encoding="ascii",
    )
    body = _widen_block(launcher_text) if widen else "    # control: no widening"
    script = tmp_path / ("widen.ps1" if widen else "control.ps1")
    script.write_text(
        "param([string] $Log, [string] $Python, [string] $Payload)\n"
        '$ErrorActionPreference = "Stop"\n'
        f"$BufferWidth = {_param_default(launcher_text, 'BufferWidth')}\n"
        '$WidthPrefix = "net-position serving transcript width:"\n'
        "Start-Transcript -Path $Log -Append | Out-Null\n"
        "try {\n"
        f"{body}\n"
        "    & $Python $Payload\n"
        f'    Write-Host "{_SENTINEL} exit=$LASTEXITCODE"\n'
        "}\n"
        "finally { Stop-Transcript | Out-Null }\n",
        encoding="ascii",
    )
    assert not log.exists(), "the fixture log must start empty"
    return script


@needs_powershell
def test_the_widening_cannot_abort_its_caller(tmp_path, launcher):
    """The property that makes this safe to ship, executed on a host with no
    console at all -- stdio is redirected here, which is precisely the case
    where a RawUI set can throw.

    Passing means the script reached the line *after* the block and exited 0,
    whether the widening succeeded, was a no-op, or was caught.
    """
    log = tmp_path / "net-position-forecast.log"
    script = _harness(tmp_path, launcher, widen=True, log=log)
    proc = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
         "-Log", str(log), "-Python", sys.executable, "-Payload", str(tmp_path / "payload.py")],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, (
        "the shipped widening aborted its caller; in production that is a lost "
        f"forecast:\n{proc.stdout}\n{proc.stderr}"
    )
    combined = proc.stdout + log.read_text(encoding="utf-8", errors="replace")
    assert "net-position serving transcript width:" in combined, (
        "nothing reported the outcome:\n" + combined
    )
    assert _SENTINEL in combined, "execution never reached the line after the block"


# ---------------------------------------------------------------------------
# 4. Executed: it actually stops the wrap, under the task's own conditions.
# ---------------------------------------------------------------------------

def _run_in_hidden_console(script: Path, log: Path, payload: Path) -> None:
    """Spawn `powershell.exe` the way the scheduled task does.

    `wscript.exe //B` is a GUI-subsystem host with no console, so `Run(cmd, 0,
    True)` gives powershell.exe a *new*, hidden console -- 120x3000 on this box.
    `STARTF_USESHOWWINDOW`/`SW_HIDE` plus `CREATE_NEW_CONSOLE` is that same
    CreateProcess call, without depending on a `.vbs` outside the repo.

    **Nothing here may redirect the child's stdio.** A transcript captures the
    console screen buffer; redirected handles bypass the console entirely, so
    native output then never reaches the log and the whole probe reads as "no
    wrapping" for the wrong reason. That is the trap that made this effect look
    unreproducible from an agent session. `subprocess` sets
    `STARTF_USESTDHANDLES` if *any* of stdin/stdout/stderr is passed -- so none
    is, not even `DEVNULL`.
    """
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0  # SW_HIDE
    rc = subprocess.call(
        [WINDOWS_POWERSHELL, "-WindowStyle", "Hidden", "-NoProfile",
         "-ExecutionPolicy", "Bypass", "-File", str(script),
         "-Log", str(log), "-Python", sys.executable, "-Payload", str(payload)],
        startupinfo=startupinfo,
        creationflags=0x00000010,  # CREATE_NEW_CONSOLE
        timeout=180,
    )
    assert rc == 0, f"the hidden-console harness exited {rc}"
    assert log.exists(), (
        "no transcript was written; the harness did not get a console and "
        "proves nothing either way"
    )


def _log_lines(log: Path) -> list[str]:
    return log.read_text(encoding="utf-8", errors="replace").splitlines()


@needs_windows_console
def test_the_control_reproduces_the_120_column_wrap(tmp_path, launcher):
    """Control, and the test that gives the next one its meaning.

    Without it, "the calibrated line is whole" can pass on a harness that never
    wrapped anything -- which is exactly how the ABL-692 runbook's check came to
    be wrong, and how the first attempt to measure this effect went inconclusive.
    """
    log = tmp_path / "control.log"
    script = _harness(tmp_path, launcher, widen=False, log=log)
    _run_in_hidden_console(script, log, tmp_path / "payload.py")

    lines = _log_lines(log)
    assert any(_SENTINEL in ln for ln in lines), "the harness did not run to the end"
    assert any(ln.startswith("2026-09-10 12:09:49,816") for ln in lines), (
        "no native output reached the transcript at all, so nothing about "
        "wrapping was measured:\n" + "\n".join(lines)
    )
    assert CALIBRATED not in lines, (
        "the control did not wrap: this box does not reproduce the cron's "
        "console, so the widening test below proves nothing"
    )
    assert any(len(ln) == WRAP and ln.startswith("2026-09-10") for ln in lines), (
        f"no native line was cut at exactly {WRAP} columns"
    )
    assert not any(
        re.search(r"calibrated s_lo=[\d.]+ s_hi=[\d.]+\)", ln) for ln in lines
    ), "the suffix survived in the control"


@needs_windows_console
def test_the_shipped_widening_stops_native_output_wrapping(tmp_path, launcher):
    """The change, executed: same harness, same bytes, the shipped block added.

    A `Select-String` for the full calibrated suffix -- the plain command that
    matched zero lines on a healthy system before ABL-732 -- now matches.
    """
    log = tmp_path / "widened.log"
    script = _harness(tmp_path, launcher, widen=True, log=log)
    _run_in_hidden_console(script, log, tmp_path / "payload.py")

    lines = _log_lines(log)
    assert any(_SENTINEL in ln for ln in lines), "the harness did not run to the end"
    assert CALIBRATED in lines, (
        "the 133-char native line is still not whole in the transcript:\n"
        + "\n".join(ln for ln in lines if "quantile" in ln or "s_hi" in ln)
    )
    assert not any(ln.startswith(" s_hi=") for ln in lines), "a continuation row remains"
    long_native = [ln for ln in lines if ln.startswith("STDOUT ") and ln.endswith(" END")]
    assert long_native, "the 174-char stdout line did not survive whole either"
    assert any("width: 120 -> " in ln for ln in lines), (
        "the log does not record that the buffer was widened from 120:\n"
        + "\n".join(ln for ln in lines if "transcript width" in ln)
    )


# ---------------------------------------------------------------------------
# 5. The stitched read is not retired, and must keep working on the mixture.
# ---------------------------------------------------------------------------

def _confirm_block() -> str:
    """The runbook's confirmation command.

    Duplicated from tests/test_abl732_transcript_wrap.py rather than imported:
    `tests/` is not a package, and the sharing convention in this repo is a
    non-`test_` helper module. The two files assert different things about the
    same block -- ABL-732 that it reads a wrapped log, this one that widening
    the writer does not break it.
    """
    text = RUNBOOK.read_text(encoding="utf-8")
    anchor = "Confirm at the next 08:00 run"
    assert anchor in text, f"the runbook no longer says {anchor!r}"
    m = re.search(r"```powershell\n(.*?)```", text[text.index(anchor):], re.DOTALL)
    assert m, "no powershell block follows the confirm heading"
    return m.group(1)


@pytest.fixture(scope="module")
def mixed_log(tmp_path_factory) -> Path:
    """What the file looks like after this change lands.

    `Start-Transcript -Append`: the wrapped history stays, and unwrapped records
    accumulate after it. A reader has to be correct on both, which is why the
    stitch rule keys on `-eq 120` plus "the next line does not open a record"
    rather than on "long".
    """
    old = [
        WITNESS,
        CALIBRATED[:WRAP],
        CALIBRATED[WRAP:],
        "2026-09-10 08:01:07,001 - energy_forecast.chronos2 - INFO - Total: 432 points",
    ]
    new = [
        "net-position serving transcript width: 120 -> 512",
        WITNESS.replace("0cd9ec26", "abcd1234"),
        CALIBRATED.replace("12:09:49,816", "08:00:51,004"),
        "2026-09-11 08:00:52,100 - energy_forecast.chronos2 - INFO - Total: 432 points",
    ]
    path = tmp_path_factory.mktemp("abl733") / "net-position-forecast.log"
    path.write_text("\n".join(old + new) + "\n", encoding="ascii")
    return path


def test_the_mixed_log_holds_both_shapes(mixed_log):
    """Vacuity guard on the fixture: one wrapped record and one whole one."""
    lines = mixed_log.read_text(encoding="ascii").splitlines()
    assert any(ln.startswith(" s_hi=") for ln in lines), "no wrapped history"
    assert any(len(ln) == len(CALIBRATED) and "s_hi=1.0091)" in ln for ln in lines), (
        "no unwrapped record: the fixture does not represent the post-change file"
    )


@needs_powershell
def test_the_runbook_command_still_works_once_the_wrap_stops(mixed_log, tmp_path):
    """The residual-risk claim, executed rather than asserted.

    Widening the writer does not retire the stitched read, and must not break
    it. A stitch rule of "join anything at least 120 long" would swallow the
    record after the now-unwrapped 133-char save line; `-eq 120` does not.
    """
    script = tmp_path / "confirm.ps1"
    rebound = re.sub(
        r'^\$log\s*=\s*".*"$',
        '$log = "{}"'.format(str(mixed_log).replace("\\", "\\\\")),
        _confirm_block(),
        count=1,
        flags=re.MULTILINE,
    )
    assert mixed_log.name in rebound, "the $log assignment was not rebound"
    script.write_text(rebound + '\n"records : " + $lines.Count\n', encoding="ascii")
    proc = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "band   : calibrated s_lo=1.0722 s_hi=1.0091" in proc.stdout, (
        "the runbook check stopped reading the suffix once output stopped "
        "wrapping:\n" + proc.stdout
    )
    assert "commit : abcd1234" in proc.stdout, (
        "the newest witness line was not read whole:\n" + proc.stdout
    )
    # 8 physical rows, one of which is a two-row wrapped record: 7 records.
    m = re.search(r"records : (\d+)", proc.stdout)
    assert m and int(m.group(1)) == 7, (
        f"stitched {m.group(1) if m else 'nothing'} records, expected 7: the "
        "rule either joined two real records or split a wrapped one\n" + proc.stdout
    )
