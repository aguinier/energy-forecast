# ABL-733 — the serving transcript stops wrapping at 120 columns

**Status:** merged 2026-09-10 (PR #120, merge `53427a17`). Nothing to install:
the change is inside `scripts/workstation/run-net-position-serving.ps1`, which
the serving checkout hard-resets to `origin/main` on every run, so it needs no
operator step.

**It takes effect at the SECOND scheduled run after the merge, not the first.**
An earlier revision of this section said "the first 08:00 run after the merge";
that was wrong, and it would have manufactured a false incident on the first
one. The launcher is the file that performs the sync, so the run that *pulls*
the change is still executing the previous version of it. See
"[When it starts serving](#when-it-starts-serving--one-run-later-than-it-looks)"
below for the mechanism, the reproduction, and what each of the two runs will
look like in the log.

No forecast-quality metric is reported here, so no scoring window and no
contamination issue applies. Everything below is a fact about what a launcher
emits, re-derivable with the commands given.

## The defect

`Start-Transcript` records the console **screen buffer**. Native (Python) output
therefore arrives in `C:\Code\able\logs\net-position-forecast.log` hard-wrapped
at `$Host.UI.RawUI.BufferSize.Width`, which is **120** for the console the
scheduled task's `wscript.exe` wrapper hands `powershell.exe`. PowerShell's own
`Write-Host` goes through a different path and is not wrapped.

ABL-732 found this through one line — the 133-char calibrated save line, which
made the ABL-692 runbook's check 2 report a defect on a correct system. It was
never one line. Re-measured on the live log, 2026-09-10, classifying a physical
line as Python-origin when it opens with a log-record timestamp
(`^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}`):

| | |
|---|---|
| physical lines | 3,352 |
| Python-origin lines | 587 |
| **truncated at exactly 120** | **171 (29%)** |
| Python-origin lines over 120 | 0 |
| distinct messages affected | **58** |
| longest record, stitched (Python) | 221 chars |
| longest record, stitched (any) | 383 chars |

The 58 include 36 occurrences of `data stops NNh short of the nominal cutoff`,
every `Built inference input for XX/net_position target=…` (221 chars, cut at
column 120 — the context window and target range are simply absent), and every
`Failed to forecast XX/net_position: <reason>`, cut at exactly the reason.

That last one is the argument against the cheaper alternative the issue offered.
Shortening the save line below 120 fixes the line that made the problem
visible and leaves every error message in the log truncated where its cause
would be.

## The change

One block in `run-net-position-serving.ps1`, immediately after
`Start-Transcript`, plus a `[int] $BufferWidth = 512` parameter.

- **512**, because the widest record this log has ever held is 383 chars. A
  width at or below that would leave it wrapping, just less often.
- **Widens only.** `BufferSize.Width` below `WindowSize.Width` throws, so an
  already-wide host is left alone rather than shrunk to the constant.
- **`try`/`catch`, and the catch does not rethrow.** Under `wscript.exe //B`
  there may be no console, and a `RawUI` set then throws.
  `$ErrorActionPreference = "Stop"` is set at the top of the launcher, so an
  uncaught throw here aborts before `& $job` and the forecast is lost to a
  logging nicety. ABL-692 settled that trade the other way: a missing vintage
  is worse than a wrapped one.
- **Reports what it did**, as `net-position serving transcript width: 120 -> 512`
  (or `unavailable, log stays wrapped (<reason>)`). The log now records the
  width it was written at, which is what makes a future read of it decidable.

### Why after `Start-Transcript`, not before

The issue proposed before. Both were measured, in a hidden console, and they
unwrap native output **identically** — so the tie is broken by evidence:
after the transcript opens, both the success and the `catch` land in the log
instead of in a stream nobody keeps. It is still the first statement in the
`try`, so an unhandled exception's own rendering — which the error formatter
also wraps — is readable too.

## Verification

### The probe

The issue recorded that this could not be checked from an agent session: the
session's host reports a 500-column buffer and routes no native output into the
transcript. **The session host was never the thing to measure.** The scheduled
task runs `wscript.exe //B` → `WshShell.Run(cmd, 0, True)`, which is
`CreateProcess` with a new console created hidden. `STARTF_USESHOWWINDOW` /
`SW_HIDE` plus `CREATE_NEW_CONSOLE` is that same call, and it reproduces the
cron's console exactly — `ConsoleHost`, `BufferSize 120x3000`,
`WindowSize.Width 120` — from an ordinary agent session.

**What hid it is stdio redirection.** A transcript captures the console screen
buffer; redirected std handles bypass the console, so the grandchild `python`
writes to a pipe and its output never reaches the log at all. Probed three ways,
same harness:

| spawn | native output in transcript |
|---|---|
| `subprocess.run(capture_output=True)` + `CREATE_NEW_CONSOLE` | **absent** |
| `stdin=DEVNULL` (sets `STARTF_USESTDHANDLES`) + `CREATE_NEW_CONSOLE` | **absent** — leaked to the parent |
| no stdio arguments at all + `CREATE_NEW_CONSOLE` | present, wrapped at 120 |

The first two are the shape that reads as "no wrapping" while measuring
nothing. `subprocess` sets `STARTF_USESTDHANDLES` if *any* of stdin/stdout/stderr
is passed, so the harness passes none — not even `DEVNULL`.

Verified independently through the production path as well: the same probe
launched via the real `wscript.exe //B //Nologo C:\Users\guill\bin\run-hidden.vbs`
gave byte-identical results.

### The measurement

Same harness, same 133-char stderr line and 174-char stdout line, one block
different:

| mode | calibrated save line in transcript |
|---|---|
| control (no widening) | `…to DB (calibrated s_lo=1.0722` + continuation ` s_hi=1.0091)` — 120 + 13 |
| widen before `Start-Transcript` | whole, 133 chars |
| widen after `Start-Transcript` | whole, 133 chars |

Both streams behave the same: the 174-char stdout line is 120 + 54 in the
control and 174 in both treatments. Nothing is padded to the new width.

### The tests

`tests/test_abl733_transcript_buffer_width.py` — **20 passed** under
`C:\Code\able\energy-forecast\.venv\Scripts\python.exe`.

That count has moved twice since this section was first written, and this
paragraph was stale for both: it said **12**, which was right for PR #120 and
wrong from the moment #122 merged. #122 added the four one-run-lag tests in
"[When it starts serving](#when-it-starts-serving--one-run-later-than-it-looks)"
below (12 → 16), and ABL-751 added the four shipped-file rehearsal tests in
"[The shipped file, through the production spawn](#the-shipped-file-through-the-production-spawn)"
(16 → 20). A test count in prose is a claim like any other; re-derive it from
the file rather than from this sentence.

Two of the original twelve execute the shipped block in a hidden console and are
the reason the rest are not just grep assertions:

- `test_the_control_reproduces_the_120_column_wrap` runs the harness **without**
  the block and asserts the transcript holds a line cut at exactly 120 and no
  whole suffix. Without it, "the line is whole" could pass on a harness that
  never wrapped anything — which is how the ABL-692 check came to be wrong.
- `test_the_shipped_widening_stops_native_output_wrapping` adds the block,
  extracted verbatim from the launcher, and asserts the 133-char line is whole.

Both are gated on a Windows console and skip on the ubuntu CI runner, which has
none; `scripts/test_floor.py`'s allowance moved 4 → 6 to say so out loud, and
6 → 10 for ABL-751's four, which are gated on the `.vbs` as well.

Six mutations, each re-read from disk to confirm it applied before the run —
the harness also asserts the tree is clean before every one, so a failed restore
cannot carry into the next:

| mutation | caught by |
|---|---|
| `$size.Width = 120` instead of `$BufferWidth` (a no-op widen) | `test_the_shipped_widening_stops_native_output_wrapping` — **and nothing else** |
| `$BufferWidth = 200` | `test_the_width_clears_the_widest_record_the_log_has_held` — and nothing else |
| drop the `-lt $BufferWidth` guard (unconditional set) | `test_the_widening_never_narrows` |
| `catch` rethrows instead of recording the reason | `test_the_widening_is_wrapped_in_its_own_try_catch`, `test_a_failure_to_widen_is_reported_not_hidden` |
| `try {` → `if ($true) {` (no guard around the RawUI set) | `test_the_widening_is_wrapped_in_its_own_try_catch`, plus both executed tests |
| move the block above `Start-Transcript` | `test_the_outcome_lands_inside_the_transcript` |

The first two are the pair worth keeping. **A no-op widen passes every
structural assertion in the file** and is caught only by execution — it is the
whole argument for the two skips on CI. **A 200-column width passes every
executed assertion** — 133 and 174 both fit under 200 — and is caught only by
the measured constant. Neither kind of test subsumes the other.

One result is honest rather than flattering: the `try {` mutation leaves an
orphan `catch`, which is a *parse* error, so the two executed tests caught it by
failing to run the script at all rather than by observing wrapped output. The
structural test caught it for the intended reason.

Planning these found a defect in the tests themselves. Both ordering tests
originally used `launcher.index("Start-Transcript")`, and this launcher's
comments name `Start-Transcript` twice before the call — so the anchor landed
inside the param block, where nothing can precede it, and the tests would have
passed however the block was placed. They now match a statement and assert the
match is not inside a comment.

### The shipped file, through the production spawn

Everything above executes an **extract**. The two hidden-console tests lift the
widening block out of the launcher into a miniature script, and the one-run-lag
tests below drive a *miniature* launcher against a miniature origin. Nothing ran
`scripts/workstation/run-net-position-serving.ps1` byte for byte, and nothing
went through the scheduled task's actual action —
`wscript.exe //B //Nologo run-hidden.vbs` → `powershell.exe -WindowStyle Hidden
-File …`. ABL-733 closed that gap by hand on 2026-09-10, once. ABL-751 turned
the one-off into four tests, because a thing measured once is not a thing
pinned.

The fixture builds a throwaway origin holding the working tree's launcher
**byte for byte** (asserted, after a `clone -c core.autocrlf=false`, so a
renormalised line ending cannot quietly substitute a different file) plus a stub
job, clones it, and runs the launcher twice through the real `.vbs`. Control
first, and the control is not "the block removed": `-BufferWidth 120` makes the
widening a genuine no-op on a 120-column console, so the pair differs by one
parameter.

What this covers that the extract cannot is the **whole-file path** — param
block, dev-checkout refusal, `Start-Transcript` on a real `-LogDir`, the
fetch/reset, the ABL-692 witness line, and `& $job -Repo` — all in the console
`wscript.exe` created. Two mutations make the difference concrete. Both leave
every structural assertion and both hidden-console tests **green**, because the
widening block itself is untouched and the extract harness never invokes a job:

| mutation | caught by |
|---|---|
| `& $job -Repo $Serving` → `… \| Out-Null` | the ABL-751 control and treatment tests — **and nothing else in the file** |
| `& $job -Repo $Serving` → `… 2>&1 > …` | all four ABL-751 tests — **and nothing else in the file** |

That is the same defect this whole issue is about, one level up: a pipeline or a
redirect takes the job's output off the console, so `Start-Transcript` never
records it and the widening is pointless. `Write-Host` survives a `| Out-Null`
and native output does not — which is why the witness-line test still passed on
the first mutation while both wrapping tests failed. The asymmetry that made
ABL-732's grep wrong is the asymmetry that makes this mutation invisible to
everything except a test that runs the real file.

They are gated on `C:\Users\guill\bin\run-hidden.vbs` and `wscript.exe`
existing, so they skip on the ubuntu runner and run on the workstation, which is
where the launcher runs in production. They are deliberately given **no
substitute spawn**: reimplementing the `.vbs` would measure a different spawn
than the one that serves, and that is precisely the substitution these tests
exist to stop relying on. The hidden-console tests above already hold the
closest faithful approximation, and those do run on CI.

Production is untouched — throwaway origin, throwaway clone, throwaway log
directory, stub job. Nothing writes to `C:\Code\able\logs`, no scheduled task is
triggered, and no forecast runs.

## When it starts serving — one run later than it looks

The serving clone hard-resets itself to `origin/main`, so this needs no operator
step. But the file being changed **is the file that performs that reset**, and a
`powershell.exe -File` script is parsed in full before its first statement runs.
So the run that *pulls* the new launcher is still executing the old one.

| run | launcher that executes | clone left at | width line in log |
|---|---|---|---|
| 2026-09-11 08:00 | `0cd9ec2` (pre-ABL-733) | `53427a17` or later | **none — expected** |
| 2026-09-12 08:00 | `53427a17` (widened) | current `origin/main` | `120 -> 512` |

`Get-ScheduledTaskInfo able-net-position-forecast` gives `NextRunTime
09/11/2026 08:00:00`; the clone's `git reflog` shows one `reset: moving to
origin/main` per run, and it currently sits at `0cd9ec2`.

**The job it invokes does not lag.** `run-net-position.ps1` is invoked with `&`
*after* the reset, so it is read from disk at that moment and a change to it
serves on the first run. The lag is specific to the launcher, and therefore
specific to this change.

Reproduced end to end rather than argued —
`test_a_running_launcher_does_not_see_its_own_update` builds a miniature origin
holding OLD and NEW copies of both scripts plus a clone parked one commit
behind, and runs the launcher twice:

```
RUN 1   LAUNCHER-VERSION: OLD      <- executes the pre-reset version
        JOB-VERSION: NEW           <- child picked up immediately
        LAUNCHER-TAIL: OLD         <- still OLD after the reset landed
        clone HEAD -> NEW
RUN 2   LAUNCHER-VERSION: NEW
```

`LAUNCHER-TAIL` is the load-bearing line: it is emitted *after* the reset has
already rewritten the file on disk, and it still reads `OLD`.

The paired control, `test_the_reset_actually_moves_the_clone`, asserts the reset
moved `HEAD`. The first draft of this probe named a branch that did not exist,
so the reset failed and both runs printed `OLD` — the right answer for the wrong
reason. A probe whose mutation silently no-ops proves nothing.

**Why this was not caught earlier: there was nothing to catch.** ABL-733 is the
first change to `run-net-position-serving.ps1` since the serving clone was
installed — `git log origin/main -- scripts/workstation/run-net-position-serving.ps1`
returns exactly two commits, `d97f168` (which created it) and `ca4278e` (this
one), and the file is byte-identical between `28abfeb` and `0cd9ec2`. No prior
run ever exercised the self-update path, so no log could have shown the lag.

## What this does not do

**It does not retire the stitched read.** The transcript is opened `-Append`, so
the 3,352 already-wrapped lines stay in the file, and this log will hold both
shapes indefinitely. The ABL-692 runbook's command is written to be correct on
both — `-eq 120` plus "the next line does not open a new record", never "long" —
and `test_the_runbook_command_still_works_once_the_wrap_stops` runs it over a
fixture holding wrapped history followed by unwrapped records.

**It is not confirmed in production yet**, and the first run after the merge is
**not** the one to confirm it on — see the section below. Confirm on the second
one, with the runbook's command plus:

```powershell
Select-String "net-position serving transcript width:" `
  C:\Code\able\logs\net-position-forecast.log | Select-Object -Last 1
```

Expected: `net-position serving transcript width: 120 -> 512`. If it reads
`unavailable, log stays wrapped (…)`, the host had no console to resize — the
forecast still ran and the log is simply as wrapped as it was before. **Zero
matches on 2026-09-11 is the expected, healthy result**, not a failure.

**It does not change what anything greps for.** `net-position serving commit:`
is untouched, and the width line is a distinct prefix, so an existing
`Select-String` for the witness matches exactly what it matched before.
