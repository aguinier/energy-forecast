> **Archived from `CLAUDE.md` on 2026-08-27** (companion to the ABL-536
> energy-dashboard-frontend trim). Historical narrative, incident forensics
> and dated measurements; `file:line` references are frozen as of the archive
> date. The durable rules distilled from this material live in the repo-root
> `CLAUDE.md`; where they conflict, the root file wins.
# What a runner reports (ABL-370)

## What a runner reports (ABL-370)

The exit code says whether a runner crashed. It does not say whether it
*produced* anything, and `forecast_daily` used to read it as if it did: exit 0
was logged `OK`, and a row count was recovered only if stdout happened to
contain `Forecast (N rows)` or `Saved N forecast records`. A run that generated
nothing prints neither, so `tso-correction` skipping all three renewable types
on a day the upstream Elia forecast has not landed printed

```
[tso-correction] OK: BE solar D+2
Total: 10, Success: 2, Skipped: 8, Failed: 0
```

— indistinguishable from a run that saved 96 rows, and its 0 vanished inside a
`Total forecasts:` sum the in-process models push into the thousands. That is
the same reporting shape that hid ABL-354.

**Every external `MODEL_RUNNERS` entry emits one line on stdout, once per run,
zero included:**

```
FORECAST_RECORDS=<n>
```

`src/runner_report.py` owns both ends — `emit_record_count()` writes it,
`parse_record_count()` reads it — so the contract cannot drift. It imports
nothing but `typing`, deliberately: `chronos-bolt-small` runs under its own venv
and importing this must never be what breaks it.

`forecast_daily` then distinguishes four outcomes, not two:

| outcome | means |
|---|---|
| `success` | exit 0, reported ≥ 1 row |
| `empty` | exit 0, reported exactly 0 rows — ran fine, produced nothing |
| `unreported` | exit 0, no count line — what it did is **unknown**, and unknown is not 0 |
| `failed` | non-zero exit, timeout, or exception |

`records` is `None` for `unreported`, and contributes nothing to
`Total forecasts:` — recording it as 0 would be a number nobody measured. The
summary gains a per-runner block and an explicit
`Runners that produced no forecasts:` callout, which is the line a silent runner
now has to appear on.

`empty` is not a failure and does not change the exit code: skipping when the
upstream forecast is genuinely absent is correct behaviour, and today D+1/D+2
for BE legitimately produce nothing (`energy_generation_forecast` for BE ends at
the reference date). The defect was never the zero — it was that the zero was
unsayable.

Adding a runner: call `emit_record_count(len(df))` on every path that exits 0,
*before* any `if not df.empty:` guard. `tests/test_runner_reporting.py` checks
that statically for each configured runner and would otherwise report your
runner as `unreported` forever.

### Skipped is a flag, not a phrase

`failed` is reported net of `skipped` — "there was no model to run" is not a
failure — but the two used to be told apart by looking for `not found` in the
error text. `chronos-bolt-small` points at a venv that does not exist on this
box, so it fails with `Executable not found: [WinError 2]`, and a runner that
could not run *at all* was counted as benign. `generate_forecast` now sets
`result['skipped'] = True` at the one place that knows (the `FileNotFoundError`
from `Forecaster.load`), and `is_skip` reads only that.

Consequence worth knowing: a default run on this box now ends
`Skipped: 1, Failed: 1` and exits 1 for BE/price, where it used to exit 0.
`chronos-bolt-small` is genuinely unrunnable here; fix the path in
`config.MODEL_RUNNERS` or set `enabled: False`, but do not read the exit 0 that
preceded it as the job having been fine.

That same handler used to log `python_exe`, a name local to
`build_runner_command` since ABL-354 — a `NameError` raised *inside* an
`except` clause, which the sibling `except Exception` does not catch. A missing
runner interpreter killed the whole daily job before it printed any summary.
`--countries BE --types price` reproduces it on the pre-fix file.
