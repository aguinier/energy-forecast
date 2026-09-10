# ABL-692 — the net-position cron runs a pinned, attributable checkout

**Status:** code landed on a branch; the one-time install is a production action
and has **not** been performed. See *Install runbook* below.

No forecast-quality metric is reported here, so no scoring window and no
contamination issue (ABL-71 / ABL-67 / ABL-111 / ABL-109) applies. Everything
below is a fact about which code executes, re-derivable with the commands given.

## The defect

The scheduled task `able-net-position-forecast` has no deploy step. It executes
a working tree directly:

```
wscript.exe //B //Nologo "C:\Users\guill\bin\run-hidden.vbs" "powershell.exe"
  -WindowStyle Hidden -NoProfile -ExecutionPolicy Bypass
  -File C:\Code\able\energy-forecast\scripts\workstation\run-net-position.ps1
```

Daily at 08:00 since 2026-07-24, as `guill`. That path is the **shared dev
checkout** — the tree agent runs use as scratch.

Measured 2026-09-10:

| fact | value |
|---|---|
| dev checkout branch / HEAD | `ABL-584-reland-abl471` / `8ac3642` |
| last time its HEAD was on `main` | `bb1fc29`, 2026-08-27 16:10 (moved off 2026-08-28 09:46) |
| `origin/main` | `34d4b57` (was `33bca88` when the issue was filed) |
| `merge-base --is-ancestor 3419031 origin/main` | **YES** |
| `merge-base --is-ancestor 3419031 HEAD` | **NO** |

`3419031` is the PR #107 merge (ABL-650 band recalibration). Commits on
`origin/main` since `bb1fc29` that touch a serving path — `scripts/forecast_chronos2.py`,
`src/`, `scripts/workstation/` — are exactly two, and neither was running:

```
4d9e51f  ABL-650: recalibrate the net-position p10-p90 band so it covers 80%
02d759a  ABL-651: intercept-only static bias correction, measured on current data
```

The serving log agrees, including **today's 08:00 run**, which post-dates the
issue: all twelve country lines log the plain form, with no calibration suffix.

```
2026-09-10 08:00:57 - energy_forecast.chronos2 - INFO -   Saved 216 quantile forecasts to DB
```

`calibrat` occurs 3× in the whole log, all three from V016's passthrough message
("…would miscalibrate it"). The ABL-650 witness suffix has never appeared.

## The fix

`scripts/workstation/run-net-position-serving.ps1` (new) is the entry point. It
hard-resets `C:\Code\able\energy-forecast-serving` to `origin/main`, writes the
resolved SHA into the transcript, then invokes the job from that tree.

The SHA line is the part that matters most. A stale tree is recoverable the
moment it is visible; before this line existed, **nothing on disk recorded which
code produced a given vintage**, so staleness was undetectable from the log.

Two deliberate choices, both against the obvious alternative:

- **A failed sync does not skip the forecast.** A missing vintage is worse than
  a one-day-stale one. The witness line is emitted either way, marked `STALE`,
  so staleness is loud rather than silent. The property this buys is
  attributability, not freshness.
- **No `git clean`.** `reset --hard` already guarantees every *tracked* file
  matches `origin/main`, which is all serving needs. `clean -xdf` would delete
  the gitignored eval reports the job writes into the tree.

`run-net-position.ps1` becomes the job, parameterised. The serving clone carries
only git-tracked code, so three gitignored inputs became parameters. **Each one
fails silently if it regresses**, which is why each has a test:

| param | default | why it is not under `$Repo` |
|---|---|---|
| `$Venv` | `…\energy-forecast\.venv` | ABL-69 pins the rail interpreter. An xgboost-3.3.0 artifact under conda 2.1.4 keeps its trees and silently resets the fitted intercept. |
| `$ModelsDir` | `…\energy-forecast\models` | 4.2 GB, gitignored. Only V014 reads it (17 MB at `models/net_position/V014`), via the `--models-dir` flag `forecast_challengers.py` already had. V010 is `fine_tune: false`, so the champion serves pretrained Chronos-2 and never resolves `MODELS_DIR`. |
| `$EvalRoot` | `…\energy-forecast\reports\net_position_eval` | ABL-30 / ABL-34 / `docs/claude/04-database.md` read this path. Moving it under the disposable clone would strand every reader on a directory that quietly stopped updating — the same failure class as ABL-692 itself. |

The `--candidate-backtest` files go the *other* way, from `$Repo`, because they
are tracked: a vintage is now scored against the backtest its own code shipped
with.

## Verification

`tests/test_abl692_serving_checkout.py` — 17 passed, under
`C:\Code\able\energy-forecast\.venv\Scripts\python.exe`. Seven mutations, each
verified to have applied before the run, each caught exactly one test:

| mutation | caught by |
|---|---|
| `$Repo` default back to the dev checkout | `test_job_does_not_default_to_the_shared_dev_checkout` |
| drop `--models-dir` | `test_job_passes_models_dir_explicitly` |
| `$EvalRoot` under the serving clone | `test_eval_reports_stay_where_their_readers_look` |
| drop the `reset --hard` | `test_launcher_fetches_and_hard_resets_to_the_pinned_branch` |
| drop the `STALE` witness line | `test_a_failed_sync_still_logs_a_sha_and_still_forecasts` |
| rename the witness prefix | `test_launcher_declares_the_witness_prefix` |
| remove the dev-checkout guard | `test_launcher_refuses_to_serve_from_the_shared_dev_checkout` |

Two of these mutations first appeared to pass; the `sed` that made them had not
matched. They were re-run with an edit that asserts a single occurrence before
writing. A mutation that passes is a claim about the test *and* about the edit.

The launcher was then **executed**, against a throwaway branch whose job script
is a stub, so no forecast was written:

- pointed at the dev checkout → refused; its branch was `ABL-584-reland-abl471`
  before and after, unchanged.
- pointed at a non-checkout directory → refused, naming the installer.
- pointed at a clone parked on an older commit **with an uncommitted tracked
  modification** → advanced to the branch head, `dirty=1` → `dirty=0`, witness
  line written *to the transcript file*, job invoked with `-Repo` set to the
  serving path.
- with `origin` broken → `WARNING … sync FAILED`, witness line marked `STALE`,
  **and the job still ran**; HEAD unchanged.

All three scripts parse under `[Parser]::ParseFile`.

## Install runbook — not yet performed

Creating the clone is inert; re-pointing the scheduled task changes what
production forecasts tomorrow morning. They are separate steps on purpose.

```powershell
cd C:\Code\able\energy-forecast-serving\scripts\workstation   # after step 1 clones it
.\install-serving-checkout.ps1                       # 1. clone + report externals
.\install-serving-checkout.ps1 -UpdateScheduledTask  # 2. re-point the task
```

Step 1 refuses to run if `origin/main` does not yet carry the launcher, so the
PR must land first. It also prints OK/MISSING for the three external inputs, so
a broken install is visible then rather than at 08:00.

Confirm at the next 08:00 run:

```powershell
Select-String "net-position serving commit:" C:\Code\able\logs\net-position-forecast.log | Select-Object -Last 1
Select-String "quantile forecasts to DB"     C:\Code\able\logs\net-position-forecast.log | Select-Object -Last 1
```

The first must name a SHA at or after `3419031`; the second must then carry the
`(calibrated s_lo=… s_hi=…)` suffix. If the first advances and the second does
not, the calibration has a second, separate problem — and that is now
distinguishable, which it was not before.

## Residual risks, stated

1. **Serving still depends on the dev checkout** for `.venv` and `models/`. An
   agent that breaks the venv breaks serving. This is not a regression — it is
   today's coupling, now explicit, named and printed by the installer instead of
   implied by a shared path. Narrowing it further means moving the 4.2 GB
   artifact tree, which is its own change.
2. **The first calibrated run resets ABL-677's clock.** Its 10-vintage window
   has to be re-based on the first calibrated `generated_at`, not on 2026-09-13.
   Until then, the ABL-650 trigger would fire against a band the fix never
   touched — reverting a change that never shipped.
3. **ABL-651 (PR #108) is in the same position** and is still `in_progress`. Its
   change is likewise not live.
