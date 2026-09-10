# ABL-692 close-out: the serving checkout publishes a calibrated band

**Read 2026-09-10 ~12:15-12:40 local (10:15-10:40 UTC).** Read-only throughout: both
databases opened with the SQLite read-only URI form. Nothing fitted, nothing promoted,
no production change. Every number below is a census of stored rows, so it is neither
in-sample nor out-of-sample -- there is no fit here to be either.

Machine record: `reports/abl_692_calibration_witness.json`
Probe: `scripts/abl692_calibration_witness.py`

## 1. What ABL-692 asked for, and why the log alone does not answer it

The unblock descriptor asked for two string checks in
`C:/Code/able/logs/net-position-forecast.log`: a `net-position serving commit:` line at
or after `3419031`, and a `(calibrated s_lo=... s_hi=...)` suffix on the next quantile
line.

Both pass (section 2). Neither is sufficient. A log line proves what the process
*said*; the forecast table is what the dashboard serves. ABL-692 exists precisely
because a merged change was inert while everything upstream of the forecast table
looked healthy, so closing it on two `Select-String` hits would repeat the original
error one layer down. Sections 3-4 are the DB-side proof.

## 2. Log and task state

Scheduled task `able-net-position-forecast`, read from `Get-ScheduledTask`:

```
Execute : C:\Windows\System32\wscript.exe
Args    : //B //Nologo "C:\Users\guill\bin\run-hidden.vbs" "powershell.exe" -WindowStyle Hidden
          -NoProfile -ExecutionPolicy Bypass -File
          C:\Code\able\energy-forecast-serving\scripts\workstation\run-net-position-serving.ps1
LastRun : 2026-09-10 12:09:13   LastTaskResult: 0   NextRun: 2026-09-11 08:00:00
```

The task points at the serving launcher, not the dev checkout. `LastRunTime` is the
task's own stamp, so the 12:09 run went through the registered action end to end and
returned 0 -- this is not a bare invocation of the script.

Witness line from that run:

```
net-position serving commit: 0cd9ec2640e0c7f887c770f020fe6f39ae1787ae
  (origin/main, committed 2026-09-10T11:02:47+02:00) Merge pull request #116 ...
```

`git -C C:/Code/able/energy-forecast-serving merge-base --is-ancestor 3419031 HEAD`
exits 0, so the PR #107 band recalibration is in the served tree. Checking ancestry
rather than equality is deliberate: the launcher hard-resets to `origin/main` every
run, so HEAD is expected to move and an equality check would go red on the next merge.

The quantile lines from the same run carry the suffix: `Saved 216 quantile forecasts to
DB (calibrated s_lo=1.0722 ...)`.

## 3. The band in the table moved by exactly the registered factors

Sidecar `C:/Code/able/data/forecasts_local.db`, `forecast_quantiles`, model
`chronos-2-V010`, type `net_position`. Two vintages exist for 2026-09-10:

| generated_at (UTC) | zones | rows | mean p10-p90 width | source |
|---|---|---|---|---|
| 2026-09-10 06:00:44 | 18 | 3888 | 2730.3 MW | scheduled run, **old dev checkout, uncalibrated** |
| 2026-09-10 10:09:24 | 18 | 3888 | 2840.4 MW | serving checkout, **calibrated** |

Both target the same day, 2026-09-12.

The whole-band ratio (1.040 pooled) is weak evidence on its own, because two runs hold
different context and the raw band moves regardless. The discriminating measurement is
the **split by half**: the calibration map is anchored at q50 and scales the lower half
by `s_lo` and the upper half by `s_hi` -- two different numbers. Per zone, all 18:

```
mean lo_ratio  1.0722    registered s_lo_applied  1.0722
mean hi_ratio  1.0091    registered s_hi_applied  1.0091
```

Not "consistent with" -- equal, in every zone, to four decimals.

**Why the comparison is exact rather than an estimate.** The q50 series is bit-identical
across the two vintages: 432 matched rows, `max|q50_new - q50_old| = 0.0`. The replica
refreshes once a day at 05:00 UTC, so both runs read the same observations, and
Chronos-2 is deterministic given its input. The raw forecast is therefore the same
series in both, and the ratios above are the applied factors themselves, not an
estimate of them. This is a clean A/B on the calibration alone, which is a stronger
result than the close condition asked for and is only available today -- once the
replica refreshes on 09-11, no later pair of vintages can be compared this way.

**The median did not move.** Against the `forecasts` table for the same vintage: 432
matched rows, `max|q50 - point| = 0.0`. The recalibration widens the band and leaves the
served point forecast bit-identical, which is the constraint PR #107 was built under.

Registration read from the serving checkout
(`experiments/net_position_quantile_calibration.json`), not from this tree, so a dev-tree
edit cannot be mistaken for what production applies:
`mode=pooled, alpha=0.1, s_lo_applied=1.0722, s_hi_applied=1.0091, fit_window
2026-08-05..2026-09-02, fit_vintages=29, fit_rows=12978`.

## 4. Contamination touching this window

Of the four standing issues, **none touches these numbers**. ABL-67 (fabricated
net_position rows) applies to GR, which is not in the served set at all. ABL-71 is not
load-bearing here. ABL-111/ABL-109 are actual-load rows and do not enter net position.
Nothing above is scored against actuals -- it is a comparison of two forecast vintages
with each other and with a registration file -- so actual-side contamination cannot
reach it.

## 5. Two findings that are not ABL-692

**5a. The first calibrated vintage is 2026-09-10 10:09:24Z, not 2026-09-11 06:00Z.**

ABL-677's read-time checklist item 2 says *"Expected first: 2026-09-11 06:00Z."* That was
right when written and is now a day off, because ABL-718 verified the install with a real
run rather than leaving it for the trigger.

**This correction could not be posted to ABL-677.** Every control-plane write in this run
returned `403 cross_issue_influence_run_context_required` -- the run is not task-bound, so
comments and status updates are unavailable to it, with or without the
`X-Paperclip-Run-Id` header. The correction is recorded here and enforced in code
(below) instead of as a comment. **ABL-677's next reader should treat this section as
that comment.**

The trap is not the date. Both vintages of 2026-09-10 target the same day, **2026-09-12**,
so the calibrated 10:09Z run is a manual re-run, not an independent 24-hour block:

1. **It must not be counted as one of the 10.** Two blocks sharing a target day are not
   independent; pooling both double-counts 09-12's actuals and tightens the interval for
   the same reason row-resampling does. ABL-677 item 6 already forbids that on the row
   axis -- this is the same error on the vintage axis.
2. **A filter phrased by date picks it up.** "Vintages on or after 2026-09-10" pools one
   calibrated and one uncalibrated vintage for a single target day. (Phrased as
   `generated_at > first_calibrated` it happens to exclude the right row, but only by
   accident of ordering -- 06:00Z sorts before 10:09Z.)
3. **It is already in the sidecar**, and `evaluate_net_position` scores every stored
   vintage whether or not this read wants it.

Suggested replacement for item 2: *count from the **scheduled** runs only. The first
calibrated scheduled vintage is 2026-09-11 06:00Z; the tenth is 2026-09-20.
`2026-09-10 10:09:24Z` is calibrated but is a manual install-verification re-run sharing
target day 2026-09-12 with the uncalibrated 06:00Z vintage -- exclude it by
`generated_at`, and do not read its existence as the window having opened early.*

**The act date of 2026-09-22 does not move**, and neither does the tenth vintage.

Rather than leave this as prose someone must remember, `duplicate_target_days()` in the
probe flags any target day served by more than one vintage, so the duplicate has to be
dispositioned rather than noticed. It fires on this case today.

**5b. PT confirmed absent, cause reproduced.** PT has produced no `chronos-2-V010`
net-position vintage since 2026-09-06 (six runs). Independently reproduced from the
replica: PT's `net_position` observations stop at **2026-09-04 21:00 UTC**, with
`fetched_at 2026-09-10 02:10` -- the ingest ran today and found nothing newer, so the
stop is upstream. The staleness guard measures the gap from the last real observation to
the hour before the target day starts: `2026-09-11 23:00 - 2026-09-04 21:00 = 170h`
against a 72h limit. It first breached on the 09-06 run (74h). The model is refusing
correctly. This is already diagnosed on ABL-693 and routed to ABL-663 (Operations); it
is recorded here only as an independent reproduction, not as a new issue.

The census also shows two coverage holes nobody counted before ABL-693:
`2026-09-01` served 9 zones, and `2026-09-02` produced no vintage at all.

## 6. Disposition

ABL-692's close condition is met, and met on the stronger form: the served tree carries
the recalibration, and the band in the forecast table moved by exactly the registered
multipliers with the point forecast unchanged.

What remains untested is only the 08:00 trigger firing on its own, which is the same
trigger that has fired daily for weeks and fired at 08:00 today. That is a delivery
question owned by ABL-718, which holds a monitor for it.

I do not deploy and I do not promote. Nothing here is a promotion recommendation: the
band recalibration was already merged under PR #107 and its revert trigger is read
separately on ABL-677 from 2026-09-22.
