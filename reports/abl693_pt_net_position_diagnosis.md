# ABL-693 — why PT left both Chronos models on 2026-09-06

**Read on 2026-09-10.** Replica `C:/Code/able/data/energy_dashboard.db` and
sidecar `C:/Code/able/data/forecasts_local.db`, both opened through the
read-only SQLite URI. Nothing was fitted and nothing was scored, so no number
here is in-sample or out-of-sample — this is a census and a log read.

## Summary

Three things, of which the second and third are corrections to the issue.

1. **PT's net-position actuals stop at 2026-09-04 21:00 UTC, upstream.** The
   ABL-650 context guard refuses a context that stale, so both Chronos models
   abstain. They are behaving correctly.
2. **It was never silent.** The refusal is logged at `ERROR` on every affected
   run, including this morning's. The 09-06..09-09 blocks have no PT line
   because they have almost no lines at all — the transcript keeps ~5 of ~155
   records per run once the block is no longer the newest. Adding a log line
   would fix nothing, and could not run anyway (ABL-692).
3. **The two models that look healthy are the ones to worry about.**
   `baseline-V012` and `xgboost-V014` are not serving PT normally; they are
   serving it off a context whose newest actual is six days old, because
   neither has a staleness guard.

## 1. The cause is upstream, and the ingest reports success

`net_position`, rows since 2026-08-01, per zone:

| zone | last observed `timestamp_utc` | rows |
|---|---|---|
| PT | **2026-09-04 21:00:00** | **838** |
| every other gate-scope zone | 2026-09-10 21:00:00 | 982 (PL 978) |

982 − 838 = 144 = 6 days × 24 h. PT has no interior holes: every day from
08-25 to 09-03 has n=24, and 09-04 has 22 (through 21:00), which is the normal
same-day leading edge — ES and FR sit at 22 rows for 09-10 for the same reason.
PT is not gappy. It stops.

The prod ingest run that should have advanced it, from `data_ingestion_log`:

```
pipeline_type=net_position country_code=PT
start 2026-09-10T02:10:04Z  end 2026-09-10T02:10:25Z
status=completed  records_inserted=45  records_updated=0  records_failed=0
error_message=NULL
```

45 rows, covering targets `[2026-09-03 01:00 .. 2026-09-04 21:00]`. The same
night, the same pipeline inserted **189** rows for SK, SI, RO, NL, LV and LT
(185 for PL), reaching 2026-09-10 21:00. So the PT fetch succeeds and returns a
short series: ENTSO-E has published nothing for the PT bidding zone past
2026-09-04. This is the ABL-663 pattern — an upstream stop that every ingest
run records as `completed` with `records_failed=0` — on a second stream.

The replica is a faithful witness for this: `able-db-sync` at 07:00 exports all
29 non-weather tables from prod and replaces the local ones transactionally
(`REFRESHED=29 tables`), so the hole is prod's, not the mirror's.

One trap worth writing down, because it cost time: `fetched_at` is **not** a
first-arrival stamp. The ingest re-fetches a rolling ~8-day window and
re-inserts it, so a target keeps the stamp of the last night it was inside that
window. That produces a clean one-day-per-day staircase on *every* zone and
reads exactly like a chronically lagging feed. ES and FR show the identical
staircase and are perfectly current. Read `MAX(timestamp_utc)`, not
`fetched_at`.

## 2. The guard, and why 2026-09-06 is the boundary

`src/chronos2/input_builder.py:711` calls
`_net_position_context_refusal_reasons`, and line 43 sets
`NET_POSITION_MAX_STALENESS_HOURS = 72`. Staleness is
`nominal_cutoff − past_cutoff`, where a D+2 target gives
`nominal_cutoff = D+1 23:00` and `past_cutoff = min(nominal_cutoff, last observed)`.

With PT frozen at 2026-09-04 21:00:

| run day | target | nominal cutoff | staleness | outcome |
|---|---|---|---|---|
| 2026-09-04 | 09-06 | 09-05 23:00 | 26 h | served |
| 2026-09-05 | 09-07 | 09-06 23:00 | 50 h | served — last PT vintage |
| 2026-09-06 | 09-08 | 09-07 23:00 | **74 h** | refused — first miss |
| 2026-09-07 | 09-09 | 09-08 23:00 | 98 h | refused |
| 2026-09-08 | 09-10 | 09-09 23:00 | 122 h | refused |
| 2026-09-09 | 09-11 | 09-10 23:00 | 146 h | refused |
| 2026-09-10 | 09-12 | 09-11 23:00 | **170 h** | refused |

Both ends of that column are pinned by logged values rather than derived: the
2026-09-10 run logged `stale_context=170h>72h` for PT, and logged
`data stops 26h short of the nominal cutoff` for the healthy zones on the same
run. The 72 h threshold buys exactly two days of grace past the normal 26 h,
which is why a feed that stopped on 09-05 removed the zone on 09-06.

**No code changed.** The serving tree `C:/Code/able/energy-forecast` is on
branch `ABL-584-reland-abl471` at `8ac3642` (2026-08-27), and
`git log --all --since=2026-08-20 -- src/chronos2/input_builder.py` is empty.
The issue's reasoning on this point holds, and it corroborates ABL-692.

## 3. It was logged. The log threw it away.

This morning, in full:

```
2026-09-10 08:01:08,343 - energy_forecast.chronos2 - ERROR -
    Failed to forecast PT/net_position:
    Refusing PT/net_position target=2026-09-12: stale_context=170h>72h
```

The 2026-09-02 line the issue quotes is not truncated in any lossy sense — the
PowerShell transcript hard-wraps at 120 columns and the remainder is the next
physical line (`n target=2026-09-04: stale_context=98h>72h`).

The reason 09-06..09-09 name no zone is that those blocks retain almost
nothing. Counting records matching `^2026-09-DD ` in
`C:/Code/able/logs/net-position-forecast.log`:

| 09-01 | 09-02 | 09-03 | 09-04 | 09-05 | 09-06 | 09-07 | 09-08 | 09-09 | 09-10 |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 14 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | **159** |

Only today's block — the newest — is intact. The five survivors are the last
five records of the run (`Saved 24 point forecasts`, the SK quantile lines,
`Total: N forecast points generated`); every per-zone line and every `ERROR`
is gone. The 09-06 block's content begins mid-line, with the fragment `7]`, so
this is a byte-level truncation and not a line-level tail. 09-02 kept its
errors only because that run produced 14 records in total.

So the issue's proposed second deliverable — a log line so the next silent drop
is loud — would not have helped, for two independent reasons:

- the line already exists and is already at `ERROR`;
- the cron executes the frozen working tree (ABL-692), so a line merged to
  `origin/main` would not run.

The loudness has to live somewhere that survives, and somewhere that is not the
serving path. That is what `scripts/abl693_zone_census.py` is.

## 4. baseline-V012 and xgboost-V014 are not fine

They kept serving PT for vintages 09-06..09-10 — 24 points a day each — from
the same frozen series, because neither carries the net-position staleness
guard. `forecast_challengers.py` only refuses a zone when it has *no* actuals
at all (`no baseline for CH,IT,NO,SE`) or no trained model; six-day-old actuals
clear both tests.

Whether abstaining beats extrapolating is a call for whoever owns those two
paths, not for this issue. The measurement consequence is not a judgement
call: **their PT rows over these vintages are not comparable to their other 18
zones** and should not be pooled with them as though they were.

## 5. What this does to ABL-677

`build_gate_scope` derives `countries_measured` from
`gate_scored["country_code"].nunique()`, and `GATE_EXCLUDED_COUNTRIES` is a
static list. A zone that stops being served therefore never appears as
*excluded* — it just contributes fewer pairs, and the pooled figure reweights
onto the zones that stayed. Nothing prints that this happened.

Census of `chronos-2-V010` and `chronos-2-V016`, 2026-08-31..2026-09-10
(`scripts/abl693_zone_census.py`, output in `reports/abl693_zone_census.json`):

| vintage day | zones served |
|---|---|
| 08-31 | 19 |
| 09-01 | **9** |
| 09-02 | **no vintage at all** |
| 09-03, 09-04, 09-05 | 19 |
| 09-06 .. 09-10 | **18** |

- PT holds 4 of 10 vintage days; row share **0.0229** against **0.0526**
  balanced, i.e. 43% of its intended weight.
- 09-01 is additionally missing AT, BE, BG, DE, ES, FI, HU, LV and PL.
- 09-02 produced `Total: 0 forecast points` — all 19 refused on a
  replica-wide staleness event — so it is absent from the sidecar entirely and
  is invisible to any per-zone check.

**The direction matters.** ABL-650's defect statement names PT at 67.2 %
coverage, one of its three worst cells. Removing a worst cell *raises* pooled
10–90 coverage. The composition shift therefore moves the pooled figure toward
the pre-registered target and away from firing the revert trigger — the
direction that hides a miss rather than inventing one. Any pooled number read
over this window without stating the panel is optimistic by an unquantified
amount, and the per-zone band-width clause cannot be evaluated for PT on
vintages 3 onward at all.

This is a population statement, not a re-read: no coverage number is computed
here, and none should be until ABL-677's own window opens.

## Contamination touching this window

- **ABL-71** (prod ingest stale, fixes undeployed) — same class as the
  mechanism above, though PT's stall is upstream of our fetch rather than in
  it. It touches this window.
- **ABL-67** (fabricated net_position rows) — `net_position_backup_abl67` tops
  out at 2026-08-11 21:00, so the removed rows fall outside 2026-08-31..09-10.
- **ABL-111 / ABL-109** (zero-as-missing actual load) — not applicable;
  net_position is a different table.

## Reproducing

```
python scripts/abl693_zone_census.py \
    --sidecar-db C:/Code/able/data/forecasts_local.db \
    --since 2026-08-31 \
    --expect-zones AT,BE,BG,CZ,DE,EE,ES,FI,FR,HR,HU,LT,LV,NL,PL,PT,RO,SI,SK
```

Exit 1 means the panel moved. `--expect-zones` is not optional in practice:
once PT has been gone longer than the window, the union over the window no
longer contains it and "the population did not change" becomes true and
useless. `tests/test_abl693_zone_census.py` pins that case.

## Not owned here

- The upstream PT stop and the ingest reporting it as `completed` belong with
  ABL-663 (same pattern, `net_position` instead of load/generation).
- The transcript losing ~150 records per run belongs with whoever owns the
  workstation cron; it is the reason this went four days unnoticed.
- The frozen serving checkout is ABL-692.
