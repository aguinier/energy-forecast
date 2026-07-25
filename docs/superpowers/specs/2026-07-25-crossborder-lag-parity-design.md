# Cross-Border Lag Parity — Design

**Date:** 2026-07-25
**Status:** Approved (design review with Guillaume)
**Repo:** energy-forecast

## Problem

The net-position input builder reads cross-border flow covariates up to
`past_cutoff = target_date 00:00 − 1h` (D+1 23:00). In production those hours do
not exist yet:

- Scheduled job (`able-net-position-forecast`) runs **08:00**, target **D+2** →
  forecast origin ≈ `T − 2d 08:00`.
- `crossborder_flows` publication lag measured on **prod** (not a sync artifact):
  `max(timestamp_utc) = 2026-07-24 12:00` at 14:12 UTC → **+26.2 h**, and the final
  hour is only partially populated (59 rows / 20 countries vs 108 / 26 for complete
  hours), so ~30 h is the safe publication lag.
- Newest flow genuinely available at origin ≈ **`T − 70h`**; the builder reads to
  `T − 1h`.

A historical backtest finds those rows in the DB, so evaluations consume data that
never existed at inference time. This is the same failure shape as the v252
retraction recorded in `forecast-promotion-gate`. Discovered by that gate while
evaluating V011 (2026-07-25); it affects **V010 and V011 equally** (shared builder),
so the V011-vs-V010 comparison stands, but the **absolute** skill of both is
unvalidated.

Target (`net_position`) and the other covariates are unaffected: day-ahead prices and
TSO load forecasts are legitimately future-dated/published ahead.

## Decision

Apply a **uniform lag of `CROSSBORDER_SERVE_LAG_HOURS = 96` (4 days)** to the
cross-border flow covariates, **identically in training and inference**: at target
hour `t` the model sees flows from `t − 96h`.

**Why uniform, not an absolute cutoff:** Chronos-2 crops training windows internally
and randomly, so an absolute "available_until" timestamp cannot be enforced per-crop.
A uniform shift is enforceable in both paths and makes train-lag = eval-lag =
serve-lag exactly, as the promotion gate requires.

**Why 96h:** the required lag varies across the target day — `T 00:00` needs 70 h,
`T 23:00` needs 93 h (origin `T − 2d 08:00`, 30 h publication lag). 96 h ≥ 93 h
worst case, with margin, and is a round 4 days.

**Gap handling:** the shift leaves the first 96 h of any series without a value →
**zero-fill** (matching the existing zero-fill convention for absent covariates and
preserving array length + the 3 homogeneous keys from
`2026-07-25-homogeneous-crossborder-covariates-design.md`).

**Accepted cost:** the model loses near-real-time flow information. V010's measured
accuracy is expected to **degrade** once this lands. That is the point — the current
number is optimistic; the post-fix number is trustworthy.

## Affected code

- **`src/chronos2/input_builder.py`**
  - New module constant `CROSSBORDER_SERVE_LAG_HOURS = 96` with a comment stating the
    derivation (30 h publication lag + D+2 horizon, 93 h worst case).
  - `_load_crossborder_flow_covariates()` — after computing the 3 aggregate hourly
    series, shift each forward by `CROSSBORDER_SERVE_LAG_HOURS` on the hourly index
    (`series.shift(96, freq="h")` semantics: the value observed at `t−96h` becomes the
    value presented at `t`). Still returns exactly the 3 keys, always.
  - Both call sites are unchanged (they consume the returned dict).
- No change to the covariate mapper, configs, or the target loader.

## Testing (TDD)

1. **Unit — lag applied:** seed flows at `t0` and `t0+1h`; assert the returned series
   presents those values at `t0+96h` / `t0+97h`, and that nothing appears at `t0`.
2. **Unit — no look-ahead:** assert that for any timestamp `t` in the output, the value
   equals the input flow aggregate at `t − 96h` (and that output before the first
   available lagged timestamp is absent/zero).
3. Existing homogeneity tests must still pass (3 keys always, empty country → 3 empty
   series).
4. **Integration:** re-run the V010 forecast and confirm it still produces 24 points
   per country and writes to the sidecar.
5. **Re-measure:** re-run `compare_experiments.py` for V010 **serve-faithfully** and
   record the new (expected worse, now honest) numbers alongside the old canonical
   ones in the verdict doc.

## Out of scope

- Re-training V011 under the corrected lag (V011 already lost by 11.7%; re-running the
  2 h fine-tune is not justified by this fix alone).
- Changing the forecast horizon or the 08:00 schedule.
- Backfilling/repairing historical `crossborder_flows` publication gaps.
