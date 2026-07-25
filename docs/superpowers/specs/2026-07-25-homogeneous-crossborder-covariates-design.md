# Homogeneous Cross-Border Covariates — Design

**Date:** 2026-07-25
**Status:** Approved (design review with Guillaume)
**Repo:** energy-forecast

## Problem

Chronos-2 **fine-tuning** (V011, net_position) trains all countries as one global
model and requires **identical covariate keys across every series**. The current
cross-border covariate loader emits **one covariate per neighbour**
(`flow__DE`, `flow__FR`, …), so the key set differs by country (AT: CH/CZ/DE/HU/IT/SI;
BE: DE/FR/GB/NL) — and even by timestamp within a country, since the neighbour set
present in the data varies over the window. Fine-tuning aborts with:

```
ValueError: All past_covariates must have same keys … Heterogeneous lists are not supported.
```

V010 **zero-shot** works only because it infers one country at a time (heterogeneity
never matters). Net-position forecasting is therefore live on V010, but the fine-tuned
V011 cannot train.

## Decision

Replace the per-neighbour covariates with a **fixed set of 3 aggregate features**,
computed per hour from the same `crossborder_flows` query, identical for every country
and every time window:

| Feature | Definition | Notes |
|---|---|---|
| `flow__total_export_mw` | Σ max(flow_mw, 0) | gross export across all borders |
| `flow__total_import_mw` | Σ max(−flow_mw, 0) | gross import across all borders |
| `flow__net_mw` | Σ flow_mw | net physical exchange (= export − import) |

**Sign convention (verified against the data):** in `crossborder_flows`, a row
`(country_from=C, country_to=N, flow_mw)` with `flow_mw > 0` means physical flow **from
C to N** (C exporting to N). Confirmed: BE @ 2026-07-24 12:00 had positive flows to GB/DE
(exporting) while `net_position` was −5223 MW — i.e. the physical-flow sum and the
market `net_position` target are genuinely distinct signals, so these features carry
independent information and are **not** a leak of the target.

These are **past covariates only** (never future) — unchanged from today; future
cross-border flows are unknown at forecast time.

## Scope

Apply **globally**: both V010 (zero-shot) and V011 (fine-tune) use the 3 aggregate
features. Rationale: homogeneous everywhere, V010/V011 directly comparable in the
evaluation, and simpler than a dual-mode loader. **Side effect:** V010's covariates
change, so its next scheduled run (`able-net-position-forecast`, 08:00) uses aggregates
instead of per-neighbour — an acceptable, arguably more robust change.

## Affected code

- **`src/chronos2/input_builder.py` → `_load_crossborder_flow_covariates()`** — the only
  function that changes. Same SQL (`SELECT country_to, timestamp_utc, flow_mw FROM
  crossborder_flows WHERE country_from=? …`); instead of `groupby(country_to)` →
  `flow__{neighbour}`, pivot to an hourly frame and compute the 3 aggregate series.
  Returns `{flow__total_export_mw, flow__total_import_mw, flow__net_mw}`.
- **Both call sites** (past-covariate loading, ~line 513 and ~line 639) iterate
  `flow_dict.items()` and add each key — so they adapt **unchanged**; only the loader's
  returned keys change.
- **`src/chronos2/covariate_mapper.py`** — the `{"source":"crossborder_flows",
  "cov_name":"crossborder_flows"}` spec is unchanged (the expansion happens in the loader).

## Data flow

Unchanged except inside the loader: query per-`country_from` rows → build an hourly
`timestamp × neighbour` frame → per-hour reductions (sum of positives, sum of negatives'
magnitude, plain sum) → 3 hourly series → aligned to the past index by the existing
`_align_to_index`. Missing neighbours simply don't contribute to the sums.

## Error handling

- **No cross-border data for a country/window** (empty query): the loader MUST still
  return all 3 keys as all-zero series — **never an empty dict**. This is the crux: a
  country returning `{}` (as today) would reintroduce the exact heterogeneity we are
  removing, so homogeneity must hold even for countries with no flow data.
- NaNs from resampling gaps → filled with 0 (no flow observed).
- **Known limitation (deferred, YAGNI):** the loader always returns 3 keys, and
  the inference path (`build_for_country`) zero-fills empty series. The training
  path (`build_training_input`) currently *skips* empty series, so a trained
  country with **zero** cross-border rows over the whole window would reintroduce
  heterogeneity. Every country currently in the net_position training set has
  flow data (validated by the all-countries smoke), so this does not occur today;
  if it ever does, the fine-tune fails **loudly** with the same homogeneity error,
  and the fix is a 2-line zero-fill in `build_training_input` mirroring
  `build_for_country`.

## Testing

1. **Unit test** `_load_crossborder_flow_covariates` against a temp SQLite DB seeded with
   mixed-sign flows and a neighbour present in only some hours: assert exactly the 3 keys,
   and correct per-hour export/import/net values; assert an empty-data country still
   returns 3 all-zero series.
2. **Smoke** re-run: `train_chronos2.py --experiment V011 --types net_position
   --countries all --steps 5` → must get past the homogeneity check and save a checkpoint.
3. **Full** fine-tune (5000 steps) + **evaluate** persistence vs V010 vs V011 (both now on
   aggregate covariates) via `compare_experiments.py`; run the `forecast-promotion-gate`
   before claiming V011 wins; conditionally promote (`run-net-position.ps1` V010→V011).

## Out of scope

- Net-position display in the dashboard UI (separate, still deferred).
- Changing the `net_position` target or the other covariates (weather, price, TSO load).
- Per-country fine-tuned models.
