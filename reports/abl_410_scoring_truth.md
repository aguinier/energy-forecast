# ABL-410 — one statement of the actual for the renewable family

**Verdict.** The scorecard's renewable-family scoring truth moves from
`energy_renewable` to `energy_generation`, matching what the dashboard has
published since ABL-399 merged (PR #30, 2026-08-13 20:05 UTC). Eight of the
fifteen live pairs do not move at all. `scorecard.py`'s strict
`hydro_run_mw + hydro_reservoir_mw` is replaced by the training-side null-aware
definition — imported, not restated.

**The finding that is not a reconciliation.** Belgium's `hydro_total` model is
not a hydro model. Its fitted target was run-of-river **plus folded pumped
storage**, and pumped storage is 84.7% of that target across the 22,641 hours
both tables carry (mean 110.91 of 130.91 MW) and 99.3% of it over the last
published window (144.68 of 145.66 MW). Scored against Belgium's actual
run-of-river it lands at **14,274% WAPE with a correlation of −0.12** — it is
anti-correlated with the quantity it is labelled as forecasting. That is not a
model that regressed; it is a model of a different series. Raised as a separate
issue, because the fix is a retrain-or-withdraw decision on a serving artifact
and this repo does not own serving.

---

## 1. Protocol

| | |
|---|---|
| Replica | `C:\Code\able\data\energy_dashboard.db`, 9,432,453,120 bytes, SQLite `mode=ro`, `uri=True` |
| Sidecar | `C:\Code\able\data\forecasts_local.db`, also `mode=ro` |
| Target window | 2026-07-11 00:00 → 2026-08-10 00:00 UTC (exclusive) — the last **published** scorecard window, so these figures sit directly beside `reports/forecast_scorecard/scorecard-2026-08-10.md` |
| Models | the `PRODUCTION_MODELS` registry snapshot, renewable family only |
| Selection | the scorecard's own: latest vintage per country + target + model + horizon band; top-of-hour actuals only; no aggregation, no interpolation |
| n | 2,760 scored pairs per pair-of-(type, country); **1,688 for FR** — see §5 |
| Sample basis | **common instants** — every column below scores the identical set of (country, target) pairs |
| In/out of sample | out of sample with respect to the stored forecasts (every one was issued before its target). The models' own fit windows are not known to this run, so no in-sample claim is made in either direction |
| Contamination | ABL-71 (prod ingest stale), ABL-67 (fabricated `net_position` rows), ABL-111/ABL-109 (zero-as-missing actual load) — **none touches this window's renewable-family actuals**: ABL-67 and ABL-111/109 are `net_position` and `energy_load` respectively, and neither table is read here. ABL-318 §3 *does* touch it, and is treated as a first-class caveat in §5 rather than as background |
| Reproduce | `scripts/abl410_scoring_truth.py`; raw output committed at `reports/abl_410_scoring_truth_measured.md` |

Read-only throughout. Nothing was fitted, corrected, promoted or deployed.

## 2. What actually changes

Common instants, so the two truth columns are the same sample.

| type | country | frozen (retired) | generation (adopted) | Δ | mean actual frozen → generation |
|---|---|---:|---:|---:|---|
| `solar` | BE | 23.42% | 23.42% | — | 2258.63 → 2258.63 |
| `solar` | DE | 62.37% | 62.37% | — | 17619.14 → 17619.14 |
| `solar` | FR | 21.49% | 21.49% | — | 6523.05 → 6523.05 |
| `wind_onshore` | BE | 192.95% | 192.95% | — | 450.01 → 450.01 |
| `wind_onshore` | DE | 61.48% | 61.48% | — | 8419.06 → 8419.06 |
| `wind_onshore` | FR | 127.14% | 127.14% | — | 2807.05 → 2807.05 |
| `wind_offshore` | FR | 108.66% | 108.66% | — | 425.04 → 425.01 |
| `biomass` | FR | 3.89% | 3.89% | — | 284.67 → 284.67 |
| `wind_offshore` | BE | 162.08% | 161.58% | −0.50 pp | 466.67 → 462.40 |
| `biomass` | BE | 69.39% | 69.76% | +0.37 pp | 202.12 → 207.38 |
| `renewable` | DE | 47.34% | 49.55% | +2.21 pp | 35046.46 → 33772.01 |
| `renewable` | FR | 26.04% | 32.82% | +6.78 pp | 14292.89 → 13497.81 |
| `renewable` | BE | 41.57% | 50.94% | +9.37 pp | 3628.27 → 3379.67 |
| `hydro_total` | FR | 15.00% | 28.70% | +13.70 pp | 4193.36 → 3458.02 |
| `hydro_total` | BE | 92.21% | **14274.51%** | ×155 | 145.66 → **1.26** |

**Eight of fifteen pairs are identical to the digit.** `solar`, `wind_onshore`,
`wind_offshore` and `biomass` are single columns carrying the same name in both
tables, and on this window they carry the same values. The whole divergence
lives in the two **re-derivations** — `renewable` and `hydro_total` — plus two
sub-percentage-point BE drifts where the frozen table's `REAL DEFAULT 0` stands
in for a measurement that is really negative (ABL-399 measured 3,895 such
offshore-wind pairs across full history).

So the issue's headline — "two surfaces, two WAPEs for one model" — is true, and
it is narrower than it looks: it is a disagreement about **what "renewable" and
"hydro" mean**, not a broad measurement discrepancy.

### 2a. The published pooled row moves for a second reason, and mostly that one

The scorecard's `horizon_band = all` row pools across countries and is
denominator-weighted, so it also moves when the *sample composition* changes.
The FR ingest gap (§5.1) drops 1,072 of FR's well-forecast hours out of the
pool, and FR is the best-forecast country in four of these six types. Decomposed
on the same window — the truth effect is the common-instant column, the rest is
composition:

| type | published before | published after | of which truth | of which composition |
|---|---:|---:|---:|---:|
| `solar` | 48.35% | 51.85% | **0.00 pp** | +3.50 pp |
| `wind_onshore` | 76.94% | 77.72% | **0.00 pp** | +0.78 pp |
| `wind_offshore` | 115.71% | 142.75% | **−0.22 pp** | +27.26 pp |
| `biomass` | 31.31% | 39.70% | **+0.62 pp** | +7.77 pp |
| `renewable` | 39.82% | 46.61% | **+3.64 pp** | +3.15 pp |
| `hydro_total` | 18.42% | 37.18% | **+18.03 pp** | +0.73 pp |

Read this before reading the new `latest.md` as a quality change. Solar and
onshore wind did not move at all as measurements; their pooled figures moved
because France left the pool. `hydro_total` is the only type whose pooled move
is essentially all truth. (The "after" column here is the scorecard's own output
— `scripts/abl410_scoring_truth.py` reproduces `latest.md`'s pooled WAPEs to the
digit, which is what makes the split above a decomposition rather than an
estimate.)

## 3. `scorecard.py:62` — a latent defect, and it changed no published number

The strict `hydro_run_mw + hydro_reservoir_mw` and the null-aware
`db._HYDRO_TOTAL_EXPR` return **identical results on every one of the fifteen
pairs**, to the digit (the `frozen strict` and `frozen null-aware` columns of
`reports/abl_410_scoring_truth_measured.md` are equal throughout). That is not
because the strict form is safe. It is because `energy_renewable` declares every
`*_mw` column `REAL DEFAULT 0`, so nothing in it is ever NULL and the
NULL-propagation the strict form was warning about cannot fire there.

Its comment — "COALESCE would turn a missing component into a fabricated zero" —
was correct in general and pointed at the wrong table. On `energy_generation`,
which has no such default, exactly one hydro component is 100% NULL for **9 of
the 24 supported countries** (`src/db.py:406`: BE/EE/FI/LT/LV/NL/SI report
run-of-river and never reservoir; GR/SE the reverse), and the strict form erases
all nine. Adopting `energy_generation` without also adopting the null-aware sum
would have silently dropped BE's `hydro_total` series entirely.

Fixed by importing `db.RENEWABLE_TYPE_COLUMNS["hydro_total"]` rather than
restating it, with a test asserting object identity, not equality. A copy is
what drifted.

## 4. Belgium's `hydro_total` model forecasts the store, not the river

The same stored forecast, scored against each component series on its own
(common instants, n = 2,760):

| country | leg | mean MW | WAPE | slope | corr |
|---|---|---:|---:|---:|---:|
| BE | frozen `hydro_run_mw` | 0.98 | 18333.94% | −7.712 | **−0.086** |
| BE | frozen `hydro_reservoir_mw` (the folded store) | 144.68 | 93.24% | 0.498 | **0.845** |
| BE | generation `hydro_run_mw` | 1.26 | 14274.51% | −9.024 | **−0.124** |
| BE | generation `hydro_reservoir_mw` | — | not measured | — | — |
| BE | generation `hydro_pumped_mw` | −74.14 | 97.55% | 0.242 | **0.722** |
| FR | frozen `hydro_run_mw` | 2326.02 | 89.89% | 3.810 | 0.720 |
| FR | frozen `hydro_reservoir_mw` | 1867.34 | 136.53% | 0.913 | 0.884 |
| FR | generation `hydro_run_mw` | 2326.02 | 89.89% | 3.810 | 0.720 |
| FR | generation `hydro_reservoir_mw` | 1132.00 | 290.18% | 2.045 | 0.872 |
| FR | generation `hydro_pumped_mw` | 6.39 | 337.74% | 0.829 | 0.838 |

`corr = 0.845` against the folded store, `corr = −0.086` against run-of-river.
The model tracks the pumped-storage leg and is mildly **anti-correlated** with
the river. Belgium reports no reservoir hydro at all — `COUNT(hydro_reservoir_mw)`
is 0 across all 49,213 of its `energy_generation` rows — so its frozen
`hydro_reservoir_mw` is the pumped-storage generation leg under another name.

The issue's per-row proof (BE `2026-01-14 08:00`, frozen reservoir 73.31 =
`energy_generation.hydro_pumped_mw` 73.31) reproduces, and generalises with one
correction worth stating: **exact equality holds on 5,550 of the 8,491 hours
where the store is net-generating, and on none of the 14,149 where it is net
pumping** — there the frozen column is ~0 (mean 0.65) while the generation
column is negative (mean −242.18). The frozen table carries the positive
generation leg; `energy_generation` carries the net, signed, per the A75 netting
that applies to every type in that table. Same conclusion, different mechanism:
the frozen column is pumped storage either way.

**Why the WAPE explodes rather than merely moving.** BE's run-of-river is a
genuinely tiny series and it is seasonal. `energy_generation` monthly means for
2026: Jan 26.03, Feb 15.03, Mar 43.36, Apr 21.05, May 25.36, Jun 13.85, **Jul
2.44, Aug 0.20**. WAPE is `Σ|e| / Σ|actual|`; keep the errors and shrink the
denominator from 145.66 MW to 1.26 MW and 92% becomes 14,274% arithmetically.
Both figures are correct. **Neither is a statement about forecast skill**, which
is why the scorecard now prints `mean actual` beside every WAPE it publishes.

## 5. Caveats that travel with every figure this change produces

1. **FR ingest gap (ABL-318 §3, open).** `energy_generation` holds no FR rows
   from 2026-06-30 23:45 to 2026-07-22 14:15 — **279 of the 720 hours** of the
   window above, which is why FR's n is 1,688 and not 2,760. Measured directly:
   distinct top-of-hour instants in-window are 720 (`energy_renewable`) against
   441 (`energy_generation`) for FR, and 720 against 720 for AT, BE and DE.
   Against this, over the 2025-10-01 → 2026-08-11 overlap era
   `energy_generation` covers 24,694 hours the frozen table does not. This is a
   specific open gap, not a coverage regression — but any FR figure in a window
   overlapping it is on a smaller sample and must say so.
2. **The models are still fitted on `energy_renewable`.** ABL-321's rejection of
   the *training* switch stands and is untouched;
   `db.RENEWABLE_TYPE_SOURCE_TABLE` remains `energy_renewable`. Where the two
   tables disagree about what the target *is* — `renewable` and `hydro_total` —
   part of the new WAPE is target mismatch rather than model error. The 2.21 to
   13.70 pp moves on `renewable`/FR-`hydro_total` are of that kind and should
   not be read as a quality regression; BE `hydro_total` is the same effect at
   its limit.
3. **This touches no promotion gate.** `ACTUAL_SPECS` is read only by
   `scorecard._load_actuals`. Both gate harnesses take their actuals from
   `RenewableFeatureBuilder._actuals` → `db.load_renewable_type_data`, which
   honours each artifact's own `training_source`. No registered gate scope, fit
   window or decision is disturbed, and nothing here is a promotion.
4. **Duplicate instants in the retired table.** `energy_renewable` holds 26,694
   duplicate instants, 26,400 of them with conflicting values, against 0 in
   `energy_generation` (ABL-399's measurement). The scorecard's
   `drop_duplicates(keep="last")` resolved those arbitrarily. Adopting
   `energy_generation` retires that non-determinism rather than fixing it in
   place.

## 6. Recommendation

1. **Adopted, in this change:** `energy_generation` as the renewable-family
   scoring truth, with null-aware sums, one definition per quantity, and the
   table printed in every scorecard under `## Scoring truth`. This is item 1 and
   item 2 of the issue. It is item 3's case as well: scoring truth and training
   source are independent post-ABL-331, and ABL-321's own decision window
   already took `energy_generation` as primary truth (`src/db.py:361`).
2. **Escalated, not decided here:** BE `hydro_total` currently serves a
   pumped-storage forecast under a hydro label. The options are a retrain on the
   corrected target or withdrawal of the pair; both are registry decisions and
   both belong to the CEO. Note that the honest BE run-of-river series averages
   0.20–43.36 MW depending on the month, so "retrain" should be weighed against
   "this country has no run-of-river fleet worth serving a model for". Until one
   of those happens, **no BE `hydro_total` WAPE against `energy_generation`
   should be quoted as a quality figure** — on either surface.
3. **Not proposed:** any change to `RENEWABLE_TYPE_SOURCE_TABLE`. ABL-321's
   measurement and the CEO's decision on it are unaffected by anything here.
