# ABL-280 — RO net position vs climatology: interim re-read and the harness for the confirmatory one

**Status: INTERIM. Not the confirmatory read.** 7 scored vintages against the
pre-registered minimum of 14. Everything below is a flag, not a finding.

Measured 2026-08-12 against the live replica and sidecar, read-only
(`file:...?mode=ro`), rail interpreter
(`C:\Code\able\energy-forecast\.venv\Scripts\python.exe`). Nothing written to
either database; the only writes are report files.

| | |
|---|---|
| model | `chronos-2-V010` (the served champion) |
| cohort | vintages at or after `FIX_DEPLOYED_UTC` = 2026-08-04 14:29 UTC |
| targets | 2026-08-07 00:00 .. 2026-08-12 21:00 UTC |
| n | 166 pairs per country, 19 gate countries (LU/GR excluded by name) |
| out-of-sample | yes — every pair is a served vintage scored against later actuals |
| baselines | serve-faithful (`net_position.as_of_for_vintage`), 28-day climatology |

Reproduce with:

```
.venv\Scripts\python.exe scripts/reread_net_position_country.py --country RO --fleet \
    --replica-db C:\Code\able\data\energy_dashboard.db \
    --sidecar-db C:\Code\able\data\forecasts_local.db --stdout
```

## 1. The filing reproduces exactly, and the cohort has not moved

Every number in the ABL-280 filing reproduces to the decimal on today's
databases — n = 166, MAE 727.2 MW, WAPE 102.6%, skill −2.6% / −23.3% / +0.5% /
+20.6% against zero / climatology / ensemble / persistence, all six
per-vintage-day rows, bias sd 721.5 MW, pooled 0.501/0.397 and within-day
0.830/0.737.

That is not a second measurement. It is the *same* measurement: the scorable
cohort on 2026-08-12 is byte-identical to the one the issue was filed on. Two
further vintages have been generated (2026-08-11 and 2026-08-12) and neither
can be scored yet, because the rail generates at D for D+2 and their targets
have no published actuals. **No new evidence about RO exists today.**

## 2. Vintages that exist are not vintages that carry evidence

`net_position.build_gate_scope` counts vintages off the *left-merged* frame, so
a vintage whose targets have no actuals yet is still counted toward
`min_live_shadow_vintages`. Measured today:

| | count |
|---|---:|
| vintages present in the gate window | **9** (8 run-days) |
| vintages contributing >= 1 scored pair | **7** (6 run-days) |

The gap is not incidental and it is not going to close: the two newest vintages
are structurally unscorable at every moment the rail is running. So the
criterion reads two higher than the evidence behind it, permanently.

**Consequence, projected.** One vintage/day accrues from a 9-vintage base on
2026-08-12 (9 vintages over 8 run-days — 2026-08-06 carries both a 06:00 and a
10:52 run). The counted total therefore reaches 14 on **2026-08-17**, which is
the date `min_live_shadow_vintages` self-clears. On that date roughly **12**
vintages will carry scored pairs. Fourteen *scored* vintages land on
**2026-08-19**.

This module declines to inherit the ambiguity: `country_reread` counts scored
vintages and labels its own output INTERIM or CONFIRMATORY accordingly. It does
**not** change the gate — that is pre-registered, and moving it is neither a
tuning knob nor mine. Flagged to the CEO as an input to the ABL-34 read; the
decision is theirs.

Caveats on the projection: it assumes the daily 06:00 UTC job keeps producing
exactly one vintage per day with no further same-day re-runs, and that the
07:00 replica sync keeps landing before the ~10:45 UTC day-ahead publication.
A second same-day run pulls the counted date in without adding a day of
evidence — which is precisely why `vintage_days` is reported beside `vintages`.

## 3. RO is not alone, and that reframes the fallback question

The issue asked whether climatology should be RO's served fallback. The sweep
across all 19 gate countries says that question is not RO-shaped:

| country | WAPE | vs zero | vs climatology | vs persistence | vs ensemble | corr pooled | corr within-day | bias sd / mean\|actual\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RO | 102.6% | −2.6% | **−23.3%** | +20.6% | +0.5% | 0.50 | 0.83 | 1.02 |
| NL | 65.9% | +34.1% | **−18.3%** | +34.6% | +11.5% | 0.30 | 0.67 | 0.65 |
| BE | 21.8% | +78.2% | −2.4% | +34.2% | +14.9% | 0.62 | 0.75 | 0.17 |
| HR | 36.5% | +63.5% | −0.2% | +32.0% | +12.5% | 0.80 | 0.86 | 0.26 |
| CZ | 28.9% | +71.1% | +0.6% | +35.8% | +11.2% | 0.89 | 0.94 | 0.25 |
| EE | 25.6% | +74.4% | +4.8% | +7.8% | +2.7% | 0.84 | 0.91 | 0.29 |
| PL | 70.9% | +29.1% | +5.4% | +22.3% | +3.0% | 0.68 | 0.74 | 0.58 |
| DE | 35.9% | +64.1% | +7.3% | +32.7% | +11.8% | 0.93 | 0.95 | 0.25 |
| ES | 31.3% | +68.7% | +7.7% | +25.6% | +12.1% | 0.93 | 0.94 | 0.13 |
| AT | 61.1% | +38.9% | +7.9% | +41.6% | +25.3% | 0.72 | 0.76 | 0.46 |
| PT | 19.4% | +80.6% | +9.4% | +21.0% | +10.5% | 0.95 | 0.95 | 0.08 |
| FR | 42.0% | +58.0% | +14.1% | +5.6% | +2.2% | 0.87 | 0.89 | 0.23 |
| SI | 24.3% | +75.7% | +19.4% | +50.2% | +35.9% | 0.73 | 0.65 | 0.21 |
| FI | 84.3% | +15.7% | +19.8% | +41.0% | +27.2% | 0.58 | 0.71 | 0.81 |
| LV | 38.7% | +61.3% | +23.7% | **−1.5%** | **−3.5%** | 0.38 | 0.66 | 0.42 |
| LT | 62.1% | +37.9% | +29.2% | +43.6% | +30.7% | 0.79 | 0.77 | 0.19 |
| SK | 42.0% | +58.0% | +29.4% | +13.3% | **−0.1%** | 0.91 | 0.97 | 0.43 |
| BG | 34.4% | +65.6% | +29.9% | +54.9% | +42.3% | 0.90 | 0.91 | 0.24 |
| HU | 21.3% | +78.7% | +53.2% | +3.3% | +18.4% | 0.93 | 0.97 | 0.23 |

- Loses to **zero**: RO (1/19) — the original filing's headline, unchanged.
- Loses to **climatology**: RO, NL, BE, HR (4/19). BE (−2.4%) and HR (−0.2%)
  are inside noise at this n. **NL at −18.3% is not.**
- Loses to **persistence**: LV (1/19). Loses to the **ensemble**: LV, SK (2/19)
  — 17/19 = 89.5% beat it, against `GATE_BASELINE_COUNTRY_FRAC` = 0.80. The
  advance read recorded on the issue reproduces.

**The correction this forces on the issue's own framing.** ABL-280 proposed
"decide whether climatology should be the served fallback for RO". A per-zone
fallback for RO alone would leave NL — which loses to climatology by 18.3% and
is the subject of ABL-66 — served by the model it also loses with. Either the
question is asked for the class of zones or it is not asked; picking RO because
RO is the one that also crosses the (decision-irrelevant) zero line would be
choosing the zone by the wrong statistic. Recommendation only; serving is not
mine, and I am not recommending a fallback at n = 166.

## 4. The level/shape split is the same defect in three zones

Ranking by `within_day corr − pooled corr` — right profile, wrong level — the
zones that separate out are **RO (+0.33), NL (+0.37), LV (+0.28)**, and by
per-day bias spread relative to signal size, **RO (1.02), FI (0.81), NL (0.65),
PL (0.58)**.

That is one coherent cluster, not three unrelated zones, and it is directly an
input to ABL-65's correction-layer design: the zones where a *static* per-country
offset is provably insufficient are the same zones with the worst skill. RO's
bias swings +259 to −1095 MW across six consecutive vintage days; a constant
fitted on that history sits near the mean of the swing and leaves both bad days
bad. NL and LV show the same signature at smaller amplitude.

I will carry this into ABL-65 rather than duplicating it here, per the CEO's
triage.

## 5. Contamination

Checked on this cohort, not assumed:

- **ABL-35 / ABL-165 (fabricated exact-zero net position)**: 0 exact-zero
  actuals in the 3,154 scored pairs. GR and LU are excluded from the gate set
  by name (`GATE_EXCLUDED_COUNTRIES`), so ABL-67/ABL-181's rows are not in the
  scored set at all.
- **ABL-71 (replica lag)**: actuals are current to 2026-08-12 21:00 UTC, so no
  window truncation. The replica was synced 2026-08-12 07:34 UTC — before that
  day's ~10:45 UTC day-ahead publication, which is why the 08-11 and 08-12
  vintages are unscorable rather than partially scored.
- **ABL-109 / ABL-111 (zero-as-missing load)**: `energy_load`, not
  `net_position`. Does not touch this measurement.

## 6. What this branch adds

- `src/evaluation/country_reread.py` — the zero baseline as a named row
  (`skill_vs_zero < 0` is pinned as identically `WAPE > 100%`), the
  level-vs-shape split, evidence-vintage counting, and the fleet sweep.
- `scripts/reread_net_position_country.py` — the entry point. Read-only on both
  databases; writes only reports.
- `tests/test_country_reread.py` — 21 cases. The load-bearing ones: the
  zero/WAPE identity holds at every bias; within-day demeaning recovers an
  injected day-level bias exactly and a genuinely wrong shape does *not* improve
  on demeaning; the minimum-vintage precondition reads scored vintages, so nine
  present with seven scored fails a threshold of eight; a country with vintages
  and no actuals reports `no_paired_actuals` rather than a flawless zero.

Nothing here changes the promotion gate, the registry, the serving path, or any
model.
