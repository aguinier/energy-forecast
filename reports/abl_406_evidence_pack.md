# ABL-406 — ABL-316 tranche 2b: the eight large-fleet `wind_onshore` pairs

**Evidence pack. Recommendation only — no promotion, no serving-registry change, no write to `forecasts`.**

Owner: Forecasting Scientist. Parent ABL-316. Registered under ABL-348
(`experiments/ABL348/config.json`), frozen; no window, band, metric, baseline,
minimum-n or source was changed by this read.

| | |
|---|---|
| Scope | `abl406-tranche2b` — ES, FI, GR, IT, NO, PL, PT, SE `wind_onshore` |
| Cells | 8 pairs × 3 primary D+2 bands = **24** |
| Algorithm | CatBoost (`ALGORITHMS["wind_onshore"]`), matching ABL-195 and ABL-348 |
| Fit window | 2026-01-14 → 2026-07-11 (exclusive) |
| Gate window | 2026-07-11 → 2026-08-10 (exclusive) — **out-of-sample** |
| Metric / baseline | WAPE vs literal seasonal-naive D-7 |
| Source table | `energy_generation` (ABL-348's registered source; **not** the harness default `energy_renewable`) |
| Gate basis | `(challenger, seasonal_naive)` |
| Replica | `C:\Code\able\data\energy_dashboard.db`, 9,432,453,120 bytes, `mode=ro` |
| Machine record | `experiments/ABL348/results_abl406_tranche2b.json` |
| Harness report | `reports/abl_406_wind_onshore_tranche2b.md` |
| Margins | `reports/abl_406_margins.json`, `reports/abl_406_margins_matched_cv.json` |

## Verdict

**FAIL — 16 of 24 cells pass.** A strict full PASS requires all 24.

That headline is the least informative number in this document. Sixteen passes
are not sixteen demonstrations of skill, and the eight failures are not eight
demonstrations of its absence. What follows is the qualification.

## 1. Per-cell gate table, with ABL-389's four model-free references

All WAPE %. `c` = constant, `k` = hour-of-day climatology; *causal* uses only the
fit window, *oracle* uses the gate window and is a hindsight bound no model could
have hit.

| pair | band | n | challenger | D-7 (bar) | c causal | c oracle | k causal | k oracle | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| ES | 24-36h | 720 | 54.27 | 41.04 | 62.07 | 41.49 | 55.14 | 27.54 | FAIL |
| ES | 36-48h | 720 | 54.18 | 41.04 | 62.07 | 41.49 | 55.14 | 27.54 | FAIL |
| ES | 48-64h | 510 | 52.43 | 38.55 | 63.93 | 44.23 | 53.09 | 26.11 | FAIL |
| FI | 24-36h | 711 | 41.12 | 59.60 | 54.63 | 53.47 | 53.41 | 51.40 | PASS |
| FI | 36-48h | 711 | 43.31 | 59.60 | 54.63 | 53.47 | 53.41 | 51.40 | PASS |
| FI | 48-64h | 504 | 46.17 | 54.86 | 51.95 | 51.52 | 50.72 | 49.90 | PASS |
| GR | 24-36h | 720 | 29.61 | 63.82 | 53.20 | 51.71 | 53.42 | 51.44 | PASS |
| GR | 36-48h | 720 | 29.58 | 63.82 | 53.20 | 51.71 | 53.42 | 51.44 | PASS |
| GR | 48-64h | 510 | 30.23 | 58.88 | 50.90 | 48.99 | 50.77 | 48.74 | PASS |
| IT | 24-36h | 716 | 71.39 | 70.64 | 92.00 | 52.32 | 90.71 | 45.09 | FAIL |
| IT | 36-48h | 715 | 71.19 | 70.65 | 91.86 | 52.29 | 90.59 | 45.08 | FAIL |
| IT | 48-64h | 505 | 66.84 | 67.23 | 82.85 | 50.76 | 83.26 | 42.42 | PASS |
| NO | 24-36h | 720 | 51.36 | 61.02 | 59.69 | 42.42 | 59.43 | 41.89 | PASS |
| NO | 36-48h | 720 | 51.56 | 61.02 | 59.69 | 42.42 | 59.43 | 41.89 | PASS |
| NO | 48-64h | 510 | 51.80 | 61.62 | 57.67 | 43.07 | 57.98 | 42.38 | PASS |
| PL | 24-36h | 720 | 54.10 | 92.76 | 61.15 | 51.19 | 59.73 | 47.40 | PASS |
| PL | 36-48h | 720 | 52.50 | 92.76 | 61.15 | 51.19 | 59.73 | 47.40 | PASS |
| PL | 48-64h | 510 | 51.35 | 94.44 | 63.91 | 52.27 | 61.53 | 48.23 | PASS |
| PT | 24-36h | 720 | 68.20 | 49.61 | 101.38 | 50.17 | 101.11 | 39.43 | FAIL |
| PT | 36-48h | 720 | 68.70 | 49.61 | 101.38 | 50.17 | 101.11 | 39.43 | FAIL |
| PT | 48-64h | 510 | 60.97 | 46.63 | 93.08 | 49.32 | 87.30 | 35.93 | FAIL |
| SE | 24-36h | 720 | 30.17 | 53.46 | 43.67 | 36.48 | 42.66 | 35.47 | PASS |
| SE | 36-48h | 720 | 30.25 | 53.46 | 43.67 | 36.48 | 42.66 | 35.47 | PASS |
| SE | 48-64h | 510 | 30.27 | 52.78 | 44.45 | 36.23 | 42.86 | 35.26 | PASS |

The incumbent reads **Not measured** in all 24 cells, correctly: re-measured
read-only on the live replica, exactly BE, AT, DE and FR carry
`renewable_type='wind_onshore'` rows in `forecasts`. None of these eight does.
That is why the gate basis is the two-column form — under the four-way basis all
24 cells would intersect to n=0.

**The climatology is the tighter reference, as on CH.** Across all 24 cells the
hour-of-day climatology beats the flat constant on both the causal and the oracle
side. Reporting only the constant would have understated the reference on every
cell — largest on PT 48-64h, where oracle constant 49.32 vs oracle climatology
35.93 is 13.4pp.

## 2. ABL-385 margins — is each comparison readable?

Every reference here is deterministic (D-7, a flat line, a climatology), so
`c_B = 0` and the published two-arm margin is a factor of √2 too wide. The gate
fits once at the pinned seed, so **k = 1**.

- **CV source:** ABL-385 §1, **wind** stream — pooled per-fit CV, p90 over 12
  (pair, algorithm, arm) units = **3.829%**. This is *not* the solar fleet
  number (5.43%); ABL-381 read its margins against a different stream's fits and
  that is the mistake being avoided here.
- **Cross-check:** restricted to the four units matching this challenger's stream
  *and* algorithm (wind_onshore / catboost) — AT 1.81%, DE 2.03%, FR 2.50%,
  BE 3.96%. Max 3.964%, against the fleet p90's 3.829%, so the fleet value is not
  being inflated by the offshore or xgboost units. Four units cannot support a
  percentile, so the maximum is used and is not called one.
- **δ_min at k=1 = 7.51%** of the challenger's own error (7.77% at the matched
  max). Read as two stochastic arms it would be 10.61% — **not used**.
- **Caveat, stated before being asked:** this is a *fleet* percentile over the
  served pairs, and **none of these eight is among them**. ABL-402 measured what
  that substitution costs on the only two pairs where both numbers exist: the
  fleet percentile ran **1.8–2.4× wider** than the pairs' own CV. So it is
  conservative in the direction that matters — a margin clearing it is readable;
  a margin failing it is *unresolved*, not absent, and needs a pair-specific CV
  to settle.

Margin vs the registered bar, as a percentage of the challenger's own error:

| pair | 24-36h | 36-48h | 48-64h | readable? |
|---|---:|---:|---:|---|
| ES | −24.38 | −24.26 | −26.47 | yes — readable **loss** |
| FI | +44.92 | +37.61 | +18.81 | yes |
| GR | +115.57 | +115.80 | +94.75 | yes |
| **IT** | **−1.05** | **−0.76** | **+0.57** | **no — all three below 7.51%** |
| NO | +18.81 | +18.34 | +18.96 | yes |
| PL | +71.44 | +76.68 | +83.90 | yes |
| PT | −27.26 | −27.78 | −23.52 | yes — readable **loss** |
| SE | +77.17 | +76.73 | +74.36 | yes |

21 of 24 comparisons against the bar are readable. The three that are not are
IT's, and they stay unreadable at the stricter matched CV.

## 3. The bar's own weakness, per pair

ABL-380 measured the mechanism on BG: a registered 93.75% D-7 bar cleared
outright by a causal constant at 82.77%, with no model. Applying the same test
here — is the registered D-7 bar looser than a causal constant?

| pair | D-7 bar | c causal | bar weaker than a flat line? |
|---|---:|---:|---|
| PL | 92.76 | 61.15 | **yes, by 31.6pp** |
| GR | 63.82 | 53.20 | **yes, by 10.6pp** |
| SE | 53.46 | 43.67 | **yes, by 9.8pp** |
| FI | 59.60 | 54.63 | **yes, by 5.0pp** |
| NO | 61.02 | 59.69 | **yes, by 1.3pp** |
| ES | 41.04 | 62.07 | no — bar is 21.0pp *tighter* |
| IT | 70.64 | 92.00 | no — bar is 21.4pp tighter |
| PT | 49.61 | 101.38 | no — bar is 51.8pp tighter |

**Stated plainly: for all five pairs that pass 3/3 — FI, GR, NO, PL, SE — the
registered bar is not what established the pass.** A flat line at the fit-window
mean would itself have cleared it. And the converse holds without exception:
every pair whose bar is *not* clearable by a constant (ES, IT, PT) failed or
tied. Across these eight pairs the gate outcome is fully predicted by the bar's
weakness, with no exceptions in either direction.

That does not void the five passes, and the bar is not re-opened after the fact.
It relocates the burden of proof onto the model-free references, which is what
ABL-389 exists for. Under that test the five separate cleanly:

- **FI, GR, SE beat all four references** — causal *and* oracle — in all nine of
  their cells. Their passes survive the bar-weakness objection outright.
- **NO and PL beat both causal references but lose to both oracle references.**
  Their passes rest on a bar a flat line clears, and a flat line at the
  gate-window median would have beaten them.

## 4. Pairs where a WAPE cannot carry a decision — report, do not decide

**IT.** All three margins against the bar are ±1.05% against a 7.51% floor. The
1/3–2/3 split across its bands is noise, not a boundary. IT is reported and not
dispositioned.

**NO — and this is the more serious of the two.** NO passes 3/3, with a readable
+18.8% margin against the bar. It is also **anti-correlated with its own target**:

| band | bias | slope | correlation |
|---|---:|---:|---:|
| 24-36h | +15.9% | −0.08 | **−0.14** |
| 36-48h | +17.3% | −0.07 | **−0.14** |
| 48-64h | +16.4% | −0.09 | **−0.16** |

A negative slope and a negative correlation in all three bands mean the model
carries no directional information about NO at all; it passes on level alone,
against a bar only 1.3pp looser than a flat line. This is not a marginal call —
it is the ABL-66 pattern (NL net position, slope 0.09, corr 0.22) with the sign
flipped. It is not a data artifact: NO's audit shows 34,176 of 34,176 intended
fit rows retained, 5,760 of 5,760 gate rows retained, zero excluded for a missing
actual or feature, and zero suspect constant runs.

**A gate pass on NO would be a false positive.** It is reported and not
dispositioned.

## 5. What the tranche actually diagnoses: a fit-to-gate level shift

The gate window (11 Jul – 10 Aug) is a summer wind lull, and the fit window
(14 Jan – 11 Jul) is not. Every one of the eight pairs falls in level between
them:

| pair | fit-window mean | gate-window median | change | challenger bias (24-36h) | D-7 skill |
|---|---:|---:|---:|---:|---:|
| PT | 1,558.7 MW | 692.6 MW | −55.6% | +46.2% | −37.5% |
| IT | 2,790.0 MW | 1,300.8 MW | −53.4% | +43.3% | −1.1% |
| ES | 6,318.9 MW | 3,861.5 MW | −38.9% | +50.0% | −32.2% |
| NO | 1,684.4 MW | 1,049.3 MW | −37.7% | +15.9% | +15.8% |
| PL | 2,180.5 MW | 1,458.1 MW | −33.1% | +37.0% | +41.7% |
| SE | 4,539.6 MW | 3,449.1 MW | −24.0% | −8.0% | +43.6% |
| GR | 1,367.8 MW | 1,151.6 MW | −15.8% | −5.1% | +53.6% |
| FI | 2,195.7 MW | 1,857.0 MW | −15.4% | −27.0% | +31.0% |

The ordering is close to monotone. Over the eight pairs, the level change ranks
against challenger bias at Spearman ρ = **−0.881** (p = 0.004) and against D-7
skill at ρ = **+0.833** (p = 0.010).

**Label: this is post-hoc and exploratory.** It was not pre-registered, n = 8,
and the eight are not independent draws — they share one European weather month.
It is a hypothesis for the next tranche to test, not a result this one
establishes. Stated as such, it is still the most useful thing in this read: the
dominant error mode is a **level shift the model does not track**, not a shape
failure. A CatBoost carrying a January-heavy wind level into a July lull predicts
high, and the three pairs whose level moved least (FI, GR, SE) are exactly the
three clean passes.

It also explains the oracle/causal gap without invoking model quality: the oracle
references are handed the new level for free, which is precisely the information
the causal ones lack.

**What this does *not* license.** A causal bias correction fitted on the fit
window could not have known the gate level either, so it would not have rescued
these cells. The honest follow-ups are a recency-weighted or rolling fit window,
and an explicit seasonal or recent-level feature — both of which change the
registration and therefore belong in a new pre-registration, not in this read.

## 6. Data and contamination caveats touching this window

- **ABL-188** (suspect constant runs) intersects the registered windows in five
  places — CZ solar, CZ `wind_onshore`, NL solar, EE solar. **None of the eight
  pairs in this tranche is among them**, and all eight audited at zero suspect
  constant runs in this run.
- **ABL-67** (fabricated net_position rows) — net position only; does not
  intersect. GR carries a provenance flag only.
- **ABL-109 / ABL-111** (zero-as-missing actual load) — load only; does not
  intersect.
- **ABL-71** (prod ingest stale, fixes undeployed) — the known wrong-write modes
  are load and net position. Carried as a provenance caveat, not as proof that
  wind ingest is pristine.
- **GR sub-hourly coverage.** GR's fit build aggregated 12,417 sub-hourly rows
  into 5,352 hourly means with **2,997 partial hours** — 56% of hours backed by
  fewer than the full 15-minute complement, against ≤8 partial hours for every
  other pair in the tranche. GR is the strongest result here (+94.8 to +115.8%
  margin), so this belongs beside it: the hourly means are computed from what
  exists and nothing is interpolated, but GR's series is materially thinner than
  its siblings' and the number should not be quoted without this line.
- **Source sensitivity of the bar.** ABL-348 records `bar_delta_pp` between the
  two source tables: IT +0.74pp, GR −0.52pp, PT −0.06pp, and 0.00 for ES, FI, NO,
  PL, SE. IT's is the largest, and IT is the pair already unreadable — a
  0.74pp bar movement is comparable to its entire ±1.05% margin.
- **Registered n shortfalls.** FI (711/711/504) and IT (716/715/505) score below
  the 720/720/510 the other six reach, from D-7-unscorable targets in the
  registration (`n_d7_scorable` 714 and 715 respectively). Both remain above
  ABL-348's registered minimum n.

## 7. Registration mechanics — and the asymmetry between the two harnesses

The wind harness has **exactly three scope-keyed registration tables** — `SCOPES`,
`SCOPE_OUTPUTS`, `GATE_BASIS` — and `check_registration_tables` at
`scripts/evaluate_wind_retrain.py` checks **all three**. A scope that registers
in two of them fails at import, not silently at read time. This tranche's entry
was committed at `d4aeeb1` **before the first fit**.

The solar harness has **six** scope-keyed tables — the same three plus
`FIT_RULES`, `SCOPE_FEATURES`, `SCOPE_TITLES` — and its
`check_registration_tables` call passes only the original three. The other three
default silently. This is deliberate and documented in the file: making a new
table required is a tax on every branch already in flight, and the comment cites
two live branches that would have produced a textually clean merge raising on
import.

Stating it rather than leaving it implied, as ABL-406 asks: **the two harnesses
are not interchangeable on this point.** On wind, "the scope is registered" means
all of it is. On solar it means three of six are, and the remaining three took a
default that nothing will tell you about.

ABL-395's contract test freezes the wind harness's 24-column `FEATURE_COLUMNS` in
`tests/feature_list_manifest.json` under `gate_harness`. The wind side has that
guard without ever having had the solar-side defect; `FEATURE_COLUMNS` was 24 at
this read and the manifest matches.

## 8. Recommendation

**No promotion, and none is proposed.** Promotion is a pre-registered gate read
plus a Board decision; this document is the first half only, and the gate reads
FAIL at the registered strictness.

Recommended dispositions for the CEO to take forward:

| pair | disposition | basis |
|---|---|---|
| **GR** | **strongest candidate** | 3/3, beats all four references, +94.8 to +115.8% margin, corr 0.78–0.81 — *carry the sub-hourly coverage caveat with it* |
| **SE** | **candidate** | 3/3, beats all four references, +74.4 to +77.2%, corr 0.56–0.61 |
| **FI** | **candidate, with calibration** | 3/3, beats all four references, but bias −27 to −33% and slope 0.32–0.39 |
| **PL** | **build and report** | 3/3 but on a bar 31.6pp looser than a flat line, and loses to both oracle references |
| **NO** | **report, do not decide** | passes 3/3 while anti-correlated (slope −0.08, corr −0.14) — a pass here would be a false positive |
| **IT** | **report, do not decide** | all three margins ±1.05% against a 7.51% floor; unreadable at either CV |
| **ES** | **negative result** | readable loss of 24–26% to the bar; still beats both causal references; bias +50% is the dominant defect |
| **PT** | **negative result** | readable loss of 24–28% to the bar; still beats both causal references; slope 0.16–0.23 |

Three of eight pairs (GR, SE, FI) produced a result that survives every check
applied here. Two (ES, PT) are clean negative results — a challenger that loses
is a finding. Three (NO, IT, PL) are cells a WAPE cannot decide, for three
different reasons, and are reported as such.

**Not a verdict on the TSO.** ABL-390 closed that: 34.2h maximum forward extent
against our 24–64h band. No TSO comparison is used as a benchmark anywhere above.
