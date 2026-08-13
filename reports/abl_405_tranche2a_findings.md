# ABL-405 — ABL-316 tranche 2a: eight continental solar countries at 27 features

Parent **ABL-316**. Registration **ABL-348** (`experiments/ABL348/config.json`),
unchanged. Scope `abl316-t2a` = BG, CH, CZ, HU, PL, RO, SI, SK on
`energy_generation`. **8 countries × 3 primary D+2 bands = 24 cells.**

Machine record: `experiments/ABL348/results_abl405_tranche2a.json`.
Harness report: `reports/abl_405_solar_tranche2a.md`.
Margin read: `reports/abl_405_gate_delta.json`
(`scripts/abl405_gate_delta.py`).
Night-floor screen: `reports/abl_405_night_floor_probe.json`.

> **No promotion is requested or implied.** No serving-registry change, no write
> to `forecasts`, no ingest change, no dashboard change, no replica write. The
> replica was opened `mode=ro` throughout. Promotion is a CEO-to-Board decision
> and this pack is evidence for it, not a step in it.

---

## 1. Protocol, and what was verified rather than trusted

Everything registered — windows, bands, metric, baseline, minimum n, source
table — is **ABL-348's and is not restated here**. Thirty-three tranches must not
become thirty-three chances to shop a window. What follows is only what is new
or was checked.

| | |
|---|---|
| fit targets | 2026-01-14 → 2026-07-11 (exclusive) |
| gate targets | 2026-07-11 → 2026-08-10 (exclusive), out-of-sample by target timestamp |
| metric / baseline | WAPE / literal seasonal-naive D-7, recomputed on the same table |
| source table | `energy_generation` |
| algorithm | CatBoost, seed 42 (`config.random_seed`) |
| interpreter | `C:\Code\able\energy-forecast\.venv\Scripts\python.exe` — Python 3.14.3 |
| replica | `C:\Code\able\data\energy_dashboard.db`, **9,432,453,120 bytes**, `mode=ro` |

**The replica was verified by size**, not by path. 9,432,453,120 bytes is the
live replica ABL-348 registered its bars against. The 3.0 GB partial snapshot at
`energy-data-gathering/energy_dashboard.db` is the nearest file to every wrong
path this module has been pointed at, and its numbers look fine.

**The hold was verified by content on `origin/main`, not off the PR.** ABL-395
merged at `1bd99e5`. `src/evaluation/solar_retrain.py` splats
`SOLAR_GEOMETRY_FEATURES = ("sun_elevation_deg", "is_night")` into
`FEATURE_COLUMNS`, and `features_for("abl316-t2a")` resolves to **27 columns with
both geometry names present** — asserted programmatically before the first fit,
not read off the diff. That was the sole gate on re-tranching the remaining solar
pairs.

---

## 2. The registration: six tables, three of which default silently

`check_registration_tables(...)` at `scripts/evaluate_solar_retrain.py` checks
**three**. The solar harness has **six**. A merge that misses one of the other
three is textually clean and fails silently, so all six are stated here with
their values.

| table | checked? | this scope | value |
|---|:---:|---|---|
| `SCOPES` | yes | **registered** | `("BG","CH","CZ","HU","PL","RO","SI","SK")` → 24 cells |
| `SCOPE_OUTPUTS` | yes | **registered** | `experiments/ABL405/artifacts`, `experiments/ABL348/results_abl405_tranche2a.json`, `reports/abl_405_solar_tranche2a.md` |
| `GATE_BASIS` | yes | **registered** | `("challenger", "seasonal_naive")` |
| `FIT_RULES` | **no** | **registered explicitly** | `exclude_impossible_night: False` |
| `SCOPE_FEATURES` | **no** | **default taken** | `DEFAULT_SCOPE_FEATURES` — 27 columns, `feature_set = legacy25+geometry` |
| `SCOPE_TITLES` | **no** | **registered** | a title, not the bare slug |

Three notes on the ones that could have gone wrong quietly.

**`GATE_BASIS` is required, not preferred.** Measured on the live replica:
`forecasts` holds solar rows for **BE, AT, DE and FR only**. All eight tranche
countries are at **0 rows**. Under `abl253`'s four-way basis the `incumbent`
conjunct would empty every intersection, all 24 cells would score n=0, and the
harness would render `UNREADABLE` — a verdict on a comparison that never
happened. The incumbent is still reported on its own intersection, where it reads
*Not measured* by construction rather than by omission.

**`FIT_RULES` is stated, not inherited.** The value is the same as
`DEFAULT_FIT_RULES` would have given, but an absence in an unchecked table is
indistinguishable from an oversight. Off is right for two independent reasons:
the BG/CH cells here are a controlled A/B against ABL-381 on the **feature vector
alone**, and turning the rule on would move the fit frame at the same time as the
column list; and ABL-348's registration does not contain the rule.

**`SCOPE_FEATURES` takes the default, and that is the point of the hold.** 27
columns. The absence is commented in the table so it reads as a decision rather
than an omission, and the run records the resolved value three ways
(`meta.feature_set`, `meta.n_features`,
`meta.feature_set_is_registered_for_scope`, which is `false` here and prints as
such in the harness report).

### ABL-404 is adjacent, live, and not fixed here

`abl316-t1b` holds **no** `SCOPE_FEATURES` row either. Because its
`SCOPE_OUTPUTS` paths are the ones ABL-381 published at, a `--scope abl316-t1b`
run *today* refits at 27 and writes the result over ABL-381's dispositioned
evidence in place — and `.gitignore:56` matches on directory name, so the
artifact half of that leaves nothing in `git status`. That is **ABL-404**
(backlog, high, unassigned), raised by ABL-402.

**This tranche routes around it and does not close it.** The BG/CH re-read is a
new scope whose three output paths are disjoint from `abl316-t1b`'s — asserted
programmatically, and `test_no_two_solar_scopes_share_an_output_path` holds it.
`artifact_dir` is deliberately `experiments/ABL405/artifacts` and **not**
`experiments/ABL316/artifacts`: BG and CH appear in both scopes, so sharing that
directory would replace the 25-feature `BG/solar/model.joblib` whose SHA-256
ABL-381's machine record cites.

ABL-381's six evidence files were SHA-256'd before this run and re-checked after;
see §7.

---

## 3. The night floor, measured for all eight

ABL-396 — the fleet-wide overnight-floor screen — has **not landed** (`todo`), so
this tranche is not blocked on it. But `scripts/abl381_night_floor_probe.py` is
already on `origin/main`, is read-only and fits nothing, so the six new countries
were screened rather than merely declared unscreened.

This is load-bearing here rather than optional. CLAUDE.md is explicit: *screen a
country's night floor before reading its solar gate — the geometry pair is a
physical prior and is worth what its actuals' respect for that physics is worth.*
This tranche fits **with** that pair in the vector for the first time, so the
floor is the single best predictor of whether it helps or hurts.

Night is `solar_features.night_mask` — the sun geometrically below −8° for the
**whole** hour at the country's capacity-weighted point — so a non-zero night
actual is not a timezone offset or a mask artefact. Threshold 1 MW, ABL-338's.
Source `energy_generation`, the table this tranche fits on.

| country | window | night hrs >1 MW | night mean | night max | **share of energy at night** |
|---|---|---:|---:|---:|---:|
| **BG** | fit | 1,168 / 1,529 (**76.4%**) | 152.33 MW | 1,097.4 MW | **6.37%** |
| **BG** | gate | 179 / 210 (**85.2%**) | 245.71 MW | 1,087.9 MW | **4.98%** |
| CH | fit | 673 / 1,465 (45.9%) | 1.32 MW | 5.8 MW | 0.05% |
| CH | gate | 0 / 186 | 0.00 MW | 0.0 MW | 0.00% |
| CZ | fit | 2 / 1,411 (0.1%) | 0.01 MW | 6.8 MW | 0.00% |
| CZ | gate | 0 / 176 | 0.00 MW | 0.0 MW | 0.00% |
| HU | fit | 439 / 1,463 (30.0%) | 1.57 MW | 20.2 MW | 0.06% |
| HU | gate | 102 / 185 (55.1%) | 2.73 MW | 20.0 MW | 0.05% |
| PL | fit | 4 / 1,378 (0.3%) | 0.20 MW | 70.3 MW | 0.00% |
| PL | gate | 0 / 152 | 0.00 MW | 0.0 MW | 0.00% |
| RO | fit | 2 / 1,483 (0.1%) | 0.01 MW | 8.5 MW | 0.00% |
| RO | gate | 0 / 200 | 0.00 MW | 0.0 MW | 0.00% |
| SI | fit | 569 / 1,504 (37.8%) | 1.63 MW | 27.9 MW | 0.26% |
| SI | gate | 162 / 183 (**88.5%**) | 3.60 MW | 26.6 MW | 0.28% |
| SK | fit | 41 / 1,449 (2.8%) | 0.24 MW | 3.6 MW | 0.10% |
| SK | gate | 97 / 177 (54.8%) | 0.86 MW | 2.9 MW | 0.18% |

**The BG and CH rows reproduce ABL-381 §5 exactly** — every count, mean, max and
energy share to the published decimal. That is the control that makes the six new
rows worth reading.

Three findings.

1. **BG remains the only country with a floor that matters, by two orders of
   magnitude.** 152–246 MW mean, up to 1,097 MW, and **5–6.4% of everything BG
   books as solar is booked in the dark**. Nothing else here exceeds 0.28%.
2. **Count and energy disagree on HU, SI and SK, and energy is the honest axis.**
   SI reads 88.5% of gate night hours above 1 MW, which looks like BG's
   signature — but its night *mean* is 3.60 MW and its night energy share is
   **0.28%**. The 1 MW threshold is absolute, and these are small fleets (SI 332
   MW, SK 115 MW gate mean), so a floor of a few MW trips the count test while
   being physically negligible. Quoting the percentage alone would put SI and SK
   in BG's class, which the energy share refutes. **BG is contaminated; HU, SI
   and SK carry a trace floor; CZ, PL and RO are clean.**
3. **CZ, PL and RO are as clean as CH** — 0.0–0.3% of night hours, 0.00% of
   energy — so they are the countries where the geometry pair should behave as it
   did on CH rather than as it did on BG.

**A defect in the probe, for whoever picks up ABL-396.**
`scripts/abl381_night_floor_probe.py` crashes with
`ValueError: Out of range float values are not JSON compliant: nan` when a
(country, source, window) has **no night rows at all** — `night_mean_mw` is a
mean over an empty selection. It reproduces on CZ against `energy_renewable`,
where the fit window has no rows, which is the normal condition of the tranche
countries in that table (CLAUDE.md: 33 of 37 pairs have under 365 days there).
The script was written for BG/CH, which are full in both. Screening the fleet —
ABL-396's whole job — hits this on the first pass. **The `energy_generation`
screen above is unaffected**; only the second source arm was dropped, and this
tranche is registered on `energy_generation`.

---

## 4. The decision margin, and why it is two numbers

ABL-385 registers

```
delta_min(k) = 1.96 * sqrt(c_A^2 + c_B^2) / sqrt(k)
```

as the minimum **readable** relative gap, where `c` is a fit's per-seed CV. Every
number below is one fit per cell, so `k = 1`, and every margin is quoted as a
**percentage of the challenger's own error** — a gap in WAPE points is not
comparable across cells whose challengers score 8% and 25%.

The margin depends on what is being compared, and using one number for both is
the mistake this section exists to prevent:

| comparison | `c_B` | margin at k=1, fleet p90 CV 5.43% |
|---|---|---:|
| challenger vs **D-7, constant, climatology** | **0** — deterministic, they do not move when the challenger is refitted | **10.64%** |
| challenger vs **another fit** (the 27-vs-25 delta in §6) | `c_A` — both arms are fitted | **15.06%** |

Quoting 15.06% against a constant is not conservatism, it is the wrong test.

ABL-385 §4 measured the matched-seed correlation the independent form ignores at
**0.113** across 48 cells, which makes 15.06% mildly conservative rather than
wrong; the correlation-adjusted value is **14.18%** and is reported beside it
below. It is never used to promote a conclusion the independent margin refuses.

**Where a pair has its own CV, the fleet percentile is the wrong bar.** ABL-402
measured BG at 1.94–2.52% and CH at 2.53–3.02% per cell — near ABL-385's fleet
*median* and roughly half its p90, so the fleet number is **~2× too wide** on
these two. Their pair-specific margins are therefore reported as a secondary
read, with two caveats stated wherever they are used: ABL-402 is **on an open PR
(#47), not merged**, and its CV was measured on the **25-feature** challenger,
where every BG/CH cell below is a 27-feature fit.
