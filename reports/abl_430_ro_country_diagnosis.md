# ABL-430 — RO is not the problem. Two different RO *inputs* are.

**Owner:** Forecasting Scientist. **Scope: diagnosis only.** No retrain, no
promotion, no gate re-read, no new registration. ABL-280 is untouched and stays
blocked on its own schedule.

**Verdict: the shared upstream cause the issue hypothesised does not exist.**
Every candidate it names — sign convention, timezone/DST offset, zone-code
mismatch, an actuals series that is not what we think it is — is measured and
ruled out on **both** targets. What is left is two *separate* input-side
deficits, one per target, and RO is a measured fleet outlier on each. The
coincidence that made this look like one hidden cause is real but shallow: RO is
bottom-of-fleet on the one input each model depends on, for two unrelated
reasons.

Reproduce everything below with:

```
.venv\Scripts\python.exe scripts/abl430_ro_country_diagnosis.py \
    --replica-db C:\Code\able\data\energy_dashboard.db --stdout
```

Machine record: `reports/abl_430_ro_diagnosis.json`.

| | |
|---|---|
| replica | `C:\Code\able\data\energy_dashboard.db`, 9,432,453,120 bytes, opened `mode=ro` |
| writes | none to either database; report files only |
| interpreter | `.venv\Scripts\python.exe` (the rail — Python 3.14.3) |
| alignment window | 2026-01-01 → 2026-08-11, n = 5,301–5,328 h |
| feature window | ABL-348's frozen fit+gate span, 2026-01-14 → 2026-08-10 |
| covariate window | V010's registered training span, 2023-01-01 → 2026-03-01 (27,720 h) |
| out-of-sample | n/a — nothing is fitted here. Every number is a measurement on stored data. |

A worktree has no `.env`, so `config.DATABASE_PATH` degrades to a bare
`\data\energy_dashboard.db`. `--replica-db` was passed explicitly and the
byte size above was checked against the live replica, not assumed — the 3.0 GB
partial snapshot in `energy-data-gathering` is the nearest wrong file to this
question and its RO rows stop in 2024.

---

## 1. Ruled out, on both targets

Each row is a test that a defect of that class would have failed.

| candidate | test | RO result | verdict |
|---|---|---|---|
| **local time / UTC offset / DST** | daily mass centroid of solar generation vs astronomical solar noon at RO's own capacity-weighted point | **−0.59 h**; fleet median −0.54 h, range [−0.86, −0.13], RO **rank 10 of 24**, z = −0.27 | ruled out |
| **time shift on wind** | lag scan, RO wind actuals vs ENTSO-E's independently published day-ahead forecast | **argmax lag = 0 h** (r = 0.9402); ±1 h gives 0.9347 / 0.9273 | ruled out |
| **time shift on net position** | lag scan, RO net position vs (total generation − load), assembled from two other tables | **argmax lag = 0 h** (r = 0.9058); ±1 h gives 0.8745 / 0.8414 | ruled out |
| **sign convention, wind** | OLS slope of the TSO publication on our actuals | **+0.912** | ruled out |
| **sign convention, net position** | OLS slope of net position on the energy balance | **+0.761** (positive = export, as for FR) | ruled out |
| **units / scale** | same two slopes; observed max vs `weather_location` fleet capacity | slope 0.912; mean 655 MW, max 2,576 MW against 5,484 MW of registered clusters | ruled out |
| **zone-code mismatch** | does RO net position stay consistent with RO's *own* generation and RO's *own* load? | r = **0.906**, 6th of 19 gate countries, above the fleet median (NL 0.59, AT 0.53, PT 0.43 are worse) | ruled out |
| **actuals not what we think** | native resolution, lag-1 autocorrelation, exact zeros, ABL-188 constant runs | 15-min native; **ac1 = 0.981** (wind) / 0.935 (net position); 0 exact-zero net-position hours; longest constant run **5 h**, against ABL-188's 24 h screen | ruled out |

The whole fleet sits near −0.5 h on the solar clock rather than 0.0, and that is
expected, not a finding: ENTSO-E stamps the *start* of a settlement interval, so
an hour's generation is booked at its opening edge and the mass centroid falls
half an hour early by construction. What matters is that RO sits inside that
distribution instead of 2–3 h away from it.

**A three-table identity is what closes the zone question.** A `net_position`
series belonging to a different bidding zone could not stay correlated at 0.906
with RO's generation and RO's load; nothing forces those three tables to agree
except them all being Romania.

### The caveat on the TSO reference, stated before it is used

`publication_timestamp_utc` on `energy_generation_forecast` is the **fetch**
time, not the publication time — a 2026-08-01 target carries 2026-08-07 — so
these `day_ahead` rows are revision-contaminated, exactly as ABL-348 records
(`tso_role: revision-contaminated context only`). **That bounds one reading and
not the other.** A revision cannot repair a sign flip, a clock offset, a unit
error or a zone swap, so the ruling-out above stands. It does mean the 24.05%
WAPE this series scores against our actuals is a *lower bound* on true
day-ahead error and **is not an achievability target**; ABL-390 separately
measured the TSO's forward extent at 34.2 h maximum, so it cannot reach the
24–64 h D+2 band at all.

### Contamination touching these windows

- **ABL-67** (fabricated `net_position` rows): RO's current rows are **byte-identical
  to `net_position_backup_abl67`** — 0 differing rows on the join. Nothing was
  rewritten for RO.
- **ABL-109 / ABL-111** (zero-as-missing `energy_load`): touches test A3, which
  reads `energy_load`. Filtered `load_mw > 0` so such rows are dropped rather than
  scored as zero. Measured for RO over the window: **0 zeros, 0 nulls in 21,200
  rows** — the filter removes nothing here.
- **ABL-188** (bit-identical constant runs): screened directly, longest RO run 5 h.
- **ABL-71** (prod ingest stale, fixes undeployed): a provenance caveat on the
  window, not proof the RO ingest is clean. Stated, not dismissed.

---

## 2. Wind: the challenger had one physical input for RO, and it is the fleet's
second-weakest

`wind_retrain.FEATURE_COLUMNS` is 24 names: **10 calendar**, **11 lag/rolling
transforms of the target itself**, and **3 weather columns**
(`wind_speed_100m_ms`, `wind_speed_10m_ms`, `temperature_c`). So a challenger can
know exactly two things about the physics — how the target repeats, and what the
country-mean wind speed is doing. Both were measured for all 18 registered
`wind_onshore` pairs on ABL-348's own fit window, against the challenger's
directional skill in the stored gate results.

| | Spearman vs mean challenger `corr` | vs mean challenger `slope` |
|---|---:|---:|
| **`corr(generation, country-mean wind speed 100 m)`** | **+0.705** | **+0.695** |
| `ac168` of the target | −0.315 | −0.278 |
| gate-window mean MW | +0.292 | +0.336 |
| `ac24` of the target | +0.049 | +0.156 |
| capacity-weighted fleet dispersion (km) | −0.010 | +0.065 |

n = 18 pairs. Spearman +0.705 is p ≈ 0.001; nothing else on the list is
distinguishable from zero.

**One country property predicts the gate challenger's directional skill, and it
is the quality of our weather covariate for that country.** The two pairs that
failed G4 are the two lowest on it:

| pair | `corr_ws100` | mean challenger `corr` | mean `slope` |
|---|---:|---:|---:|
| **NO** (ABL-406) | **0.329** | −0.148 | −0.078 |
| **RO** (ABL-417) | **0.366** | −0.056 | −0.021 |
| LV | 0.596 | +0.107 | +0.050 |
| … | … | … | … |
| GR | 0.725 | +0.799 | +0.666 |
| NL | 0.832 | +0.538 | +0.395 |

RO carries a second handicap NO does not: its target's own history is the least
informative in the fleet — **`ac24` = 0.196 and `ac168` = 0.085**, the lowest
`ac24` of the 18 (next is BG at 0.278; NO is mid-pack at 0.469). So for RO
*both* informative feature families are near-empty, while for NO only the weather
one is.

### That low persistence is real Romanian weather, not a data defect

This is the check that separates "RO's series is broken" from "RO's wind is
like that". The **TSO's independently produced forecast** for RO has
**`ac24` = 0.230** against the actual's 0.196 — two series from different
producers agreeing that Romanian wind barely persists day to day. The same
comparison for NO is 0.507 / 0.469 and for PL 0.513 / 0.463. And RO's actual is
physically smooth throughout (ac1 0.981, ac2 0.941, ac3 0.891), so this is
decorrelation at the synoptic scale, not noise in the ingest.

### "Anti-correlated" overstates what was measured — and the D-7 baseline is the anti-correlated one

Fisher z of each correlation against its own null:

| band | n | challenger `corr` | z | D-7 `corr` | z |
|---|---:|---:|---:|---:|---:|
| 24-36h | 720 | −0.015 | **−0.41** | −0.204 | **−5.53** |
| 36-48h | 720 | −0.046 | **−1.24** | −0.204 | **−5.53** |
| 48-64h | 510 | −0.108 | −2.44 | −0.147 | −3.33 |

**Two of the challenger's three bands are indistinguishable from zero.** Only
48-64h is readably negative. What *is* readably negative in all three bands is
the **registered D-7 baseline**, at roughly four times the challenger's z — while
persistence (+0.008 to +0.155) and both climatologies (+0.134 to +0.159) stay
weakly positive. The correct statement
is not "the model is anti-correlated with its target" but **"on RO in this
window the entire persistence family is anti-informative, and the challenger —
which is mostly built from it — lands at zero."** G4 caught something real; it
just is not a defect of the model.

### The intuitive explanation for RO's weak weather covariate is wrong

`weather_data` holds **one series per country** — for RO a 48-point national
aggregate — while RO's fleet is 46% in Dobrogea on the Black Sea coast and 44%
inland. The obvious hypothesis is that a national mean averages decorrelated wind
regimes. **Measured and not supported:** capacity-weighted fleet dispersion has
Spearman −0.010 against challenger skill. ES (296 km spread) and SE (298 km)
score 0.622 and 0.636, while single-cluster CH, BG and HU score 0.378, 0.475 and
0.505. RO at 170 km is mid-pack on dispersion and second-worst on skill. The
covariate is weak for RO; *why* it is weak is not fleet geometry, and this
diagnosis does not claim to have found the reason.

---

## 3. Net position: RO is the fleet minimum on the one flow leg the loader reads

`src/chronos2/input_builder._load_crossborder_flow_covariates` queries
`country_from = ?` **only**. That is a documented defect (ABL-28, in the
function's own docstring): `flow__total_import_mw` is a constant zero for every
country and `flow__net_mw` duplicates gross export. Fleet-wide it was A/B'd at
0.8% of MAE and filed rather than fixed.

The question this diagnosis asks is what it costs a country whose **outbound**
legs are the sparse ones. Coverage over V010's registered training span:

| country | net_position | da_price | tso_load_fc | weather | **xb outbound (READ)** | xb inbound (not read) |
|---|---:|---:|---:|---:|---:|---:|
| **RO** | 100.0 | 96.7 | 6.0 | 98.5 | **65.7** | 100.0 |
| SK | 100.0 | 97.2 | 6.0 | 98.5 | 68.5 | 100.0 |
| FR | 100.0 | 98.1 | 99.9 | 98.5 | 84.1 | 100.0 |
| LV | 100.0 | 97.3 | 6.0 | 98.5 | 89.4 | 94.7 |
| *12 of 19 others* | 100.0 | 91–100 | 6 / 99.9 | 98.5 | **100.0** | 100.0 |

**RO is the fleet minimum on exactly one input family, and it is the one the
net-position covariate loader reads.** It is unremarkable on all five others.
(The `tso_load_forecast` 6.0% and the >90% price figures are shared across many
countries — not RO-specific, and not this issue.)

Three separate holes make up that 65.7%:

1. **Six entire months with zero outbound rows** since 2025-01 — 2025-01,
   2025-10, 2026-01, 2026-02, 2026-04, 2026-06 — plus 2026-07 partial (317 of
   744 h). Against HU 2 zero months, BG 1, CZ 0, PL 0.
2. **Structural asymmetry.** RO→BG / RO→HU / RO→RS / RO→UA hold ~20,320 rows
   each over the full history; BG→RO and RS→RO hold 30,949 and 30,205.
3. **Three physically existing interconnectors never appear at all:**
   `RO→MD`, `MD→RO`, `UA→RO`. Romania has been Moldova's principal electricity
   supplier since 2022, so a continuous several-hundred-MW export leg is
   invisible to the covariate while being fully present in the `net_position`
   target it is meant to explain.

Measured on the 557 hours where every leg we *do* hold is present, RO's net
position agrees with its own flow sum at r = 0.843 with a **+329.2 MW mean
residual** — the right sign and the right order for a missing Moldova export.

**How much of ABL-280 this explains, stated honestly.** The residual's
**day-to-day spread is 256.5 MW** (24 days, range −207 to +760 MW) against the
**721.5 MW** per-vintage-day bias sd ABL-280 measured on RO. That is a
contributor of the right shape — an unobserved, day-varying component of exactly
the quantity being forecast — but it is about a third of the bias in sd terms and
roughly an eighth in variance. **It does not account for ABL-280 on its own, and
this pack does not claim it does.**

**And it is a training-span defect, not a serving-time one.** In the ABL-280
scored cohort (2026-08-01 → 08-14) RO's outbound coverage is **96.8%, exactly
the fleet median**. So the model was fitted with a flow covariate that was blank
for months at a time, and is now served one that is populated — a train/serve
mismatch rather than a missing input at inference. Those are different repairs
and the distinction matters to whoever picks this up.

---

## 4. What this changes, and what it does not

**Nothing here moves a gate, a registration, a model or a served forecast.** The
ABL-417 and ABL-406 packs stand as written; RO's B grade and NO's B grade are
correct readings of what was measured. What changes is the *interpretation*:

- **G4 is working.** It caught two pairs whose challengers carry no directional
  information. But on RO it is not evidence of a broken model — it is evidence
  that neither of the challenger's two physical feature families says anything
  about Romanian wind, which is measurable **before** fitting anything.
- **`corr(generation, country-mean wind speed)` is a pre-fit screen.** At
  Spearman +0.705 over 18 pairs it predicts challenger directional skill better
  than fleet size, target persistence or fleet geometry. It costs one query.
  Whether to register it as a precondition on future tranches is a
  pre-registration question, not a change to make after seeing this result — I
  am not proposing it as a bar here.
- **Do not quote "RO is anti-correlated with its target"** without the z column.
  Two of three bands are noise around zero; the readably anti-correlated series
  in that cell is the registered D-7 baseline.

## 5. Two upstream items, reported and not fixed

Both are ingest/pipeline territory. Per the issue's own instruction — *"if the
cause turns out to be upstream data, report it and stop"* — I am reporting them
and stopping. Neither is mine to fix and neither is actioned here.

1. **`crossborder_flows` is missing RO's Moldova legs entirely** (`RO→MD`,
   `MD→RO`) and `UA→RO`, and drops RO's outbound legs for whole months at a time
   (6 of the last 20). This is the largest per-country coverage hole in the gate
   set.
2. **`_load_crossborder_flow_covariates` reads only the outbound leg.** Already
   filed as ABL-28 and correctly costed at 0.8% of fleet MAE — but that average
   hides that the cost lands on whichever country has asymmetric leg coverage,
   and today that is RO by a wide margin. Reading the `country_to` leg would
   both fix the constant-zero import series and make RO's coverage 100%.

Neither is a forecasting-model change, and I have not made one.
