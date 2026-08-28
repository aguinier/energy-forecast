# ABL-246 — D+1 load-serving evidence pack: TSO vs ML, per country

**Author:** Forecasting Scientist · **Date:** 2026-08-28 · **Status:** evidence, no serving change

Machine record: `reports/abl_246_tso_d1_load.json`
Reproduce: `.venv\Scripts\python.exe scripts/abl246_tso_d1_load_pack.py --replica-db C:\Code\able\data\energy_dashboard.db --json-out reports/abl_246_tso_d1_load.json --csv-out <path>`

---

## 1. Headline

On genuinely as-issued vintages, **the TSO day-ahead load forecast beats our
production ML forecast in 20 of 23 evaluable countries**, readably, and usually
by a wide margin — fleet median WAPE **3.26% (TSO D+1) against 8.85% (ML D+2)**.

The finding that should worry us more is the baseline one: **our ML load
forecast readably loses to a D-7 seasonal naive in 10 of 23 countries and
readably beats it in 1.** That is a model-quality problem independent of
anything the TSO does, and it is the same conclusion ABL-128 §2 reached by a
different route.

**And the premise this issue was gated on does not survive.** ABL-246 was held
14 days because the pre-archive `energy_load_forecast` kept only post-revision
TSO values, which was expected to have flattered the TSO. Measured, that
optimism is **0.016 pp mean, 0.283 pp worst (GR), above 0.1 pp in 2 of 23
countries** — two orders of magnitude smaller than the TSO-vs-ML gap it was
supposed to threaten. TSO load day-ahead values are essentially never revised.
The ABL-128 §2 load numbers were not materially optimistic; they reproduce.

---

## 2. Protocol

| | |
|---|---|
| **Window** | target hours 2026-08-13 08:00 → 2026-08-28 00:00 UTC (16 target days, 14 complete) |
| **Sample** | **8,436** scored (country, hour) pairs; 338–353 per country |
| **Countries** | 24 (`config.SUPPORTED_COUNTRIES`); **23 evaluable**, NL excluded — §5 |
| **Source** | `forecast_vintage_archive` (ABL-184), `first_seen_at >= 2026-08-12` |
| **Basis** | **out-of-sample** throughout, except the one row explicitly labelled in-sample in §6 |
| **Truth** | `energy_load`, aggregated to hourly means |
| **Reads** | replica opened `file:...?mode=ro`; no write path touched |

**Arms**, all scored on one identical (country, hour) intersection:

- `tso_d1_last` — TSO day-ahead, latest vintage seen **before the target's local
  market day opened**. The honest D+1 arm.
- `tso_d1_first` — earliest such vintage.
- `tso_final` — last vintage ever seen, revisions included. Not available at
  D+1; stands in for the pre-archive read.
- `ml_d2` — our production forecast in the scorecard's registered **24–64h**
  D+2 band, latest leak-free vintage.
- `d7_naive` — actual load at target − 168h.

**The D+1 cut is a market day, not a UTC day and not a lead.** Two rules were
tried first and both are wrong, in opposite directions, and both would have
changed the answer:

- A **UTC-day** cut drops the target hours where the UTC day and the local
  market day disagree — 2 hours/day for every CEST country, 1 for PT, 3 for the
  EET fleet. Invisible in a per-country total.
- A flat **`lead >= 24h`** cut is worse. A day-ahead forecast published at 11:00
  on D-1 leads target hour D 00:00 by thirteen hours, not twenty-four;
  "day-ahead" names the delivery day, never a fixed lead. Requiring 24h deletes
  the early hours of every delivery day and keeps only the evening — DE fell
  from 22 scored hours/day to 8, all late — turning each per-country WAPE into
  an hour-of-day-biased statistic over a different hour set per country.

The market-day rule keeps all 24 hours (576 pairs/day on complete days) and is
exact. `MARKET_TIMEZONE` is declared per country with **no default**; an
unregistered country raises. It caught AL on the first run — the archive tracks
34 countries where our fleet is 24.

**The arms are not horizon-matched and cannot be.** TSO day-ahead leads its
delivery day by a median 13.9–15.9h here; our D+2 arm by 27.8h. That asymmetry
is the product difference this issue is about, and it runs **against** the TSO —
the safe direction for a result that favours it.

**Leads are lower bounds.** The archive stamps `first_seen_at` (when our poller
saw a value), never the TSO's own publication time, so every lead quoted here
understates how early the forecast existed.

---

## 3. Per-country results

WAPE %, lower is better. `Δ vs ML` is the paired **daily** WAPE difference
(TSO − ML) with a Student-t 95% interval over that country's target days;
negative favours TSO, **bold** where the interval excludes zero.

| Country | n | TSO D+1 | ML D+2 | D-7 | Δ vs ML [95% CI] | Recommendation |
|---|---|---|---|---|---|---|
| AT | 352 | 4.41 | 9.81 | **3.12** | **−5.74 [−10.64, −0.84]** | Serve TSO — D-7 caveat |
| BE | 353 | 3.55 | 7.03 | 4.97 | **−3.26 [−5.40, −1.12]** | Serve TSO |
| BG | 349 | 3.50 | 6.95 | 7.86 | **−3.31 [−5.52, −1.09]** | Serve TSO |
| CH | 352 | 3.26 | 6.91 | 7.49 | **−3.66 [−4.96, −2.36]** | Serve TSO |
| CZ | 352 | 1.68 | 7.96 | 2.14 | **−6.41 [−9.61, −3.20]** | Serve TSO |
| DE | 352 | 3.43 | 7.76 | 4.20 | **−4.39 [−7.06, −1.71]** | Serve TSO |
| EE | 345 | 12.22 | 10.26 | 11.11 | +2.14 [−0.48, +4.76] | **Hold** |
| ES | 353 | 2.01 | 9.60 | 5.10 | **−7.37 [−10.61, −4.12]** | Serve TSO |
| FI | 352 | 1.97 | 2.62 | 3.05 | **−0.70 [−1.39, −0.01]** | Serve TSO |
| FR | 352 | 2.51 | 6.92 | 4.55 | **−4.59 [−7.16, −2.02]** | Serve TSO |
| GR | 353 | 2.47 | 10.65 | 17.12 | **−8.11 [−12.85, −3.37]** | Serve TSO |
| HR | 353 | 3.34 | 10.15 | 9.08 | **−6.59 [−9.23, −3.94]** | Serve TSO |
| HU | 353 | 4.19 | 9.16 | 10.43 | **−4.72 [−7.54, −1.91]** | Serve TSO |
| IT | 353 | 3.26 | 10.33 | 10.99 | **−7.37 [−10.26, −4.48]** | Serve TSO |
| LT | 353 | 13.67 | 8.85 | 8.62 | **+4.50 [+1.14, +7.86]** | **Do not serve TSO** |
| LV | 338 | 6.33 | 10.20 | 6.84 | **−4.00 [−6.20, −1.80]** | Serve TSO |
| NO | 353 | 1.94 | 2.86 | **1.71** | −0.93 [−1.97, +0.12] | **Hold** |
| PL | 352 | 2.99 | 9.63 | 4.72 | **−6.86 [−10.45, −3.27]** | Serve TSO |
| PT | 352 | 1.69 | 7.50 | 4.86 | **−5.88 [−8.21, −3.54]** | Serve TSO |
| RO | 353 | 3.01 | 11.31 | 8.87 | **−8.14 [−12.75, −3.53]** | Serve TSO |
| SE | 353 | 2.52 | 5.91 | 3.63 | **−3.25 [−4.62, −1.88]** | Serve TSO |
| SI | 353 | 5.00 | 15.66 | 5.33 | **−10.32 [−15.61, −5.04]** | Serve TSO |
| SK | 352 | 2.33 | 7.84 | 3.15 | **−5.57 [−8.05, −3.08]** | Serve TSO |
| *NL* | *353* | *22.96* | *21.69* | *21.02* | *+0.50 [−4.03, +5.02]* | *Not evaluable — §5* |

**Tally (23 evaluable):** serve TSO 19 · serve with D-7 caveat 1 · hold 2 · do
not serve 1.

Fleet medians: TSO D+1 **3.26** · TSO final **3.26** · ML D+2 **8.85** ·
D-7 **5.10**.

---

## 4. The three negative findings

**4.1 Our ML load model loses to "same hour last week."** Paired daily,
ML − D-7 is readable in 11 of 23 countries and favours ML in exactly **one**
(GR, −6.60 pp). It readably loses in **10**: AT +6.96, CZ +5.96, DE +3.59,
ES +4.39, LV +3.39, PL +5.02, PT +2.71, SE +2.45, SI +10.07, SK +4.71 pp. The
remaining 12 are not readable over this window. A D+2 product is entitled to be
worse than a D+1 one; it is not entitled to be worse than a lag.

**4.2 The revision-optimism premise is refuted.** `tso_d1_last` vs `tso_final`
differ by a mean of **0.016 pp** across 23 countries, max **0.283 pp** (GR), and
exceed 0.1 pp in only 2. The 14-day archive gate was a reasonable precaution and
it bought a real answer — the answer is that the effect it guarded against is
not there for load. Note this is *revision* optimism specifically: the TSO does
issue multiple pre-delivery vintages (median 2–4 per target) and the later ones
are better, by up to 3.21 pp (LV) and 2.79 pp (HR). Using the *first* pre-day
vintage instead of the last would have understated the TSO.

**4.3 Even the TSO does not always beat the lag.** D-7 readably beats TSO D+1 in
AT (+1.22) and LT (+4.68). AT is flagged in the table for that reason: adopting
TSO there is still an improvement on what we serve, but it is not the best
available cheap forecast.

---

## 5. NL is not evaluable, and this is a finding

NL is the only country where the **truth series** fails a screen, and it fails
it decisively. Where the TSO forecast, our ML forecast and the D-7 baseline —
three independent predictors — all sit far above the actual, the suspect is the
truth, not three models at once. Counting hours with
`actual < 0.5 x min(all three predictors)`:

| | NL | every other country |
|---|---|---|
| orphan hours | 22 / 353 (**6.23%**) | **0 / 8,083 (0.00%)** |

The mechanism is behind-the-meter solar. NL's `energy_load` midday trough falls
to **0.17x** the country median and deepens monotonically with the solar year —
hour-12 monthly means run 12,817 MW (Jan) → 5,024 (Mar) → 3,332 (Apr) → 3,951
(Aug), with minima to 73 MW — while the TSO's own day-ahead forecast, our ML
forecast and D-7 all sit at 9–10 GW through it. `energy_load` is a net-of-BTM
series; the forecasts are not on that basis.

Consequence: **every NL load number this programme has published against
`energy_load` is measuring a basis mismatch, not forecast skill** — the
scorecard included. NL's 22.96 / 21.69 / 21.02 WAPEs above are uninterpretable
and must not be read as "NL is hard". Filed separately (§8); it is out of scope
here and it is not mine to fix.

---

## 6. LT and EE are level errors, and only one of them is recoverable

Both countries where the TSO does not win are **bias**, not shape:
LT −13.44%, EE +6.49% of mean load, against ±0.0–3.1% everywhere else.

Correcting each country's TSO series by its mean error over **prior days only**
(leak-free; first day drops out, hence n):

| | raw | debiased (causal) | debiased (**in-sample**) | n | ML D+2 | D-7 |
|---|---|---|---|---|---|---|
| LT | 13.67 | **8.81** | 8.86 | 337 | 8.85 | 8.62 |
| EE | 12.22 | **11.78** | 11.42 | 329 | 10.26 | 11.11 |

The in-sample column is an upper bound on the achievable gain and is **not** a
forecastable result; it is shown only because it brackets the causal one.

LT's TSO is recoverable — a trailing-mean correction removes 4.86 pp and brings
it to parity with our ML (8.81 vs 8.85), though still not past D-7 (8.62). EE's
is not: 0.44 pp, and it stays behind both. Across the other 21 countries the
same correction is neutral-to-harmful (BG 3.50 → 4.27, GR 2.47 → 2.80), so a
**fleet-wide** TSO bias correction is not warranted on this evidence.

---

## 7. Caveats

- **TSO is a D+1 product only. It cannot replace the D+2 forecast.** Nothing in
  this pack bears on D+2, where the TSO has no product to offer. Whatever is
  decided here, **the D+2 question stays open** and §4.1 says it is the more
  urgent one.
- **15 target days is short.** Every claim is carried by a paired daily
  interval over k = 14–16 days rather than by a WAPE point estimate, for that
  reason. The two `HOLD` verdicts are cells where 15 days is genuinely not
  enough, not cells where the answer is "no".
- **Contamination.** ABL-111/ABL-109 (zero-as-missing actual load): **1 row**
  in the whole window, dropped, immaterial. ABL-67 (fabricated net_position) does
  not touch load. **ABL-71** (prod ingest stale, fixes undeployed) touches the
  ingest that feeds both the archive and `energy_load`; it applies equally to
  every arm, so it cannot manufacture the TSO-vs-ML gap, but it is not
  quantified here.
- **The archive's own cap.** 17 genuine issued days exist; the 2026-08-11
  bucket (13.7M rows, target days back to 2018) is the go-live backfill of
  retained post-revision values and is excluded throughout. Confirmed
  independently — matches the CEO's 2026-08-28 operator check.
- **TSO plausibility guard** (ABL-431/458) ran on every archive read:
  **0 refusals** in this window.
- **21 of 24 countries serve CatBoost; AT, BE and FR serve XGBoost.** The issue
  says "our CatBoost D+2"; that is 21/24. The `ml_model` column in the JSON
  carries the per-country truth.
- Replica read-only throughout. No model was loaded, trained or promoted; no
  registry, serving path or ingest was touched.

---

## 8. What I recommend, and what I am not doing

**Recommendation to the CEO — for a Board decision, not for me to execute:**

1. **Serve the TSO day-ahead forecast in the D+1 slot for the 19 countries**
   marked *Serve TSO*, plus **AT** with the §4.3 caveat recorded. This is P5 and
   needs a pre-registered gate read plus a Board decision; **I am not promoting
   anything.**
2. **Hold EE and NO** pending a longer window. Re-read at ~30 target days.
3. **Do not serve TSO in LT** as-is. A trailing-bias correction reaches parity
   with our ML but not past D-7; LT is a modelling problem, not a sourcing one.
4. **Treat §4.1 as the priority, not this pack.** Adopting the TSO at D+1
   improves the D+1 slot and leaves a D+2 product that loses to a lag in 10 of
   23 countries. Sourcing the D+1 slot externally does not fix that.

**Follow-ups I am filing rather than doing here** (one issue = one landable
change):

- **NL load basis mismatch** (§5) — affects the scorecard and any published NL
  load metric, not just this pack. Ingest/scoring-truth question, not a model
  one.
- **D+2 load model vs D-7** (§4.1) — the ten readable losses, per country.

---

*Every number here is out-of-sample except the one column in §6 labelled
in-sample. Window, n, baseline and contamination are stated in §2 and §7.*
