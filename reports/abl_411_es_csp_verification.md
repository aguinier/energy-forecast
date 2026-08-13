# ABL-411 — ES overnight solar: the CSP reading is CONFIRMED, with one refinement

**Author:** Forecasting Scientist · **Date:** 2026-08-13 · **Status:** read-only
verification. No fit, no promotion, no change to `src/solar_clamp.py` or to the
`exclude_impossible_night` rule.

**Machine record:** `reports/abl_411_es_csp_probe.json`
**Tool:** `scripts/abl411_ree_solar_termica_probe.py`
**Replica:** `C:\Code\able\data\energy_dashboard.db`, 9,432,453,120 bytes, opened
`mode=ro`. Nothing was written to either database. The Red Eléctrica series was
fetched, compared and discarded to a scratch cache — no ingest path was added.

---

## 0. The question, and the answer

ABL-396 §3 argued that ES's overnight `solar_mw` is genuine concentrated-solar-power
(CSP) dispatch rather than contamination, and named its own falsifier: Red
Eléctrica publishes *solar térmica* separately from *solar fotovoltaica*, and if
that series does not account for ES's overnight MW the reading is wrong.

I fetched it.

> **Over 3,196 night hours, Red Eléctrica's own PV + CSP split accounts for
> 98.55% of the MW the replica books for ES when the sun is down.** Hour-by-hour
> MAE is **5.55 MW** against a mean night level of **263.5 MW**. The dominant
> term is CSP: **80.1%** of the annual night energy, rising to **91.4% in July**.

The CSP reading stands. ES's overnight solar is not contamination, and the two
mechanisms that would treat it as impossible would be destroying real generation.

**The refinement, which is a partial falsification and matters.** It is not
*all* CSP. REE's own *solar fotovoltaica* series reports **44–59 MW at sun
elevations of −40° to −49°**, where photovoltaics cannot generate. That accounts
for **18.5%** of ES's annual night floor. It is a TSO-side estimation artifact in
REE's published PV series, faithfully mirrored by ENTSO-E and by our ingest — not
a replica defect, but not physical generation either. §4 sizes it.

---

## 1. Protocol

| | |
|---|---|
| **Hourly source** | `demanda.ree.es/WSvisionaMovilesPeninsulaRest/resources/demandaGeneracionPeninsula?curva=DEMANDAQH`, 5-minute peninsular generation, fields `solFot` / `solTer` / `sol` |
| **Daily source** | `apidatos.ree.es` `generacion/estructura-generacion`, `time_trunc=day`, technologies `Solar photovoltaic` / `Thermal solar` |
| **Hourly window** | **2025-01-01 → 2025-12-10 UTC**, n = **8,220** joined hours, of which **3,196** night hours |
| **Daily window** | **2026-01-14 → 2026-08-10**, n = **209** days — the ABL-348 registered fit + gate pair |
| **Replica table** | `energy_generation.solar_mw`, country `ES`, resampled to hourly mean |
| **Night** | `src/solar_features.py::night_mask('ES', ...)` — sun below −8.0° for the whole hour at (39.125, −4.129). Bit-identical to the ABL-337 serving clamp's predicate and to ABL-396's |
| **Baseline** | None. This is a measurement against an external reference series, not a model comparison |
| **In/out of sample** | **Out of sample by construction** — nothing was fitted |

**Reproduction.** The probe was re-run from a **cold HTTP cache** — every REE day
re-fetched — and the JSON came back identical except for its own timestamp and
the `gate_window` block added on that second pass. Nothing here depends on a
cached response or on a one-off console session.

**Timestamps.** REE's `ts` is Europe/Madrid local. Two independent checks: the
response pads the requested local day by three hours each side (361 × 5 min =
30 h), and PV peaks at 14:00 local = 12:00 UTC. The autumn fold is disambiguated
in the label itself — REE writes the repeated hour as `2A:mm` (still CEST) and
`2B:mm` (CET) — so nothing is inferred from ordering.

**Perimeter.** REE's peninsular system and the ENTSO-E `ES` bidding zone are both
peninsular Spain; the Canary and Balearic systems are outside both.
`SOLAR_REPRESENTATIVE_POINTS['ES']` is central peninsula. §2 measures the
perimeter match rather than assuming it.

**Contamination touching these windows.** **ABL-188** (`energy_renewable`
zero-fill) is live and is why the headline runs on `energy_generation` — §6
reports the other table separately and, incidentally, confirms ABL-188 against an
external reference. **ABL-71 / ABL-67 / ABL-111** are net-position and load
defects and do not touch a solar actuals read.

---

## 2. The series are the same series

Before any per-hour claim, the alignment. Over all 8,220 hours:

| statistic | value |
|---|---|
| best lag, scanned ±3 h | **0 h** (MAE 277 MW at lag 0; 1,105 MW at +1 h, 1,631 MW at −1 h) |
| correlation, replica vs REE (PV + CSP) | **0.9974** |
| MAE | 277.1 MW on a **6,192.5 MW** mean, i.e. 4.5% |
| mean replica − mean REE | **−1.2 MW** on 6,192.5 MW |

So ENTSO-E's ES `solar_mw` **is** REE's `solFot + solTer`, hour-aligned, with no
perimeter gap worth naming. That is the aggregation claim ABL-396 inferred from
the B16 production-type definition, now measured.

**Reproduction check against ABL-396.** ABL-396 reported an ES night mean of
**599 MW in July** and **42 MW in December**, read off the aggregate. This probe,
independently, on the same table: **598.8 MW** (July 2025, n = 217 night hours)
and **42.2 MW** (December 2025 to the 10th, n = 119). Same series, same numbers.

I used `solFot + solTer` throughout, never REE's own `sol` field: `sol` agrees
with its components to rounding except on **2025-04-28**, the Iberian blackout,
where it disagrees by up to 3,002 MW across 18 hours.

---

## 3. The finding

Restricted to the **3,196 night hours** — the sun below −8° for the whole hour:

| | mean MW | share of replica night energy |
|---|---|---|
| **replica `solar_mw`** | **263.5** | 100% |
| REE *solar térmica* (CSP) | **211.0** | **80.09%** |
| REE *solar fotovoltaica* | **48.6** | **18.46%** |
| unexplained residual | 3.8 | 1.45% |

- **REE PV + CSP explains 98.55% of the replica's night energy.**
- Hour-by-hour residual: MAE **5.55 MW**, median **2.00 MW**, p95 **21.88 MW**.
- Correlation replica vs CSP at night: **0.984**. Against PV: 0.287.
- **18.79%** of REE's *entire annual CSP output* is generated at night, against
  **0.329%** of its PV output. CSP is a night-dispatching technology in the data,
  exactly as the storage hypothesis requires.

### The discharge curve, decomposed

ABL-396 read a monotone eight-hour discharge curve off the aggregate. Split into
its two components (2025 night hours, mean MW by hour):

| hour UTC | sun elev | replica | CSP | PV |
|---|---|---|---|---|
| 18 | −16.4° | 154.3 | 93.8 | 58.6 |
| 19 | −22.8° | 216.5 | 161.8 | 51.2 |
| 20 | −28.5° | 298.0 | 235.4 | 58.5 |
| 21 | −33.4° | **365.7** | **303.6** | 58.5 |
| 22 | −41.8° | 340.3 | 283.1 | 53.6 |
| 23 | −47.4° | 317.1 | 263.8 | 49.7 |
| 00 | −48.9° | 290.9 | 241.3 | 46.1 |
| 01 | −45.3° | 267.6 | 219.9 | 44.7 |
| 02 | −38.2° | 246.6 | 197.9 | 44.2 |
| 03 | −29.1° | 214.8 | 163.1 | 44.2 |
| 04 | −25.6° | 85.5 | 44.4 | 38.3 |
| 05 | −19.6° | 46.1 | 11.2 | 36.0 |

The evening ramp and the overnight decay are **CSP**, which rises from 93.8 MW at
dusk to a 303.6 MW peak and decays to 11.2 MW by 05:00 — a charge-and-discharge
shape. **PV is a near-flat 36–59 MW floor across the entire night**, including at
−48.9°. That flat component is the refinement in §4.

### It is not a stuck value

A forward-filled or stuck series would also look flat overnight, and ES's winter
nights *are* flat — 2026-03-06 holds 384 MW for five hours. So: of 344 nights in
2025, only **4** have a replica range ≤ 4 MW (the ES quantum). On those four,
REE's own PV + CSP has a mean range of **2.5 MW at a mean level of 22.6 MW**,
against the replica's **22.2 MW**. The flatness is the TSO's, not ours.

### Why the charge-coupling correlation existed

ABL-396's strongest single argument was a within-month detrended correlation of
**r = +0.515** over 585 days between a day's daylight energy and that same
night's energy. This probe reproduces the effect (**r = +0.649**, n = 343 days,
2025, `energy_generation`) and identifies its cause directly: the detrended
correlation between the replica's **night energy** and **REE's CSP night
energy** is **r = +0.966**. The night series essentially *is* the CSP series, so
it inherits CSP's charge coupling. The inference was correct.

---

## 4. The refinement: part of the floor is a TSO PV-estimation artifact

REE reports **44–59 MW of *solar fotovoltaica* at sun elevations of −40° to
−49°**, every night, all year. PV cannot generate there. This is REE estimating
non-metered distributed generation, and the estimate does not go to zero at
night. ENTSO-E folds it into B16 and our ingest mirrors it faithfully.

Its weight is strongly seasonal, because CSP output collapses in winter while
the PV floor does not:

| month | night hrs | replica MW | CSP MW | PV MW | CSP share | PV share | residual MW |
|---|---|---|---|---|---|---|---|
| 2025-01 | 357 | 57.1 | 31.5 | 25.4 | 55.2% | 44.5% | +0.2 |
| 2025-02 | 308 | 109.9 | 68.9 | 37.7 | 62.7% | 34.3% | +3.3 |
| 2025-03 | 302 | 158.9 | 113.2 | 42.7 | 71.3% | 26.9% | +3.0 |
| 2025-04 | 245 | 287.4 | 215.5 | 68.0 | 75.0% | 23.7% | +3.9 |
| 2025-05 | 227 | 465.5 | 382.4 | 78.1 | 82.2% | 16.8% | +5.0 |
| 2025-06 | 210 | 434.2 | 370.5 | 56.1 | 85.3% | 12.9% | +7.6 |
| **2025-07** | 217 | **598.8** | **547.3** | 46.6 | **91.4%** | **7.8%** | +4.9 |
| 2025-08 | 247 | 495.0 | 444.5 | 42.7 | 89.8% | 8.6% | +7.8 |
| 2025-09 | 284 | 404.3 | 343.0 | 54.4 | 84.9% | 13.5% | +6.8 |
| 2025-10 | 320 | 202.5 | 150.1 | 46.9 | 74.1% | 23.2% | +5.5 |
| 2025-11 | 360 | 99.8 | 38.6 | 61.1 | 38.7% | 61.2% | +0.2 |
| 2025-12 | 119 | 42.2 | 21.3 | 22.1 | 50.6% | 52.5% | −1.3 |

**Sizing it against ABL-396's headline.** ABL-396 measured **1.35%** of ES's
solar energy booked at night over the registered windows; this probe measures
**1.655%** over calendar 2025. Applying July's split — the right seasonal
analogue for a July-August gate window — the 1.35% decomposes to roughly

- **≈1.23 pp real CSP dispatch**,
- **≈0.11 pp REE PV-estimation floor**,
- **≈0.01 pp unexplained**.

So the correction to ABL-396 is real but small, and it does not change any
decision: the overwhelming majority of what the clamp would zero and the fit rule
would drop is genuine generation. **ABL-396's ranking does not need re-reading.**
ES stays where it is, and stays labelled "do not fix" — the label should read
"real CSP dispatch plus a small TSO PV-estimation floor" rather than "real
generation" flat.

---

## 5. The registered 2026 windows, at daily resolution only

**A limit, stated first.** REE's 5-minute archive serves **2021-01-01 through
2025-12-14**; from 2025-12-15 it answers `curva DEMANDAQH no valida` while the
sibling live endpoint still returns a current timestamp — an archive gap, not an
outage. **The ABL-348 registered windows cannot be reached at hourly
resolution.** Everything in §2–§4 is therefore measured on calendar 2025 and
carried forward on the argument that the ES CSP fleet did not change. The
coarser check below is what the registered windows themselves support.

Over **2026-01-14 → 2026-08-10**, n = **209** days, `energy_generation`:

- **Aggregation identity holds:** replica daily solar energy 187,445 MWh/day
  against REE PV + CSP 190,040 MWh/day, correlation **0.9989**, mean gap −1.8%.
- **Night energy fits inside the daily CSP budget on 189 of 209 days**, median
  ratio night/CSP **0.186** — i.e. about 19% of daily CSP energy is booked at
  night, matching the 18.79% measured hourly in 2025.
- **In the ABL-348 gate window (2026-07-11 → 2026-08-10, half-open — the
  registered fit window ends where this begins): 30 of 30 days**, mean ratio
  **0.185**, **max 0.261**, mean night 3,626 MWh against mean daily CSP
  19,434 MWh. So no day in the window comes within a factor of three of
  exhausting the CSP budget. Machine record:
  `daily_check[].gate_window`. **Both replica source tables agree to the digit
  here** — `energy_renewable` and `energy_generation` return identical ES night
  energy over these 30 days, so ABL-410's choice of scoring truth does not move
  ES's night floor in the window ABL-419 reads on. (They differ over the full
  209-day span, where ABL-188's zero-fill still bites: 207 usable days against
  209.)
- Detrended daily correlation of night energy against CSP **+0.668**, against PV
  **+0.272**.

**The 20 exceptions are a product artifact, and I can only mostly clear them.**
All 20 fall in January–March. Calibrating the daily `apidatos` series against the
5-minute archive where both exist (n = 344 days, correlation 0.9982) shows
`apidatos` under-reports CSP precisely in the low-output regime those days sit in:

| apidatos daily CSP | n days | median ratio archive / apidatos |
|---|---|---|
| ≤ 200 MWh | 5 | **3.17×** |
| 200–1,000 MWh | 34 | 1.09× |
| 1,000–3,000 MWh | 36 | 0.98× |
| 3,000–10,000 MWh | 103 | 0.99× |
| > 10,000 MWh | 166 | 0.99× |

So a winter day whose "CSP budget" looks exceeded by 53× is a day whose budget
was measured with the wrong instrument. That explains the cluster; it does not
*prove* those specific 2026 nights are clean, because the 5-minute archive does
not reach them. Given they are winter nights at 40–600 MWh — the low-amplitude
regime, far from any gate window — I am not proposing further work on them, but
they are the one thing this pack does not fully close.

---

## 6. Two side observations

**ABL-188 is confirmed against an external reference.** On the
`energy_renewable` overlap (2025-11-08 → 2025-12-10, 768 hours) that table runs
**786 MW below** REE all-hours (correlation 0.860), against `energy_generation`'s
**−6 MW** (correlation 0.9988). The cause is exactly ABL-188's zero-fill: 141
zero hours against 4, and the largest divergences are midday hours where
`energy_renewable` reads **0.0** while `energy_generation` and REE both read
~18,000 MW. REE now supplies the true values those zeros replaced, if anyone
wants them.

**A hypothesis, not a finding.** If REE's night PV floor is an estimation
artifact rather than something specific to Spain, other TSOs may publish the same
kind of floor, and part of the small night floors ABL-396 measured elsewhere
(EE 0.72%, and the ≤0.28% group) could be the same artifact rather than ingest
error. I have measured **one** TSO. Checking a second would be cheap — it is the
same probe pointed at a different published split — but it is not in this issue's
scope and nothing here supports the general claim yet.

---

## 7. Recommendation on scope (the CEO's call, not mine)

The question put to me was whether the ABL-337 clamp and ABL-376's
`exclude_impossible_night` should become **per-country registered properties**
rather than fleet-wide constants. My read:

**Yes — but as one property, with no default, gated on a measured criterion.**

1. **The warrant is a physical claim, and it is false for ES.** Both mechanisms
   rest on "the sun is down, so production is impossible". §3 shows that premise
   is wrong for ES by 263.5 MW on average and by 365.7 MW at the 21:00 UTC peak.
   A fleet-wide constant is not a conservative simplification here; it is a claim
   about physics that the data refutes for one country.

2. **One property, not two.** The clamp and the fit rule already share the
   `night_mask` predicate. If they take separate country lists they can disagree
   about which hours are dark-but-real, and a model could then be fitted on rows
   the clamp will later zero. Register one property and have both read it.

3. **No default value.** This repo has been bitten precisely here: ABL-376 and
   ABL-395 added registration tables that **default silently instead of
   aborting**, and a merge that misses one is textually clean. A fourth
   silently-defaulting property would be the same trap with a bigger blast
   radius, because the default that reads "night is impossible" is the one that
   destroys signal. Make registration mandatory and abort when a scope does not
   declare it.

4. **Gate it on a measured criterion, not an opinion.** ABL-396 already produced
   the right statistic — `pct_of_total_energy_at_night`, which is simultaneously
   the width of the band an all-hours read can sit in and a hard lower bound on
   any served forecast's WAPE. Registering the property should be mechanical:
   measure `f`, and where it is material, require the country to declare which of
   the two readings applies. That keeps the decision out of the reviewer's
   judgement.

5. **The blast radius today is one country, which argues for doing it properly
   once.** On ABL-396's ranking ES is the only country where the night floor is
   real generation. BG (4.98%) is a genuine defect and EE (0.72%) is small;
   everything else is ≤0.28%. So this is not a sprawling migration — it is one
   registered property with one non-default member. Special-casing ES in the
   clamp would be cheaper this week and would leave the physics claim wrong in
   the code.

**What I am not recommending.** Not that ES be exempted from the clamp by a
constant, not that ES enter a tranche — that stays blocked on the scope decision
regardless of this result — and not that anyone treat the 0.11 pp PV-estimation
component as worth a separate mechanism. It is inside the noise of the decision.

---

## 8. Limits

- The hourly evidence is **calendar 2025**; the registered ABL-348 windows are
  covered only at daily resolution (§5), because REE's 5-minute archive stops at
  2025-12-14.
- The 20 winter days where night energy exceeds the daily CSP budget are
  explained by a calibrated product difference, not cleared by direct
  measurement.
- REE's peninsular perimeter is assumed to equal the ENTSO-E `ES` bidding zone.
  The −1.2 MW mean gap on a 6,192.5 MW mean (§2) is the evidence for that; it is
  strong but it is an inference.
- Nothing here is a promotion recommendation, and nothing here changed
  `src/solar_clamp.py` or the `exclude_impossible_night` rule.
