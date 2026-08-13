# ABL-396 — Overnight generation floor: all 24 solar countries screened

**Author:** Forecasting Scientist · **Date:** 2026-08-13 · **Status:** read-only screen, no fit, no write

**Machine record:** `reports/abl_396_night_floor_screen.json`
**Tool:** `scripts/abl381_night_floor_probe.py --countries all` (ABL-381's probe, extended — not a second probe)
**Replica:** `C:\Code\able\data\energy_dashboard.db`, 9,432,453,120 bytes, opened `mode=ro`.
`net_position` reaches 2026-08-14 21:00, i.e. a live day-ahead mirror, not the 3.0 GB decoy.

> **Windows.** ABL-348's registered pair, unchanged and not selected here:
> fit `2026-01-14 → 2026-07-11`, gate `2026-07-11 → 2026-08-10`.
> **Night** is `solar_features.night_mask` — the sun below −8° for the *whole*
> hour at the country's capacity-weighted point — i.e. bit-identical to the
> predicate the ABL-337 serving clamp uses. **Threshold** is 1 MW, ABL-338's,
> so counts are directly comparable to the FR pack.
>
> **Contamination touching this window:** ABL-188 (`energy_renewable`
> zero-fill) is live and is why every number is reported per source table and
> the attribution runs on the two tables' common index — see §4. ABL-71 /
> ABL-67 / ABL-111 are net-position and load defects and do not touch a solar
> actuals read. All figures are **out-of-sample by construction**: nothing was
> fitted here.

---

## 1. Headline

**BG is not typical — it is an outlier by a factor of 3.7, and the class it
belongs to is smaller than feared.** Of the 19 ABL-348 solar candidates plus the
4 incumbents and NO, exactly **three** carry a night floor big enough to move a
gate read by more than half a WAPE point, and **one of those three is real
generation, not a defect**.

1. **BG 4.98%** — confirmed, and still the worst by a wide margin.
2. **ES 1.35%** — the largest new finding, and §3 argues it is **genuine
   concentrated-solar-power dispatch**, not contamination. Do not "fix" it.
3. **EE 0.72%** — a real, small, persistent floor.

Everything else is at or under **0.28%**. **All six of ABL-405's new countries
(CZ, HU, PL, RO, SI, SK) are in that clean group** — see §5, which is the
operational answer that issue is waiting on.

A fourth finding the issue did not anticipate: **NL's night series is uniformly
*negative*** — 1,544 of 1,544 night hours in the two windows, −1.47 to −0.12 MW.
Different defect, different owner (§6).

---

## 2. What the ranking number is, and why it is rankable

Scope item 4 asks which countries a floor *would* move a gate read for. That is
answerable model-free, and the answer is a single column.

Let `f` be the share of a window's total |energy| booked at night, and `W` the
daylight-only WAPE of any challenger. Night actuals bound the all-hours WAPE
exactly at both ends:

| challenger behaviour at night | all-hours WAPE |
|---|---|
| predicts **0** — what the ABL-337 clamp forces, on this same predicate | `W·(1−f) + f` |
| reproduces the floor **perfectly** | `W·(1−f)` |

So **`f` is the full width, in WAPE percentage points, of the interval an
all-hours read can occupy relative to the daylight-only read of the same
challenger** — and separately a **hard lower bound on the WAPE of any served
solar forecast**, because the clamp cannot beat zero against a floor.

**Checked against the one country where a real gate read exists.** BG's gate `f`
is 4.98% at a daylight-only WAPE of 18.90%, so the band is **[17.96%, 22.94%]**.
ABL-381 measured the all-hours read at **18.89%** — inside the band, near the
floor-reproducing end, which is the same conclusion that pack reached
independently from the model's own night predictions (224.78 MW against 225.13
MW actual). The bound is not decorative; it predicts the one case we can check.

The probe reports it as `wape_floor_pct_if_clamped`, taken on **|MW|**. The
pre-existing `pct_of_total_energy_at_night` is signed and, on NL, reports
*negative* and sorts NL as the cleanest country in the fleet.

---

## 3. The ranking

Sorted by gate-window `wape_floor_pct_if_clamped`, `energy_generation` (the table
ABL-348 registers). `role`: **405** = ABL-405 tranche 2a, **348** = ABL-348
candidate, **serve** = has a live solar model.

| # | cc | role | **gate f (pp)** | fit f (pp) | gate night >1 MW | gate max MW | gate mean MW | attribution |
|--:|---|---|---:|---:|---:|---:|---:|---|
| 1 | **BG** | 348 | **4.980** | 6.375 | 85.2% | 1,087.9 | 245.7 | upstream |
| 2 | **ES** | 348 | **1.352** | 1.338 | **100.0%** | 992.0 | 515.5 | upstream |
| 3 | **EE** | 348 | **0.718** | 0.971 | 79.1% | 76.0 | 12.6 | upstream |
| 4 | SI | **405** | 0.275 | 0.261 | 88.5% | 26.6 | 3.6 | upstream |
| 5 | SK | **405** | 0.185 | 0.104 | 54.8% | 2.9 | 0.9 | upstream |
| 6 | FR | serve | 0.106 | 0.139 | 11.3% | 365.5 | 25.5 | upstream |
| 7 | HU | **405** | 0.053 | 0.057 | 55.1% | 20.0 | 2.7 | upstream |
| 8 | LV | 348 | 0.041 | 0.418 | 15.5% | 20.0 | 0.7 | upstream |
| 9 | NL | 348 | 0.040 | 0.600 | 0.0% | **−0.1** | **−0.1** | no_floor |
| 10 | SE | 348 | 0.033 | 0.057 | 92.7% | 3.6 | 1.6 | upstream |
| 11 | LT | 348 | 0.018 | 0.013 | 3.9% | 14.4 | 0.6 | upstream |
| 12 | PT | 348 | 0.009 | 0.449 | 3.3% | 21.2 | 0.4 | **source_mapping** |
| 13 | FI | 348 | 0.002 | 0.285 | 33.3% | 2.3 | 0.8 | upstream |
| — | AT BE CH CZ DE GR HR IT NO PL RO | | **0.000** | ≤0.054 | ≤1.0% | ≤101.7 | ≤12.6 | — |

**Only BG's band is wide enough to threaten a verdict.** BG's own pass cleared
D-7 by 5.45pp (18.90% vs 24.35%) against a **4.98pp** band — comparable, and the
reason it survived is that the model reproduced the floor rather than that the
floor was small. Every other country's band is **≤0.72pp** against margins that
run 4–6pp in the reads published so far, so no other gate verdict in the
programme is at risk from this. That is the direct answer to scope item 4.

### ES is real generation, and this is the one that matters most to get right

ES ranks second and would be the obvious next candidate for a night-floor fix.
**It should not get one.** Five independent lines all say the output is genuine
concentrated-solar-power (CSP) dispatch from thermal storage:

1. **It tracks the charge.** Within-month detrended correlation between a day's
   daylight energy and *that same night's* energy is **r = +0.515** over **585
   days**. Removing the month mean removes the seasonal confound, so this is a
   day-to-day charge→discharge relationship. Every other country tested is at
   ~0: BG +0.084, EE +0.084, SI +0.152, SK +0.140, HU +0.174. **Contamination
   cannot produce this** — a stuck, forward-filled or mismapped value has no
   reason to know how sunny that particular day was.
2. **It is seasonal in the right direction and by the right amount** — night mean
   **42 MW in December 2025 → 599 MW in July 2025**, tracking the solar season.
3. **The overnight profile is a discharge curve, not noise** — July gate window,
   monotone from 663 MW at 20:00 UTC through 630 / 585 / 551 / 523 / 484 / 446 to
   384 MW at 03:00, at sun elevations reaching **−28.7°**.
4. **The magnitude fits the fleet.** Spain operates ~2.3 GW of CSP with molten-salt
   storage — the only large such fleet in Europe. A 385–992 MW overnight output
   is an ordinary capacity factor for it.
5. **It is not misreported batteries.** ES reports `energy_storage_mw`
   separately, and it averages **1.6 MW**.

ENTSO-E aggregates CSP and PV into one production type (B16), so real CSP output
lands in `solar_mw` with nothing to distinguish it.

**Three consequences, and the second is a production finding:**

- **Do not apply ABL-376's `exclude_impossible_night` rule to ES.** That rule's
  warrant is "the sun says this cannot exist". For CSP the sun says no such
  thing, and the rule would delete ~1.3% of ES's energy and the most predictable
  part of its evening ramp.
- **The ABL-337 serving clamp would destroy real signal on ES.** The clamp zeroes
  every night hour unconditionally on the same predicate. Were ES solar promoted
  to serving today, the clamp would impose a floor of **1.35% WAPE** on it for
  nothing — and by §2 that is a *lower* bound. ES is not served today, so nothing
  is being lost right now; this is a **precondition on ever serving ES solar**,
  and it belongs to whoever owns the clamp, not to this screen.
- **`is_night` is a feature ES's model should be allowed to use, not fitted
  around.** ES is precisely the country where the ABL-395 geometry pair carries
  real information at night rather than a physical zero.

**How to falsify this.** Red Eléctrica publishes *solar térmica* separately from
*solar fotovoltaica*. If that series does not account for ES's overnight MW,
this reading is wrong. I did not fetch it — this screen is read-only against the
replica and adding an external source is an ingest question. Recommended before
ES enters any tranche.

---

## 4. Upstream, or introduced by the source mapping (scope item 3)

**Verdict: upstream, essentially everywhere.** Switching source table fixes
nothing.

The comparison runs on the **intersection** of the two tables' hourly indices,
restricted to hours finite in both, so the two are compared on rows they both
report rather than through summaries over different row sets. The discriminator
is relative **energy** per band, not a row count or a max.

> **Why not the 1 MW threshold on both bands.** My first cut applied it to
> daylight too and got BG wrong — it flagged `series_differ` on a 935 MW daylight
> gap that is **one revised day** (2026-02-14; 27 hours of 2,722, the other 99.0%
> bit-identical). 1 MW is the correct *absolute* test for a night hour, whose
> honest value is exactly 0, and a category error for a daylight hour running to
> 5,000 MW. The tolerance for the relative test is a documented reporting choice,
> not a registered constant, and every raw per-band difference is in the JSON so
> a reader can re-verdict without re-running.

Gate window: **13 upstream, 10 no_floor, 1 source_mapping** (PT, on a floor of
0.009pp — immaterial). Fit window adds a handful of `source_mapping` and
`series_differ` verdicts, and the direction is consistent: `energy_generation`
carries night energy that `energy_renewable` does not.

| cc | window | gen night MWh | ren night MWh | verdict |
|---|---|---:|---:|---|
| DE | fit | 17,663.3 | 23.8 | source_mapping |
| PT | fit | 14,693.6 | 43.7 | source_mapping |
| LV | fit | 4,058.4 | 319.0 | series_differ |
| CH | fit | 1,921.5 | 0.0 | series_differ |
| SI | fit | 2,453.4 | 2,248.9 | source_mapping |
| HU | fit | 2,278.0 | 2,581.7 | source_mapping |
| EE | fit | 7,084.1 | 7,550.4 | series_differ |
| GR | fit | 107.0 | 394.0 | series_differ |

Cadence is identical within every pair above, so ABL-332 resolution is not the
confounder.

> **A zero in `energy_renewable` is not evidence of a clean feed.** ABL-188
> established that its mapper initialises every renewable column to 0.0 before
> consulting the source frame, so "no night floor" there may mean "no
> measurement" rather than "measured zero". **Do not read the DE and PT rows as
> "switch source to fix it".** CH is the worked example of why the daylight band
> had to be measured too: its `energy_renewable` carries **4.0% less daylight
> energy** than `energy_generation`, so its zero night floor comes with a
> materially different daytime series and is scored `series_differ` — not
> attributable — rather than being credited as a fix.

---

## 5. ABL-405 (tranche 2a) is clear to proceed

ABL-405 asked to quote this result if it landed. It has. Its six new countries:

| cc | gate f (pp) | fit f (pp) | gate night >1 MW | gate max MW | reading |
|---|---:|---:|---:|---:|---|
| SI | 0.275 | 0.261 | 88.5% | 26.6 | prevalent, immaterial |
| SK | 0.185 | 0.104 | 54.8% | 2.9 | prevalent, immaterial |
| HU | 0.053 | 0.057 | 55.1% | 20.0 | negligible |
| CZ | 0.000 | 0.000 | 0.0% | 0.0 | clean |
| PL | 0.000 | 0.002 | 0.0% | 0.0 | clean |
| RO | 0.000 | 0.001 | 0.0% | 0.0 | clean |

**None of the six can move a gate cell by more than 0.28 WAPE points**, against
D-7 bars of 18.35% (HU), 19.14% (SK) and 22.28% (SI). **Screened, and no
tranche-2a verdict is at risk from a night floor.**

Worth stating because it is counter-intuitive: **SI and SK have a floor in most
night hours (88.5% and 54.8%) that is nonetheless immaterial** — their peaks are
26.6 and 2.9 MW. Prevalence and materiality are different questions, and this is
why the screen ranks on energy share rather than on the ABL-338 row count.
Ranking on `% of night rows > 1 MW` would put SE (92.7%) and SI above ES.

---

## 6. NL is a different defect: a uniformly negative night series

NL reads **negative at every single night hour** — 1,390/1,390 in the fit window
(−1.47 to −0.12 MW) and 154/154 in the gate window (−0.13 to −0.12 MW). Never
zero, never positive.

This is not a floor and the 1 MW threshold does not see it: `% of night rows > 1
MW` is 0.0%, and the signed energy share reads **−0.60%**, which sorts NL as the
cleanest country in the fleet. It is small in magnitude and does not threaten a
gate read (|f| = 0.040pp at the gate). It is reported because it is a **distinct
class** — a small negative offset applied to a quantity that is physically
non-negative — and because a screen that only looked for positive floors would
have certified NL as spotless. `n_night_negative` now makes it visible.

Not adjudicated here: whether this is a metering-convention artefact (own-consumption
netting) or an ingest sign error. It needs a data owner, not a forecasting fix.

---

## 7. DE's floor is fit-side only, and DE serves

DE — a **live serving** solar country — carries a night floor in **87.8% of fit-window
night hours, max 101.7 MW, mean 12.6 MW**, and **exactly zero** in the gate
window (0/160 hours). So the contamination stops somewhere between 2026-07-11
and the fit window's end.

Materially it is small (fit f = 0.038pp), so it does not threaten a read. It is
flagged for two reasons: it lands on a country whose model is in production, and
it is a reminder that **a clean gate window does not certify the fit window** —
these are separate measurements and this screen reports both for exactly that
reason. Dating the transition is a follow-up, not this screen.

---

## 8. Method notes and limits

- **The probe was extended, not replaced** (scope item 1). Re-running ABL-381's
  own invocation through the modified script reproduces
  `reports/abl_381_night_floor_probe.json` with **no original field changed** —
  verified field-by-field after every edit in this branch. New fields are
  additive: daylight statistics, `wape_floor_pct_if_clamped`,
  `n_night_negative`, `n_missing_rows`, `native_resolution`, `source_comparison`.
- **NaN is missing, never zero.** CZ's `energy_renewable` carries 93 NaN hours in
  the fit window. Every statistic is taken over finite rows and the dropped count
  is reported. Reading a hole as 0.0 would invent a night zero — precisely the
  direction being measured — and would flatter a table into looking clean.
- **Two windows only.** The registered ABL-348 fit and gate windows. A floor
  outside them cannot move a tranche read but would still contaminate a longer
  fit; §7's DE result shows floors do start and stop. A full-history sweep is
  cheap and not run here.
- **One representative point per country.** `solar_geometry` uses a single
  capacity-weighted point, so `night_mask` is approximate for wide fleets. It is
  conservative — night requires the sun below threshold for the *whole* hour — so
  it under-counts rather than over-counts, and the ES elevations (−14.6° to
  −28.7°) are far outside any plausible error.
- **`f` bounds a WAPE displacement; it does not predict one.** Where a
  challenger lands inside its band depends on how faithfully it learns the floor.
  BG landed near the floor-reproducing end. Do not read column `gate f` as an
  expected error change.
- **Nothing here was fitted, and no verdict is proposed.** Read-only, per the
  ABL-67/ABL-210 rule the issue restates: repair beats delete, and nothing in
  this screen mutates a table.

## 9. Recommended next steps

1. **ABL-405 proceeds** on all six new countries, quoting §5. No action needed.
2. **ES is gated on the CSP question, not on a data fix.** Before ES solar enters
   any tranche, settle §3 against Red Eléctrica's *solar térmica* series. If it
   confirms, ES needs the clamp and the ABL-376 rule **disapplied**, which is a
   Founding Engineer change to shared serving code and a CEO call — not a
   tranche-local one.
3. **BG stays the one genuine problem**, unexplained by CSP (no fleet, r = +0.084)
   and unfixable by source switch (§4). Its pass stands, on ABL-381's daylight-only
   re-score; a fit-side exclusion would remove 76–85% of its night rows and is a
   pre-registration question for whoever re-reads that scope.
4. **EE** (0.718pp) is the only other country worth a look before it is gated.
5. **NL and DE** (§6, §7) are data-owner items and should be filed as such rather
   than carried as forecasting caveats.
