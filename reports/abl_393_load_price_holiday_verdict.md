# ABL-393 — do the four holiday features help load and price?

**load: HELP → keep them. price: NO_EFFECT → exclude them.**

The prior was right about load and wrong to be extended to price. `src/features.py`
has called these four *"high impact for load forecasting"* since the initial commit,
and on load that claim is **vindicated**: 8 of 8 registered cells favour them, every
one at 8/8 seeds, at a paired mean of **−12.73% all-hours MAE**. On price — same
window, same countries, same algorithms, same seeds — the effect is **−1.09% and not
material in 6 of 8 cells**, and DE moves the wrong way at both algorithms.

None of the 48 serving load/price artifacts carries any of the four. **This is a
finding, not a promotion:** no serving-registry change, no retrain, no artifact
written. Whether the 24 load countries are retrained to pick these up is the CEO's
decision.

| | registration | machine record | raw arms |
|---|---|---|---|
| | [`experiments/ABL393/config.json`](../experiments/ABL393/config.json) | [`abl_393_holiday_verdict_tables.{json,md}`](abl_393_holiday_verdict_tables.md) | [`abl_393_load_price/`](abl_393_load_price/) |

Registered and frozen in git at `12ccbe5` **before any registered arm was fitted**.
384 fits, all under `.venv` Python 3.14.3 / xgboost 3.3.0, against the live replica
(`C:\Code\able\data\energy_dashboard.db`, 9,432,453,120 bytes — verified, not the
3.0 GB `energy-data-gathering` snapshot). No write of any kind to the replica; this
issue's only outputs are reports and experiment JSON.

---

## 1. What was measured

| | |
|---|---|
| **contrast** | `control` (the four holiday names present) vs `control_noholiday` (**exactly** the serving feature list — 26 names on load, 25 on price) |
| **primary statistic** | paired by seed: `δ_s = 100 × (MAE_hol,s − MAE_nohol,s) / MAE_nohol,s` on all holdout hours. **Negative = holidays better.** |
| **seeds** | 101, 103, 107, 109, 113, 127, 131, 137 — the repo's standing eight (ABL-376/395), inherited not chosen |
| **materiality** | ≥7 of 8 seeds agreeing in sign (two-sided sign test p ≤ 0.0703; p = 0.0078 at 8/8). No percentage threshold — ABL-386's reason |
| **cells** | (type, window, country, algorithm): 3 registered groups × 4 countries × 2 algorithms = 24 |
| **windows** | `spring` 2026-04-30..2026-06-12 primary for both types; `winter` 2025-12-06..2026-01-18 a registered replication for **load only** |
| **countries** | AT, BE, DE, FR — ABL-386's set, held fixed on purpose |
| **algorithms** | catboost and xgboost, each at its own defaults under `--force-algorithm`; both are live in both serving fleets (load 20/4, price 19/5) |

There is **one harness**, not two: `scripts/abl338_solar_holdout.py`, ABL-386's own,
extended to accept `--type load|price`. Re-running ABL-386's committed DE catboost
invocation under this issue's code reproduces **all 12 arm-fits field for field**;
the change adds JSON keys and nothing else. Only the verdict *rule* is new, because
ABL-386's rule was unpaired range disjointness over three seeds and this one is the
paired test its own corrections block asked for.

`control_noholiday` really is the served list — checked twice, once from ABL-386's
drift JSON and once directly against `models/` under `.venv`: **all 48 load and price
artifacts equal `get_feature_columns(type, include_holidays=False)` name for name and
in order.** There is no geometry on these types, so unlike solar there is no second
gap to confound the contrast. It is a refit, though, not the live model — the serving
artifacts are fitted through roughly today and scoring them on either holdout would
score them in-sample. **No number here may be phrased as beating a serving model.**

---

## 2. Load — HELP, and it is not a small effect

**spring, all hours, paired over 8 seeds.** Every cell 8/8, p = 0.0078.

| cell | MAE holidays (MW) | serving list | paired δ | own seed spread hol/nohol |
|---|---:|---:|---:|---:|
| catboost/AT | 170.5 | 208.5 | **−18.21%** ± 3.90 | 8.94% / 4.57% |
| catboost/BE | 193.2 | 220.4 | **−12.32%** ± 1.18 | 3.78% / 1.37% |
| catboost/DE | 1,414.4 | 1,665.7 | **−15.08%** ± 1.41 | 4.51% / 1.95% |
| catboost/FR | 1,115.8 | 1,234.3 | **−9.59%** ± 2.15 | 5.74% / 4.53% |
| xgboost/AT | 162.1 | 189.8 | **−14.56%** ± 2.54 | 4.49% / 5.48% |
| xgboost/BE | 197.7 | 219.6 | **−9.97%** ± 1.74 | 4.25% / 5.11% |
| xgboost/DE | 1,328.9 | 1,601.2 | **−16.99%** ± 1.86 | 4.66% / 4.36% |
| xgboost/FR | 1,115.2 | 1,175.5 | **−5.12%** ± 1.82 | 4.02% / 4.50% |

The right-hand column is the reason a one-seed read is worthless here: each arm moves
1.4–8.9% of its own mean across seeds with nothing changed, which on the weakest cell
(xgboost/FR, −5.12%) is larger than the effect.

**Pairing did not change the load/spring reading, and that is worth stating rather
than implying otherwise.** Scope item 2 asked for the paired statistic because
ABL-386's unpaired range test was the weaker instrument. Applied to these eight cells
the old test also calls all eight disjoint — the load effect is simply too large to
need the better instrument. Pairing changed the reading in **4 of the 24 cells across
the whole read**, and they are exactly the cells where the effect is comparable to the
seed spread: winter xgboost/DE (unpaired overlap, paired 8/8 for), both winter FR
cells (unpaired overlap, paired 1/8 — material *against*), and price catboost/AT
(unpaired overlap, paired 8/8 for). On price, where all the effects live inside the
seed spread, the unpaired test finds **1** material cell where the paired test finds
2. So the instrument upgrade earned its place on the marginal cells and on the price
arm, not on the headline.

**Winter replicates it.** 6 of 8 cells favour the holiday features, all at 8/8, sum(d)
= +4 → HELP. Both windows agree in direction, so the registered replicate-downgrade
does not fire and spring stands.

**The gain lands where the mechanism says it must.** `holiday_affected` rows
(holiday, bridge day, or within a day of one) are 20.5–27.3% of the spring holdout
and carry **66–98% of the total error saved**. On holiday rows alone the effect is
−15% to −60%. A gain spread evenly over ordinary rows would have been an effect in
search of an explanation; this one is not.

**Against no model at all.** The four ABL-389 references all cover 24/24 hours here,
so they are scored on exactly the challenger's rows. The holiday arm beats D-7 by
49.5–61.6% and — the number that matters, per the standing ABL-381 ask — beats the
**hour-of-day climatology chosen with hindsight** by 57.8–70.8%. Load is strongly
diurnal, so a flat line is a formality (the model beats it by 73.8–85.3%, and the
gap between the constant and the climatology is how much of load is forced diurnal
structure); the climatology is the honest reference, and the model clears it
comfortably. This is real skill, not a diurnal artifact.

### The two cells that go the other way

FR winter is +1.83% (catboost) and +0.90% (xgboost), both at 1/8 — material *against*
the holiday features. These are the two thinnest cells in the whole read and the
registration said so before the fit: FR load is missing all of **2026-01-01** in that
window (a 26 h hole), leaving its winter holdout with exactly **one** holiday,
Christmas, and 11.5% holiday-affected rows against AT's 29.5%. Both FR winter cells
also show a net loss overall, so there is no gain to apportion. Read them as the
weakest evidence in the pack, not as a counter-finding — FR spring, on a window with
four holidays, is −9.59% and −5.12% at 8/8.

---

## 3. Price — NO_EFFECT, and the country split is the interesting part

**spring, all hours.** 2 of 8 cells material, both favouring the holiday features;
mean δ **−1.09%**, range −3.74% to +3.40%.

| cell | MAE holidays (EUR/MWh) | serving list | paired δ | k/8 | material |
|---|---:|---:|---:|---:|---|
| catboost/AT | 19.33 | 20.09 | −3.74% | 8/8 | **yes** |
| catboost/BE | 19.15 | 19.49 | −1.72% | 6/8 | no |
| catboost/DE | 24.55 | 24.20 | **+1.49%** | 3/8 | no |
| catboost/FR | 19.61 | 19.92 | −1.50% | 6/8 | no |
| xgboost/AT | 20.68 | 21.10 | −1.93% | 6/8 | no |
| xgboost/BE | 21.09 | 21.35 | −1.16% | 5/8 | no |
| xgboost/DE | 26.77 | 25.91 | **+3.40%** | 2/8 | no |
| xgboost/FR | 18.71 | 19.41 | −3.58% | 8/8 | **yes** |

Every arm's own seed spread here is 3.3–8.2%, which is larger than almost every
effect in the table. That is the honest summary: on price these four features move
less than a reseed does.

**But the holiday-row read is not a null — it is a split**, and it is worth recording
because it is the opposite of what a one-number verdict suggests. On `holiday` rows:

- **AT −12.9% / −10.3% and FR −10.7% / −15.0%**, all four at 8/8 — materially better.
- **DE +4.1% / +14.6%**, both at **0/8** (p = 0.0078) — materially **worse**. BE is
  worse too (+4.2% / +5.5%, 1/8).

So the features genuinely help AT and FR price on the days the mechanism reaches, and
genuinely hurt DE. Pooled, that cancels to nothing. The registered rule reads the
all-hours group verdict and it is NO_EFFECT; the registered interpretation rule for
"all-hours null, subsets not null" is to report both numbers with the all-hours one
leading, which is what this section does. **It is not promoted to HELP**, and DE
moving materially in the harmful direction is the reason a per-type exclusion is the
right call rather than a per-country one this read is not powered to make.

**DE price also loses to an oracle climatology** (xgboost, −0.4%): an hour-of-day
median with no model and no weather in it scores better than the fitted model on that
cell. That changes no verdict here — this issue has no gate, and the holiday contrast
is measured *within* the model — but it bounds what the cell is worth, and it is the
ABL-380 failure mode showing up on a type that has no gate to catch it. DE price is
the worst cell against every reference in the table (D-7 +23.8%, climatology oracle
−0.4% to +8.0%, against AT/FR's +20.7% to +27.1%).

---

## 4. Contamination

| issue | touches this read? | how it was handled |
|---|---|---|
| **ABL-109 / ABL-111** zero-as-missing actual load | **No — measured, not assumed.** `energy_load` with `data_quality='actual'` holds **0 zero-valued and 0 null rows** for AT/BE/DE/FR over the entire history (AT 92,684 rows from 2024-01-01; BE 197,810 and DE 197,798 from 2021-01-01; FR 87,542 from 2021-01-01). | Nothing to handle. Worth measuring because `db.load_energy_data` applies **no `> 0` guard** on the training path, unlike the scorecard's ABL-35 read — a zero-as-missing row would have entered the fit as a real 0 MW target. Bounded to these four countries; the other 20 load artifacts are not covered. |
| **ABL-71** prod ingest stale, fixes undeployed | **Yes**, and it does not cancel out of the absolute numbers. Its known wrong-write modes are load and net position. | Both arms score identical rows, so a contaminated actual moves both MAEs together and **cancels out of the paired difference** — the verdict is safe. It does not cancel out of any absolute MAE quoted here; every one carries this caveat. |
| **ABL-67** fabricated net_position rows | No — net position only. | — |
| **ABL-188** constant-run screen | No — `load_renewable_type_data` only; neither aggregate type goes through it. | — |
| price zeros and negatives | Not contamination. `energy_price` carries 297–1,355 exact zeros and 1,724–3,256 negative hours per country over its history — real market outcomes. | Retained. This is why MAE and not WAPE is the statistic anything is decided on for price. |

**A data defect this read had to work around, currently unowned.** `energy_price` is
missing **2,236 h for AT and 2,483 h for DE**, almost all of it 2025-09 to 2025-12
(AT's single largest hole is 1,651 h from 2025-09-09; DE's is 1,309 h from
2025-09-25), plus a 91 h hole from 2025-12-23 that removes Christmas from both.
`energy_load` over the same span misses one 27–29 h outage on 2026-02-15 common to
all four majors, plus 26 h of FR over New Year 2026. Inventory:
[`abl_393_source_gaps.json`](abl_393_source_gaps.json).

This matters beyond coverage, because **`create_lag_features` shifts by `days × 24`
rows**, which is a day only on a gapless hourly frame — so every row for a fortnight
after a hole carries D-1/D-7/D-14 lags reaching across it. The AT and DE price holes
sit inside the *fit* window of the spring read; identical in both arms, so the
contrast is unaffected, but the absolute price MAE is worse than a clean series would
give. Filed separately for whoever owns ingest.

---

## 5. Why December was rejected for price, and how the windows were chosen

The rule was fixed and published before any candidate was scored: a window is
admissible on **calendar holiday density** and **source coverage**, both model-free,
and on nothing else. Then measured:

- **December is not the densest holiday window of the year** for three of these four
  countries. `winter` holds AT 5, BE 2, DE 3, FR 1 holiday days; `spring` holds AT 4,
  BE 4, DE 3, FR 4 — Labour Day, Ascension, Whit Monday, FR's 8 May and AT's Corpus
  Christi all fall in the second. The issue's premise that December is "the densest
  holiday cluster of the year" does not hold on the `holidays` calendar. What December
  has instead is a contiguous low-demand fortnight, which is what
  `days_to_holiday`/`days_from_holiday` mark — hence keeping it for load.
- **Winter coverage disqualifies price outright**: AT and DE are **67.3%** covered
  there, behind the holes above. Registered as *rejected*, with the reason, rather
  than quietly dropped.
- `spring` is 100% covered on all eight pairs and is **inherited, not selected** —
  ABL-375 chose it before any of its own arms were fitted, on a criterion having
  nothing to do with holidays. Using it means the solar read and this one differ in
  the target and in nothing else.

Because a favourable winter number on DE load was seen (as a smoke test, at seeds
outside the registered set) *before* the registration was written, winter was
registered as a **replication that can weaken the load verdict and never create one**.
The disclosure, with the numbers, is in the registration. It did not end up mattering:
spring alone reads HELP at 8/8 cells.

---

## 6. What this does not establish

- **Four countries of 24.** AT, BE, DE, FR. The other 20 load and 20 price artifacts
  are not covered, and neither is the ABL-109/111 zero screen. ABL-407 measured that
  6 of the 66 drifted artifacts have unexplained provenance and that those 6 are
  **BE/DE/FR × {load, price}** — six of the eight spring pairs read here.
- **Not serve-faithful.** Features come from the training-time pipeline, whose lags
  and rolling windows are anchored at the target hour; at serving they are anchored at
  the generation instant (ABL-183). Every arm carries that identically, so the
  contrast is sound and the absolute MW/EUR are optimistic against the rail. The four
  holiday features are exempt by construction — all are calendar functions of the
  target timestamp, so they are identical in both paths, and they are among the most
  serve-faithful features in the set (a public holiday is known years ahead).
- **No horizon bands, no gate.** This answers "do these four features change the fit",
  not "does load clear a promotion gate" — there is no promotion gate for load or
  price to clear. That absence is itself worth noting given §3's oracle-climatology
  result on DE price.
- **Two windows, one of them load-only.** Spring is late spring; winter is the
  Christmas cluster. Neither is a claim about the rest of the year.
- **A refit is not a serving model.** The 5.1–18.2% load gain (12.7% mean over the
  eight spring cells) is between two refits on an identically truncated window. It is
  not a forecast of what retraining the fleet would deliver, which depends on the
  serving window, the tuned per-country hyperparameters this read deliberately
  replaced with defaults, and the serve path.

---

## 7. Recommendation

**Load — KEEP the four names in `get_feature_columns('load')`.** This is the
registered mapping for HELP. The measurement says the repo's own documented claim is
correct, and that on **the four countries measured** these four features are worth
5.1–18.2% of MAE on a like-for-like refit — while all 24 serving load artifacts carry
none of them. **No serving change follows from this issue.** A retrain of the load
fleet is a CEO decision and would need its own read on the remaining 20 countries;
this pack covers four, and the effect already varies 3.5-fold across those four.

**Price — EXCLUDE them from `get_feature_columns('price')`** on parsimony, per the
registered NO_EFFECT mapping. Note the registered rationale for NO_EFFECT contains a
clause about the `"high impact for load forecasting"` comment being contradicted; that
clause was written expecting NO_EFFECT to be the *load* outcome and does not apply
here — load read HELP, so the comment is vindicated. Recorded as a visible correction
in the registration; the recommended action is unchanged.

Two things worth a decision that this issue deliberately does not take:

1. **AT and FR price do benefit on holiday rows** (−10% to −15%, 8/8) while DE is
   materially worse. A per-country price feature list is a different kind of change
   from a per-type one and is not something an eight-seed read on four countries
   should propose.
2. **DE price loses to an hour-of-day climatology chosen with hindsight.** That is an
   ABL-380-shaped finding on a type with no gate to surface it, and it is unrelated to
   holidays.
