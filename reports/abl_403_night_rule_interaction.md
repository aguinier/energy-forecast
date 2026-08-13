# ABL-403 — geometry x `exclude_impossible_night`: the 2x2, on BG and CH

**The question was whether the night axis moves only when both changes are
present. It does not. The exclusion rule moves it enormously on its own — and in
the wrong direction. On BG it doubles night MAE (44.8 → 105.9 MW, t = +9.6, 8/8
seeds), costs 1.4–1.9pp of gate-band WAPE, and consumes 47% of the gate margin
that ABL-405's PASS was carrying. There is a readable interaction, but its sign
is "geometry makes the rule do less damage", not "geometry makes the rule work".**

**Recommendation: register `exclude_impossible_night: False` for every remaining
ABL-316 solar tranche, ES and EE included.**

Machine record: `reports/abl_403_night_rule_interaction.json`.
Probe: `scripts/abl403_night_rule_interaction_probe.py`.
Statistics guard: `tests/test_abl403_interaction_stats.py`.

---

## 1. Why this ran, and what it was asked

ABL-395 added ABL-338's geometry pair to the gate feature list and found nothing
on the night-negative axis. ABL-376 found its own exclusion rule **27x more
effective** on FR's night level once the model had `is_night` to see with. Both
are single-factor reads of a two-factor design, and ABL-395 ran the cell that
carries neither change. The CEO's ask was for a registration rule, not four
numbers, and specifically:

> does the night axis move only when both are present? [...] If that interaction
> is real, say so and tell me what to register for the remaining tranches. If it
> is not, say that just as plainly.

It is not. Section 4 says so plainly, and sections 5–6 say what to register.

## 2. Protocol

Four arms, per country, per seed. One vintage frame per country, built once at
the 27-name superset and shared by every fit, so the arms differ only in the
column list and the fit-row filter:

| arm | features | `exclude_impossible_night` |
|---|---:|---|
| `f25_off` | 25 (the list through ABL-381) | off — the control, ABL-395's `f25` |
| `f27_off` | 27 (+ `sun_elevation_deg`, `is_night`) | off — ABL-395's `f27` |
| `f25_on` | 25 | **on** |
| `f27_on` | 27 | **on** |

Exclusion is applied to the **fit** frame after `finite_training_rows`, exactly
as `evaluate_solar_retrain.py:727` applies it, so the two audits partition the
dropped rows. **The gate frame is never filtered, at any arm** — that asymmetry
is ABL-376's rule and section 5 is about what it costs when the filtered share
gets large. All four arms scored **2,730 identical selected rows**; the probe
asserts gate-row identity across all four arms and fit-row identity across the
geometry axis within each filter level, because an arm scored on different rows
is not an A/B.

Windows, source, schedule, bands, algorithm and basis are ABL-348's registered
ones: fit 2026-01-14 → 2026-07-11, gate 2026-07-11 → 2026-08-10 (both exclusive),
`energy_generation`, CatBoost, primary bands 24-36h / 36-48h / 48-64h, basis
`(challenger, seasonal_naive)`, out of sample by target timestamp. Replica
`C:\Code\able\data\energy_dashboard.db` at **9,432,453,120 bytes** — the live
replica, not the 3.0 GB partial snapshot — opened read-only. No write of any
kind was made to it, and no artifact was saved.

Seeds are ABL-376's eight registered ones `(101, 103, 107, 109, 113, 127, 131,
137)`, reused verbatim: frozen before that issue's first fit, disjoint from the
gate's 42, and the same set ABL-395 swept, so this read is commensurable with
both. **64 fits, 10 minutes.**

**Reading rule, fixed before the run and not moved after it.** Every contrast is
paired within seed. Each is reported against a null built from control-arm
(`f25_off`) fits — the same fit, one integer apart — sized to the number of
independent fits the statistic combines: control-vs-control **pairs** for a
simple effect, control **quadruples** for the interaction. An effect no larger
than its null max is not distinguishable from seed noise in this design. The
seed count was not extended.

## 3. The control reproduces ABL-395 exactly, and the rule's bite is not symmetric

**Reproduction.** The two `off` arms are a like-for-like refit of ABL-395 at the
same windows, source, schedule, algorithm and seeds. Over all 8 seeds x 2
countries x 3 bands, the largest disagreement with the published JSON is
**0.005pp** on both the night-negative rate and every band WAPE — that is
ABL-395's 2-decimal rounding against this run's 3, i.e. the fits are identical.
Everything below rests on that.

**What the rule actually removes.** Both countries start from 34,176 fit rows.

| | night fit rows | excluded | share of night rows | distinct targets | max excluded | mean night actual (fit) | mean night actual (gate) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **BG** | 12,232 | **9,344** | **76.4%** | 1,168 | **1,097.4 MW** | 152.3 MW | **225.1 MW** |
| **CH** | 11,720 | 5,384 | 45.9% | 673 | **5.8 MW** | 1.3 MW | **0.00 MW** |

Two things in that table were not in the design.

- **BG's 76.4% confirms ABL-396 §9.3's estimate to the point** ("would remove
  76–85% of its night rows"), and 25.3% of BG's *gate* rows are night rows
  carrying a 225 MW mean. So the rule forbids the fit from learning a pattern
  that a quarter of the score is measured on.
- **CH is not the no-op the issue assumed.** Its *gate* night actuals are exactly
  0.00 MW, but its *fit* window carries small non-zero values, and at
  `IMPOSSIBLE_NIGHT_THRESHOLD_MW = 1.0` the rule drops **45.9% of CH's night fit
  rows for a maximum reading of 5.84 MW** against a ~5 GW fleet (0.1%). CH is a
  clean control on *effect* — section 4 shows it measures nothing — but it is not
  a clean control on *row count*, and a threshold that discards half a country's
  night rows on sub-6 MW readings deserves its own look (§6).

## 4. The 2x2

Arm means over the eight seeds. `rule@f25` and `rule@f27` are the two simple
effects of the exclusion rule; `interaction` is their difference — the quantity
this issue exists to measure, positive meaning the rule moves the metric further
up when geometry is present.

### 4a. BG — the informative cell

| quantity | f25_off | f27_off | f25_on | f27_on | rule@f25 | rule@f27 | interaction |
|---|---:|---:|---:|---:|---:|---:|---:|
| **night MAE (MW)** | 44.84 | 46.70 | 105.88 | 93.56 | **+61.05** | **+46.86** | **−14.19** |
| night bias, pred − actual (MW) | −2.08 | −1.55 | +88.45 | +73.79 | **+90.53** | **+75.34** | −15.20 |
| night WAPE (%) | 19.92 | 20.74 | 47.03 | 41.56 | **+27.12** | **+20.82** | **−6.30** |
| night rows predicted negative (%) | 20.09 | 21.63 | 12.97 | 9.86 | **−7.12** | **−11.78** | −4.66 |
| WAPE 24-36h (%) | 19.41 | 19.87 | 20.85 | 21.77 | **+1.45** | **+1.90** | +0.45 |
| WAPE 36-48h (%) | 19.18 | 19.63 | 20.55 | 21.51 | **+1.37** | **+1.89** | +0.52 |
| WAPE 48-64h (%) | 20.68 | 20.91 | 22.23 | 22.83 | **+1.55** | **+1.92** | +0.37 |
| daylight WAPE 24-36h (%) | 19.43 | 19.87 | 19.51 | 20.77 | +0.08 | +0.90 | +0.82 |
| daylight WAPE 36-48h (%) | 19.17 | 19.59 | 19.20 | 20.50 | +0.03 | +0.91 | +0.88 |
| daylight WAPE 48-64h (%) | 20.83 | 21.03 | 20.85 | 21.86 | +0.03 | +0.83 | +0.80 |

Evidence on the four contrasts that carry the finding:

| contrast | mean | sd | t | seeds | sign p | null max (p95) | outside null |
|---|---:|---:|---:|---:|---:|---:|---|
| night MAE, rule@f25 | +61.05 | 18.01 | **+9.59** | **8/8 up** | **0.0078** | 6.96 (6.65) | **yes** |
| night MAE, rule@f27 | +46.86 | 14.98 | **+8.85** | **8/8 up** | **0.0078** | 6.96 (6.65) | **yes** |
| night MAE, interaction | −14.19 | 18.97 | −2.11 | 7/8 down | 0.0703 | 11.31 (9.37) | **yes** |
| night-negative %, interaction | −4.66 | 6.29 | −2.09 | 6/8 down | 0.2891 | 19.28 (15.36) | no |

Per seed, night MAE (MW) — the effect is not carried by one draw:

| seed | f25_off | f27_off | f25_on | f27_on | rule@f25 | rule@f27 | interaction |
|---|---:|---:|---:|---:|---:|---:|---:|
| 101 | 45.41 | 46.83 | 111.30 | 104.39 | +65.89 | +57.56 | −8.34 |
| 103 | 45.26 | 44.73 | 105.02 | 75.92 | +59.76 | +31.20 | −28.56 |
| 107 | 42.62 | 47.21 | 98.04 | 81.00 | +55.43 | +33.79 | −21.63 |
| 109 | 42.34 | 49.20 | 136.90 | 116.92 | +94.56 | +67.72 | −26.84 |
| 113 | 46.96 | 44.81 | 77.65 | 105.41 | +30.69 | +60.59 | **+29.91** |
| 127 | 42.73 | 44.89 | 105.24 | 85.23 | +62.52 | +40.34 | −22.17 |
| 131 | 49.30 | 48.26 | 100.05 | 77.40 | +50.75 | +29.14 | −21.61 |
| 137 | 44.08 | 47.66 | 112.86 | 102.21 | +68.78 | +54.56 | −14.23 |

### 4b. CH — the control, and it measures nothing

| quantity | f25_off | f27_off | f25_on | f27_on | rule@f25 | rule@f27 | interaction |
|---|---:|---:|---:|---:|---:|---:|---:|
| night MAE (MW) | 7.92 | 7.32 | 8.50 | 8.40 | +0.58 (p 0.29) | +1.09 (p 0.73) | +0.51 (p 0.73) |
| mean prediction at night (MW) | −5.59 | −3.71 | −4.64 | −4.00 | +0.95 (p 0.73) | −0.30 (p 0.29) | −1.25 (p 0.73) |
| night rows predicted negative (%) | 77.05 | 73.20 | 72.64 | 71.86 | −4.41 (p 1.00) | −1.35 (p 1.00) | +3.07 (p 1.00) |
| WAPE 24-36h (%) | 8.35 | 8.22 | 8.35 | 8.14 | +0.01 (p 0.73) | −0.08 (p 1.00) | −0.09 (p 0.73) |
| WAPE 36-48h (%) | 8.30 | 8.06 | 8.31 | 7.94 | +0.01 (p 0.73) | −0.12 (p 1.00) | −0.13 (p 0.73) |
| WAPE 48-64h (%) | 8.70 | 8.47 | 8.68 | 8.46 | −0.02 (p 0.73) | −0.01 (p 0.73) | +0.01 (p 0.73) |

Not one exclusion or interaction contrast on CH reaches its null or a sign-test
p below 0.29. Night WAPE is *Not measured* on CH rather than reported as a
number: the gate window's night actuals are exactly 0.00 MW, so the denominator
does not exist.

What CH does reproduce is **ABL-395's geometry result, and it survives the rule**:
−0.239pp at 36-48h and −0.229pp at 48-64h with the rule off (both 8/8, p =
0.0078), and −0.365pp / −0.218pp with it on. The geometry pair is a small clean
accuracy gain on CH either way. Nothing here argues against ABL-395.

## 5. What it says

**5a. The premise is refuted: the night axis moves from the rule alone, hugely.**
`rule@f25` on BG's night MAE is **+61.05 MW at t = +9.59, 8/8 seeds** — the
largest effect anywhere in this design, produced by the arm with *no* geometry.
"The night axis moves only when both are present" is not a description of these
data.

**5b. The interaction is real on the night level, and it is damage mitigation.**
−14.19 MW night MAE, 7/8 seeds, sign p = 0.0703, clearing a conservative 4-fit
null of 11.31 (p95 9.37); night WAPE the same shape at −6.30pp against a 5.02
null. So ABL-376's mechanism is confirmed **in structure** — the geometry pair
does change what the exclusion rule does to the model — but the rule's effect is
+61 MW without geometry and +46.9 MW with it. Both are large and both are worse.
Geometry recovers 23% of a self-inflicted wound; it does not turn the rule into a
gain.

On the axis the issue was originally framed around, the **night-negative rate,
no interaction is readable at all** (−4.66pp, 6/8, p = 0.29, inside a 19.28pp
null).

**5c. The night-negative rate would have called this a success. That is the
methodological finding.** On BG the rule *improves* the night-negative rate from
20.09% to 12.97% (and to 9.86% with geometry, 8/8 seeds, p = 0.0078) while
**doubling night MAE and pushing night bias from −2.1 MW to +88.5 MW**. A read
that dispositioned this on the sign metric alone — which is the metric ABL-381
§4 and ABL-395 both reported — would have adopted a rule that made the forecast
substantially worse and shown a number moving the right way. **Never disposition
a night-floor change on the negative-prediction rate alone; it cannot see the
level.**

**5d. The rule costs BG half its gate margin.** The D-7 bar is identical across
arms by construction (it does not depend on the fit):

| band | D-7 bar | `f25_off` margin | `f27_on` margin | margin lost |
|---|---:|---:|---:|---:|
| 24-36h | 24.40% | **+4.99pp** | +2.63pp | **−47%** |
| 36-48h | 24.40% | **+5.22pp** | +2.89pp | **−45%** |
| 48-64h | 24.99% | **+4.31pp** | +2.16pp | **−50%** |

BG still clears D-7 at 8/8 seeds in every arm, so this is not a verdict flip. It
is half the cushion, on the one country ABL-396 identified as having a band wide
enough to threaten a verdict at all.

**5e. Why the asymmetry stops working at BG's ratio.** ABL-376 registered the
rule as fit-side only: refuse to train on values the sun says are impossible,
still score against whatever the source reports, so the challenger cannot delete
the rows it is held to account on. That is sound — and on FR it removed **113
contaminated targets**. On BG it removes **1,168 targets, 76.4% of the night fit
rows**, while **25.3% of the scored gate rows are night rows at a 225 MW mean**.
At that ratio the asymmetry is no longer a discipline; it is a penalty the fit
inflicts on itself. **You cannot forbid a model to learn what you still grade it
on, once "what you still grade it on" is a quarter of the score.**

Note this holds *whether or not* BG's floor is genuine. ABL-396 §9.3 puts BG as
"the one genuine problem", unexplained by CSP (no fleet, detrended r = +0.084)
and unfixable by source switch. Grant that entirely: the rows are contaminated,
and removing them from the fit still costs 1.4–1.9pp of WAPE, because the *score*
keeps them. The fix for contaminated actuals is upstream repair (ABL-67/ABL-210's
"repair beats delete"), not a fit-side filter under a score that never learned
about it.

**5f. It refutes ABL-395 §5c's proposed mechanism for BG.** That section
hypothesised BG's +0.44pp geometry regression came from `is_night` being a *lie*
on BG — telling the model the sun is down on hours booking 225 MW — and predicted
that removing those rows would rescue geometry on BG. It does the opposite.
Geometry's cost on BG roughly **doubles and becomes significant** when the rule
is on:

| band | geometry effect, rule **off** | geometry effect, rule **on** |
|---|---:|---:|
| 24-36h | +0.463 (6/8, p 0.29) | **+0.914 (8/8, p 0.0078)** |
| 36-48h | +0.440 (6/8, p 0.29) | **+0.960 (8/8, p 0.0078)** |
| 48-64h | +0.234 (6/8, p 0.29) | **+0.600 (8/8, p 0.0078)** |

Removing the "lying" rows aggravates the regression instead of curing it, which
is not what that mechanism predicts. A reading consistent with the data — offered
as a hypothesis, not a result — is that once the night rows are gone, `is_night`
becomes a near-perfect flag for *absent from training*, and the model applies a
near-zero night level learned from the 2,888 retained sub-1 MW rows to a gate
window whose night is 225 MW. **ABL-395 §5c should be marked as tested and not
supported**, and BG's +0.44pp remains unexplained and still not significant.

## 6. Recommendation — what to register

1. **`exclude_impossible_night: False` on every remaining ABL-316 solar tranche,
   including ES and EE.** No new scope should turn it on. It is the default
   already (`DEFAULT_FIT_RULES`), so this is a recommendation to leave it alone
   with a measured reason rather than an unexamined one — which is what ABL-396
   §9.3 left open for whoever re-read BG.
2. **ES specifically: the rule must stay off, and this is now measured rather
   than argued.** ABL-396 §3 and my ABL-411 Red Eléctrica read both say ES's
   overnight output is real CSP dispatch. BG shows what the rule does to a
   country whose night rows carry real MW *even when those MW are contaminated*;
   on ES they are genuine generation, so the case is strictly stronger. This is
   the direct answer to the ABL-411 step-3 question about whether the rule
   becomes a per-country registered property: **it should not become one on the
   strength of anything measured so far.**
3. **If the rule is ever wanted on a scope, it needs a score-side decision beside
   it.** §5e is the general statement: a fit-side exclusion is only defensible
   when the excluded rows are both (a) genuinely contaminated and (b) a small
   enough minority that the score is not dominated by them. FR under ABL-376
   meets both. BG meets (a) at best and fails (b) by a wide margin. There is no
   screen for (b) today; `f` from ABL-396's screen is the closest thing and would
   serve.
4. **`IMPOSSIBLE_NIGHT_THRESHOLD_MW = 1.0` is worth re-deriving if the rule is
   ever used.** It discards 45.9% of CH's night fit rows on readings that never
   exceed 5.84 MW against a ~5 GW fleet. That costs nothing measurable here, but
   it is not the targeted instrument the constant's docstring describes.
5. **Nothing here changes ABL-395's adopted feature list.** The geometry pair is
   still a small clean gain on CH (§4b) and still ambiguous on BG. `abl253` and
   `abl376` keep their `SCOPE_FEATURES` pins. No dispositioned read moves.

## 7. Power, and what this cannot say

The CEO asked for the ambiguity and the power rather than an extended seed count,
so:

- **8 seeds bounds the sign test at p = 0.0078.** 7/8 gives 0.0703 and 6/8 gives
  0.2891. The interaction on night MAE is 7/8, so **p < 0.05 was unreachable for
  it with one dissenting seed** — seed 113 reverses it (+29.91 MW). Calling that
  interaction *suggestive and clearing its null* rather than *established* is the
  honest reading, and no more seeds were run to move it.
- **The main effect needs no such hedging.** +61.05 MW at 8/8 with t = +9.59
  against a 6.96 MW null is not a marginal result.
- **Two countries, one 30-day summer holdout, one algorithm.** BG and CH bracket
  the night-floor question; they do not sample the fleet. The magnitude of the
  BG effect should be expected to depend on `f`, and this run does not measure
  that dependence.
- **The interaction null is deliberately conservative** — built from four
  *independent* control fits, where the real statistic combines four fits sharing
  a seed and therefore positively correlated. An effect clearing it has cleared
  more than it strictly had to. The p95 is quoted beside every max for that
  reason; the pre-registered rule was the max.
- **No mechanism is established for §5f**, only a refutation of the one ABL-395
  proposed.

## 8. Caveats and boundary

- Out of sample by target timestamp; gate targets were never fitted. All four
  arms scored identical rows, asserted in the probe.
- 69.3% of fit rows carry a degraded `lag_1d` — the serve-faithful schedule,
  identical in every arm, and the dominant feature-quality limit on these fits.
- Contamination: ABL-67 is net-position only, ABL-109/ABL-111 are load only,
  ABL-71's known wrong-write modes are load and net position. **None touches
  solar in this window.** BG's night floor (ABL-381 §5, ABL-396 §9.3) is a solar
  data defect none of those four covers, it is live in this window, and it is the
  subject of the run rather than a confounder that was overlooked.
- **Not a gate read. It dispositions nothing.** No promotion, no serving-registry
  change, no ingest change, no dashboard change, no replica write, no sidecar
  write, no artifact saved. It writes under no registered scope's
  `report_out`/`json_out`/`artifact_dir`. ABL-381's and ABL-405's PASSes stand as
  read, on the artifacts they were read on.
