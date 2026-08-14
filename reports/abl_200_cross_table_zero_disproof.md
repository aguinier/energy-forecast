# ABL-200 — a zero in `energy_renewable` is adjudicated against `energy_generation`, not against a duration

**Status:** implemented at the single training read site; **no promotion, no refit, no gate re-read.**
**Measured:** live replica `C:\Code\able\data\energy_dashboard.db`, read-only, **2026-08-14**.
**Regenerable:** `scripts/abl200_cross_table_zero_census.py` (census, read-only),
`scripts/abl200_before_after_backtest.py` (paired A/B). Do not quote this file
without re-running the census — this database self-repairs, and an approved
remediation once acted on an enumeration of which 48% of the rows had already
healed four minutes later.

**Contamination touching these windows:** ABL-188 (the zero-fill this issue is
about) throughout; ABL-71/ABL-67/ABL-109/ABL-111 are not implicated — none of
them writes `energy_renewable`'s renewable columns, and the rule below fires on
no `net_position` or `energy_load` row by construction.

---

## 1. Headline

Two findings, and the first one is that **the issue's own premise overstates the
harm by two orders of magnitude on the pair it was filed about.**

1. **BE `wind_offshore`'s ~120 unguarded zero-runs are, in the main, not
   defects.** Re-measured today there are **105** flat-zero runs of 6 h or
   longer (the issue says ~130) and **9** of them clear the 24 h threshold — that
   second figure reproduces exactly. But of the **1,432 rows** inside those 105
   runs, `energy_generation` reads **negative on 1,378 and positive on 54**, and
   only **2 of the 105 runs** contain a single positive sibling value. A negative
   sibling is A75 netting — an idle farm drawing house load — and a gross `0.0`
   is the *correct* reading of it. Across all 2,214 of BE `wind_offshore`'s exact
   zeros: 56 have a positive sibling, **45 of those are already excluded by the
   existing 24 h rule**, and the cross-table rule's marginal contribution for
   this pair is **10 rows**.

2. **The rule is still worth landing, and the reason is fleet-wide rather than
   BE.** At the registered calibration quantile it excludes **564 rows across 38
   of 120 country/stream pairs** that the duration rule does not reach — cases
   like GR `wind_onshore` reading `0.0` against 3,059 MW in the twin, and ES
   `wind_onshore` against 17,500 MW.

**The 2,175 MW worked example is already excluded today.** It is BE
`wind_offshore` 2025-11-17 04:00, and it sits inside one of the 9 runs the 24 h
rule already catches (`rule24 = True`). The new rule's marginal contribution to
that window is the 11 rows on **2025-11-14 16:00 → 2025-11-15 17:00**, siblings
5.8–424.5 MW.

**ABL-198's own adjudicated window does not fire, and should not.** BE
`wind_offshore` 2026-03-08 09:00 → 2026-03-10 00:00, 40 rows: the twin reads
**−11.4 to −29.7 MW on all 40**. Already caught by the 24 h rule; correctly not
touched by this one.

---

## 2. The rule

For a series read from `energy_renewable`, a row whose value is exactly `0.0` is
**disproved** when `energy_generation` — the NaN-preserving twin of the same
fetch — reports, at the identical parsed instant, a value that is

- **non-NULL**, and
- **strictly positive**, and
- **larger than the disagreement the two tables routinely show on this pair.**

A disproved row becomes `NaN` — unadjudicated-missing, the encoding
`exclude_suspect_constant_runs` already uses and the encoding
`load_renewable_type_data` already gives two contradictory duplicate spellings.
**Nothing is repaired and no stored row is touched**; this is a read-site
exclusion, so the underlying term stays recoverable (precedent: ABL-412).

Run length does not enter the test at all.

### 2.1 Why the floor is calibrated per pair rather than chosen

Three measurements, each of which kills a simpler rule.

**A magnitude test must be one-sided.** `energy_generation` is signed
net-of-consumption; `energy_renewable` is floored and holds **no negative value
in any of the 120 pairs**. So `|sibling|` would null BE `wind_offshore`'s 2,158
netting zeros — the bulk of the very runs this issue was filed about.

**The two tables also disagree by revision vintage, in both directions, and by
wildly different amounts.** Measured over instants where `energy_renewable` is
positive: **32 of 100 comparable pairs agree bit-for-bit at least 99% of the
time**, while NL `wind_onshore` has `energy_generation` *higher* at 83.5% of
instants (median +311.8 MW — the ABL-439 seam). No single MW floor, and no floor
keyed to fleet size, can serve both.

**There is no empty band to hang a global threshold on.** Sibling value ÷ fleet
p99.5, over all 18,900 raw candidates:

| q05 | q10 | q25 | q50 | q75 | q90 | q95 | q99 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000065 | 0.000119 | 0.000217 | 0.000478 | 0.005937 | 0.052617 | 0.139509 | 0.568791 | 1.018 |

Continuous across four decades, no gap wider than 4× anywhere, and that one down
at a ratio of 1e-7. This is the **opposite** of ABL-431, whose 3.0 tolerance sits
inside a measured empty band 2.3× wide. A fleet-scaled floor here would be a
convention wearing a measurement's clothes.

So the floor is the pair's own inter-table disagreement:

> `floor = q0.99( |energy_renewable − energy_generation| )` over the instants
> where `energy_renewable` is **strictly positive**

— strictly positive so the zero-fill defect cannot raise the bar that would catch
it. Bit-identical pairs get a floor of exactly `0.0` and any positive sibling
disproves; vintage-divergent pairs set their own high bar and the rule falls
quiet on them, which is the right outcome for a pair whose two tables are known
to hold different vintages. Nobody chooses a number for NL.

### 2.2 The two registered constants

**`SIBLING_DISPROOF_QUANTILE = 0.99`.** Not a knife edge — the fleet-wide
marginal exclusion count moves 896 / 739 / **564** / 416 at q = 0.90 / 0.95 /
0.99 / 1.00, and no acceptance case in §4 changes verdict anywhere in that range.
0.99 is registered because the tail this floor exists to exclude is exactly the
1% of instants where one table has been revised and the other has not, and
because the conservative direction is refusing to null a row. **q = 1.00 was
considered and rejected**: one contaminated calibration row would set an
unreachable floor and silently disable the rule for that pair forever (DE
`wind_onshore`'s largest inter-table disagreement is 21,364 MW).

**`SIBLING_DISPROOF_MIN_CALIBRATION_ROWS = 1000`**, below which the rule
**refuses to adjudicate** and carries the reason (ABL-431's `evaluable` pattern).
**This one is a measurement.** 20 of the 120 pairs have a calibration population
of exactly **0** — the all-zero series, landlocked countries whose
`wind_offshore_mw` is `0.0` forever — and the smallest non-zero population is
**2,559**. Anything in (0, 2559) is the same rule today. Without the refusal
those 20 pairs would take a floor of 0.0, and any sibling value at all would
delete a new fleet's first output: the failure ABL-431 declined to risk in the
identical situation.

### 2.3 Two implementation properties that are load-bearing

**Alignment is on parsed instants, never on the stored string.**
`energy_renewable` stores BE's 2025-11-09 → 2025-11-25 rows in the ISO
`2025-11-14T16:00:00` form while `energy_generation` stores every row in the
`2025-11-14 16:00:00` form. A SQL `JOIN ... ON g.timestamp_utc =
r.timestamp_utc` returns NULL for **all 540** of them — including every row of
the worked example this rule exists for, which would have read as "no sibling,
nothing to adjudicate" instead of as 424 MW of hidden generation. (Relatedly:
those 540 rows are 180 duplicated instants and **all 180 disagree**, so ABL-321's
existing collapse already nulls them — which is why the 2025-11-17 18:00 →
2025-11-18 23:00 edge of that window needs no rule at all.)

**The new guard runs strictly *after* `exclude_suspect_constant_runs`.** That
guard measures a run over the observations present, so nulling rows inside a long
flat run first would split it at the new gap and drop both halves under
`min_run_hours` — rows the 24 h guard excludes today would start entering
training. In the registered order the two are strictly additive.
`test_the_new_rule_runs_after_the_duration_rule_and_cannot_weaken_it` pins it.

---

## 3. Extent, measured

`scripts/abl200_cross_table_zero_census.py`, 120 pairs, whole history, at the
registered q = 0.99. "Marginal" = excluded by this rule and **not** already
excluded by the 24 h rule.

| | |
|---|---:|
| pairs censused | 120 |
| pairs the rule refuses to adjudicate (no calibration population) | 20 |
| pairs where it fires | 38 |
| **marginal rows excluded** | **564** |
| marginal rows inside ABL-348's fit window (2026-01-14 → 2026-07-11) | **170** (24 pairs) |
| marginal rows inside ABL-348's gate window (2026-07-11 → 2026-08-10) | **0** |

Largest ten:

| pair | observed | exact 0.0 | already excl. (24 h) | floor (MW) | **marginal** | largest disproof (MW) |
|---|---:|---:|---:|---:|---:|---:|
| FI `hydro_total` | 29,101 | 92 | 0 | 0.40 | 92 | 2,153.1 |
| FI `solar` | 29,760 | 769 | 0 | 0.40 | 47 | 161.4 |
| FI `wind_onshore` | 29,760 | 64 | 0 | 0.40 | 31 | 2,258.7 |
| FI `biomass` | 29,037 | 28 | 0 | 0.41 | 28 | 1,198.7 |
| NO `wind_offshore` | 26,409 | 1,000 | 0 | 0.00 | 28 | 3.0 |
| NO `wind_onshore` | 26,409 | 28 | 0 | 0.00 | 28 | 2,151.8 |
| NO `hydro_total` | 25,741 | 28 | 0 | 902.36 | 28 | 21,729.4 |
| NL `wind_onshore` | 25,812 | 355 | 197 | 1,315.32 | 28 | 2,032.8 |
| GR `wind_onshore` | 13,818 | 26 | 0 | 190.00 | 26 | 3,059.0 |
| EE `hydro_total` | 23,761 | 3,945 | 12,113 | 0.00 | 25 | 0.7 |

The full 38-row table and every field is in `reports/abl_200_census.json`.

### 3.1 The gate window is untouched; some fit windows are not

**This changes the training set of 15 pairs under 7 registered scopes, and per
ABL-401 a gate re-read against a changed training set is a new
pre-registration.** Flagging it as the issue instructed; **no gate has been
re-read and no artifact refitted.**

| scope | pairs hit | rows |
|---|---|---|
| `abl406-tranche2b` | 4/8 | ES (6), FI (3), GR (24), IT (3) `wind_onshore` |
| `abl417-tranche2e` | 4/8 | EE (21), LT (1), LV (1), NL (1) `wind_onshore` |
| `abl316-t2d` | 3/6 | FI (12), LT (1), LV (2) `solar` |
| `abl316-t2c` | 2/5 | ES (6), GR (5) `solar` |
| `abl253` / `abl376` | 1/3 | FR `solar` (1) |
| `abl322-pilot` | 1/2 | NL `wind_offshore` (1) |
| `abl195`, `abl380-tranche1a`, `abl435-tranche2f`, `abl316-t1b`, `abl316-t2a` | 0 | — |

Three qualifiers on that, in decreasing order of comfort:

- **Gate truth and gate rows are byte-identical between the two arms.** Zero
  marginal rows fall in the gate window, so nothing about how a published cell
  was *scored* moves.
- **No gate row's D-7 baseline moves either.** The latest marginal row anywhere
  is **2026-07-03 00:45**; the earliest gate row's seasonal-naive lookback
  reaches back to 2026-07-04. The 168 h rolling features have the same bound.
- **The causal references do move**, and this is the one that is not free.
  `constant_causal` / `climatology_causal` at `fit_window` levelling read the
  whole fit window, which is where all 170 rows are; ABL-437's `trailing_28d`
  variant reaches back to 2026-06-13 for the first gate row and so picks up the
  tail of them. Any re-read must say which levelling it used, per ABL-437.

---

## 4. Acceptance cases

The three classes the issue registered, plus the ones the measurements added.
All in `tests/test_cross_table_zero_disproof.py`.

| # | case | required | result |
|---|---|---|---|
| 1 | BE `wind_offshore` 2025-11-14/15, twin at 5.8–424.5 MW | **excluded** | excluded (10–11 rows depending on q) |
| 2 | FR `wind_offshore` 2023-01-01 → 05-31, twin NULL for the identical span | **not** excluded by this rule | not excluded; still caught by the 24 h rule |
| 3 | genuine overnight solar zero, twin also 0 | **not** excluded | not excluded |
| 4 | BE `wind_offshore` 2026-03-08/10, twin at −11 to −30 MW (A75 netting) | **not** excluded | not excluded |
| 5 | all-zero series, no calibration population | refuse to adjudicate | `evaluable = False`, reason carried |
| 6 | vintage-divergent pair, disproof inside its own noise | **not** excluded | not excluded, floor set automatically |
| 7 | ordering: does the new rule weaken the 24 h rule? | must not | strictly additive in the registered order |
| 8 | excluded row's value | `NaN`, not the twin's number | `NaN`; caller's frame not mutated |

---

## 5. Before/after backtest

Paired A/B on ABL-348's registered windows, `scripts/abl200_before_after_backtest.py`.
Arm A reconstructs the pre-ABL-200 training set; arm B is the rule as landed.
Same algorithm (XGBoost, `config` defaults, early stopping off), same protocol,
**5 seeds per arm**, scored on identical gate rows. **Not a gate read**: it
registers no scope, writes no artifact and refits no serving pair.

Pairs chosen as the three where the rule removes the most fit-window rows: GR
`wind_onshore` (in `abl406-tranche2b`), EE `wind_onshore` (in
`abl417-tranche2e`), and IT `wind_offshore`, which sits under no registered scope
and is the control for "does a pair with no gate move at all".

Gate rows are **identical** between arms on all three pairs, as §3.1 predicts.

| pair | band | n | before | after | paired Δ | sd(before) | sd(after) | seeds better | \|Δ\| as % of own error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GR `wind_onshore` | 24-36h | 720 | 27.76% | 27.90% | +0.139pp | 0.70 | 0.59 | 2/5 | 0.50% |
| GR `wind_onshore` | 36-48h | 720 | 29.03% | 29.23% | +0.206pp | 0.62 | 0.52 | 3/5 | 0.71% |
| GR `wind_onshore` | 48-64h | 510 | 29.14% | 29.41% | +0.265pp | 0.54 | 0.80 | 2/5 | 0.90% |
| EE `wind_onshore` | 24-36h | 720 | 48.50% | 47.98% | **−0.512pp** | 0.76 | 0.38 | 4/5 | 1.07% |
| EE `wind_onshore` | 36-48h | 720 | 48.05% | 47.95% | −0.094pp | 0.50 | 0.55 | 4/5 | 0.20% |
| EE `wind_onshore` | 48-64h | 510 | 51.90% | 51.45% | −0.445pp | 0.41 | 1.01 | 3/5 | 0.86% |
| IT `wind_offshore` | 24-36h | 644 | 179.11% | 179.21% | +0.094pp | 2.50 | 5.91 | 3/5 | 0.05% |
| IT `wind_offshore` | 36-48h | 642 | 173.64% | 173.35% | −0.289pp | 3.71 | 5.18 | 2/5 | 0.17% |
| IT `wind_offshore` | 48-64h | 455 | 163.20% | 166.05% | +2.846pp | 3.46 | 4.57 | 1/5 | 1.71% |

Fit rows: GR 32,999 → 32,935 (−64); EE 32,987 → 32,823 (−164); IT 30,104 →
30,040 (−64). (Vintage-expanded rows, so one excluded hourly observation
contributes several.)

**Verdict: no readable effect in either direction, and that is the expected
result rather than a disappointing one.** Every one of the nine cells is inside
its own seed spread; the largest, IT 48-64h at +2.846pp, sits against seed
standard deviations of 3.46 and 4.57. Quoted the way ABL-406 quotes a margin —
against the challenger's own error — the biggest movement is **1.71%**, where
ABL-385's registered readability floor for the wind stream is **7.51%**
(`c_B = 0`, every reference here being deterministic). The seeds-better column
runs 1/5 to 4/5; a 4/5 one-sided sign test is p = 0.1875 and nothing reaches 5/5.

Two caveats a reader should carry:

- **This is the right size of change to expect.** The rule removes 21–26 hourly
  observations from fit windows of ~5,300 hours. A backtest that showed a large
  effect there would be evidence of something *else* moving, not of this rule
  working. The case for landing it is the correctness argument in §2 and the
  worked examples in §4, not a WAPE win.
- **IT `wind_offshore` is not a usable model in either arm.** 163–179% WAPE is
  worse than forecasting a flat zero, on both sides of the comparison. It is
  included here because it is the unscoped control and because its arms are
  paired, not as a claim about that pair. It sits under no registered scope; the
  finding is noted and belongs to whoever picks up that stream.

---

## 6. What this does not do

- **No promotion and no registry change.** The release call is the CEO's.
- **No ingest change.** ABL-414 is CEO-decided "no ingest change"; ABL-268's
  guard covers new writes. This is about history already on disk.
- **No row deleted or rewritten.** Read-site exclusion only.
- **No gate re-read**, and §3.1 names every scope for which one would now be a
  new pre-registration.
- **No `min_run_hours` change.** The 24 h duration rule is untouched, and item 1
  of the issue's ask — "should wind's threshold differ from solar's?" — is
  answered *no*, on the evidence in §1: the runs a lower threshold would have
  caught are, on BE `wind_offshore`, corroborated-consistent 103 times out of 105.
- **FR's 2023 window is not nulled retroactively** (item 1's second half). The
  rule declines to judge it, which leaves the existing 24 h rule's exclusion in
  place — the same practical outcome as nulling it, with no stored row touched
  and no claim made about whether the fleet existed.

### 6.1 A second, unguarded read site — found, not fixed here

`src/chronos2/input_builder.py:93-98` maps every renewable type to
`energy_renewable` and reads it with **its own SQL**, not through
`load_renewable_type_data`. That read has none of the guards this file has been
discussing: no ABL-188 duration rule, no ABL-200 disproof, no ABL-321
duplicate-instant collapse, and no ABL-332 hourly aggregation. Its `hydro_total`
is the NULL-propagating `(hydro_run_mw + hydro_reservoir_mw)` rather than
`db._HYDRO_TOTAL_EXPR`, which by the argument recorded over that constant erases
all rows for the nine countries reporting exactly one component.

**Not fixed in this PR**, deliberately: the issue registers the change at "the
single training read site, `energy-forecast/src/db.py` around line 653", and one
reviewable diff was the instruction. Its live blast radius is also currently
small — the `chronos-2` runner is `enabled: False` in `config.MODEL_RUNNERS`, so
no renewable forecast is being produced through it today. It is filed as a
follow-up rather than folded in.
