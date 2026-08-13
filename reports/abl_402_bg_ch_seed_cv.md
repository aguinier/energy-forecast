# ABL-402 - the per-fit seed CV of ABL-381's challenger on BG and CH solar

Generated 2026-08-13 19:00 UTC. Replica `C:/Code/able/data/energy_dashboard.db`, opened read-only. Interpreter: the rail (`.venv`, Python 3.14.3, CatBoost).

**20 seeds**, frozen and committed before the first fit at `bf8e1cc01d98` (2026-08-13T20:37:57+02:00), disjoint from 42: `[211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317]`. Seed 42 is fitted once per pair as a reproduction control and is excluded from every CV below.

Registration is ABL-348's, read and not re-derived: fit 2026-01-14 -> 2026-07-11, gate 2026-07-11 -> 2026-08-10, source `energy_generation`, basis ['challenger', 'seasonal_naive'], **25 features** (legacy25 (ABL-381's challenger), pinned via LEGACY_FEATURE_COLUMNS).

## 0. What this settles

ABL-381 read its two climatology margins against **ABL-385's fleet percentile**, because no
pair-specific CV existed for BG or CH. This measures the pair-specific CV and re-reads them.

1. **CH's headline margin is readable at one seed, and not marginally.** CH's own per-fit CV
   on the 24-36h cell is **3.02%** (95% CI 2.30-4.42%), so `delta_min(1) = 5.92%` of its own
   error. The margin is **10.48%** at the pinned seed and **8.88%** at the 20-seed mean. It
   clears on both. ABL-381's "marginal at p90" was an artefact of the fleet percentile, which
   is **1.8-2.4x wider** than either of these pairs' actual CV. The qualification ABL-381
   section 9 recommended for a Board reading can be dropped for this cell.

2. **BG's withdrawal is confirmed, and it should be stated more strongly than "unreadable".**
   At the pinned seed BG's 24-36h margin is +1.43% of its own error against a 4.94% bar. At
   the 20-seed mean it is **+0.04%** -- a dead tie, needing over 12,000 seeds to resolve. On
   the 48-64h band the mean margin is **negative** (-0.28%): the challenger is behind the
   hindsight climatology there. BG has not been shown to beat the average day, and the point
   estimate at its central value is that it does not beat it at all.

3. **New, and not in ABL-381: CH's weakest cell does not resolve either.** ABL-381 quoted the
   24-36h cell. On **48-64h** CH's margin is +3.59% at the pinned seed and **+1.74%** at the
   mean, against a 4.97% bar -- **not readable**, and it would take 9 seeds. So CH beats an
   hour-of-day climatology on the two shorter bands and is unresolved on the longest one.

4. **The metric caveat ABL-381 raised against itself does not bite.** WAPE and daylight-MAE
   CVs agree to within 0.10pp on every cell (section 2).

**None of this moves the disposition.** The registered bar is seasonal-naive D-7, which is
deterministic, and both pairs clear it by 19.9-36.8% -- readable at one seed several times
over under any CV here. What is bounded is the *reference* comparison, which is the point of
reporting a reference.

## 1. The reproduction control - is this ABL-381's challenger?

Seed 42 is the gate's pinned seed. If the rig is fitting the model ABL-381 published,
it returns ABL-381 section 3's table. Published values are quoted there to 2 decimals,
so agreement is bounded by their own rounding.

| country | band | n | challenger here | published | delta | oracle climatology here | published | delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BG | 24-36h | 720 | 18.8852% | 18.89% | -0.0048 | 19.1549% | 19.15% | +0.0049 |
| BG | 36-48h | 720 | 18.5989% | 18.60% | -0.0011 | 19.1549% | 19.15% | +0.0049 |
| BG | 48-64h | 510 | 20.0272% | 20.03% | -0.0028 | 20.3830% | 20.38% | +0.0030 |
| CH | 24-36h | 720 | 8.1617% | 8.16% | +0.0017 | 9.0172% | 9.02% | -0.0028 |
| CH | 36-48h | 720 | 8.0072% | 8.01% | -0.0028 | 9.0172% | 9.02% | -0.0028 |
| CH | 48-64h | 510 | 8.3943% | 8.39% | +0.0043 | 8.6958% | 8.70% | -0.0042 |

**Reproduced.** Largest disagreement across all 12 readings: **0.0049pp**, which is within the rounding of the published table. The measurement below is of ABL-381's challenger and not of a neighbouring model.

**Determinism, checked rather than assumed.** The whole sweep was run twice -- once before
and once after `origin/main` (ABL-381, ABL-395) was merged into this branch, the second time
with the 25-feature list pinned explicitly rather than inherited. **286 paired readings,
largest disagreement 0.00e+00.** That is what licenses reading the spread below as a seed
effect: the only thing moving between these fits is `random_seed`. It also witnesses that the
explicit pin resolves to exactly what the pre-ABL-395 tree resolved to on its own.

## 2. The measured per-fit CV

`cv = sd/mean` across the seeds, sample sd (ddof=1). The interval is ABL-385's chi-square one:
a sd from n draws is itself an estimate, and at 20 seeds it is uncertain by about -24%/+46%.

| country | band | n | mean WAPE | sd | **CV** | 95% CI on CV | seed range | daylight MAE CV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BG | 24-36h | 720 | 19.146% | 0.482 | **2.52%** | 1.91-3.68% | 1.774pp | 2.60% |
| BG | 36-48h | 720 | 18.935% | 0.447 | **2.36%** | 1.80-3.45% | 1.658pp | 2.46% |
| BG | 48-64h | 510 | 20.441% | 0.397 | **1.94%** | 1.48-2.84% | 1.351pp | 2.04% |
| CH | 24-36h | 720 | 8.282% | 0.250 | **3.02%** | 2.30-4.42% | 0.977pp | 3.02% |
| CH | 36-48h | 720 | 8.247% | 0.249 | **3.02%** | 2.30-4.41% | 0.942pp | 3.05% |
| CH | 48-64h | 510 | 8.547% | 0.217 | **2.53%** | 1.93-3.71% | 0.868pp | 2.53% |
| **BG** | **pooled D+2** | 1,950 | 19.453% | 0.437 | **2.25%** | 1.71-3.28% | 1.593pp | 2.34% |
| **CH** | **pooled D+2** | 1,950 | 8.354% | 0.229 | **2.74%** | 2.08-4.01% | 0.880pp | 2.75% |

**Against the fleet percentile ABL-381 had to use** (solar, ABL-385): median 2.32%, p80 4.47%, p90 5.43%.
Both pairs sit near the fleet **median** and far below p90 -- the percentile was
conservative here by roughly a factor of two, which is exactly the direction ABL-385
warned a fleet number could be wrong in and the reason it said to prefer a pair-specific CV.

**The metric caveat ABL-381 raised against itself is empirically small.** ABL-385 reads solar
on daylight MAE while these margins are whole-window WAPE. Measured on the same rows and the
same fits, the two CVs differ by at most 0.10pp on BG and 0.03pp on CH. The relative spread
does transfer between the two metrics on these pairs.

## 3. The margins re-read

One arm is fitted and the reference is deterministic arithmetic on the actuals, so `c_B = 0`
and `delta_min(k) = 1.96 * c_A / sqrt(k)` -- the two-fitted-arm margin over sqrt(2).
Each margin is read twice: at the pinned seed 42 (what ABL-381 published) and at the
20-seed mean (the challenger's actual central value).

| country | band | oracle climatology | margin @ seed 42 | as % of own | readable? | margin @ mean | as % of own | readable? | delta_min(1) | seeds needed @ mean |
|---|---|---:|---:|---:|:--:|---:|---:|:--:|---:|---:|
| BG | 24-36h | 19.155% | +0.270pp | +1.43% | **no** | +0.009pp | +0.04% | **no** | 4.94% | 12,118 |
| BG | 36-48h | 19.155% | +0.556pp | +2.99% | **no** | +0.220pp | +1.16% | **no** | 4.63% | 16 |
| BG | 48-64h | 20.383% | +0.356pp | +1.78% | **no** | -0.058pp | -0.28% | **no** | 3.80% | 179 |
| CH | 24-36h | 9.017% | +0.856pp | +10.48% | yes | +0.735pp | +8.88% | yes | 5.92% | 1 |
| CH | 36-48h | 9.017% | +1.010pp | +12.61% | yes | +0.771pp | +9.34% | yes | 5.92% | 1 |
| CH | 48-64h | 8.696% | +0.301pp | +3.59% | **no** | +0.148pp | +1.74% | **no** | 4.97% | 9 |

## 4. Every published cell is a favourable draw

Seed 42 against the mean of the 20, in units of the cell's own seed sd:

| country | band | seed 42 | mean | difference | in sd |
|---|---|---:|---:|---:|---:|
| BG | 24-36h | 18.885% | 19.146% | -0.261pp | -0.54 |
| BG | 36-48h | 18.599% | 18.935% | -0.336pp | -0.75 |
| BG | 48-64h | 20.027% | 20.441% | -0.414pp | -1.04 |
| CH | 24-36h | 8.162% | 8.282% | -0.120pp | -0.48 |
| CH | 36-48h | 8.007% | 8.247% | -0.239pp | -0.96 |
| CH | 48-64h | 8.394% | 8.547% | -0.153pp | -0.71 |

**6 of 6 cells land on the favourable side of their own mean**, by 0.48 to 1.04 sd.

This is *not* six independent draws and must not be read as one: the three bands of a pair
come from a single fit per seed, so there are effectively **two** draws here, and two
favourable draws is a coin flip twice. It is not evidence that 42 is a special seed, and
nothing here suggests the seed was chosen -- it is `config.CATBOOST_PARAMS`'s default and
predates every one of these pairs.

What it *does* establish is narrower and still worth stating: **every margin in ABL-381's
published table is larger than the same margin at the challenger's central value**, on both
pairs and all three bands. A reader taking the published cells as the model's typical
performance is reading a number that is 0.12-0.41pp optimistic.

## 5. Independent cross-check

ABL-395 measured the same quantity on the same two pairs at the same 25 features over
ABL-376's eight registered seeds (101-137), for a different purpose. It is the only external
check on this CV, and the seed sets are disjoint from these 20.

| country | band | ABL-402 mean +- sd (20 seeds) | CV | ABL-395 mean +- sd (8 seeds) | CV | CV intervals overlap |
|---|---|---:|---:|---:|---:|:--:|
| BG | 24-36h | 19.146 +- 0.482 | 2.52% | 19.407 +- 0.471 | 2.43% | yes |
| BG | 36-48h | 18.935 +- 0.447 | 2.36% | 19.185 +- 0.460 | 2.40% | yes |
| BG | 48-64h | 20.441 +- 0.397 | 1.94% | 20.675 +- 0.425 | 2.06% | yes |
| CH | 24-36h | 8.282 +- 0.250 | 3.02% | 8.346 +- 0.154 | 1.85% | yes |
| CH | 36-48h | 8.247 +- 0.249 | 3.02% | 8.299 +- 0.155 | 1.87% | yes |
| CH | 48-64h | 8.547 +- 0.217 | 2.53% | 8.704 +- 0.160 | 1.84% | yes |

All six overlap. The agreement is close on BG (2.52 vs 2.43%) and looser on CH, where this
read finds **3.02%** against ABL-395's **1.85%** -- a factor of 1.6. Neither is wrong: an
8-seed CV carries a -34%/+96% interval, and the two intervals overlap across their whole
width. It is a useful reminder that **the CV is itself a noisy estimate**, and the reason
this report quotes intervals rather than points.

Every conclusion below is stated against **this read's larger CH CV**, which is the
conservative choice: CH's margin clears its bar on the wider of the two measurements.

## 6. What this sizes for the remaining 33 tranches

They face this question identically, and the answer is now cheap to state.

**The bar, in one line.** Against a deterministic reference (`c_B = 0`), at the gate's single
pinned seed, a margin is readable at two-sided 95% only if it exceeds **`1.96 * c`** of the
challenger's own error. On the two pairs measured here `c` is **1.9-3.0%**, so the bar is
**roughly 4-6% of the challenger's own error**. Below that, a one-seed climatology comparison
says nothing in either direction.

**Three practices this supports, none of which needs a new registration:**

1. **Quote every reference margin as a percentage of the challenger's own error, not in pp.**
   BG's 0.26pp and CH's 0.86pp are a 3.3x ratio in pp and a 7.5x ratio in the units that
   decide readability. The pp figure is the one that misleads.
2. **Report the weakest cell, not the headline one.** CH's 24-36h cell resolves and its
   48-64h cell does not, from the same pair in the same run. A pack that quotes one cell in
   prose is quoting the one it happened to look at first.
3. **A pair-specific CV costs about 8 minutes and replaces a percentile that was 2x wrong
   here.** Frame building dominates (~4 min/pair); the 20 refits are ~80 s/pair. Where a
   margin lands anywhere near the bar, measure it rather than reading the fleet number.

**What does not transfer.** This CV belongs to *these two pairs at 25 features*. It is not a
fleet constant and the remaining 33 should not adopt it as one -- ABL-385's own spread across
16 solar units runs 1.4-6.1%, a factor of four. Note also that a solar gate fit from ABL-395
forward is a **27-feature** challenger; a CV measured here does not automatically carry to it.

## 7. Contamination

- **ABL-337** (physically impossible night solar actuals): the scope registers
  `exclude_impossible_night: False`, so the fit frame is unfiltered, identically to
  ABL-381's read. BG's overnight floor (ABL-381 section 5) is inside these fits, as it was
  inside the published ones. Measuring the spread under a different fit rule would not be a
  spread of the read being re-read.
- **ABL-188** constant-run screen: applied by `db.load_renewable_type_data` to whatever table
  is read, so it is identical across seeds by construction.
- **ABL-71**: provenance caveat only; known wrong-write modes are load and net position.
- **ABL-67, ABL-109/ABL-111**: net position and load. Neither touches this scope.

**Why contamination matters less here than usual, and where it does not.** Every quantity in
section 2 is a spread *within* a fixed (pair, band, arm) cell across seeds only. Contamination
identical across the seeds of a cell shifts the cell's level and cancels out of its CV. The
margins in section 3 have no such protection -- they are level comparisons, and they inherit
ABL-381's contamination exposure unchanged.

## 8. Boundaries

- Evidence only. No promotion, no serving change, no registry change, no ingest change.
- The replica is opened read-only and nothing is written to it.
- **No artifact is written and no dispositioned cell is re-scored.** `save_gate_artifact` is
  never called; `experiments/ABL316/artifacts` is untouched. Pinned by
  `tests/test_abl402_seed_cv.py::TestBoundary`.
- None of the five registration tables is edited. `abl316-t1b`'s rows in `SCOPES`,
  `GATE_BASIS`, `SCOPE_OUTPUTS`, `FIT_RULES` and `SCOPE_TITLES` are read, never written.
- This changes no pre-registered gate. It supplies an error bar for a *reported reference*
  comparison; the registered bar is seasonal-naive D-7 and it is untouched.

