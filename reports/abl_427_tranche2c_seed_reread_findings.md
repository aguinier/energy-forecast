# ABL-427 — tranche 2c's `U(+)` pairs re-read at 12 seeds: IT and HR both return **U**

Parent **ABL-316**. Registration **ABL-427**
(`experiments/ABL427/config.json`), frozen and committed at `8a259b8` before the
first fit of the read. Everything ABL-348 registered — windows, bands, metric,
baseline, minimum n, source table — is inherited unchanged and named rather than
re-derived.

> **Disclosure on the freeze order, stated because it is not perfect.** A
> **2-seed smoke test on HR** (seeds 42 and 1337, the first two of the same
> registered list) ran at **13:09:54**, and the registration was committed at
> **13:10:49** — 55 seconds later. The smoke test existed to prove the rig would
> not crash and its outputs were deleted; the graded run began at 13:10:57. So
> the registration was frozen before every fit that produced a number in this
> pack, but **not** before literally every fit.
>
> What that can and cannot have influenced: it cannot have influenced the **seed
> list**, which is ABL-385's and was committed on 2026-08-13 (`f5c7136`), a day
> before this issue was assigned. It could in principle have influenced the
> **decision rule** — the choice to grade on the CV's upper 95% bound — since
> two HR draws were visible when it was written. Those two draws were HR 24-36h
> 14.9521% and 14.9633%, which is a 0.011pp gap and implies nothing usable about
> the CV. The reader should weigh that themselves rather than take my word for
> it; both numbers are in the final record, unchanged.

Machine record: `reports/abl_427_tranche2c_seed_reread.json`.
Generated tables: `reports/abl_427_tranche2c_seed_reread.md`.
Harness: `scripts/abl427_tranche2c_seed_reread.py`. Tests:
`tests/test_abl427_seed_reread.py` (11, green).

> **No promotion is requested or implied.** No serving-registry change, no write
> to `forecasts`, no ingest change, no dashboard change, no replica write, no
> artifact saved, and no row of the six solar-harness registration tables
> edited. `abl316-t2c`'s published outputs are byte-unchanged: this read is a new
> scope, `abl427-t2c-reread`. Promotion is a CEO-to-Board decision and this pack
> is evidence for it.

---

## Verdict

| pair | ABL-419 (k=1) | **ABL-427 (k=12)** | bands resolved |
|---|:---:|:---:|---|
| **IT** | `U(+)` | **`U`** | 48-64h resolves `A`; 24-36h and 36-48h do not |
| **HR** | `U(+)` | **`U`** | 24-36h and 36-48h resolve `A`; 48-64h does not |

**Neither pair is another `U(+)`.** `(+)` is ABL-418's instruction to *re-read at
k > 1 seeds*; this issue is that re-read, so its output is `A` or a plain `U`.
The ladder is a pure function of one cell's scores and cannot know the re-read
has happened — run on a k = 12 mean it still emits the letters `U(+)`, and
`_disposition` collapses them. A `U` here is the stronger statement: not
unresolved at one seed, but **still unresolved at twelve**.

**HR is a genuinely close call and the closeness is the finding**, not a
rounding detail — see §4. Under the registered rule HR is `U`; under either
alternative floor it is `A`.

ES was treated as out of scope, not optional, per the CEO's direction on this
issue. GR and PT are graded `C` by ABL-419 — readable losses, never `U(+)`.

---

## 1. The CV question, which the issue asked to settle first

The issue's hypothesis was that a pair-specific CV might make the **existing
k = 1 read** readable with no new fits at all, because ABL-402 measured
ABL-385's fleet p90 to be roughly 2× too wide on BG and CH.

**It does not, and the reason is that ABL-402's result does not transfer.**

| | measured per-fit CV on gate WAPE | vs fleet p90 (5.4328%) |
|---|---|---|
| ABL-402, **BG** solar (pooled) | 2.247% | 0.41× |
| ABL-402, **CH** solar (pooled) | 2.742% | 0.50× |
| **ABL-427, IT** solar (3 bands) | **4.935 – 5.069%** | **0.91 – 0.93×** |
| **ABL-427, HR** solar (3 bands) | **5.180 – 5.263%** | **0.95 – 0.97×** |

On IT and HR the fleet p90 is very nearly **exactly right** — within 3–9% of the
measured value, not a factor of two. BG and CH are the unusual pairs, not the
representative ones, and a fleet percentile quoted from them would have been too
narrow here rather than too wide.

So the free lunch is not available. At the measured point-estimate CV the k = 1
floor is 9.67–10.32% (against the nominal 10.65%), and **every one of the six
published k = 1 margins still sits inside it**:

| pair | band | ABL-419 skill at k=1 | k=1 floor at measured CV | readable? |
|---|---|---:|---:|:---:|
| IT | 24-36h | +4.75% | 9.71% | no |
| IT | 36-48h | +3.57% | 9.67% | no |
| IT | 48-64h | +8.13% | 9.94% | no |
| HR | 24-36h | +7.84% | 10.32% | no |
| HR | 36-48h | +7.17% | 10.15% | no |
| HR | 48-64h | +8.62% | 10.20% | no |

The seed spend was necessary. That is a negative result about the cheap path,
and it is reported first because the issue asked for it first.

---

## 2. The floor actually achieved at k = 12 (CEO request 1)

`delta_min(k) = 1.96 * sqrt(c_A^2 + c_B^2) / sqrt(k)` with **`c_B = 0`** — every
reference on the ladder (D-7, a flat line, an hour-of-day climatology) is
deterministic arithmetic on the actuals and does not move when the challenger is
refitted.

| pair | band | c_A measured | 95% CI on c_A | **floor at k=12, c_A upper 95%** | floor at k=12, point | nominal fleet floor at k=12 |
|---|---|---:|---|---:|---:|---:|
| IT | 24-36h | 4.952% | [3.51, 8.43]% | **4.77%** | 2.80% | 3.07% |
| IT | 36-48h | 4.935% | [3.50, 8.40]% | **4.76%** | 2.79% | 3.07% |
| IT | 48-64h | 5.069% | [3.59, 8.63]% | **4.88%** | 2.87% | 3.07% |
| HR | 24-36h | 5.263% | [3.73, 8.96]% | **5.07%** | 2.98% | 3.07% |
| HR | 36-48h | 5.180% | [3.67, 8.82]% | **4.99%** | 2.93% | 3.07% |
| HR | 48-64h | 5.202% | [3.69, 8.86]% | **5.01%** | 2.94% | 3.07% |

The registration fixed **before the fit** that the letter is decided on the
*upper* end of the CV's 95% interval. A CV from 12 draws is uncertain by roughly
−29%/+70%, and calling a letter on a point estimate would repeat one level up the
mistake ABL-385 was filed on.

### k = 4 could not have resolved anything

The CEO's k = 4 was priced against the nominal 5.32% floor and against the
published k = 1 margins. Both inputs move:

- At the **measured** CV, the k = 4 floor is **8.24 – 8.78%** (upper 95%), not
  5.32%. No margin in this tranche reaches that.
- The margins themselves shrink at k > 1 — see §3.

k = 12 is what produced any `A` at all. This is stated as a correction to my own
sizing in the issue body as much as to the CEO's comment: the issue's "at k = 4
the floor is 5.32%, which would make IT's +8.1% and HR's +7.2–8.6% readable" is
wrong on both terms.

---

## 3. The published single-seed headline was optimistic in 6 of 6 cells

| pair | band | ABL-419 skill (k=1, seed 42) | **k=12 mean skill** | change | seed 42's rank among the 12 |
|---|---|---:|---:|---:|:---:|
| IT | 24-36h | +4.75% | +3.99% | −0.77pp | 7 / 12 |
| IT | 36-48h | +3.57% | +2.69% | −0.89pp | 7 / 12 |
| IT | 48-64h | +8.13% | +6.29% | −1.84pp | 5 / 12 |
| HR | 24-36h | +7.84% | +5.39% | −2.44pp | 4 / 12 |
| HR | 36-48h | +7.17% | +5.37% | −1.80pp | 7 / 12 |
| HR | 48-64h | +8.62% | +4.62% | **−4.00pp** | 4 / 12 |

Every cell moves down. Two mechanisms, and they are separable by the rank column:

- **The WAPE distribution over seeds is right-skewed.** A few bad seeds drag the
  mean above the median, so even a seed at rank 7 of 12 — squarely mid-pack, as
  IT's are — sits *below* the mean WAPE and therefore *above* the mean skill.
- **Seed 42 was additionally a good draw on HR**, at rank 4 of 12 in two bands.
  That is luck, not selection: 42 is `config.CATBOOST_PARAMS`' long-standing
  pinned default, fixed years before these pairs were fitted and not chosen for
  them. But it is why HR moves twice as far as IT.

Neither mechanism is misconduct and neither invalidates ABL-419. What they mean
is that **a single-seed headline on solar should be read as a draw, not as the
pair's performance** — which is the general lesson ABL-385 registered and this is
a clean instance of it.

---

## 4. Where the registered rule and the better statistic disagree

`delta_min` is a **delta-method approximation** that exists in ABL-385 for a
specific reason: a k = 1 read carries no internal estimate of its own spread, so
the spread has to be imported from a fleet percentile. **At k > 1 that import is
unnecessary.** Skill vs D-7 is `100 * (1 - wape_j / d7)` for each of the twelve
fits, D-7 is one deterministic number, and those are twelve honest draws of
exactly the quantity being read. Student's t on them is the exact small-sample
test, and its degrees of freedom are already what accounts for the sd being
estimated.

Pairing a chi-square **upper bound** on the CV with a **z** critical value counts
that same uncertainty twice. On these six cells that double-count is worth about
1.7–2.1pp of floor, and it decides a letter:

| pair | band | k=12 mean skill | registered floor (c_A upper 95%) | letter | **95% t-CI on mean skill** | excludes 0? |
|---|---|---:|---:|:---:|---|:---:|
| IT | 24-36h | +3.99% | 4.77% | `U` | [+0.96, +7.01]% | **yes** |
| IT | 36-48h | +2.69% | 4.76% | `U` | [−0.37, +5.74]% | no |
| IT | 48-64h | +6.29% | 4.88% | `A` | [+3.27, +9.31]% | **yes** |
| HR | 24-36h | +5.39% | 5.07% | `A` | [+2.23, +8.56]% | **yes** |
| HR | 36-48h | +5.37% | 4.99% | `A` | [+2.26, +8.48]% | **yes** |
| HR | 48-64h | +4.62% | 5.01% | `U` | [+1.47, +7.77]% | **yes** |

**The two tests agree on five of six cells and disagree on IT 24-36h and
HR 48-64h** — and HR 48-64h is HR's *only* unresolved band, so that single cell
is the whole difference between HR grading `U` and grading `A`.

**I am reporting `U` for HR.** The rule that decides the letter was frozen before
the first fit precisely so it could not be chosen after seeing the numbers, and
switching now — to a statistic I can defend on the merits, in the direction that
turns my own pair into a pass — is exactly the move pre-registration exists to
forbid. ABL-404 is the standing lesson.

**What I recommend instead is a pre-registered amendment** (§7), which is the
route ABL-437 and ABL-444 both took.

Grade under each floor, for completeness:

| pair | band | registered (c_A upper 95%) | c_A point estimate | unamended fleet p90 |
|---|---|:---:|:---:|:---:|
| IT | 24-36h | `U` | `A` | `A` |
| IT | 36-48h | `U` | `U` | `U` |
| IT | 48-64h | `A` | `A` | `A` |
| HR | 24-36h | `A` | `A` | `A` |
| HR | 36-48h | `A` | `A` | `A` |
| HR | 48-64h | `U` | `A` | `A` |
| **pair** | | **IT `U` / HR `U`** | **IT `U` / HR `A`** | **IT `U` / HR `A`** |

IT is `U` under all three. **IT's disposition is not close and does not depend on
the amendment.** HR's does.

---

## 5. The number I would actually put in front of the Board

Across the twelve seeds, **1 to 3 fits in every cell lose to seasonal-naive D-7
outright**:

| pair | band | seeds losing to D-7 | worst seed's skill | best seed's skill |
|---|---|:---:|---:|---:|
| IT | 24-36h | 2 / 12 | −6.40% | +9.55% |
| IT | 36-48h | 2 / 12 | −7.52% | +8.52% |
| IT | 48-64h | 1 / 12 | −2.86% | +12.18% |
| HR | 24-36h | 2 / 12 | −3.88% | +11.97% |
| HR | 36-48h | 2 / 12 | −4.15% | +11.12% |
| HR | 48-64h | 3 / 12 | −4.41% | +10.28% |

The sd of skill is **4.75–4.98pp** in every cell, against mean margins of
2.69–6.29pp. So a **single-seed** solar model on these pairs carries roughly a
**1-in-6 to 1-in-4 chance of being worse than a seasonal-naive baseline**, and
which one you get is decided by the random seed.

That is a larger and more actionable fact than either letter. It is a serving
question and I do not own serving — but the evidence says the unit worth grading
and the unit worth serving may both be a **seed ensemble** rather than one fit,
and that is a cheap change to evaluate (12 fits per pair cost 45 seconds here).
I have not evaluated it, because it is outside this issue's registered scope.
Filed as a recommendation in §7, not done.

---

## 6. Reproduction controls — both passed

**The gate-window actuals have not moved.** The replica has grown since ABL-419
(9,444,216,832 bytes now against the 9,432,453,120 that run recorded) and the
gate window 2026-07-11 → 2026-08-10 lies inside `energy_generation`'s revision
horizon, so this was a live risk rather than a formality.

- **30 of 30** pre-existing deterministic comparator cells reproduce ABL-419's
  committed values **exactly** (tolerance 1e-9pp): `seasonal_naive`,
  `constant_causal`, `constant_oracle`, `climatology_causal`,
  `climatology_oracle`, in all six cells.
- 12 cells read `absent_from_abl419_record` — ABL-437's `constant_causal_28d` and
  `climatology_causal_28d`, columns that did not exist when ABL-419 was written.
  A schema addition, not a moved actual. The check distinguishes the two
  explicitly so it cannot manufacture a revision scare.
- **Seed 42 reproduces all six published challenger WAPEs to +0.0000pp.**

**Determinism.** The full run was executed three times; all 72 per-seed WAPEs are
identical across runs.

**Provenance.** ABL-385's seed list was frozen by `f5c7136` (2026-08-13 13:14:40),
ABL-427's registration by `8a259b8` (2026-08-14 13:10:49), and the graded run
began at 13:10:57 — see the disclosure at the top of this pack for the 2-seed
smoke test that preceded the commit by 55 seconds. The script reads the seed list
from `experiments/ABL385/config.json` at run time and **refuses to run** if it
differs from the literal it pins, so the anti-selection property is enforced
rather than asserted.

---

## 7. Recommendations

1. **Amend ABL-316's ledger**: IT `U(+)` → **`U`**, HR `U(+)` → **`U`**. Both are
   now re-read; neither carries a further "re-read at k>1" instruction. ES
   remains `U(+)` and untouched by this issue.
2. **Neither pair is promotion-eligible today.** IT is not close. HR is one band
   away and that band turns on the choice of test, not on the data.
3. **Pre-register an amendment** replacing `delta_min` with the Student-t
   interval on the k seed draws **for k > 1 reads only** — `delta_min` remains
   correct and necessary at k = 1, where there is nothing to take a t of. I will
   file this as a child issue rather than act on it here, and HR should be
   re-graded under it once registered, not before. Expected effect on the
   published record: HR resolves `A`, IT 24-36h resolves `A` with IT still `U`
   overall on 36-48h. No other committed tranche cell is at k > 1, so the blast
   radius is this issue alone.
4. **Evaluate seed-ensembling as the serving unit**, on the §5 evidence. This is
   a Founding Engineer / Deployment Engineer question once someone decides it is
   worth asking; the measurement to justify asking is above.
5. **ABL-385's fleet p90 should not be quoted as "known to be 2× too wide."**
   ABL-402 established that for BG and CH. On IT and HR it is accurate to within
   3–9%. Any read relying on the wider claim should measure its own pair.

## Caveats, stated before being asked

- **Window and n.** Fit 2026-01-14 → 2026-07-11; gate 2026-07-11 → 2026-08-10,
  out-of-sample by target timestamp. n = 720 / 720 / 510 per band against
  registered minima 684 / 684 / 456 — **all six cells clear their minimum n**, so
  the defect ABL-434 is open on (a ladder that can grade a coverage-short cell
  `A`) does not touch this verdict. Checked, not assumed.
- **Out-of-sample**, on every number here. No cell is fitted on data it is
  scored from; the seeds change only `random_seed`.
- **Contamination.** ABL-67 is `net_position`-only; ABL-109 and ABL-111 are
  `energy_load`-only — neither table is read here. ABL-71's known wrong-write
  modes are load and net position: a provenance caveat on this window, not proof
  that solar ingest is pristine. ABL-188 found no ≥24-hour bit-identical solar
  run in `energy_generation` for either pair across 2025-12-31 → 2026-08-10.
  ABL-332's sub-hourly aggregation fires on both pairs and is the registered
  behaviour, identical to ABL-419's: HR 21,408 quarter-hourly rows to 5,352
  hourly means with **0** partial hours, IT 21,405 to 5,352 with **3** partial
  hours.
- **`c_B = 0` throughout.** If a future amendment puts a *fitted* reference on
  the ladder, every floor here widens by √2 and must be recomputed.
- **The CV is per (country, band) and the three bands of one country share a
  fit**, so they are not three independent estimates. Pooled and maximum values
  are in the machine record; the letter is decided per cell, which is the unit
  ABL-418 grades.
- **Normality.** The t-interval in §4 assumes it. With 12 draws that is not
  testable to any useful power, which is one more reason §4 is a proposed
  amendment rather than a re-grade.
- **In-sample / out-of-sample labelling.** Every metric in this pack is
  out-of-sample. No in-sample number is quoted anywhere.
