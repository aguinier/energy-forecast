# ABL-443 — DE/NL `wind_offshore` at the corrected reference: NL's A is earned, DE's was not

**Read:** `reports/abl_443_offshore_trailing_reread.md` (report),
`reports/abl_443_offshore_trailing_reread.json` (machine record).
**Registration:** `experiments/ABL443/config.json` and
`reports/abl_443_trailing_reference_registration.md`, committed in the previous
commit, before the read computed a column.
**Scope:** `abl443-offshore-trailing` — a new scope. ABL-436's `abl322-pilot` read
stands byte-unchanged (§6).

Window: ABL-348's gate window **2026-07-11 → 2026-08-10** (30 d), fitted
2026-01-14 → 2026-07-11. **n = 720 / 720 / 510** per band per pair, 6 cells,
**6/6 reconstructed to 1e-09**. Out-of-sample by target timestamp, **one seed, one
holdout**. Baseline: seasonal-naive D-7 (G1), unchanged. Source table
`energy_generation`. **No refit** — this is arithmetic over ABL-436's committed
record plus two recomputed references.

---

## 1. The headline

| pair | ABL-436 (fit-window reference) | ABL-443 (trailing 28d) | why |
|---|:---:|:---:|---|
| **NL** `wind_offshore` | A | **A** | every condition holds, every margin readable |
| **DE** `wind_offshore` | A | **B** | fails G3 at 48-64h — and **all six** of its G2/G3 margins are inside the floor |

**The letters undersell the difference.** The number that matters is not which
letter moved, it is what happened to the margins:

| pair | G2 margin (level) | | G3 margin (shape) | |
|---|---:|---|---:|---|
| | **published** | **corrected** | **published** | **corrected** |
| DE 24-36h | +11.14pp | **+0.33pp** | +10.34pp | **+0.52pp** |
| DE 36-48h | +11.75pp | **+1.32pp** | +10.96pp | **+1.32pp** |
| DE 48-64h | +12.72pp | **+1.03pp** | +10.18pp | **−0.47pp** |
| NL 24-36h | +32.80pp | **+21.78pp** | +30.73pp | **+21.57pp** |
| NL 36-48h | +31.91pp | **+20.77pp** | +29.81pp | **+20.68pp** |
| NL 48-64h | +31.44pp | **+19.46pp** | +27.89pp | **+18.80pp** |

Readability floor: **7.51%** (ABL-385 `delta_min(k=1)` with `c_B = 0`, correct
against deterministic references).

- **DE: every one of its six G2/G3 margins sits inside the floor** — the widest is
  +1.32pp against a 7.51% floor. Its published +10 to +13pp margins were **the
  reference's mis-levelling, not the model's skill**. DE has not demonstrated that
  it predicts the level or the shape, in any band, in **either direction**. The
  single negative cell (−0.47pp) is equally unreadable: this is *not measured*,
  not *measured worse*.
- **NL: every margin stays readable**, at roughly 2.5–2.9× the floor after losing
  a third of its published width. NL's A is earned against a reference that is
  actually a level estimate.

## 2. What the correction actually did

The trailing reference sits between the fit-window one and the oracle in **all six
cells**, as §6 of the registration predicted, and the residual mis-levelling drops
by roughly two-thirds:

| pair | band | `c causal` | `c 28d` | `c oracle` | inflation causal → 28d |
|---|---|---:|---:|---:|---:|
| DE | 24-36h | 74.40% | 66.33% | 62.86% | 18.36% → **5.52%** |
| DE | 36-48h | 74.40% | 66.54% | 62.86% | 18.36% → **5.85%** |
| DE | 48-64h | 75.79% | 66.83% | 62.32% | 21.62% → **7.25%** |
| NL | 24-36h | 89.97% | 77.29% | 71.74% | 25.40% → **7.73%** |
| NL | 36-48h | 89.97% | 77.32% | 71.74% | 25.40% → **7.77%** |
| NL | 48-64h | 92.99% | 79.16% | 73.27% | 26.91% → **8.03%** |

**It reduces the mis-levelling; it does not remove it.** 5.5–8.0% residual remains
in every cell, and it is *larger* on NL than on DE despite NL's raw inflation being
worse — NL's level is still falling through the gate window faster than a 28-day
window can track. Do not quote the corrected reference as exact. The levels, from
the read's §5:

| pair | `c causal` (fixed) | `c 28d` min → max (mean) | `c oracle` |
|---|---:|---|---:|
| DE | 3263.8 MW | 1780.9 → 2865.9 MW (2343.6) | 2006.0 MW |
| NL | 1704.1 MW | 946.3 → 1226.5 MW (1129.4) | 870.3 MW |

The fit-window constant is **1.63×** (DE) and **1.96×** (NL) the correctly-levelled
one, and the trailing window moves across a 1.6× and 1.3× range within the gate
window — it is doing real work, not wearing a new name.

## 3. Is that level change real, or a revision vintage?

**Real seasonality. Screened, not assumed.** This mattered because ABL-439 found
NL `wind_onshore`'s 3× level shift on this same table (`energy_generation`) was a
**revision vintage**, not a fleet change — and a level shift of that shape is
exactly what the inflation diagnostic keys on. If these two pairs carried the same
artifact, the trailing window would be tracking a data seam and the challenger
would have been *fitted through it*.

Monthly means, both source tables, `wind_offshore`, over the fit + gate span
(2026-01-14 → 2026-08-10), replica read-only:

| month | DE gen | DE renew | ratio | NL gen | NL renew | ratio |
|---|---:|---:|---:|---:|---:|---:|
| 2026-01 | 5487.7 | 5476.1 | 1.00 | 1953.4 | 1944.8 | 1.00 |
| 2026-02 | 5491.1 | 5457.5 | 0.99 | 2750.1 | 2699.1 | 0.98 |
| 2026-03 | 3110.4 | 3102.4 | 1.00 | 1763.5 | 1759.2 | 1.00 |
| 2026-04 | 2560.6 | 2547.1 | 0.99 | 1521.5 | 1523.2 | 1.00 |
| 2026-05 | 2041.8 | 2041.8 | 1.00 | 1194.7 | 1255.8 | 1.05 |
| 2026-06 | 2083.0 | 2083.0 | 1.00 | 1464.8 | 1500.2 | 1.02 |
| 2026-07 | 2700.7 | 2700.7 | 1.00 | 1136.5 | 1145.7 | 1.01 |
| 2026-08 | 2336.4 | 2336.4 | 1.00 | 963.7 | 977.1 | 1.01 |

**The two tables agree to within 0–5% in every month, for both countries.** There
is no 2–3× seam of the ABL-439 kind on either pair. The level change is the winter →
summer offshore cycle: DE falls **2.7×** from February to May, NL **2.3×** from
February to May. That is the seasonal case ABL-437 was registered against, cleanly,
and this is the first pair set where the mechanism has been screened against its
confounder rather than inferred.

(Monthly **means** against a gate-window **median** oracle — the two are not the
same statistic, so read the table for the *seam*, which is what it screens for, not
as a second estimate of the level.)

## 4. What this does and does not say about DE

**Says:** DE still clears the registered bar readably. G1 is +24.05 to +26.11pp
against seasonal-naive D-7, far outside the floor, on all three bands. ABL-436's
**PASS stands** — the gate is D-7 and DE beats it. The B is about what that PASS
entitles the pair to, not about whether it passed.

**Says:** DE's grade is now the same shape as its already-recorded oracle problem.
ABL-436 §5 recorded that DE loses to `constant_oracle` in all three bands (−4.45 to
−6.15pp) and to `climatology_oracle` in all three (−5.69 to −7.89pp, the 48-64h one
**readable**). Re-derived here unchanged. Under the corrected causal reference the
picture is consistent: DE beats a flat line at the right level by an amount too
small to read, and loses to a hindsight one.

**Does not say:** that DE is worse than published. Every DE WAPE is byte-identical
to ABL-436's. Nothing about the challenger changed; the reference it was compared
against did.

**Does not say:** that DE fails. A −0.47pp G3 margin against a 7.51% floor is
**not demonstrated at one seed**. Resolving it needs k > 1 seeds under ABL-385's
protocol, not a verdict.

## 5. What it says about NL

NL is the stronger pair on every reading available:

- G1 +25.10 to +27.97pp, readable.
- G2/G3 +18.80 to +21.78pp at the corrected reference, readable at ~2.5–2.9× the
  floor.
- Beats **both** oracle references in all three bands readably (+12.99 to +15.73pp
  vs the oracle flat line; +11.87 to +14.71pp vs the oracle climatology).

Per ABL-418's ladder and this read, **NL `wind_offshore` is grade A on a corrected
reference, and it is one of the few pairs in the ABL-316 programme that beats the
oracles by a readable margin.** On the ABL-406/ABL-417 pattern — where a pass was
predicted by bar weakness — NL is the counter-case: its bar was *not* weaker than a
flat line, and it passed anyway.

## 6. Evidence hygiene

- **Zero deletions** in `experiments/` and `reports/` across this branch.
- ABL-436's record and both its reports are **byte-identical** to `origin/main`, by
  git blob hash: `results_abl436_offshore_reread.json` `a9d0e814cd8a`,
  `abl436_preregistration_recheck.json` `5faff5f069f1`,
  `abl_436_offshore_reference_grade.md` `c394f938886e`,
  `abl_436_offshore_evidence_pack.md` `a4bcebf6afec`. Also unchanged:
  `reports/abl_322_pilot_gate.md` `3da919cb47b0`, `reports/abl_418_retro_grade.{json,md}`
  `49343de23825` / `c74c9e9209f2`.
- ABL-437's registration, re-read and script are byte-identical to their branch.
- The refusal is **in the script** (`PROTECTED` in
  `scripts/abl443_offshore_trailing_reread.py`), derived from the registration and
  tested per path — not in the operator's memory. Verified firing.
- The read is **deterministic**: a second run reproduces the JSON exactly, modulo
  `generated_at`.
- Guards: `tests/test_abl443_offshore_trailing.py`.

## 7. Limits and contamination

- **One seed, one 30-day summer holdout.** Everything inside the 7.51% floor is
  unresolved, not decided. That is six of DE's twelve reported margins.
- **The corrected reference is not exact.** 5.5–8.0% residual inflation per cell,
  reported above rather than assumed away.
- **`constant_oracle` is not a per-cell optimum** — it is the median of the whole
  gate window, scored on a band subset (ABL-437 §1). It bounds; it does not gate.
- **G2/G3 remain sign tests.** ABL-437 declined to widen them to a floor test and
  this read does not either. Every margin is printed with its readability label,
  and the label moves no letter. Whether that widening should happen is a live
  question this read makes harder to defer — see §8.
- **Contamination:** ABL-67 is net-position-only; ABL-109 / ABL-111 are load-only.
  Neither intersects `wind_offshore`. ABL-71's known wrong-write modes are load and
  net position — a provenance caveat, not proof that wind ingest is pristine. The
  ABL-439 revision-vintage question is **screened and negative** for both pairs
  (§3). This read touches no ingest, no source table and no window.
- **The incumbent column is `Not measured` by construction** — DE and NL hold zero
  rows in `forecasts`. Unchanged from ABL-436.

## 8. What follows

Nothing here promotes anything, and nothing here is a promotion recommendation.
Two things for the ledger (ABL-316 §4.1/§5) and one for the CEO:

1. **NL `wind_offshore` — A, on a corrected reference, beating both oracles
   readably.** The strongest offshore result the programme has.
2. **DE `wind_offshore` — B.** Clears D-7 readably; level and shape not
   demonstrated at one seed, either way. Belongs beside the other pairs whose PASS
   is carried by a weak bar.
3. **For the CEO, not decided here:** ABL-437's re-read moved 11 pair-records on
   flip margins as tight as 0.36pp, and DE here moves on 0.47pp. That is now
   several pairs whose letter turns on a margin nobody can read at one seed. Adding
   a readability floor to G2/G3 would be a third registration change and must be
   pre-registered *before* it is read against any pair — proposing it after seeing
   which letters it moves is the defect this apparatus exists to prevent. Filing it
   is a CEO call.
