# ABL-435 — ABL-316 tranche 2f: BG and CH `wind_onshore` re-read against the reference suite and the ABL-418 ladder

**Gate disposition: PASS, 6/6 cells — and it reproduces ABL-380 bit-exactly.**
**Graded disposition: BG wind_onshore = A. CH wind_onshore = A.**
**The issue's premise does not survive the ladder as registered: CH is *not* a G2 failure, and the
reason it is not is itself the finding. See §4.**

Gate read: `reports/abl_435_wind_onshore_tranche2f.md` (harness-generated).
Machine record: `experiments/ABL348/results_abl435_tranche2f.json`.
Registration: `experiments/ABL348/config.json`, frozen at ABL-348 and not re-derived here.
Scope: `abl435-tranche2f`, registered in `scripts/evaluate_wind_retrain.py` at commit `ebd6aa6`.

No promotion, no serving-registry change, no ingest change, no dashboard change, no replica
write, no write to `forecasts`. Promotion remains CEO-to-Board.

---

## 1. What this issue was for, and what it actually settles

Tranche 1a (ABL-380) was the only ABL-316 wind read with **no model-free reference in its
committed machine record and no grade**. It was fitted 2026-08-13 at 08:32Z — before ABL-389
added the four references (`75adff8`) and before ABL-418 registered the G1–G4 ladder
(`5bf2f4f`). ABL-418 could retro-grade tranches 2a and 2b by arithmetic because their stored
results files already carried `constant_causal` and `climatology_causal` as columns. Tranche
1a's carries `challenger, seasonal_naive, incumbent, persistence` and nothing else, so **G2 and
G3 cannot be derived from it at all**. The grade needed the columns; the columns needed a run.

That run is done. Three things came out of it, in ascending order of importance:

1. The read **reproduces ABL-380 to the digit** (§2). That is worth having on its own — it is
   the first end-to-end reproduction of a dispositioned gate read in this programme.
2. BG and CH both grade **A** (§3).
3. **The `A` on CH is not what an `A` normally means, and the ladder as registered does not
   catch it** (§4). That is a finding about the ladder, not about CH, and it is the durable
   output of this issue.

---

## 2. Reproduction — bit-exact, and what it does and does not prove

Same two pairs, same fit window (2026-01-14 → 2026-07-11 exclusive), same gate window
(2026-07-11 → 2026-08-10 exclusive), same bands, same WAPE metric, same literal seasonal-naive
D-7 bar, same registered minimum n, same `energy_generation` source, same catboost at
`random_seed: 42`. **Nothing about the registration was moved.** The re-read is a new scope with
new output paths, which is the mechanism doctrine prescribes for re-reading a dispositioned
scope; ABL-380's record is untouched (§7).

| pair | band | n | challenger ABL-380 | challenger ABL-435 | D-7 ABL-380 | D-7 ABL-435 |
|---|---|---:|---:|---:|---:|---:|
| BG | 24-36h | 720 | 56.8624% | **56.8624%** | 93.7529% | **93.7529%** |
| BG | 36-48h | 720 | 56.8236% | **56.8236%** | 93.7529% | **93.7529%** |
| BG | 48-64h | 510 | 57.7558% | **57.7558%** | 89.3213% | **89.3213%** |
| CH | 24-36h | 720 | 47.4180% | **47.4180%** | 59.2647% | **59.2647%** |
| CH | 36-48h | 720 | 44.9881% | **44.9881%** | 59.2647% | **59.2647%** |
| CH | 48-64h | 510 | 44.3095% | **44.3095%** | 59.8129% | **59.8129%** |

**Maximum absolute difference over every cell × every comparator × every field
(`n`, `wape_pct`, `mae`, `bias_pct`, `slope`, `correlation`): 0.0.** Fit audit identical too —
34,176/34,176 retained rows and 23,674 degraded lag-1d rows for both pairs, unchanged.

Preconditions were re-measured before fitting (`scripts/abl380_tranche_precheck.py`, read-only)
and land on ABL-380 §1 exactly: BG 720/720 gate hours, D-7 93.75%, mean 108.9 MW; CH 720/720,
D-7 59.26%, mean 12.9 MW; **zero ABL-188 constant runs** in fit, gate or feature-lookback
window for either pair; 720/720 hours bit-identical against `energy_renewable`, max abs
difference 0.0 MW.

Two honest limits on what this proves:

- **The artifact SHA-256 does not witness it and cannot.** BG moved
  `eb0f63d8…` → `86f8c565…` and CH `5d2ec407…` → `6339dc3b…`. `Forecaster.save` stamps
  `saved_at`, so a byte-identical model serialises to a different hash on every run. The
  reproduction claim rests on prediction equality, which is the thing that matters and is
  measured above. Do not read the hash change as a refit that landed somewhere else.
- **It is a reproduction of a deterministic pipeline on unchanged data, not a robustness
  result.** It says the harness, the replica and the seed still produce what they produced
  yesterday. It says nothing about a different seed, and ABL-385 is explicit that a one-seed
  wind gap under 7.51% is not readable at all.

**One number moved, and it is the one already labelled unreliable.** CH's TSO WAPE went
27.7790% → **27.2601%** (−0.52pp) on an unchanged n=1,950; BG's is unchanged to four decimals.
The TSO series comes from a replacement table with no first-seen vintages, so it carries
revisions by construction — that caveat is in every wind report this programme has produced,
and this is the first time it has been *observed* rather than asserted. It remains context and
cannot support promotion.

---

## 3. The gate read and the graded disposition

Readability floor for wind at k=1: **7.5054%** (ABL-385's `delta_min` with `c_B = 0`).

| pair | band | challenger | D-7 | skill vs D-7 | `constant_causal` | `climatology_causal` | slope | corr | gate | grade |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| BG | 24-36h | 56.86% | 93.75% | +39.35% | 82.77% | 81.03% | 0.268 | 0.575 | PASS | **A** |
| BG | 36-48h | 56.82% | 93.75% | +39.39% | 82.77% | 81.03% | 0.261 | 0.563 | PASS | **A** |
| BG | 48-64h | 57.76% | 89.32% | +35.34% | 86.90% | 82.72% | 0.249 | 0.502 | PASS | **A** |
| CH | 24-36h | 47.42% | 59.26% | +19.99% | 79.07% | 77.82% | 0.078 | 0.138 | PASS | **A** |
| CH | 36-48h | 44.99% | 59.26% | +24.09% | 79.07% | 77.82% | 0.086 | 0.170 | PASS | **A** |
| CH | 48-64h | 44.31% | 59.81% | +25.92% | 78.36% | 73.47% | 0.131 | 0.246 | PASS | **A** |

No cell is near the floor: the smallest G1 margin is CH's +19.99%, 2.7× the floor, and it holds
on the own-error denominator too (+24.98%). So there is no `U` cell and the
skill-vs-own-error sensitivity ABL-418 §2 asks for changes nothing here.

**BG's bar weakness is confirmed exactly as the issue states it.** `constant_causal` scores
82.77% against BG's registered D-7 bar of 93.75% — a flat line at the fit-window mean clears
BG's bar with no model at all, and the harness flags `bar_weaker_than_a_flat_line = yes` for
all three BG bands. The pass is real anyway, on the ladder and on the oracle references (§4),
but **the D-7 bar is not what established it**, and no future reading of BG should cite the
+39% skill figure without this line beside it.

---

## 4. The finding: CH grades `A`, and that is the ladder failing rather than CH passing

The issue predicted CH would come back a **G2 failure** — "challenger 47.42% vs an oracle
constant at 40.29% … under the ABL-418 ladder that is a G2 failure, so CH cannot be grade A."
**It does not, and the reason is a registration detail worth being precise about rather than
arguing around.**

**G2 is registered on `constant_causal`, not `constant_oracle`.** ABL-418 says so in terms —
*"Causal references only. The two oracle references stay reported and gate nothing: an oracle is
not causally available, so losing to one bounds what a verdict means rather than voiding it."*
CH's causal constant is **79.07%**; the challenger at 47.42% beats it by 40.03%. G2 holds. The
40.29% in the issue is the **oracle** constant, which by registration gates nothing. Grading CH
`B` on that column would be applying a criterion the ladder does not have, three tranches after
it was pre-registered and on a pair whose old numbers are already published — which is the
precise shape of shopping a registration, even though the direction is conservative. I am not
doing it.

**G4 holds on sign, which is all G4 tests.** Per-band slope 0.078 / 0.086 / 0.131 and
correlation 0.138 / 0.170 / 0.246 — all strictly positive. The `slope 0.094, corr 0.176` the
issue quotes are ABL-380's *all-D+2* figures (the `country_d2` roll-up, n=1,950), not the gate
cells the ladder grades; they are also positive, so G4 holds either way. CH is not ABL-406's NO,
which was genuinely anti-correlated.

**So why is the `A` misleading? Because both causal references are mis-levelled by roughly a
factor of two, and in the direction that flatters the challenger.**

| pair | `constant_causal` = fit-window mean | `constant_oracle` = gate-window median | ratio |
|---|---:|---:|---:|
| BG | 141.54 MW | 74.69 MW | **1.90×** |
| CH | 21.97 MW | 10.68 MW | **2.06×** |

Both fleets ran at about twice the level in the fit window that they ran at in the gate window.
`constant_causal` and `climatology_causal` are built from the fit window by construction — that
is what makes them causal — so on these two pairs they are flat lines at roughly double the
truth, and **G2 and G3 are weak tests here for a mechanical reason that has nothing to do with
the model.** Re-run the same two comparisons against the correctly-levelled (oracle) forms and
the two pairs separate cleanly:

| pair | band | vs `constant_oracle` | vs `climatology_oracle` |
|---|---|---:|---:|
| BG | 24-36h | **+10.84%** win | **+9.02%** win |
| BG | 36-48h | **+10.90%** win | **+9.08%** win |
| BG | 48-64h | **+4.77%** win | **+3.72%** win |
| CH | 24-36h | **−17.68%** loss | **−24.12%** loss |
| CH | 36-48h | **−11.65%** loss | **−17.76%** loss |
| CH | 48-64h | **−10.87%** loss | **−16.98%** loss |

**BG beats the best flat line and the best average day available with hindsight, in every band.
CH loses to both, in every band.** That is ABL-380 §4's finding, now measured per band instead
of once, and it is the distinction the causal ladder does not make. The harness prints it
unprompted — `lost_to_a_model_free_reference` names all three CH cells against both oracles in
`reports/abl_435_wind_onshore_tranche2f.md` — so the qualifier is in the machine record and the
generated report, not only here. It is reported and it gates nothing, exactly as registered.

**The bar-weakness flag reads `no` on CH, and it reads `no` for the wrong reason.** This is
ABL-417's finding reproducing on a third pair. The flag asks whether `constant_causal` clears
the registered D-7 bar; CH's `constant_causal` is 79.07% against a 59.26% bar, so the flag says
the bar was not weak. But that 79.07% is the mis-levelled number. The **correctly-levelled**
constant scores 40.29% against the same 59.26% bar — so a hindsight flat line clears CH's bar
outright, and CH's bar *is* weak. Read `constant_oracle` beside the flag; on this pair the flag
alone is wrong.

---

## 5. Should CH be withdrawn from any list of ABL-316 passes?

Asked plainly in the issue, so answered plainly.

**Yes — from any list used to count programme progress or to support a promotion. No — from the
record, and the grade does not withdraw it either.** Concretely:

- **CH's `PASS 3/3` stands as arithmetic** and reproduces exactly. It is not withdrawn, and
  nothing here re-opens it.
- **CH's `A` must not be read as promotion-eligible.** ABL-418 defines `A` as "promotion-eligible,
  *subject to any named data hold*". CH carries three named holds, all measured: it loses to both
  oracle references in all three bands (§4); its slope of 0.078–0.131 says it responds to
  variation at roughly a tenth of the true amplitude; and its gate-window mean is **12.9 MW**,
  which ABL-348 registered as not decision-grade (`CH_wind_onshore_is_not_decision_grade`) and
  which the CEO restated on ABL-348. A WAPE of 45.7% on a 12.9 MW series is an MAE of about
  5.9 MW.
- **CH should therefore be excluded from the tranche pass tally**, which is what ABL-380 §9.2
  already recommended and what has not been done in the places that quote `PASS 6/6`.
- **BG is unaffected by this and passes on merit.** It clears the ladder, and it clears both
  oracle references, which is the test CH fails.

The programme-level statement, which I would rather make once here than have re-derived: **the
`A` grade is not currently sufficient to carry a promotion recommendation on a pair whose fleet
level shifts materially between the fit and gate windows.** Two of the two pairs in this tranche
have such a shift, and on one of them the ladder returns `A` for a model that a hindsight flat
line beats by 7.1pp.

---

## 6. Recommendation to the CEO

1. **BG `wind_onshore`** — accept as a reproduced, graded `A`, with two caveats attached and not
   detachable: the D-7 bar was cleared by a causal constant with no model (§3), and the TSO
   forecast beats the challenger (50.14% vs 57.06%, n=1,950). Evidence, not a promotion.
2. **CH `wind_onshore`** — reported, grade `A`, **gates nothing and must not be counted in a
   tranche pass tally** (§5). The `A` is carried by a reference mis-levelled 2.06×.
3. **Correct the standing record.** Anywhere ABL-316 quotes tranche 1a as an unqualified
   `PASS 6/6` — the `ledger` document on ABL-316 §4.4 is the consolidated position — CH's three
   cells should carry the §4/§5 qualifier. I have not edited that document from this issue; say
   the word and I will, or it can ride with whoever next revises it.
4. **Consider a pre-registered amendment to the ABL-418 ladder — CEO's call, not mine to adopt.**
   This is the third instance of one pattern (ABL-406's bar weakness, ABL-417's mis-levelled
   `constant_causal`, this). A candidate amendment: where
   `constant_causal / constant_oracle` (as MW levels) falls outside some registered band, flag
   the cell's G2/G3 as **not evaluable** rather than passed — the ABL-421 `SCOPE_NOT_EVALUABLE`
   precedent and the net-position gate's `INCOMPLETE` rule, one level down. It must be
   pre-registered before the pairs it would re-grade are re-read, for the reason ABL-418 itself
   gives. **I am not proposing to grade any oracle reference**: an oracle is not causally
   available and putting it on the ladder would be a different and worse mistake.
5. **The TSO result is unchanged and still outranks the model work** (ABL-380 §5). TSO beats the
   challenger on both pairs again — BG 50.14% vs 57.06%, CH 27.26% vs 45.71%. Note CH's TSO
   moved 0.52pp between two reads of the same window, which is the revision problem that makes
   this a feature-ingest question with an ingest owner, not something this harness can adopt.

---

## 7. Evidence hygiene — ABL-380's record is byte-unchanged

Proven by blob hash against `origin/main` (`e261669`), not by inspection:

| file | `origin/main` blob | working tree blob |
|---|---|---|
| `experiments/ABL348/results_abl380_tranche1a.json` | `8aa200eb0bcee1b0a4740d3ae923cdd7371ed194` | **identical** |
| `reports/abl_380_wind_onshore_tranche1a.md` | `845037ed5e33af1a6abea4a72ccac84c3782a6a5` | **identical** |
| `reports/abl_380_tranche1a_findings.md` | `6cde4f61158da8af2cc6527110031bfcc0459237` | **identical** |

Swept as whole directories rather than by filename glob: **all 336 tracked files under
`reports/` and `experiments/` are byte-identical to `origin/main`**, which also covers tranche
2b's and 2e's records.

`git diff --cached --numstat origin/main` — **zero deletions on every path**:

```
1127	0	experiments/ABL348/results_abl435_tranche2f.json
 103	0	reports/abl_435_wind_onshore_tranche2f.md
  86	0	scripts/evaluate_wind_retrain.py
  86	0	tests/test_gate_scope_registration.py
```

(plus this file). The two CatBoost artifacts under `experiments/ABL435/artifacts/` are correctly
excluded by `.gitignore:56`, confirmed with `git check-ignore -v`.

**Registration precedes the fit, checkable in git.** The scope commit `ebd6aa6` is timestamped
**2026-08-14T00:24:04Z**; the first fit started **00:24:10Z** and the run ended **00:25:57Z**.

**Scope registration tables, counted at head rather than trusted.** The wind harness carries
**three** scope-keyed tables — `SCOPES`, `GATE_BASIS`, `SCOPE_OUTPUTS` — and
`check_registration_tables(SCOPES=…, GATE_BASIS=…, SCOPE_OUTPUTS=…)` covers **all three**, so
there is no silently-defaulting table on this side. The seven the issue warns about are the
*solar* harness's (`FIT_RULES`, `SCOPE_FEATURES`, `SCOPE_TITLES`, `SCOPE_NOT_EVALUABLE` beyond
the three), of which only three are import-checked; no solar table was touched here. Four new
tests pin the 2f registration, including that it may not share any of tranche 1a's three output
paths.

---

## 8. Contamination and limits

- **ABL-188 does not touch either pair in any of the three windows** — measured, zero constant
  runs in fit, gate and feature-lookback, not assumed.
- **ABL-67 is net-position-only; ABL-109/ABL-111 are load-only.** Neither intersects these wind
  targets. **ABL-71**'s known wrong-write modes are load and net position; that is a provenance
  caveat, not proof that wind ingest is pristine.
- Every metric here: window 2026-07-11 → 2026-08-10 exclusive (30 d), n = 720 per 720-band cell
  and 510 for 48-64h, n = 1,950 per pair on the all-D+2 roll-up, baseline literal seasonal-naive
  D-7, **out-of-sample by target timestamp** (gate targets were never fitted), source
  `energy_generation`, replica `C:\Code\able\data\energy_dashboard.db` at 9,432,453,120 bytes —
  the same byte size ABL-348's bars and ABL-380's read were measured on, and not the 3.0 GB
  stale snapshot.
- One 30-day summer holdout, one seed per cell. Not a year-round robustness claim, and under
  ABL-385 a wind gap below 7.51% would not be readable at one seed at all.
