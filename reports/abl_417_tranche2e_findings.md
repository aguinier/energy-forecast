# ABL-417 — tranche 2e evidence pack: the eight small-fleet `wind_onshore` pairs

**Owner:** Forecasting Scientist. **Parent:** ABL-316. **Scope id:** `abl417-tranche2e`.
**Status: REPORT-ONLY. No cell in this pack recommends serving, whatever it grades.**

Machine record: `experiments/ABL348/results_abl417_tranche2e.json`.
Gate table: `reports/abl_417_wind_onshore_tranche2e.md`.
Margin read: `reports/abl_417_margins.json` (`scripts/abl406_margin_read.py`, reused).

---

## 1. What was run, and under what registration

Eight `wind_onshore` pairs — **CZ, EE, HR, HU, LT, LV, NL, RO** — × 3 primary D+2
bands = **24 cells**. This closes ABL-316's wind half: 2 (tranche 1a, ABL-380)
+ 8 (tranche 2b, ABL-406) + 8 here = the 18 `wind_onshore` countries ABL-348
registers.

Fitted and scored under the **frozen** registration at
`experiments/ABL348/config.json`. Nothing in it was touched — not the windows,
bands, metric, baseline, minimum n, nor the source table. ABL-348
`voids_this_registration` on any of those, and this pack changes none of them.
The report-only framing is a *reading* rule, exactly as CH was handled in tranche
1a: these pairs are fitted and scored **identically** to tranche 2b, on the same
bar, and the caveat lives here rather than in the scoring.

| item | value |
|---|---|
| fit window | 2026-01-14 → 2026-07-11 (exclusive) |
| gate window | 2026-07-11 → 2026-08-10 (exclusive) |
| metric | WAPE, `sum(abs(error)) / sum(abs(actual))` |
| baseline (the gate) | literal seasonal-naive D-7, recomputed on the same table |
| source table | `energy_generation`, passed explicitly via `--renewable-source` |
| algorithm | CatBoost (`ALGORITHMS["wind_onshore"]`) |
| primary bands | 24-36h, 36-48h, 48-64h |
| gate basis | `(challenger, seasonal_naive)` |
| interpreter | `.venv\Scripts\python.exe` — Python 3.14.3, xgboost 3.3.0, catboost 1.2.10 |
| replica | `C:\Code\able\data\energy_dashboard.db`, 9,432,453,120 bytes, opened `mode=ro` |

The worktree has no `.env`, so `config.DATABASE_PATH` degrades to a bare
`\data\energy_dashboard.db`. That was **verified, not assumed**, and
`--replica-db` was passed explicitly. All three output paths were passed
explicitly too; `experiments/*/results.json` and `artifacts/` are gitignored, so
an empty `git diff` would prove nothing about where a run wrote.

### The gate basis excludes the incumbent, and that was re-measured

Tranche 2b's basis comment asserts incumbent-absence for *its* eight countries. I
re-measured it for mine on the live replica rather than inheriting it. Across the
whole `forecasts` table, exactly **BE (32,068)** and **AT / DE / FR (31,056
each)** carry `renewable_type='wind_onshore'` rows. All eight of CZ, EE, HR, HU,
LT, LV, NL, RO carry **zero** — while each holding 65,088–65,232 forecast rows of
*other* types. So "the country is absent from the table" is not the explanation,
and a four-way basis would have looked plausible right up to an n=0 intersection
in all 24 cells. The incumbent column reads `Not measured` by construction.

---

## 2. What this pack is not

It is not a gate read, and the pass count is not its headline.

ABL-348 warned that these bars make a pass uninformative; ABL-406 turned that
warning into a measurement. Across its eight pairs the gate outcome was **fully
predicted** by whether a causal constant clears the registered bar on its own —
five weak bars gave five passes, three strong bars gave three failures or ties,
**no exceptions** — and NO passed 3/3 while **anti-correlated with its own
target** (slope −0.08, corr −0.14).

The registered D-7 bars for these eight run **86.78% (EE) to 125.38% (HU)**: a
baseline whose error is of the same order as the series itself. A PASS against
that carries close to zero information about the model.

What makes the tranche worth fitting anyway is **ABL-418's ladder**, imported
from `src/evaluation/gate_grading.py` — not reimplemented here, and already wired
into the wind harness by ABL-418 at `attach_grades`. G1 is the condition these
weak bars degrade; **G2, G3 and G4 are untouched by bar weakness and stay
readable**, which is what lets each pair be characterised honestly without
pretending the bar did the work.

Readability floor for wind at k=1 seeds: **7.51%** (`1.96 * c_A / sqrt(k)` with
`c_B = 0`, since every reference on the ladder is deterministic).

---

## 3. Two references qualify every grade below, and both were measured here

### 3.1 `constant_causal` is mis-levelled high on all eight — G2 is softer than it sounds

`constant_causal` is a flat line at the **fit-window** mean. The fit window
(14 Jan – 11 Jul) is windier than the gate window (11 Jul – 10 Aug, the summer
wind minimum), so that flat line sits systematically **above** the series it is
scored against, on every one of the eight:

| country | fit-window mean MW | gate-window mean MW | fit / gate |
|---|---:|---:|---:|
| CZ | 78.1 | 63.3 | 1.23× |
| EE | 141.7 | 101.8 | 1.39× |
| HR | 335.9 | 226.0 | 1.49× |
| HU | 65.5 | 41.6 | 1.57× |
| LT | 598.5 | 381.3 | 1.57× |
| LV | 34.9 | 34.6 | **1.01×** |
| NL | 728.0 | 224.6 | **3.24×** |
| RO | 682.9 | 492.2 | 1.39× |

Mean ratio **1.61×**. Measured on the live replica through the harness's own
loader (`db.load_renewable_type_data`, `source='energy_generation'`), so it is the
same series the fits and the gate use.

**Why this matters for reading G2.** Beating a flat line that is 61% too high on
average is an easier test than "beats a flat line" sounds. G2 remains a real
condition — a model that cannot clear a mis-levelled constant is in serious
trouble — but a G2 pass should be read against `constant_oracle`, the
*gate-window*-levelled flat line, which ABL-389 registered as a reported
reference for exactly this reason. **LV (1.01×) is the one pair where the causal
constant is already well-levelled, so LV's G2 is the most meaningful of the
eight; NL's (3.24×) is the least.**

This is a property of the *fit/gate window split*, not of any model, and it is not
a registration change: `constant_causal` was always defined as the fit-window
mean, and no gate criterion moved.

### 3.2 NL's target is signed — real, and quantified as inert here

`energy_generation`'s renewable columns are net of Actual Consumption for **every**
production type (ABL-412's diagnosis, recorded as a decision in ABL-414), so the
target can go negative. The fold-in asked what that does to NL's WAPE denominator.
Measured, before the margin surfaced it:

| series | window | n | neg hours | min MW | `sum\|a\| / \|sum a\|` |
|---|---|---:|---:|---:|---:|
| NL wind_onshore | gate | 720 | 62 (8.6%) | −13.0 | **1.0044** |
| NL wind_onshore | fit | 4,272 | 5 (0.12%) | −4.2 | — |
| NL solar (contrast) | gate | 720 | 237 (32.9%) | −0.4 | 1.0013 |

**The answer is that it does almost nothing, for a structural reason.** WAPE's
denominator is `sum(abs(actual))` — the absolute value is taken **element-wise**
(`src/evaluation/scorecard.py`, `denom = float(np.sum(np.abs(a)))`). Sign
cancellation therefore cannot deflate the denominator at all. The only residual
effect is that negatives make `sum|a|` exceed `|sum a|`, and for NL wind that gap
is **0.44%** — two orders below the 7.51% readability floor, so it cannot move
NL's grade.

Two corrections to the framing this inherited, both measured above:

- NL's negatives are **8.6% of gate hours on wind**, not "every night hour" —
  that description belongs to NL *solar*, where 32.9% of hours are negative.
- The denominator effect is nonetheless **larger on wind (0.44%) than on solar
  (0.13%)**, because NL solar's negatives are a −0.4 MW floor while wind's reach
  −13.0 MW. The pair with more negative hours is not the pair with the bigger
  denominator effect.

NL is the **only** one of the eight with any negative hour in the gate window;
CZ, EE, HR, HU, LT, LV and RO have none (minima 8.0, 4.0, 0.0, 0.0, 3.9, 0.0 and
0.0 MW respectively).

### 3.3 CZ: the TSO publishes nothing, and it is an availability finding

CZ `wind_onshore` is the pair where the TSO publishes no wind forecast at all.
Re-measured on the live replica, and it is **stronger** than the issue stated:
across `energy_generation_forecast` at `forecast_type='day_ahead'`, CZ carries
**105,112 rows, every one of them non-null and every one of them exactly 0.0**,
maximum 0.0, against a real fleet averaging 63.3 MW in the gate window. (The
issue's "36,192 rows over a year" is a narrower slice of the same fact.)

So **CZ's coverage claim is an availability claim, not a quality one**: for CZ
there is no TSO wind forecast to be better or worse than, and a model that
produces anything at all is the only forecast that exists. That is worth stating
separately from the other seven, and it is *not* a TSO benchmark — ABL-390 closed
that question for every pair (34.2h maximum forward extent against our 24-64h
D+2 band).

---

## 4. Contamination touching this window

- **ABL-67** (fabricated `net_position` rows) — does not intersect: different
  forecast type.
- **ABL-109 / ABL-111** (zero-as-missing actual **load**) — does not intersect.
- **ABL-71** (prod ingest stale, fixes undeployed) — a provenance caveat on this
  window rather than proof the wind ingest is clean. Stated, not dismissed.
- **ABL-188** (bit-identical constant runs) — the screen is applied by
  `db.load_renewable_type_data` to whichever table is read, and this run reads
  `energy_generation`. ABL-348 records the CZ `wind_onshore` fit-window hit
  (−86h) as **`energy_renewable`-only**, so it does not apply to this arm. The
  harness re-runs the audit per pair against the table actually fitted; the
  `constant_runs` block in the machine record is authoritative and is reported in
  §5.

### An out-of-window TSO anomaly, flagged and inert here

While measuring §3.3 I found that HU's day-ahead `wind_onshore_mw` reaches
**140,996 MW** against a fleet averaging 41.6 MW — 96 rows above 10 GW, all
clustered on **2026-02-04, 21:45–22:45**. That is a ~3,400× magnitude anomaly and
looks like an ingest defect.

**It does not touch this read**, and I checked rather than assumed: it is outside
the gate window entirely, and TSO is context-only under ABL-348
(`tso_role: revision-contaminated context only; not a gate criterion`) — never a
feature, never a fit input, and not on the ladder. Within my gate window HU's TSO
column is sane (n=2,880, 1.9–194.8 MW, mean 37.0). A search of all 400 company
issues found nothing covering a magnitude anomaly in `energy_generation_forecast`;
the nearest five concern coverage, publication time and revisions. Raised to the
CEO as an ingest-owner item; it is not mine to fix and it changes no number here.

---

## 5. Result: 24/24 PASS on the gate, and the pass count is the least informative number here

Run generated 2026-08-13 22:57 UTC; 8 pairs fitted in 454.6 s. Verdict `PASS`,
**24/24 primary cells** clearing the registered D-7 bar. **All 24 cells meet
their registered minimum n** (684/684/456; the tightest are EE 36-48h at exactly
684 and EE 48-64h at 475).

That headline is exactly what ABL-406's mechanism predicts on bars this weak, and
it must not be read as model strength. The rest of this section is what the cells
actually say.

### 5.1 Grades — the ladder splits a uniform PASS into 5 A and 3 B

| pair | gate | grade | failed | gate-window mean MW | D-7 bar | skill vs D-7 (3 bands) |
|---|:---:|:---:|---|---:|---:|---|
| CZ | 3/3 PASS | **A** | — | 63.3 | 86.4% | 48.2 / 47.9 / 44.9% |
| EE | 3/3 PASS | **A** | — | 101.8 | 85.8% | 50.2 / 50.6 / 47.0% |
| HR | 3/3 PASS | **A** | — | 226.0 | 97.7% | 24.2 / 29.8 / 31.3% |
| LT | 3/3 PASS | **A** | — | 381.3 | 100.5% | 43.9 / 44.1 / 38.5% |
| NL | 3/3 PASS | **A** | — | 224.6 | 94.9% | 17.8 / 13.4 / 13.3% |
| HU | 3/3 PASS | **B** | G2, G3 | 41.6 | 124.2% | 15.5 / 15.1 / 16.5% |
| LV | 3/3 PASS | **B** | G2, G3 | 34.6 | 97.5% | 8.7 / 7.4 / 6.4% |
| RO | 3/3 PASS | **B** | **G4** | 492.2 | 103.7% | 23.3 / 22.6 / 19.1% |

LV's B is its 24-36h band; its **36-48h and 48-64h bands grade `U`** — skill 7.4%
and 6.4%, both inside the 7.51% floor, so those two cells are unreadable at one
seed. Neither is `U(+)`: G2 and G3 fail in both, so the disposition is *not*
"re-read at k>1 seeds". The pair takes the worse B.

### 5.2 RO reproduces ABL-406's NO failure mode — a second pair, same shape

**RO passed 3/3 while anti-correlated with its own target.**

| band | challenger WAPE | D-7 | skill | slope | corr |
|---|---:|---:|---:|---:|---:|
| 24-36h | 79.5% | 103.7% | +23.3% | **-0.005** | **-0.015** |
| 36-48h | 80.3% | 103.7% | +22.6% | **-0.016** | **-0.046** |
| 48-64h | 79.8% | 98.6% | +19.1% | **-0.042** | **-0.108** |

Negative slope *and* negative correlation in all three bands. This is the ABL-406
NO result (slope -0.08, corr -0.14) recurring on a second pair — and RO is the
largest fleet in this tranche at 491.9 MW. It is not a one-off, and G4 is the
condition that catches it: without the ladder, RO reads as a clean 3/3 PASS with
19-23% skill.

**A model anti-correlated with its target beat a D-7 baseline purely on level.**
That is the entire content of RO's pass, and it is the most important finding
here.

### 5.3 The registered bar established nothing on any of these eight

The bar-weakness flag fires on 7 of 8 pairs — a causal flat line clears the
registered D-7 bar on its own for CZ, EE, HR, HU, LT, LV and RO.

NL is the sole exception, and **it is an exception for the wrong reason.** NL's
causal constant scores 225.5% WAPE, so naturally it does not clear a 94.9% bar —
but that is the 3.24x fit/gate level shift of section 3.1, not a strong bar.
Levelled correctly, NL's bar falls too.

Stated on the properly-levelled reference, the result is unanimous:

> **An oracle flat line beats the registered D-7 bar in 24 of 24 cells, on all
> eight pairs.**

So on this fleet the registered bar is beaten by a constant everywhere, and no
PASS in section 5.1 rests on it. That is the measurement behind "report-only",
not an opinion about it.

### 5.4 Against the four model-free references, only EE and LT win outright

Bands (of 3) in which the challenger beats each reference:

| pair | constant causal | constant **oracle** | climatology causal | climatology **oracle** |
|---|:---:|:---:|:---:|:---:|
| CZ | 3/3 | 3/3 | 3/3 | **0/3** |
| EE | 3/3 | 3/3 | 3/3 | **3/3** |
| HR | 3/3 | 2/3 | 3/3 | **0/3** |
| HU | 0/3 | 0/3 | 0/3 | 0/3 |
| LT | 3/3 | 3/3 | 3/3 | **3/3** |
| LV | 0/3 | 0/3 | 0/3 | 0/3 |
| NL | 3/3 | **0/3** | 3/3 | 0/3 |
| RO | 3/3 | **0/3** | 3/3 | 0/3 |

Fleet totals: the challenger is ahead in **24/24** cells against the registered
bar, 18/24 against the causal constant, 18/24 against the causal climatology,
**11/24** against the oracle constant and **6/24** against the oracle
climatology.

Two qualifications the grade column alone does not carry:

- **CZ and HR grade A but lose to an oracle climatology in every band.** Their A
  is real against the causal references — the causally available ones, and the
  only ones on the ladder — but an hour-of-day climatology computed on the gate
  window itself would have beaten them in all three bands.
- **NL grades A while losing to *both* oracle references in all three bands.**
  Its A rests entirely on beating a causal constant that is 3.24x mis-levelled.
  Of the five A pairs, NL's is the weakest, and it is weakest for a reason that
  is a property of the window split rather than of the model.

Oracles are **reported and never gating**, exactly as ABL-389 registered them: an
oracle is not causally available, so losing to one bounds what a verdict means
rather than voiding it. **EE and LT are the only two pairs that beat all four
references in all three bands** — the only two where the model demonstrably
carries information beyond level and daily shape.

### 5.5 Contamination: clean on this arm, measured per pair

The harness re-ran the ABL-188 constant-run screen against the table actually
fitted (`energy_generation`) for each pair. **Zero suspect constant runs on all
eight.** ABL-348 records a CZ `wind_onshore` fit-window hit (-86h) as
`energy_renewable`-only, and this arm does not read that table — confirmed rather
than assumed. No `PERFORMANCE PASS - HOLD FOR CONTAMINATION ADJUDICATION`
condition arises.

---

## 6. Can a WAPE carry a decision here? Per pair, explicitly

The precedent chain is published, and this tranche extends it: **12.9 MW** (CH
wind, ABL-348 `CH_wind_onshore_is_not_decision_grade`) -> **114.8 MW** (SK solar,
ABL-405, the `**SK**:` bullet — *"report it, do not decide"*) -> here. Tranche 2b
was cut at **700 MW** to be decision-grade. Every pair here is below that line,
and half are below the SK line.

| pair | gate-window mean MW | vs SK line (114.8 MW) | can a WAPE carry a decision? |
|---|---:|---|---|
| LV | 34.6 | **below** | **No.** Below the published non-decisional line, and it grades B — loses to a flat line in all 3 bands, 2 bands unreadable. |
| HU | 41.6 | **below** | **No.** Below the line, and B — beaten by a flat line and by climatology in every band. |
| CZ | 63.3 | **below** | **No** on the denominator, despite an A. Reported for coverage and for the availability finding in section 3.3. |
| EE | 101.8 | **below** | **No** on the denominator, despite the cleanest result in the tranche (A, beats all four references 3/3). |
| NL | 224.6 | above | **No.** Above SK but far below 700 MW, and its A rests on a 3.24x mis-levelled constant; it loses to both oracles 3/3. |
| HR | 226.0 | above | **No.** Above SK, far below 700 MW; A, but loses to oracle climatology 3/3. |
| LT | 381.3 | above | **Weakest no.** The strongest case in the tranche — A, beats all four references 3/3 — but still under half the 700 MW line. |
| RO | 492.2 | above | **No, emphatically.** Largest fleet here, and anti-correlated with its own target in all 3 bands (section 5.2). A decision on RO's WAPE would be a decision on a model with negative directional skill. |

**The honest answer on all eight is no**, as the issue anticipated — but not for
one reason. Four fail on the denominator alone (LV, HU, CZ, EE); three fail on
the references despite clearing the bar (NL, HR, and HU/LV again on G2/G3); and
RO fails on direction. **LT and EE are the only pairs whose model behaviour would
justify a further look**, and both are blocked by fleet size, not by quality.

---

## 7. Disposition

**Report-only. No cell here recommends serving, and nothing in this pack is
promotion-eligible.** Promotion is a pre-registered gate read plus a Board
decision. Grade `A` means "promotion-eligible subject to any named data hold" —
and every A pair in section 5.1 carries a named hold in section 6.

What this tranche delivers:

1. **Coverage.** ABL-316's `wind_onshore` half is closed — 18 of 18 registered
   countries fitted and read (2 + 8 + 8).
2. **A second reproduction of the anti-correlation failure mode**, on RO. ABL-406
   found it once and warned it would recur; it recurred in the next tranche, on
   the largest pair in it.
3. **A measurement that the registered D-7 bar is beaten by a constant in 24/24
   cells** on this fleet, which settles the "is a weak-bar pass informative"
   question for these eight rather than arguing it.
4. **A reading correction for the bar-weakness flag** (section 5.3): it can read
   False because the causal constant is mis-levelled by the fit/gate seasonal
   split rather than because the bar is strong. NL is the worked example, and the
   oracle constant is the check.

Suggested follow-ups — **not** actioned here, and none blocking this pack:

- **RO's anti-correlation** deserves the treatment ABL-406's NO got. Two pairs
  now show it; one shared cause is likelier than two coincidences.
- **The fit/gate level shift** (section 3.1, 1.01-3.24x) makes `constant_causal`
  a systematically soft G2 on any summer-gated wind tranche. Worth registering
  whether G2 should carry the oracle constant as a reported sensitivity — a
  question for a *future* pre-registration, explicitly not a change to ABL-418's
  registered ladder after seeing these results.
- **HU's day-ahead TSO magnitude anomaly** (section 4) — an ingest-owner item,
  outside this window and inert for this read.
