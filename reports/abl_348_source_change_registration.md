# ABL-348 — Does a change of training source void the ABL-195 / ABL-253 registration?

**Disposition: VERDICT DELIVERED + GENERIC PRE-REGISTRATION FROZEN. No model
trained, no promotion, no serving change, nothing written to either database.**

Generated: 2026-08-12 UTC.
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), opened
`mode=ro`, `uri=True`. Sidecar not opened.
Interpreter: `.venv\Scripts\python.exe` — Python 3.14.3, xgboost 3.3.0 (the rail).
Reproduce with `scripts/abl348_source_registration_probe.py`; raw output in
`reports/abl_348_probe.json`. Registration: `experiments/ABL348/config.json`.

Every number below was taken through `db.load_renewable_type_data`, the same
loader both gate harnesses train and score from — so it measures the series a
harness would actually see: `data_quality='actual'` only, NULL dropped,
duplicate instants collapsed (disagreeing spellings nulled), ABL-188
suspect-constant-run screen applied. Hand-rolled SQL would have measured a
different series than the one under registration.

Scope: the **37 tranche pairs** — every ABL-318 `verdict == TRAIN` pair (39)
minus the two ABL-322 pilot pairs (DE/NL `wind_offshore`), which carry their own
registration at `40debf3`.

---

## 0. Answer in three lines

1. **The windows, holdout boundary, metric, baseline and acceptance band survive
   a change of source unchanged.** The depth shortfall is real and reproduces
   exactly — and lies entirely *outside* the registered window.
2. **The registered minimum n does not survive, for two named pairs.** FI solar
   is broken by the source change itself; EE solar is broken on both sources.
3. **A fit on `energy_generation` still needs its own pre-registration** — for
   reasons of pair set and named source, not of window validity. It is frozen
   here, once, generically, in `experiments/ABL348/config.json`. **The four level
   pairs cannot reuse ABL-195/ABL-253 either**, but they are the cleanest first
   tranche under the new registration, with one caution (§5).

---

## 1. The CEO's measurement reproduces exactly — and it does not reach the window

`reports/abl_348_probe.json`, all 37 pairs, unbounded first reported instant per
table. Grouped as the ABL-348 description groups them:

| `energy_renewable` starts | `energy_generation` starts | shortfall | n | pairs |
|---|---|---:|---:|---|
| 2021-01-01 | 2021-01-01 | **0 d** | 4 | BG solar, BG onshore, CH solar, CH onshore |
| 2025-10-05 | 2021-01-01 | 1,738 d | 8 | FI onshore, IT solar/onshore, PL solar/onshore, RO solar/onshore, SE onshore |
| 2025-10-05 | 2021-12-14 | 1,390 d | 1 | SE solar |
| 2025-10-05 | 2022-12-31 | 1,009 d | 1 | FI solar |
| 2025-11-07 | 2021-01-01 | 1,771 d | 4 | GR solar/onshore, HR solar/onshore |
| 2025-11-08 | 2021-01-01 | 1,772 d | 5 | ES solar/onshore, LV onshore, PT solar/onshore |
| 2025-11-08 | 2023-12-31 | 678 d | 1 | LV solar |
| 2025-11-09 | 2021-01-01 | 1,773 d | 8 | CZ solar/onshore, EE onshore, LT solar/onshore, NL solar/onshore, NO onshore |
| 2025-11-16 | 2021-01-01 | 1,780 d | 5 | EE solar, HU solar/onshore, SI solar, SK solar |

Every row the description gives is confirmed to within a day (boundary
convention). The three source-specific wrinkles are confirmed: SE solar
(`energy_generation` from 2021-12-14), FI solar (2022-12-31), LV solar
(2023-12-31) are shorter absolute histories than the table-vs-table column
suggests. The table also completes the description's summary, which did not
enumerate GR solar/onshore, HR solar/onshore, HU onshore, LT onshore or FI solar
in its shortfall rows. **33 of 37 pairs do not span the same period. Confirmed.**

**And it does not bear on the registered window.** The registration reaches back
to **2025-12-31** — the fit window opens 2026-01-14 and the feature builder
takes a 14-day lookback:

| | measured |
|---|---|
| latest `energy_renewable` first instant, over all 37 | **2025-11-16 23:00** |
| latest `energy_generation` first instant, over all 37 | **2023-12-31 22:00** |
| earliest instant the registration reaches | 2025-12-31 00:00 |
| pairs where *either* table begins after that | **0 of 37** |

The 678–1,780 day shortfall is **prefix history the registered window never
touches**. It is a strong argument for a *different, longer* fit window
(see §6). It is not an argument that the registered one has become invalid.

## 2. The bar barely moves, because inside the window the two tables are largely one series

Literal seasonal-naive D-7 WAPE over the registered gate window
(2026-07-11 → 2026-08-10, 720 hourly `:00` targets), computed independently on
each table. This is the **bar**, and it is the thing that mechanically changes
when the target changes table.

- **26 of 37 pairs: Δ = 0.00 pp.**
- **All 37: |Δ| ≤ 1.17 pp.** Largest movers: EE onshore 1.17, NL onshore 1.00,
  IT onshore 0.74, GR solar 0.71, GR onshore 0.52; every other pair ≤ 0.12.

Why: within the gate window the tables mostly hold **identical values**. Over
hours observed in both, 28 of 37 pairs are bit-identical in every single hour.

| pair | hours in both | % bit-identical | mean abs diff | as % of level | max abs diff |
|---|---:|---:|---:|---:|---:|
| **GR solar** | 720 | **81.2%** | **130.36 MW** | **4.88%** | 1,342.0 MW |
| GR wind_onshore | 720 | 70.4% | 12.55 MW | 0.93% | 158.0 MW |
| NL wind_onshore | 720 | 68.9% | 1.49 MW | 0.66% | 13.4 MW |
| PT solar | 720 | 77.4% | 3.78 MW | 0.32% | 239.0 MW |
| PT wind_onshore | 720 | 75.1% | 1.21 MW | 0.15% | 132.9 MW |
| NL solar | 720 | 65.0% | 0.05 MW | 0.08% | 1.0 MW |
| HU solar | 720 | 78.5% | 0.28 MW | 0.02% | 9.1 MW |
| the other 30 | — | **100.0%** | 0.000 MW | 0.000% | 0.0 MW |

For scale: ABL-195 and ABL-253 decided their cells on skill margins of **+24%
to +53% relative**. A bar that moves by at most 1.17 pp on a 12–125% baseline is
not a bar a challenger can be shopped against. **The acceptance band —
"challenger WAPE < literal seasonal-naive D-7 WAPE" — is a *relative,
self-referential* criterion: no absolute WAPE number is registered, and the
baseline is recomputed on whatever series the challenger is scored against. That
form is what makes it source-portable.** An absolute-threshold band would not
have been.

**GR solar is the one genuine exception.** At 4.88% of level and a 1,342 MW max
disagreement, the two tables there are not the same series and the source choice
is substantive. Reported per-pair; see §5.

## 3. What does *not* survive: the registered minimum n

The registered minimum n — **684 / 684 / 456** for 24-36h / 36-48h / 48-64h,
being 95% of the intended 720 / 720 / 480 — is a property of the **data**, not of
the protocol. It was derived from the run schedule assuming complete hourly
coverage. Coverage differs by table, so this is the one registered quantity a
source change can and does break.

`n_d7_scorable` below is the count of gate target hours where both the actual and
its D-7 lag exist. It is an **upper bound on the n of the 24-36h and 36-48h
bands**, since each intends all 720 gate hours and the harness scores on a finite
intersection that is a subset of this. (The 48-64h band selects a 480–510 row
subset, so the relation there is proportional rather than a hard bound.)

| pair | `energy_renewable` n | `energy_generation` n | vs 684 |
|---|---:|---:|---|
| **FI solar** | 714 | **650** | **passes on the old source, fails on the new** |
| **EE solar** | **630** | **630** | **fails on both** |
| all other 35 | 702–720 | 702–720 | pass |

- **FI solar is the one pair broken by the source change alone.** `energy_generation`
  holds 663 of the 720 gate hours against `energy_renewable`'s 717. This is the
  ABL-322 §3.3 phenomenon on a second pair: `energy_generation` is not uniformly
  more complete.
- **EE solar fails regardless of source.** The cause is an ABL-188 exclusion — a
  44.8-hour bit-identical zero run, 2026-07-21 00:00 → 2026-07-22 20:45, present
  **identically in both tables** and screened out of both. Not a source question,
  but it must be declared *before* the tranche runs, or it will read as a gate
  failure when it is an evaluability failure.

Both are declared in `experiments/ABL348/config.json` as **NOT-EVALUABLE at the
registered minimum n**, pre-committed, before any challenger for either pair
exists.

### ABL-188 inside the registered windows

The screen is applied to both sources by `db.load_renewable_type_data`, so
handling is identical between arms. Five hits fall inside the registered windows,
and **four of the five are `energy_renewable`-only**:

| pair | table | window | cost |
|---|---|---|---:|
| CZ solar | `energy_renewable` only | fit | −93 h |
| CZ wind_onshore | `energy_renewable` only | fit | −86 h (179.87 MW held 86.5 h, 2026-02-11→02-15) |
| NL solar | `energy_renewable` only | fit | −45 h (0 MW held 44.2 h, 2026-01-28→01-30) |
| EE solar | **both tables** | **gate** | −45 h each |

So on this evidence the source change **removes** screened contamination from
three pairs' fit windows and **adds** none. That is a point in favour of
`energy_generation`, independent of depth.

### Contamination statement for these windows

- **ABL-67** (fabricated `net_position`) — does not intersect: different table
  and target. GR is the country involved; see the GR provenance flag in §5.
- **ABL-109 / ABL-111** (zero-as-missing actual load) — load only, does not intersect.
- **ABL-71** (prod ingest stale, fixes undeployed) — known wrong-write modes are
  load and net position. A provenance caveat on these targets, not proof that
  solar/wind ingest is pristine.
- **ABL-188** — *does* intersect, as tabulated above, and is handled identically
  on both arms.

## 4. Verdict on the three questions asked

### 4.1 Do the registered windows survive a source change?

**The windows, the holdout boundary, the metric, the baseline and the acceptance
band survive unchanged** — the shortfall lies outside the window (§1) and the
bar moves by ≤ 1.17 pp because inside the window the tables are largely the same
series (§2). **The registered minimum n does not survive** for FI solar (broken
by the source change) and EE solar (broken on both sources) (§3).

### 4.2 Does a fit on `energy_generation` need its own pre-registration?

**Yes — and it is frozen here, once, for all 37 pairs.** Not because the windows
moved. Three reasons a re-used ABL-195/ABL-253 registration cannot carry:

1. **Those registrations name their pairs, and the 37 are not among them.**
   ABL-195 registers `{wind_offshore: [BE, FR], wind_onshore: [BE, DE, FR]}`;
   ABL-253 registers `{solar: [BE, DE, FR]}`. A pre-registration is a commitment
   over a *stated* pair set. Reading a gate for BG solar against a registration
   that never names BG is not a strict reading of an existing gate — it is an
   unregistered gate wearing a registered one's numbers.
2. **Neither names a source table.** Both predate ABL-331; their implicit source
   is the global default, `energy_renewable`. Now that the source is a
   per-artifact property, a registration that does not name its table is exactly
   the ambiguity ABL-321 and ABL-331 were written to end.
3. **The two n exceptions must be declared before the fit, not after.**

### 4.3 Can the four level pairs (BG, CH) reuse the existing registration?

**No — but not for a data reason, and the distinction matters.**

On data they are flawless. Both tables start 2021-01-01; 720/720 gate hours in
each; 720/720 D-7-scorable in each; **bit-identical in all 720 co-observed
hours**; D-7 bar identical to two decimal places (BG solar 24.40 / 24.40, BG
onshore 93.75 / 93.75, CH solar 12.67 / 12.67, CH onshore 59.26 / 59.26). No
ABL-188 hit in either window. If any four pairs could reuse a registration on
data grounds, these are they.

They still cannot, for reason 4.2(1): **neither ABL-195 nor ABL-253 names BG or
CH.** That holds however well the data behaves. Under the new generic
registration they need **no adjustment, no exception and no caveat**, which is
what makes them the cheapest first tranche.

## 5. Sequencing note on the first tranche — one caution

Recommended first tranche, in order: **BG solar, CH solar, BG wind_onshore**,
then **CH wind_onshore** as a build-and-report pair rather than a decision pair.

| pair | gate-window mean | ABL-318 max | D-7 bar |
|---|---:|---:|---:|
| BG solar | 1,439.2 MW | 4,408.1 MW | 24.40% |
| CH solar | 1,331.0 MW | 4,477.0 MW | 12.67% |
| BG wind_onshore | 108.9 MW | 737.7 MW | 93.75% |
| **CH wind_onshore** | **12.9 MW** | 198.6 MW | 59.26% |

**CH wind_onshore's gate-window mean is 12.9 MW.** A WAPE computed on a ~13 MW
series is dominated by a fleet too small for the number to carry a promotion
decision either way, and its 59.26% D-7 bar is not a meaningful reference. It is
registration-compatible and cheap to run; it is not decision-grade. Reporting it
as a passed or failed cell alongside BG solar would give it weight it cannot
bear. This is a recommendation on how to *read* the cell, not a change to the
registration — the band stays as registered.

### The band is the same; the task behind it is not

Worth carrying into every tranche evidence pack, because it changes what a
pass rate means. ABL-253 read solar against D-7 bars of **22.8–32.9%** (BE, DE,
FR). The tranche's solar bars are far tighter in the south — **IT 7.11%, ES
11.78%, CH 12.67%, PT 13.09%, HR 16.43%** — because Mediterranean July/August
clear-sky solar is nearly D-7 periodic. Conversely several small wind fleets have
very loose bars: **HU 125.38%, RO 104.14%, LT 100.36%, HR 99.58%, LV 97.11%**.

Same registered band, materially harder task in southern solar and a nearly
automatic pass in small-fleet wind. **Expect a lower solar pass rate in the south
and do not read it as model quality; expect wind passes in the small-fleet
countries and do not read them as model strength.** This is a fact about the
band's meaning under a new pair set, which is what this issue asked to
establish — it is not a reason to move the band, and the band is not moved.

### GR — carry the source disagreement into its evidence pack

GR solar is the only pair where the tables materially disagree inside the window
(§2: 4.88% of level, max 1,342 MW). For GR the source choice is substantive
rather than cosmetic. Registered on `energy_generation` like every other pair,
but its evidence pack must report the disagreement, and a GR gate read must not
be treated as interchangeable between sources. Separately, GR is the country
whose `net_position` actuals are fabricated zeros (ABL-67) — a different table
and target, recorded here as a provenance flag only.

## 6. The distinction that actually matters, and the trap it hides

**The registration survives a change of *table*. It does not survive a change of
*window* — and the extra window is the entire reason to want the new table.**

Nothing in §1–§3 licenses fitting the 37 pairs on 2021-01-01 → 2026-07-11. That
is a different experiment: a ~5-year fit against a ~6-month one, on a different
number of seasons, with different feature-lag behaviour. It may well be better.
It is not the ABL-195/ABL-253 protocol and must not be reported as it.

The trap is the ordering. Fit the short window, see a cell fail, then refit on
five years and report the pass — that is shopping, and it is available here
precisely because the long window is genuinely attractive. The registration
therefore:

- **fixes the primary fit window at the inherited 2026-01-14 → 2026-07-11**, so
  tranche results stay directly comparable to ABL-195/ABL-253 and to the ABL-322
  pilot; and
- **permits a long-history arm only as a second arm declared before fitting**,
  with both arms reported whenever the long arm is run.

Whether to fund the long arm — 37 pairs × ~5 years of fit — is a sequencing and
budget call for the CEO, not a registration question. I am not asking for it
here.

## 7. Limits

- No model was trained. No challenger score exists for any of the 37 pairs, in
  either arm. That is what makes this a pre-registration rather than a
  rationalisation.
- All D-7 figures are pooled over the gate window on the plain hourly `:00`
  series. They are the pre-committed bar; the harness's per-band,
  finite-intersection D-7 will differ slightly and remains authoritative for a
  gate read.
- `n_d7_scorable` bounds the two 684-bands directly; the 48-64h band's n is
  proportional rather than bounded, so a pair marginally under 684 may still
  clear 456 in that band. FI solar and EE solar are declared NOT-EVALUABLE on the
  684-bands specifically.
- The gate is one 30-day summer holdout. Out-of-sample by target timestamp; not a
  year-round robustness claim. For the southern solar pairs, a July/August window
  is the most D-7-favourable month of the year, which is part of why those bars
  are tight.
- The `raw_*_prescreen` counts in the JSON conflate duplicate-instant collapse
  with ABL-188 exclusions for `energy_renewable`; `energy_generation` has no
  duplicate instants, so its screen cost is pure ABL-188. The four attributions
  in §3 were confirmed against the loader's own logged run boundaries.
- No production deploy, serving-registry change, model promotion, ingest change,
  dashboard change, replica write or sidecar write was performed.

## 8. Recommendation to the CEO

1. **Sequence BG/CH first, as you proposed.** They are registration-compatible
   under `experiments/ABL348/config.json` and carry no exception. Read CH
   wind_onshore as build-and-report, not as a decision cell (§5).
2. **Accept the two declared NOT-EVALUABLE pairs** — FI solar and EE solar — as
   pre-committed exclusions at the 684-bands, rather than letting them surface
   later as gate failures.
3. **Do not fold ABL-333/ABL-334's incumbent migration into a new-pair tranche.**
   Your FR `wind_offshore` observation is confirmed: `energy_renewable` spans
   1,319 days (2023-01-01 → 2026-08-12) against `energy_generation`'s 1,168
   (2023-05-31 → 2026-08-12). An incumbent moved to `energy_generation` loses
   151 days of history, which is the opposite trade from every tranche pair.
4. **Decide the long-history arm deliberately, before any tranche fits** (§6),
   or explicitly defer it. Deciding it after a tranche result is seen voids the
   registration.

Promotion is not decided here and is not recommended here. This issue produced
one document, one registration and one probe; no model, no serving change, no
Board ask.
