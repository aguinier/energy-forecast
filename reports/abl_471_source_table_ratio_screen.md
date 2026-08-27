# ABL-471 — the ABL-439 source-table ratio screen on the four unscreened pairs, and LV `solar` explained

**Report only.** No refit, no re-grade, no new gate scope. Dispositions below are
stated as screen outcomes against the ABL-316 standing rule; the decision is the
CEO's.

Record: `reports/abl_471_source_table_ratio_screen.json`.
Script: `scripts/abl471_source_table_ratio_screen.py` (imports ABL-439's readers
directly, so the rows are read the way ABL-439 read them).

## Protocol

| | |
|---|---|
| replica | `C:\Code\able\data\energy_dashboard.db`, 10,220,126,208 bytes, opened `mode=ro` |
| measure | hourly-mean `energy_generation / energy_renewable` over **co-observed hours only** |
| aggregation | raw rows averaged into hours before any comparison (ABL-332) |
| windows | `abl439_comparator` 2026-05-01→2026-07-01 · `abl348_fit_window` 2026-01-14→2026-07-11 · `abl348_gate_window` 2026-07-11→2026-08-10 |
| n | 1,461–1,464 / 4,234–4,245 / 663–720 co-observed hours per pair per window |
| in/out of sample | neither — this is a property of the stored series, not a model score |
| contamination | **ABL-188** (`energy_renewable` zero-fills a type ENTSO-E did not return) touches the head of `energy_renewable`'s coverage: NL `wind_offshore` 2025-11 has 505 hours against `energy_generation`'s 720 at `zero_fraction` 0.224. That month is excluded from every window above and is reported as coverage, not as level. **ABL-67** (fabricated `net_position` rows) and **ABL-111 / ABL-109** (zero-as-missing actual *load*) touch neither of these tables nor these columns. **ABL-71** — ingest staleness, fixes undeployed — is the same *family* of effect as the vintage split measured here; this report **measures and reports** that split rather than correcting it, and no window is silently trained or scored through it. |

**Five pins reproduce to 4 dp** against ABL-439's committed record before anything
new is reported: NL `wind_onshore` 2.4647, NL `solar` 1.6269, GR `solar` 0.7945,
CH `wind_onshore` 1.0747, LV `solar` 1.1143. Same replica, same rows.

## 1. The five ratios

| pair | comparator window | **fit window** | gate window | fit−gate | verdict |
|---|---:|---:|---:|---:|---|
| DE `wind_offshore` | 1.0000 | **1.0017** | 1.0000 | +0.0017 | **basis-consistent** |
| NL `wind_offshore` | 0.9648 | **0.9922** | 0.9912 | +0.0010 | **basis-consistent** — see §3 |
| EE `solar` | 1.1695 | **1.1764** | 1.0000 | **+0.1764** | **basis-INCONSISTENT** |
| FI `solar` | 1.0000 | **1.0000** | 1.0000 | 0.0000 | **basis-consistent** |
| LV `solar` | 1.1143 | **1.1706** | 1.0000 | **+0.1706** | **basis-INCONSISTENT** |
| *NL `wind_onshore`* (reference) | *2.4647* | *2.2592* | *0.9933* | *+1.2659* | *the pair ABL-439 diagnosed* |

The reference row is carried through the identical protocol so the comparison in
§2 is measured rather than asserted. Every window here is 94.26–94.28%
pre-boundary on the fit side and **0.00%** on the gate side (§2.4).

Band membership, both readings, because they disagree on one pair:

| pair | in 0.99–1.07 on comparator | in 0.99–1.07 on fit window | flagged by ABL-439's own rule (\|r−1\|>0.15) |
|---|:---:|:---:|:---:|
| DE `wind_offshore` | yes | yes | no |
| NL `wind_offshore` | **no** (0.9648) | yes (0.9922) | no |
| EE `solar` | no | no | **yes** |
| FI `solar` | yes | yes | no |
| LV `solar` | no | no | **yes** |

> **0.99–1.07 is a description, not the rule.** Ledger §5.6 reports it as where the
> unaffected pairs landed. The screen's actual decision is
> `abs(ratio - 1) > SWEEP_MATERIAL_RATIO`, i.e. **±15%** — the constant of that
> name in `scripts/abl439_reporting_basis_probe.py`. Both are tabulated above so a
> descriptive range does not silently become a promotion criterion.

## 2. LV `solar` = 1.1143 — explained: it is the ABL-439 revision vintage

**Not a basis change, not a resolution change, not a content difference.** It is the
defect ABL-439 already diagnosed on NL `wind_onshore` — same table, same backfill,
same 28-day rule — showing up on a second stream.

Four measurements, each ruling out one alternative:

1. **Resolution is ruled out by cadence.** LV carries **1.000 rows/hour on both
   tables** — the only one of the five pairs that is hourly on both sides. There is
   no cadence difference to weight. (Both sides are averaged to hours first
   anyway, so cadence could not move this ratio.)

2. **A genuine content difference is ruled out by convergence.** The two tables
   disagree until **2026-06-30 20:00 UTC** and are **bit-identical for the 1,240
   consecutive hours after it** — max absolute difference **0.0 MW**. Two tables
   carrying genuinely different quantities do not become identical to the last bit
   on a Tuesday evening. EE `solar` converges at **the same instant**.

3. **The direction is fixed by the TSO reference.** The TSO's own day-ahead
   forecast is published by the same TSO for the same fleet and is not derived
   from the actuals, so dividing by it removes the weather. Over the fit window
   `energy_renewable / TSO` is flat at **0.93–1.03** every month, while
   `energy_generation / TSO` runs **1.80, 1.82, 1.33, 1.12, 1.04, 1.11** and then
   drops to **0.955** at the convergence instant. One series is stable against an
   independent reference; the other is not, and stops moving exactly when the two
   tables meet.

4. **The mechanism is in the fetch log, not in Latvia.** Age-at-fetch of the
   stored rows, over 2026-01-14→2026-08-10, split at the convergence instant:

   | table | before boundary | after boundary |
   |---|---|---|
   | `energy_generation` | n=4,029, min **28.89** d, median 112.81 d | n=960, median 8.81 d, max **28.85** d |
   | `energy_renewable` | n=3,999, median **6.86** d | n=960, median **6.87** d |

   The split is exact and has **no overlap**. `energy_renewable` is only ever
   written at a flat ~6.9-day lag — it always holds ENTSO-E's *first* publication.
   `energy_generation` was backfilled on **2026-07-29** (48,691 LV rows covering
   2021-01-01→2026-07-23), so it holds the **~28-day-revised** publication for
   every instant older than that backfill minus ~28.9 days, and the first
   publication for everything newer. The boundary is
   `2026-07-29 − 28.9 days` — a fact about our fetch schedule, and nothing
   physical. Identical structure on all six pairs in §1 (see §4).

**What it cost the LV `solar` gate read.** Measured on co-observed hours,
ABL-348's fit window is **94.27%** pre-boundary (3,999 of 4,242 hours) and its
gate window is **0.00%** pre-boundary (0 of 717) — so the pair was **fitted on the
revised basis and scored on the first-publication basis**, which sits **17.1%
lower**. That is ABL-439's harm exactly, at **+0.1706** where the same measurement
on NL `wind_onshore` gives **+1.2659**. And, as there, **the gate actuals are provisional**: no
gate row has yet aged past the ~28-day horizon and been re-fetched, so they are
expected to move up onto the other basis at the next backfill — which moves the
gate read with them.

Per the issue's instruction, that is where this stops. **LV `solar` does not
clear the screen**; the disposition is the CEO's. (It was held anyway at +5.87,
inside the ABL-444 readability floor, so no shipping decision turns on it.)

**EE `solar` is the same finding** — ratio 1.1764, same convergence instant, same
provenance split, `energy_generation / TSO` inflated to 1.38/1.32/1.37/1.26 while
`energy_renewable / TSO` sits at 1.11/1.09/1.22/1.03. It is separately held on
coverage (§4.3b / ABL-434), so again nothing turns on it.

## 3. NL `wind_offshore` — the pair that decides a shipment

**It clears, and the caveat has to be stated with it.**

The two windows disagree, and this is the pair where that matters:
**0.9648** on the window ABL-439 used for the other 37, **0.9922** on ABL-348's
registered fit window. 0.9648 is 2.4 pp below the descriptive band's floor.

Three things settle it against the vintage defect the screen exists to find:

- **fit−gate discontinuity = +0.0010.** A steady offset between two tables voids
  nothing: the model is fitted *and* scored on the registered `energy_generation`,
  so only a *change* of basis between the two windows can invalidate a gate read.
  Here that change is **one tenth of one percentage point**. On LV it is 17.1, on
  NL `wind_onshore` 126.6.
- **No convergence, in either direction.** The tables still disagree at
  **2026-08-20 18:00** — only 29 co-observed hours before the end of the data —
  by up to 13.0 MW. The vintage signature is *disagree, then converge*. This is a
  persistent few-percent disagreement, spread across hours (the 10 largest hours
  are 5.2% of the total absolute difference; 1,158 of 1,464 comparator hours have
  gen < ren), not a step.
- **Both tables move together against the TSO.** Monthly
  `|gen/TSO − ren/TSO|` is ≤0.024 in **six of the nine** clean months and ≤0.075 in
  eight, peaking at 0.168 in 2026-05. On LV the same spread reaches **0.84**.
  (2025-11's 1.885 is the ABL-188 partial-coverage head and is excluded everywhere
  above.)

So the 0.9648 is real but is **not** the ABL-439 defect: it is a small persistent
difference between two tables, present in the fit and gate windows alike, and far
inside the screen's own ±15% rule on both windows.

**Consequence.** Under the §14.6 standing rule this is the hold named in §14.5
clearing, with no new Board card. NL `wind_offshore` clears both oracle references
readably (+12.99% vs the oracle flat line, +11.87% vs oracle climatology in its
worst band, against a 7.51% wind floor). The disposition is the CEO's.

**One caveat I am not dropping.** NL `wind_offshore` has the same provenance split
as every other pair, and its gate window is 100% first-publication. Its revision
happens to be worth +0.17% rather than +17%, *on the fit window we can see* — the
gate window's own revision has not landed yet. It is expected to be as small, and
it is not yet measured.

## 4. Why the four were unscreened — two causes, and a ledger correction

Neither is a property of the data; both are ABL-439's `_programme_context` scope.

1. **DE / NL `wind_offshore` — a directory the glob does not name.** The screen
   globs `experiments/ABL348/results_*.json`; the offshore record is
   `experiments/ABL322/results_abl436_offshore_reread.json`.

   **Ledger §5.6 attributes this to the record being gitignored (ABL-440). That is
   not what happened to this file.** It is tracked at `origin/main`, committed by
   ABL-436 in `c2126b8`, and `git check-ignore` matches no rule against it.
   ABL-440 is about the path the `abl322-pilot` scope is *registered to write* —
   `experiments/ABL322/results.json`, which `experiments/*/results.json` does
   swallow — and ABL-436 sidestepped it by writing a differently-named file.
   **The screen was reachable all along; ABL-440 was never blocking it.**

2. **EE / FI `solar` — a band they have no cell in.** The screen takes one horizon
   band (`--programme-band`, default `24-36h`). These two pairs are the ABL-434
   coverage cases: of three bands only `48-64h` gates at all, so the band filter
   drops them **silently** rather than reporting them as unscreened.

Both are the ABL-431 lesson again — a sweep's blind spots live in its scope
declaration — so this script names its five pairs explicitly instead of
discovering them from a glob.

**The vintage structure is programme-wide, not pair-specific.** All five pairs show
the identical provenance split (`energy_generation` ≥28.8 days old at fetch before
the boundary, ≤28.9 after; `energy_renewable` flat at ~6.9 days throughout), and
every one of them is 94.26–94.28% pre-boundary on the fit window and 0.00% on the
gate window. What differs is only whether ENTSO-E's revision actually *changed*
the values for that pair — the fit−gate column of §1, in percentage points:
**+126.6** on NL `wind_onshore`, **+17.6** on EE `solar`, **+17.1** on LV `solar`,
**+0.17** on DE `wind_offshore`, **+0.10** on NL `wind_offshore`, **0.00** on FI
`solar`. A pair is not safe because it escaped the backfill — every pair was
backfilled — it is safe because its revision was empty. The clean pairs and the
affected ones are separated by two orders of magnitude with nothing in between,
so no pair here is a borderline call.

## 5. What this does and does not license

- It **does** settle §5.6's open question on LV `solar`, and adds EE `solar` to the
  same finding.
- It **does** complete the screen: all 41 ledger pair-records are now screened.
- It **does not** re-grade anything. Whether a 17.1% basis discontinuity should move
  a letter is ABL-437's question, and no ladder was re-read here.
- It **does not** promote anything. NL `wind_offshore` clearing the screen is a
  screen outcome reported under the standing rule; the shipping decision is the
  CEO's.
