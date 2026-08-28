# ABL-247 — does a TSO day-ahead vintage earn its place as a model *feature*?

**Author:** Forecasting Scientist · **Date:** 2026-08-28 · **Status:** evidence, no serving change

Machine record: `reports/abl247/abl247_tso_feature_backtest.json`
Pre-registration: document key `abl247-prereg` on ABL-247, written **2026-08-14**, eleven days before the data existed.
Reproduce:

```
.venv\Scripts\python.exe scripts/abl247_tso_feature_backtest.py ^
    --db C:\Code\able\data\energy_dashboard.db --out reports/abl247
```

---

## Headline

**The registered test passes, and it should not be quoted on its own.** Adding
the TSO day-ahead vintage as a feature rejects the null `c = 0` for all four
series at 0–24h — but the fitted combiner puts a weight of `b ≈ 0.02–0.11` on our
own forecast and `c ≈ 0.88–1.17` on the TSO's, and **it is worse than simply
using the TSO series alone in 6 of 8 (type, band) cells.** The feature reduces to
"use the other forecast", which is ABL-246's result, not a new one.

**Recommendation: do not adopt the TSO series as a model feature on this
evidence.** The one apparent exception — a combiner beats *both* arms in the two
countries where the TSO is weak (LT −6.87 pp, EE −3.69 pp) — routes to **ABL-283**
rather than to a modelling follow-up: LT and EE are two of its six suspect TSO
zones, and this pack reproduces their poor TSO accuracy independently (§6.3).

**The larger finding is negative and belongs to the renewable models:** on the
same rows, our solar model loses to an hour-of-day climatology by 18 pp, wind
onshore loses to a 28-day trailing constant by 15.7 pp, and wind offshore is
1.9x worse than that constant (§8.1).

**Two process findings:** the first run of this harness was invalid and is
documented rather than discarded quietly (§2), and the pre-registration's
48–64h re-scope is **confirmed** at 16 days — coverage there is exactly 0.00%
(§5).

---

## 0. The question, and what it is not

ABL-246 asked *"is the TSO better than us?"* and answered it: on D+1 load the TSO
beats our production ML in 20 of 23 evaluable countries, 3.26% against 8.85%
median WAPE. That is a **rival baseline** result and it stands.

This issue asks a different question: *"does our forecast **plus** the TSO's,
combined, beat our forecast alone on the same rows?"* A series can be a better
standalone forecast and still add nothing as a feature — if it carries no
information our own model has not already extracted. The registered estimator is

```
y  ~=  a + b*f_ours + c*f_tso          null hypothesis: c = 0
```

fit per (forecast_type, band), blocked leave-one-day-out CV.

---

## 1. Protocol

| | |
|---|---|
| **Window** | target days 2026-08-13 → 2026-08-28 (16 complete genuine target days; archive holds 17) |
| **Source** | `forecast_vintage_archive` (ABL-184), `source='tso'`, `model_name='tso-day_ahead'` |
| **Feature rule** | for each of our forecast rows, the cutoff is **its own `generated_at`**; the feature is the latest TSO vintage with `first_seen_at <= generated_at` |
| **Selection order** | per target **instant**, then averaged to the hour — never the reverse |
| **Basis** | **out-of-sample** throughout: every prediction scored is out-of-fold |
| **Truth** | `scorecard.ACTUAL_SPECS`, aggregated to hourly means |
| **Reads** | replica opened `file:...?mode=ro`; sidecar untouched; no serving, ingest or registry change |

**Leads are lower bounds.** The archive stamps `first_seen_at` — when our poller
saw a value — never the TSO's own publication time. Every coverage and lead
figure here understates how early the forecast existed, which biases **against**
the feature. That is the safe direction for a result that favours it.

**The go-live backfill is excluded, and the exclusion is exact.** 13,728,487 rows
carry `first_seen_at = 2026-08-11` and span target days 2018-12-31 → 2026-08-13;
they are retained post-revision values, not vintages. Counted naively they read
as years of vintage history that does not exist. Measured directly: **0 rows** of
that bucket survive the `target_day >= 2026-08-13` floor.

---

## 2. The first run was invalid — root cause and fix

The first live run of this harness completed with exit 0 and a full record.
**Every number in it was discarded.** It is documented here rather than quietly
re-run, because the failure mode is the kind that survives a green suite.

Two reported numbers were arithmetically impossible:

| | first run said | true |
|---|---|---|
| median feature lead, load 48–64h band | **54.07 h** | 47.07 h is the archive's **maximum** forward lead |
| feature present, load 48–64h | 1,273 rows | must be **0** |

A row 48–64h ahead of its cutoff cannot carry a day-ahead feature. Measured on a
raw positional read: maximum forward lead **47.07h (load) / 45.32h (solar)**,
with **0.0000%** of targets beyond 48h — consistent with ABL-390's ~34h ceiling.

**Root cause.** `normalize_ts` returned `pd.to_datetime(pd.Series(list(values)))`,
and `pd.Series(list(...))` carries a fresh positional `RangeIndex`. Every read
filters rows *before* parsing (`.isin(SUPPORTED_COUNTRIES)`, `.dropna()`), so the
frame being assigned back into has a **gappy** index, and
`frame["target"] = normalize_ts(frame["raw"])` aligned label-to-label against a
positional index.

It did not raise and it did not produce nulls. It produced **real timestamps
attached to the wrong rows**. `first_seen` was assigned after
`concat(ignore_index=True)` on a clean `RangeIndex` and stayed correct while
`target` did not — and that asymmetry is the only reason the corruption was
visible at all. Post-guard, the corrupted frame carried **16.33%** of rows at
lead > 48h with a maximum of **427.81h**, against 0.0000% and 47.07h in a raw
read of the same rows.

**Not the guard.** `guard_tso_frame` / `guard_tso_series` are clean: 0 refusals on
load, the pre-guard frame already carried correct leads, and `guard_series`
copies and masks in place without reordering, so its positional assign-back is
safe.

**Confined to this script.** The three other `pd.Series(list(...))` uses in the
repo — `model_free_reference.py:273`, `solar_features.py:106/138/228` — build an
index or go straight to `.to_numpy()` positionally and are never assigned back
into a filtered frame. No published number outside ABL-247 is affected.

**Fix** (`1e79174`): at the source, not at the five call sites, so a later read
cannot reacquire it. Three tests, the first **mutation-checked** — reverting
`normalize_ts` fails it with row 4's timestamp landing on row 2, exactly the
corruption signature.

---

## 3. Contamination statement

| issue | touches this work? | handling |
|---|---|---|
| **ABL-111 / ABL-109** zero-as-missing actual load | **Yes**, on the load target | `0.0` treated as missing and dropped; **1 row of 37,969 in-window (0.0026%)** |
| **ABL-71** prod ingest stale | **Yes**, bounded | archive `max(first_seen_at)` 2026-08-28T14:11:02Z (**3.18h** stale); truth current to 2026-08-28 13:30 UTC |
| **ABL-431 / ABL-458** TSO plausibility | **Yes**, on the feature | every archive read guarded before any resample, lag or merge; **0 refusals on all four series** — the HU `wind_onshore` 140,996 MW cluster sits on 2026-02-03/04 and is excluded by the `target_day >= 2026-08-13` floor, so the guard is armed here but has nothing to refuse in this window |
| **ABL-67** fabricated `net_position` | No | different series, out of scope |
| **ABL-277 / ABL-505 / ABL-506** NL gross-vs-net load basis | **Yes**, on NL load | NL load results are not interpretable; see §7 |

---

## 4. Estimator and power — why an affine combiner, not a retrain

The archive is the hard cap on training history for any TSO-feature model, and
it holds **17 genuine target days**. A gradient-boosted refit carrying one new
feature would train on those days against production models trained on months:
that arm measures the *training window*, not the feature. The pre-registration
fixed the ladder before any of this was visible.

**Primary — affine combiner.** `y ~= a + b*f_ours + c*f_tso`, three parameters,
which 16 days can carry. Fit per (forecast_type, band). Blocked
**leave-one-day-out** CV: the independent unit is the target **day**, not the
row, because within-day errors are strongly autocorrelated and European weather
correlates across countries on top of that. Every prediction scored is
out-of-fold — the day it lands on took no part in the fit that produced it.

**The interval on `c` is a delete-one-day cluster jackknife** with the standard
`(k-1)/k` inflation. A plain t-interval over leave-one-out estimates would
understate the variance badly, because each fold's estimate reuses `k-1` of the
same days.

**Collinearity is reported, not hidden.** Both arms forecast the same quantity,
so they are near-collinear by construction and the *split* of a shared effect
between `b` and `c` is weakly identified even where the pair jointly fits well.
That does not invalidate the registered test — `c = 0` is exactly the question
"does adding `f_tso` help", and the CV WAPE delta is measured on predictions,
which are stable under collinearity — but it is why an interval on `c` can be
wide beside a real WAPE movement. `corr(f_ours, f_tso)` and the VIF sit beside
every pooled fit.

**Secondary — the NaN-native retrain.** Pre-registered as exploratory and
underpowered, controlled against a matched no-feature retrain on the identical
days, never against the production model. **Not run.** The pre-committed rule in
§5 of the pre-registration is that a series whose `c` interval includes 0 at
0–24h does not escalate to the retrain arm hoping for a win, and that rule binds
regardless of which way the primary went.

**Refused — the retained-series proxy.** Ruled out by measurement, not
assumption. See §8.

---

## 5. Coverage, re-derived at 16 days (prereg §6.1)

The pre-registration's §1 figures were provisional at **n = 2 publication days**
and the CEO instruction was explicit that they must not be carried forward.
These replace them. Two denominators are reported because §1 used one and the
backtest uses the other; neither substitutes for the other.

**(a) Panel denominator** — of the target hours our own production runs actually
forecast, how many carry a feature:

| type | band | rows | countries | days | present | **coverage** | med. lead h | med. age at cutoff h |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| load | 0–24h | 6,574 | 21 | 16 | 6,554 | **99.70%** | 15.14 | 0.15 |
| load | 24–48h | 7,832 | 21 | 16 | 3,071 | **39.21%** | 27.14 | 1.07 |
| load | 48–64h | 5,149 | 21 | 16 | 0 | **0.00%** | — | — |
| solar | 0–24h | 785 | 3 | 16 | 769 | **97.96%** | 16.32 | 0.32 |
| solar | 24–48h | 826 | 3 | 16 | 114 | **13.80%** | 26.32 | 1.82 |
| solar | 48–64h | 594 | 3 | 16 | 0 | **0.00%** | — | — |
| wind_onshore | 0–24h | 1,026 | 10 | 16 | 979 | **95.42%** | 14.32 | 0.32 |
| wind_onshore | 24–48h | 1,118 | 3 | 16 | 163 | **14.58%** | 26.32 | 0.39 |
| wind_onshore | 48–64h | 734 | 3 | 16 | 0 | **0.00%** | — | — |
| wind_offshore | 0–24h | 639 | 3 | 16 | 619 | **96.87%** | 14.39 | 0.32 |
| wind_offshore | 24–48h | 745 | 2 | 16 | 133 | **17.85%** | 26.32 | 0.39 |
| wind_offshore | 48–64h | 489 | 2 | 16 | 0 | **0.00%** | — | — |

**(b) §1's own horizon-grid denominator** — standing at a cutoff, over every
target hour in `(cutoff, cutoff+64h]`, measured independently on a raw
positional read across all 34 archive countries: **0–24h 87.57% · 24–48h 28.89%
· 48–64h 0.00%**, against the provisional 78.1 / 31.5 / 0.0.

### The §1 re-scope is confirmed, not overturned

**48–64h coverage is exactly 0.00% — 0 rows of 6,966 across all four series.**
This is not a sampling artifact: the archive's maximum forward lead is **47.07h
(load) / 45.32h (solar)**, with **0.0000%** of targets beyond 48h. A day-ahead
product structurally cannot reach the top third of our horizon. The CEO's
decision on 2026-08-14 not to backtest that band was correct, and is now
confirmed at 16 days rather than 2.

**0–24h coverage is much better than the provisional figure** (99.70% vs 78.1%
for load); **24–48h is close to it** (39.21% vs 31.5%).

---

## 6. Result — the registered test, and what it actually means

### 6.1 The registered test rejects `c = 0` everywhere

Pooled per (type, band); blocked leave-one-day-out CV; `c` interval by
delete-one-day cluster jackknife. All out-of-sample.

| type | band | n | days | c-hat | 95% CI on c | WAPE null (CV) | WAPE combiner (CV) | delta pp | 95% CI on delta | verdict |
|---|---|---:|---:|---:|---|---:|---:|---:|---|---|
| load | 0–24h | 6,554 | 16 | 0.878 | [0.771, 0.985] | 7.06 | 3.69 | **−3.37** | [−4.58, −2.16] | excludes 0 |
| load | 24–48h | 3,071 | 16 | 0.966 | [0.846, 1.087] | 8.74 | 3.41 | **−5.33** | [−7.15, −3.45] | excludes 0 |
| solar | 0–24h | 769 | 16 | 1.006 | [0.955, 1.057] | 44.83 | 7.45 | **−37.38** | [−42.46, −29.56] | excludes 0 |
| solar | 24–48h | 114 | 15 | 0.853 | [0.596, 1.110] | 33.69 | 22.63 | **−11.05** | [−21.70, −1.68] | excludes 0 |
| wind_onshore | 0–24h | 979 | 16 | 0.923 | [0.859, 0.988] | 58.89 | 13.99 | **−44.90** | [−63.85, −38.05] | excludes 0 |
| wind_onshore | 24–48h | 163 | 15 | 0.937 | [0.807, 1.067] | 56.02 | 18.48 | **−37.54** | [−69.43, −18.64] | excludes 0 |
| wind_offshore | 0–24h | 619 | 16 | 1.064 | [0.913, 1.215] | 80.60 | 37.36 | **−43.24** | [−110.75, −25.16] | excludes 0 |
| wind_offshore | 24–48h | 133 | 15 | 1.171 | [0.859, 1.483] | 94.04 | 42.96 | **−51.08** | [−199.14, −25.00] | excludes 0 |

By the pre-committed criterion — *"if the CI on c includes 0 at 0–24h, the
feature has not earned its place"* — **the feature earns its place in all four
series.** Reported as the registered test requires. §6.2 is why that sentence
must not be quoted on its own.

### 6.2 The mechanism is replacement, not combination

**Read `b` beside `c`.** The fitted combiners are:

| type | band | a | **b** (our forecast) | **c** (TSO) | corr(ours, TSO) | VIF |
|---|---|---:|---:|---:|---:|---:|
| load | 0–24h | −7.75 | **0.112** | 0.878 | 0.995 | 95.7 |
| load | 24–48h | 55.32 | **0.023** | 0.966 | 0.991 | 55.8 |
| solar | 0–24h | 71.16 | **−0.086** | 1.006 | 0.854 | 3.7 |
| solar | 24–48h | 18.31 | **0.105** | 0.853 | 0.927 | 7.1 |
| wind_onshore | 0–24h | −22.59 | **0.040** | 0.923 | 0.688 | 1.9 |
| wind_onshore | 24–48h | −121.09 | **0.053** | 0.937 | 0.785 | 2.6 |
| wind_offshore | 0–24h | −79.83 | **0.087** | 1.064 | 0.096 | 1.0 |
| wind_offshore | 24–48h | −188.83 | **0.099** | 1.171 | −0.017 | 1.0 |

**`b` is within 0.11 of zero in every fit and `c` is within 0.18 of one.** The
fitted "combiner" is, to a good approximation, *the TSO forecast plus an offset*.
Our own forecast is weighted out.

For load the VIF is 95.7 / 55.8, so the *split* between `b` and `c` is weakly
identified there and the individual coefficients should not be over-read. That
caveat does not apply to solar and wind, where VIF is 1.0–7.1 and `b ≈ 0` is
firmly estimated — and for wind_offshore `corr(ours, TSO) = 0.096 / −0.017`, i.e.
our offshore forecast is essentially **uncorrelated** with the TSO's.

So the claim has to be tested directly: **is the combiner better than simply
using the TSO alone?** Pooled (volume-weighted) WAPE, identical
availability-matched rows:

| type | band | ours | **TSO alone** | **combiner (CV)** | combiner − TSO |
|---|---|---:|---:|---:|---:|
| load | 0–24h | 6.81 | **3.57** | 3.69 | **+0.12 (worse)** |
| load | 24–48h | 8.28 | **3.33** | 3.41 | **+0.08 (worse)** |
| solar | 0–24h | 43.67 | 7.69 | **7.45** | −0.24 |
| solar | 24–48h | 31.70 | **21.42** | 22.63 | +1.21 (worse) |
| wind_onshore | 0–24h | 62.84 | 14.34 | **13.99** | −0.35 |
| wind_onshore | 24–48h | 64.67 | **17.03** | 18.48 | +1.45 (worse) |
| wind_offshore | 0–24h | 125.34 | **36.40** | 37.36 | +0.96 (worse) |
| wind_offshore | 24–48h | 149.83 | **39.39** | 42.96 | +3.57 (worse) |

**In 6 of 8 cells the combiner is worse than the TSO series on its own**, and in
the 2 where it wins the margin is under 0.4 pp. Per country on load 0–24h the
combiner beats the TSO alone in **8 of 21**.

**The honest reading: the TSO day-ahead series carries real information our
models do not have, but almost none of it is *incremental* to our forecast.**
Nearly all of the measured gain is the gain ABL-246 already reported — the TSO
is simply a better forecast — reached here by an independent route under a
leak-free feature protocol. Adding it as a *feature* is not measurably better
than using it as a *forecast*, and is usually slightly worse.

### 6.3 The one place combination adds over replacement

The combiner beats the TSO alone precisely where the TSO is **weak**, which is
what a hedge should do. Load 0–24h:

| country | ours | TSO alone | combiner (CV) | combiner − TSO |
|---|---:|---:|---:|---:|
| **LT** | 8.46 | 14.67 | **7.80** | **−6.87** |
| **EE** | 10.59 | 13.14 | **9.45** | **−3.69** |
| *NL* | *24.23* | *25.35* | *23.41* | *−1.94 — not evaluable, §7* |
| NO | 2.29 | 1.89 | 1.45 | −0.44 |
| SI | 11.79 | 4.86 | 4.45 | −0.40 |

LT and EE are exactly the two countries ABL-246 flagged **"do not serve TSO"** and
**"hold"**. A routed design that serves the TSO where it wins and our model where
it does not would capture most of this; a fitted combiner captures it
automatically and does somewhat better in those two.

**This is a hypothesis, not a finding.** It is a per-country point comparison
selected after seeing the direction, on 2 of 21 countries, and I have **not**
attached a paired interval to it. The registered comparison is
combiner-vs-null, not combiner-vs-TSO; putting an interval on a post-hoc
comparison after seeing which way it went is the practice a pre-registration
exists to prevent.

**And it is probably not a modelling opportunity at all.** LT and EE are two of
the six zones **ABL-283** already carries as *suspect TSO load-forecast zones*,
split out of ABL-277. ABL-283 measured them on prod over 2026-08-04..11 at
**LT 14.2%** and **EE 10.8%** MAPE; this pack measures the same series on a
different window (2026-08-13..28) under first-seen vintages at **LT 14.67%** and
**EE 13.14%** WAPE. Those are independent reproductions of an open data-quality
question, and the ranking of LT and EE as the fleet's two worst TSO zones is the
same in both.

So the most likely reading of "the combiner hedges in LT and EE" is that it is
absorbing a **defective TSO series** that ABL-283 exists to establish or clear —
not that combination has found signal. Fitting a hedge on top of a series that
may be wrong at source would bake the defect into a model. **The correct next
step is ABL-283, not a modelling follow-up**, and I have not filed one.

---

## 7. Evaluability — what these renewable numbers can and cannot carry

The country counts in §5 overstate what is evaluable. Filtering to cells with
more than one target day:

| type | countries in panel | **countries with >1 day** | evaluable set |
|---|---:|---:|---|
| load | 21 | 21 | 20 usable (NL excluded) |
| solar | 3 | **3** | BE, DE, FR |
| wind_onshore | 10 | **3** | BE, DE, FR |
| wind_offshore | 3 | **2** | BE, FR |

For wind_onshore, 5 of the 10 countries (BG, CZ, EE, FI, GR) carry **n = 8–14
rows on a single day** and 2 (LT, SE) carry none; they produce no interval and
are excluded from every conclusion. `wind_offshore` NL is n = 14 on one day.

**The binding constraint is our own production coverage, not the archive.**
Measured on `forecasts` over the window: our production model writes rows for
**21 of 24** countries for load, but only **6** for solar (BE, CH, CZ, DE, FR,
RO), **10** for wind_onshore and **3** for wind_offshore. The TSO archive carries
33–34 countries throughout. So the renewable arms of this issue are underpowered
because we do not forecast those countries, not because the vintages are missing.

**NL is not evaluable** for load. ABL-277 located the divergence upstream in
ENTSO-E's A65 documents; ABL-505 / ABL-506 own the energy-forecast half.
`energy_load` is net of behind-the-meter solar while the forecasts are not, so
NL's 24.23 / 25.35 / 23.41 are uninterpretable and are excluded from every tally
above. ABL-246 reproduced this independently on the same window.

**24–48h is availability-confounded** and the pre-registration says so in
advance. Coverage there is 13.80–39.21%, and the composition term — our own arm
on all rows minus the same arm on matched rows — is large: solar **−10.55 pp**,
wind_offshore **+13.59 pp**, against **+0.03 to −4.69 pp** at 0–24h where
coverage is 95–99.7%. The matched subset at 24–48h is a different, easier
population, not a better forecast. Per the pre-committed rule, **no serving
recommendation is made for 24–48h.**

---

## 8. Negative findings, reported as promptly as the positive ones

**8.1 Our renewable models lose to model-free references.** On the same
availability-matched rows, pooled, 0–24h:

| type | ours | seasonal-naive (D-7) | climatology 28d (causal) | constant 28d (causal) |
|---|---:|---:|---:|---:|
| solar | **43.67** | 36.32 | **25.70** | 78.90 |
| wind_onshore | **62.84** | 58.02 | 48.50 | **47.11** |
| wind_offshore | **125.34** | 92.96 | **66.11** | 67.34 |
| load | 6.81 | 6.80 | 8.69 | 13.16 |

Solar loses to an hour-of-day climatology by 18 pp. Wind onshore loses to a
**28-day trailing constant** by 15.7 pp. Wind offshore is **1.9x worse than a
constant**. Per country, BE wind_onshore is **194.63%** WAPE and BE wind_offshore
**183.41%** against TSO figures of 23.82 and 39.26 on the identical rows.

The climatology is the ABL-437 causal reference imported, not re-implemented, so
solar is graded against hour-of-day rather than a flat line — the comparison
that certifies nothing. Load is the one series that holds its own, and it only
ties D-7 (6.81 vs 6.80), which reproduces ABL-246 §4.1 on a different band.

**This is a model-quality finding independent of anything the TSO does**, and it
is the more consequential half of this pack. It belongs to the renewable model
work, not to this issue.

**8.2 The `tso-week_ahead` series stays closed.** Pre-registered §8 as a closed
negative, not pending data. Not referenced anywhere in the harness. Every
post-go-live target carries negative lead.

**8.3 The retained-series proxy stays refused — but the premise behind it is
weaker than the pre-registration assumed.** Prereg §2 measured that 39.6% of load
targets carry a revision worth ~3.25% of level, and concluded training history is
hard-capped at the archive. ABL-246 subsequently measured the quantity that
actually matters — the **WAPE effect** of scoring on post-revision rather than
first-seen values — at **0.016 pp mean, 0.283 pp worst across 23 countries**. A
revision that moves 3.25% of level on 40% of targets and 0.016 pp of WAPE is a
revision that is essentially unbiased.

Those two measurements are not in conflict; they answer different questions. But
the *inference* in prereg §2 — "there is no proxy path to the pre-archive
months" — does not follow from the revision-share number alone, and ABL-246's
number weakens it substantially. **I have not acted on this.** The
pre-registration is binding on this run and it refuses the proxy; re-opening a
design choice after seeing the data is exactly what it exists to prevent. It is
recorded here as a deviation *candidate* and carried into §9 for a future
pre-registration to settle.

---

## 9. Recommendation

**No serving change is recommended in this issue, and none is made.** Promotion
is a pre-registered gate read plus a Board decision; this is evidence.

1. **Do not add the TSO day-ahead series as a model feature on this evidence.**
   The registered test passes, but the fitted combiner sets `b ≈ 0` and is worse
   than the TSO alone in 6 of 8 cells. A feature that reduces to "use the other
   forecast" should be adopted as the other forecast, not as a feature — it is
   simpler, has no fitted parameters to drift, and scores better.

2. **The decision-relevant question is ABL-246's, and it is already before the
   CEO.** This pack independently corroborates it on a second window under a
   leak-free feature protocol: TSO 3.57 vs ours 6.81 pooled WAPE on load 0–24h.

3. **The LT/EE hedge routes to ABL-283, not to a modelling follow-up.** The one
   thing here ABL-246 does not already answer is that a combiner beats *both*
   arms where the TSO is weak (LT −6.87 pp, EE −3.69 pp vs TSO alone). But LT and
   EE are two of the six **suspect TSO load-forecast zones** ABL-283 already
   carries, and this pack independently reproduces their poor TSO accuracy on a
   different window under first-seen vintages (§6.3). A hedge fitted on a series
   that may be defective at source would bake the defect into a model. **No
   modelling issue filed**; the corroborating measurement is added to ABL-283
   instead.

4. **Priority for the renewable models is §8.1, not the TSO.** Solar, wind
   onshore and wind offshore all lose to causal model-free references on the same
   rows. A feature cannot rescue a model that a 28-day constant beats.

5. **48–64h needs no further vintage work.** Coverage is 0.00% and the cause is
   structural. Nothing accrues that changes it.

### Caveats carried

- **16 target days**, 2026-08-13 → 2026-08-28; the independent unit is the
  (country, target-day) pair, so effective n is materially below row counts.
- Every metric here is **out-of-sample** (out-of-fold under day-blocked CV).
- Leads and coverage are **lower bounds** — `first_seen_at` is our poller's
  stamp, not the TSO's publication time.
- Contamination touching this window is stated in §3; ABL-111/109 removed 1 row
  of 37,969, ABL-431/458 guard refused 0 rows on all four series over the window.
- The 48–64h band was **measured but not fitted**, per the binding re-scope.
- The secondary NaN-native retrain arm was **not run**, per the pre-committed
  rule in §4.
