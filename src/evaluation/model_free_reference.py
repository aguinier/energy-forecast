"""ABL-389: the model-free references both gate harnesses report.

ABL-380 (tranche 1a) passed 6/6 and reported, against its own passing result,
two things the gate read could not have told anyone:

* **CH wind_onshore cleared all 3 cells while scoring 47.42% WAPE, against a
  flat line at the gate-window median at 40.29%** — the fitted model was 7.1pp
  *worse* than a constant. Slope 0.094, correlation 0.176.
* **BG's registered D-7 bar of 93.75% is cleared outright by a causal constant
  at the fit-window mean, 82.77%** — with no model at all. BG still passes on
  merit (56.86%), but the bar did not establish that.

The D-7 baseline was registered when the gated pairs had incumbents and real
seasonal structure. On a low-capacity-factor onshore pair it certifies close to
nothing, and 33 more such pairs are queued behind tranche 1a — all with zero
rows in `forecasts`, all otherwise dispositioned against the same weak floor.

So both harnesses now print, beside every gate cell, what that cell's PASS is
actually worth. Two predictors, each in a causal and an oracle form, bounding
the question from opposite sides:

``constant_causal`` / ``constant_oracle``
    a flat line, at the **fit-window mean** and at the **gate-window median**.
    The causal form is the fair "no model" floor, using only what a forecaster
    had before the gate window opened; the oracle form is the hindsight upper
    bound on what *any* constant could have achieved, since the median minimises
    ``sum|a - c|``. That is what catches a model losing to a flat line it could
    never have known.
``climatology_causal`` / ``climatology_oracle``
    the same idea, per **hour of day** — the fit-window hourly mean and the
    gate-window hourly median. Still no model, still available at forecast time
    in the causal form, but able to represent the one structure a constant
    cannot represent at all.

**Why both, rather than the constant alone.** The Forecasting Scientist measured
the constant across ABL-381's six solar cells and found it useless there: every
form scores 63–95% WAPE, because a flat line cannot represent a diurnal cycle
and on solar the diurnal cycle *is* the signal. A comparator that always loses
by 70pp is a formality, not a bar — the same defect one level up as the one this
issue was opened to fix. Re-measured independently here on 2026-08-13 against
the replica, over ABL-348's registered windows, whole gate window per pair:

===============  =============  =============  ============  ============
pair             const causal   const oracle   clim causal   clim oracle
===============  =============  =============  ============  ============
BG solar               75.30%         73.49%        41.98%       19.15%
CH solar               95.08%         94.65%        37.53%        9.02%
BG wind_onshore        82.77%         63.78%        81.03%       62.50%
CH wind_onshore        79.07%         40.29%        77.82%       38.20%
===============  =============  =============  ============  ============

The two wind constants reproduce ABL-380's published numbers to the decimal,
which is what establishes that this module computes what that read computed.

The climatology is the tighter reference on **both** technologies, as it must
be — a constant is the one-bucket degenerate case of a climatology, and the
median minimises the same sum within each bucket independently. So this is not a
solar-only concession: CH wind's challenger at 47.42% loses to its oracle
climatology by 9.2pp, where the constant put the gap at 7.1pp. It sharpens the
finding that motivated the issue rather than softening it.

The constant is kept beside it because the *pair* carries what neither carries
alone. ``constant_*`` measures predicting the level; ``climatology_*`` measures
predicting the level and the daily shape. The gap between them is how much of a
series is forced diurnal structure — about 1.5pp on CH wind, about 86pp on CH
solar — which is worth reading per pair before dispositioning 33 of them.

**These are reported references, never gate criteria.** All four belong to
`REPORTED_COMPARATORS` and to no `GATE_BASIS` entry; the PASS rule, the
registered bands, the bars, the windows and `experiments/ABL348/config.json` are
untouched, and a pair that clears D-7 while losing to a reference still reads
PASS — beside the number that qualifies it. Moving a bar after seeing a result
is exactly what the pre-registration apparatus exists to prevent, and it does
not become acceptable because the change would be a conservative one.
`tests/test_gate_model_free_reference.py` pins both halves of that.

ABL-437: the causal references are levelled on a trailing window too
--------------------------------------------------------------------

The two ``*_causal`` references above are levelled on the **fit** window and
scored on the **gate** window, and ABL-348's fit window is
2026-01-14 -> 2026-07-11 against a gate window of 2026-07-11 -> 2026-08-10.
That is a winter-to-summer average scored against high summer, and wind is
seasonal, so on a wind pair the "causal constant" is not an estimate of the gate
window's level at all. Measured across every committed tranche record, worst
band per pair, as the gap between the causal constant and the correctly-levelled
oracle constant:

* NL ``wind_onshore`` 225.54% vs 73.85% -- **205% inflated**. A flat line at the
  fit-window mean scores three times worse there than forecasting zero.
* PT 102%, CH 96%, IT 76%, ES 50%, BG 43%, LT 39%, HR 38%, PL 22%, CZ 21%.
* Every solar pair sits at 0-8%: on solar the diurnal cycle dominates the
  denominator, so a level error barely moves the WAPE. **This is a wind
  problem.**

G2 and G3 are registered on exactly these two references, so where the level
moves between the two windows both become strawmen and the grade is inflated for
free. That is the third instance of one pattern -- ABL-406 (bar weakness),
ABL-417 (mis-levelled ``constant_causal`` on RO), ABL-435 (BG/CH).

So this module attaches **two more causal references**, levelled the way a
forecaster actually would:

``constant_causal_28d`` / ``climatology_causal_28d``
    the same flat line and hour-of-day mean, taken over the
    :data:`TRAILING_WINDOW_DAYS`-day window ending at **the row's own
    ``generated_at``** -- the forecast issue instant. Strictly causal by
    construction, and not by a new argument: it is the same anchor, the same
    inclusive hour-floored bound and the same ABL-188-filtered series that
    ``wind_features._rolling_features`` already uses for
    ``target_value_roll_168h_mean``, which is one of the challenger's own input
    features. The reference therefore uses no information the challenger did not
    have.

Four things about the form, each of which was a choice:

* **28 days, and the two forms share the window.** A constant is a climatology
  with one bucket, so levelling them differently would break the reading that
  the gap between them is forced diurnal structure. A shared window has to serve
  the climatology, which needs enough samples per hour-of-day bucket to be a
  level rather than noise -- 28 days gives 28, where 7 would give 7 and re-create
  the strawman defect in a new place. 28 days is also four whole weeks, so
  day-of-week composition is balanced, and it is short enough to sit inside one
  season, which is the whole point.
* **The window is in the column name.** Two reads levelled on different windows
  must not wear the same name, for the reason two reads on different source
  tables must not: changing the parameter changes the column, and a record says
  which it carries.
* **Nothing is removed.** ``constant_causal`` and ``climatology_causal`` keep
  their names, their definitions and their published values, so every letter
  already graded still means what it meant. The new columns are reported beside
  them, which makes the inflation itself a printed diagnostic
  (:func:`level_inflation`) rather than something a human has to go looking for.
* **The oracles stay off the ladder.** An oracle is not causally available. The
  fix for a strawman causal reference is a correctly-levelled causal reference,
  not a hindsight one.

Which pair of causal references the ABL-418 ladder reads is a **per-scope
registration** (``CAUSAL_LEVELLING`` in each harness), not a global. Every scope
whose record is already published is pinned to :data:`FIT_WINDOW` so its read
still reproduces; the default for a new scope is :data:`TRAILING_28D`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


#: The flat-line comparator column names, in report order. Named once so a
#: harness cannot report one and score the other.
CONSTANT_COMPARATORS = ("constant_causal", "constant_oracle")

#: The hour-of-day comparator column names, likewise.
CLIMATOLOGY_COMPARATORS = ("climatology_causal", "climatology_oracle")

#: ABL-437's trailing-window causal pair, in report order.
TRAILING_COMPARATORS = ("constant_causal_28d", "climatology_causal_28d")

#: Every reference this module attaches, in report order. A harness adds this to
#: `REPORTED_COMPARATORS` and never to `GATE_BASIS`.
MODEL_FREE_COMPARATORS = (CONSTANT_COMPARATORS + CLIMATOLOGY_COMPARATORS
                          + TRAILING_COMPARATORS)

#: The trailing window, in days. It is in the two column names above on purpose:
#: changing it here without changing them would let two reads levelled on
#: different windows wear one name. `tests/test_abl437_causal_levelling.py`
#: holds the two together.
TRAILING_WINDOW_DAYS = 28

#: The two registered levelling forms for the *causal* references, i.e. which
#: pair of columns G2 and G3 are scored against. The oracle references are
#: unaffected by either: they are hindsight bounds and are on no ladder.
FIT_WINDOW = "fit_window"
TRAILING_28D = "trailing_28d"
CAUSAL_LEVELLINGS = (FIT_WINDOW, TRAILING_28D)


def _window_values(actuals: pd.Series, start, end) -> pd.Series:
    """The target series over ``[start, end)``, finite values only.

    Half-open, matching every other window in this protocol. The builder loads
    actuals over a span that runs 14 days before the fit window (for lags) and,
    because ``_load_actuals_series`` asks the loader for ``end + 1 day``, can
    reach past the gate window too — so both bounds are applied here rather than
    assumed from the series.
    """
    if actuals.empty:
        return actuals.astype(float)
    index = pd.DatetimeIndex(actuals.index)
    inside = (index >= pd.Timestamp(start)) & (index < pd.Timestamp(end))
    return actuals[inside].dropna()


def constant_reference_levels(actuals: pd.Series, fit_start, gate_start,
                              gate_end) -> dict:
    """The two constant-predictor levels, in MW, for one pair.

    ``None`` when the window holds no finite observation. A level that cannot be
    measured is not a number: the column it fills is then all-NaN, its own
    intersection with the gate basis is empty, and it reads ``Not measured``
    rather than standing in for a measurement nobody made.
    """
    fit = _window_values(actuals, fit_start, gate_start)
    gate = _window_values(actuals, gate_start, gate_end)
    return {"constant_causal": float(fit.mean()) if len(fit) else None,
            "constant_oracle": float(gate.median()) if len(gate) else None}


def _hourly(values: pd.Series, aggregate: str) -> dict:
    """One level per hour of day present, from ``values``.

    Hours absent from the window are absent from the mapping — not zero, and not
    filled from a neighbouring hour. A country that reports nothing at 03:00 has
    no 03:00 climatology, and the rows at that hour must drop out of this
    comparator's intersection rather than be scored against an invented level.
    """
    if not len(values):
        return {}
    grouped = getattr(values.groupby(pd.DatetimeIndex(values.index).hour), aggregate)()
    return {int(hour): float(level) for hour, level in grouped.items()
            if np.isfinite(level)}


def climatology_reference_levels(actuals: pd.Series, fit_start, gate_start,
                                 gate_end) -> dict:
    """The two hour-of-day levels sets, in MW, for one pair.

    ``{}`` — never a level — when the window holds no finite observation, for
    the same reason ``constant_reference_levels`` returns ``None``.

    The causal form is the fit-window **mean** and the oracle form the
    gate-window **median**, matching the constants: the mean is what a
    forecaster would have fitted, and the median is what minimises ``sum|a - c|``
    within each bucket, which makes the oracle a true upper bound on any
    hour-of-day predictor rather than merely a good one.
    """
    return {"climatology_causal": _hourly(_window_values(actuals, fit_start, gate_start), "mean"),
            "climatology_oracle": _hourly(_window_values(actuals, gate_start, gate_end), "median")}


def _trailing_window(actuals: pd.Series, as_of, window_days: int) -> pd.Series:
    """The finite target values over the trailing window ending at ``as_of``.

    Hour-floored and **inclusive** at the anchor, spanning ``window_days * 24``
    hours -- character for character the bound
    ``wind_features._rolling_features`` applies to
    ``target_value_roll_168h_mean``. That is deliberate and is the whole
    causality argument: this reference sees exactly the series the challenger's
    own rolling features were built from, at that row's own issue instant, so
    there is no new claim about what was knowable when.
    """
    anchor = pd.Timestamp(as_of).floor("h")
    start = anchor - pd.Timedelta(hours=window_days * 24 - 1)
    index = pd.DatetimeIndex(actuals.index)
    return actuals[(index >= start) & (index <= anchor)].dropna()


def trailing_reference_levels(actuals: pd.Series, as_of_values,
                              window_days: int = TRAILING_WINDOW_DAYS) -> dict:
    """``{as_of: {"constant": level | None, "climatology": {hour: level}}}``.

    One entry per **distinct** issue instant, because the eight pre-registered
    run instants are shared by every target hour of a day: a 30-day gate window
    has some 240 of them and some 2,160 rows, so levelling per instant rather
    than per row is the same numbers at a ninth of the work.

    ``None`` and ``{}`` carry the same meaning they carry everywhere else in
    this module -- a level nobody could measure is not a number, the column is
    NaN there, and those rows leave that comparator's own intersection.
    """
    levels = {}
    for as_of in pd.DatetimeIndex(pd.Series(list(as_of_values)).unique()):
        window = _trailing_window(actuals, as_of, window_days)
        levels[as_of] = {"constant": float(window.mean()) if len(window) else None,
                         "climatology": _hourly(window, "mean")}
    return levels


def _trailing_summary(levels: dict, window_days: int) -> dict:
    """What a record keeps of a per-row level: its range, not 240 numbers.

    A trailing reference is a *set* of levels, one per issue instant, so unlike
    the two fit-window levels it cannot be written down as one number. The
    summary is what the report renders and what a later reader checks a WAPE
    against; the per-instant levels are reproducible from the same series by
    :func:`trailing_reference_levels` and are deliberately not duplicated into
    every record.
    """
    constants = [entry["constant"] for entry in levels.values()
                 if entry["constant"] is not None]
    hours = [len(entry["climatology"]) for entry in levels.values()]
    return {"window_days": window_days, "as_of_count": len(levels),
            "constant_min_mw": min(constants) if constants else None,
            "constant_max_mw": max(constants) if constants else None,
            "constant_mean_mw": float(np.mean(constants)) if constants else None,
            "climatology_hours_min": min(hours) if hours else None,
            "climatology_hours_max": max(hours) if hours else None}


def attach_trailing_references(frame: pd.DataFrame, actuals: pd.Series,
                               window_days: int = TRAILING_WINDOW_DAYS
                               ) -> tuple[pd.DataFrame, dict]:
    """Attach ABL-437's two trailing-window causal columns, keyed on ``generated_at``.

    A frame with no ``generated_at`` column has no issue instant to level on, so
    both columns are all-NaN and read ``Not measured``. That is the safe
    direction and not an accident: a condition that could not be measured is not
    satisfied (ABL-418), so a cell missing the column cannot grade ``A`` on it.
    It never silently falls back to the fit-window level, which would restore the
    defect this reference exists to remove.
    """
    result = frame.copy()
    constant_name, climatology_name = TRAILING_COMPARATORS
    if "generated_at" not in result.columns or result.empty:
        result[constant_name] = np.nan
        result[climatology_name] = np.nan
        return result, {"window_days": window_days, "as_of_count": 0,
                        "constant_min_mw": None, "constant_max_mw": None,
                        "constant_mean_mw": None, "climatology_hours_min": None,
                        "climatology_hours_max": None}
    as_of = pd.DatetimeIndex(result["generated_at"])
    levels = trailing_reference_levels(actuals, as_of, window_days)
    hours = pd.DatetimeIndex(result["target_ts"]).hour
    result[constant_name] = [
        np.nan if levels[stamp]["constant"] is None else levels[stamp]["constant"]
        for stamp in as_of]
    result[climatology_name] = [
        levels[stamp]["climatology"].get(int(hour), np.nan)
        for stamp, hour in zip(as_of, hours)]
    return result, _trailing_summary(levels, window_days)


def attach_model_free_references(frame: pd.DataFrame, actuals: pd.Series,
                                 fit_start, gate_start,
                                 gate_end) -> tuple[pd.DataFrame, dict]:
    """Attach all four model-free comparator columns to a scored frame.

    Every reference is represented as a *column* and scored by exactly the path
    ``seasonal_naive`` and ``persistence`` are scored by — never as a special
    case inside the scorer. That is what gives each one its own intersection with
    the gate basis, and therefore what makes a reference that cannot be measured
    read ``Not measured`` instead of emptying the cell (ABL-322/ABL-378). It is
    why this is done this way and why it must not be shortcut later.

    A constant is one number per pair broadcast over every row, so its
    intersection equals the basis intersection wherever the level exists and is
    empty wherever it does not. **A climatology is 24 numbers, so unlike every
    other comparator here it can be *partially* measurable**: a row whose hour of
    day never appeared in the source window gets NaN, drops from this
    comparator's intersection alone, and lowers only its own ``n``. Read that
    ``n`` before comparing a climatology WAPE to a challenger WAPE — scored on
    different rows, they are not the same measurement. The alternative, filling
    the missing hour from its neighbours, would be interpolating a data point to
    close a visual gap, and is not on the table.

    **ABL-437 attaches its trailing pair here too, for every scope**, and which
    pair the ladder *reads* is registered per scope elsewhere. Reporting all six
    unconditionally is the same decision ABL-389 made when it added the first
    four to scopes read before they existed: a column costs a column, and the
    alternative -- attaching the reference only where it is graded -- would mean
    a record could not show the inflation it was graded around.

    Returns the frame and the levels, so the run can record what it used.
    """
    result = frame.copy()
    levels = {**constant_reference_levels(actuals, fit_start, gate_start, gate_end),
              **climatology_reference_levels(actuals, fit_start, gate_start, gate_end)}
    for name in CONSTANT_COMPARATORS:
        level = levels[name]
        result[name] = np.nan if level is None else float(level)
    # `.map` on a plain dict leaves NaN wherever the hour is missing, which is
    # exactly the wanted behaviour and the reason it is not a `.replace`.
    hours = pd.Series(pd.DatetimeIndex(result["target_ts"]).hour, index=result.index)
    for name in CLIMATOLOGY_COMPARATORS:
        result[name] = hours.map(levels[name]).astype(float)
    result, trailing = attach_trailing_references(result, actuals)
    levels["trailing"] = trailing
    return result, levels


def comparator_wape(scores: dict, name: str):
    """A comparator's WAPE, or ``None`` if this record does not carry it.

    ``.get`` rather than ``[]`` so that re-rendering a ``results.json`` written
    before ABL-389 prints ``Not measured`` for the new columns instead of
    raising. An absent measurement and an unmeasurable one read the same, which
    is correct: neither is a number.
    """
    return (scores.get(name) or {}).get("wape_pct")


def level_inflation(scores: dict, name: str = "constant_causal"):
    """How much worse a causal constant scores than the correctly-levelled one.

    ``constant_causal / constant_oracle`` as a ratio of WAPEs, minus one, in
    percent -- the quantity ABL-437 was opened on, and 205% at its worst (NL
    ``wind_onshore``: 225.54% against 73.85%). It is a property of the
    *reference*, not of the challenger: a flat line cannot be a fair "does it
    predict the level" test when the level it sits at is not the level of the
    window it is scored on.

    This is the half of the not-evaluable proposal that survives into the
    adopted form. Instead of spending the number on an abstention, it is printed
    beside every cell, per causal reference, so the residual mis-levelling of the
    trailing form is visible too rather than assumed away. ``None`` where either
    side was not measured, or where the oracle scored no error at all.
    """
    causal = comparator_wape(scores, name)
    oracle = comparator_wape(scores, "constant_oracle")
    if causal is None or oracle is None or not oracle:
        return None
    return 100.0 * (causal / oracle - 1.0)


def reference_prose() -> list[str]:
    """The paragraph that says these four columns are not a second gate."""
    return [
        "Model-free reference (ABL-389) — four predictors with no model in them, reported beside every cell. "
        "`constant_causal` is a flat line at the **fit-window mean**, the honest \"no model\" floor, using only what was "
        "knowable before the gate window opened; `constant_oracle` is a flat line at the **gate-window median**, a hindsight "
        "upper bound on what any constant could have achieved. `climatology_causal` and `climatology_oracle` are the same two "
        "forms taken **per hour of day** — the fit-window hourly mean and the gate-window hourly median — which is the tighter "
        "reference on every pair measured so far, because a constant is a climatology with one bucket. Read the pair together: "
        "the constant says whether the model predicts the *level*, the climatology says whether it predicts the level *and the "
        "daily shape*, and the gap between them is how much of this series is forced diurnal structure.",
        "",
        "All four are **reported references and not gate criteria**: none is in the gate basis, none can move a cell's verdict, "
        "and a pair that clears its D-7 bar while losing to one still reads PASS. They are the number that qualifies the PASS — "
        "a challenger that does not beat `climatology_oracle` has not demonstrated skill beyond the average day, and a D-7 bar "
        "that `constant_causal` clears on its own was not a demanding bar. **Check each reference's own n before comparing it "
        "to the challenger.** A climatology is 24 levels, so an hour of day absent from its source window leaves those rows "
        "unscored for that column alone; scored on different rows, two WAPEs are not the same measurement. Nothing is "
        "interpolated to close that gap.",
        "",
        f"**Trailing-window causal reference (ABL-437).** `constant_causal_{TRAILING_WINDOW_DAYS}d` and "
        f"`climatology_causal_{TRAILING_WINDOW_DAYS}d` are the same flat line and the same hour-of-day mean, levelled over "
        f"the **{TRAILING_WINDOW_DAYS} days ending at each row's own `generated_at`** instead of over the whole fit window. "
        f"They exist because the fit window here runs winter to summer and the gate window is high summer, so on a seasonal "
        f"series the fit-window mean is not an estimate of the gate window's level: a causal constant reads up to **205% "
        f"worse than the correctly-levelled oracle constant** (NL `wind_onshore`, 225.54% against 73.85%), which inflates a "
        f"G2/G3 pass for free. The trailing form is strictly causal by construction — same anchor, same inclusive "
        f"hour-floored bound and same filtered series as the challenger's own `target_value_roll_168h_mean` feature — so it "
        f"uses no information the challenger did not have. The `level inflation` column prints the residual per cell. "
        f"**Which pair the grade ladder reads is registered per scope**; the fit-window pair keeps its name, its definition "
        f"and every value already published.",
    ]


def levels_table(training: list[dict], key: str | None = None) -> list[str]:
    """The per-pair reference levels in MW, so the WAPEs above can be checked.

    The two constants in full, and the climatology summarised by its range and
    the count of hours it covers — 24 levels per pair times two forms would be a
    table nobody reads, and the hourly levels are in ``results.json`` in full.
    The **hours covered** stay in the markdown even so: that is the number that
    says whether a climatology WAPE was scored on the same rows as the cell.

    Skips rows with no recorded level, so a record written before ABL-389 (or a
    test fixture that predates it) renders without them rather than raising.
    """
    rows = [row for row in training if row.get("model_free_reference_mw")]
    if not rows:
        return []
    heading = "| type | country |" if key else "| country |"
    rule = "|---|---|" if key else "|---|"
    lines = ["", "Reference levels used, from the same ABL-188-filtered target series the gate actuals and the "
             "D-7/persistence baselines come from — no refit, no second read, no additional upstream fetch. The hourly "
             "levels behind the climatology columns are in `results.json` in full; `h` is how many of the 24 hours of the "
             "day that level set covers, and anything below 24 means those rows were dropped from that column's n:", "",
             f"{heading} constant causal | constant oracle | climatology causal | climatology oracle "
             f"| constant causal {TRAILING_WINDOW_DAYS}d |",
             f"{rule}---:|---:|---:|---:|---:|"]
    for row in rows:
        levels = row["model_free_reference_mw"]
        prefix = f"| {row[key]} | {row['country']} |" if key else f"| {row['country']} |"
        cells = [_fmt_level(levels.get(name)) for name in CONSTANT_COMPARATORS]
        cells.extend(_fmt_hourly(levels.get(name)) for name in CLIMATOLOGY_COMPARATORS)
        cells.append(_fmt_trailing(levels.get("trailing")))
        lines.append(f"{prefix} {' | '.join(cells)} |")
    return lines


def _fmt_level(level) -> str:
    return "Not measured" if level is None else f"{level:.2f} MW"


def _fmt_trailing(summary) -> str:
    """ABL-437's trailing constant as its range over the issue instants.

    A range and not a single number, because that is what the reference *is*:
    one level per forecast issue instant. A record that shows only its mean
    would read like a flat line and hide the tracking that makes it correct.
    """
    if not summary or summary.get("constant_min_mw") is None:
        return "Not measured"
    return (f"{summary['constant_min_mw']:.2f}–{summary['constant_max_mw']:.2f} MW "
            f"({summary['as_of_count']} as-of)")


def _fmt_hourly(levels) -> str:
    """An hourly level set as its range and its coverage, or ``Not measured``."""
    if not levels:
        return "Not measured"
    values = [float(v) for v in levels.values()]
    return f"{min(values):.2f}–{max(values):.2f} MW ({len(values)}h)"


#: The oracle references, with the sentence each one's loss means. Both are
#: hindsight bounds, so being beaten by either is a statement about the
#: challenger and never about the verdict.
_ORACLES = (
    ("constant_oracle", "constant",
     "A flat line at the gate-window median scores better than the fitted model there, so whatever those cells earn over the "
     "D-7 bar they earn by predicting close to the level and varying little around it — and not even at the best level."),
    ("climatology_oracle", "climatology",
     "An hour-of-day median — the average day, with no model and no weather in it — scores better than the fitted model "
     "there. This is the weaker claim to lose and the stronger one to win: a challenger that beats it is doing something no "
     "table of hourly averages can do."),
)


def lost_to_a_model_free_reference(rows: list[dict], label) -> list[str]:
    """Name every cell whose challenger is beaten by an oracle reference.

    Stated in prose for the same reason the harness already states "the TSO
    forecast is better than the challenger": a reader who sees PASS in the gate
    column will not otherwise compare two numbers in a sixteen-column table, and
    ABL-380 is the proof — 6/6 PASS was reported, correctly, on a pair whose
    fitted model was worse than a flat line. This changes no verdict.

    Each oracle is reported separately rather than collapsed into "beaten by
    something", because they mean different things: losing to the constant says
    the model failed to beat the level, losing to the climatology says it failed
    to beat the average day. A pair can do the second while passing the first.
    """
    lines = []
    for name, noun, explanation in _ORACLES:
        beaten = []
        for row in rows:
            challenger = row["scores"]["challenger"]["wape_pct"]
            oracle = comparator_wape(row["scores"], name)
            if challenger is not None and oracle is not None and oracle < challenger:
                beaten.append((label(row), challenger, oracle))
        if not beaten:
            continue
        lines.append(f"- **The challenger loses to a {noun} chosen with hindsight in {len(beaten)} cell(s).** {explanation} "
                     "This does not change any verdict above; it bounds what the verdict means:")
        lines.extend(f"  - {cell}: challenger {challenger:.2f}% vs oracle {noun} {oracle:.2f}% "
                     f"({challenger - oracle:+.2f}pp)" for cell, challenger, oracle in beaten)
    return lines
