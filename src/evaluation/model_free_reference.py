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
"""

from __future__ import annotations

import numpy as np
import pandas as pd


#: The flat-line comparator column names, in report order. Named once so a
#: harness cannot report one and score the other.
CONSTANT_COMPARATORS = ("constant_causal", "constant_oracle")

#: The hour-of-day comparator column names, likewise.
CLIMATOLOGY_COMPARATORS = ("climatology_causal", "climatology_oracle")

#: Every reference this module attaches, in report order. A harness adds this to
#: `REPORTED_COMPARATORS` and never to `GATE_BASIS`.
MODEL_FREE_COMPARATORS = CONSTANT_COMPARATORS + CLIMATOLOGY_COMPARATORS


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
    return result, levels


def comparator_wape(scores: dict, name: str):
    """A comparator's WAPE, or ``None`` if this record does not carry it.

    ``.get`` rather than ``[]`` so that re-rendering a ``results.json`` written
    before ABL-389 prints ``Not measured`` for the new columns instead of
    raising. An absent measurement and an unmeasurable one read the same, which
    is correct: neither is a number.
    """
    return (scores.get(name) or {}).get("wape_pct")


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
             f"{heading} constant causal | constant oracle | climatology causal | climatology oracle |",
             f"{rule}---:|---:|---:|---:|"]
    for row in rows:
        levels = row["model_free_reference_mw"]
        prefix = f"| {row[key]} | {row['country']} |" if key else f"| {row['country']} |"
        cells = [_fmt_level(levels.get(name)) for name in CONSTANT_COMPARATORS]
        cells.extend(_fmt_hourly(levels.get(name)) for name in CLIMATOLOGY_COMPARATORS)
        lines.append(f"{prefix} {' | '.join(cells)} |")
    return lines


def _fmt_level(level) -> str:
    return "Not measured" if level is None else f"{level:.2f} MW"


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
