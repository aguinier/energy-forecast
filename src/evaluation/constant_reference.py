"""ABL-389: the constant-predictor reference both gate harnesses report.

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
actually worth. Two levels, bounding the question from opposite sides:

``constant_causal``
    the **fit-window mean** — the fair "no model" floor, using only information
    a forecaster had before the gate window opened.
``constant_oracle``
    the **gate-window median** — the hindsight upper bound on what *any*
    constant could have achieved, which is what catches a model that loses to a
    flat line it could never have known.

**These are reported references, never gate criteria.** They belong to
`REPORTED_COMPARATORS` and to no `GATE_BASIS` entry; the PASS rule, the
registered bands, the bars, the windows and `experiments/ABL348/config.json` are
untouched, and a pair that clears D-7 while losing to a constant still reads
PASS — beside the number that qualifies it. Moving a bar after seeing a result
is exactly what the pre-registration apparatus exists to prevent, and it does
not become acceptable because the change would be a conservative one.
`tests/test_gate_constant_reference.py` pins both halves of that.

The definitions here reproduce ABL-380's hand-rolled read exactly, measured
against the replica on 2026-08-13 over its registered windows: BG causal 82.77%
/ oracle 63.78%, CH causal 79.07% / oracle 40.29%, CH fit-window mean 21.97 MW
and gate-window mean 12.91 MW.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


#: The two comparator column names, in report order. Named once so a harness
#: cannot report one and score the other.
CONSTANT_COMPARATORS = ("constant_causal", "constant_oracle")


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


def attach_constant_references(frame: pd.DataFrame, actuals: pd.Series,
                               fit_start, gate_start,
                               gate_end) -> tuple[pd.DataFrame, dict]:
    """Attach the two constant-predictor columns to a scored frame.

    A constant is one number per pair broadcast over every scored row, so its
    own intersection with the gate basis equals the basis intersection wherever
    the level exists, and is empty wherever it does not. That property falls out
    of representing the reference as a *column* — scored by exactly the path
    ``seasonal_naive`` and ``persistence`` are scored by — instead of as a
    special case inside the scorer, which is why it is done this way and why
    that must not be shortcut later.

    Returns the frame and the levels, so the run can record what it used.
    """
    result = frame.copy()
    levels = constant_reference_levels(actuals, fit_start, gate_start, gate_end)
    for name in CONSTANT_COMPARATORS:
        level = levels[name]
        result[name] = np.nan if level is None else float(level)
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
    """The paragraph that says these two columns are not a second gate."""
    return [
        "Constant-predictor reference (ABL-389) — `constant_causal` is a flat line at the **fit-window mean**, the honest "
        "\"no model\" floor, using only what was knowable before the gate window opened. `constant_oracle` is a flat line at "
        "the **gate-window median**, a hindsight upper bound on what any constant could have achieved. Both are **reported "
        "references and not gate criteria**: neither is in the gate basis, neither can move a cell's verdict, and a pair that "
        "clears its D-7 bar while losing to a constant still reads PASS. Read them as the number that qualifies the PASS — "
        "a challenger that does not beat `constant_oracle` has not demonstrated dynamic skill, and a D-7 bar that "
        "`constant_causal` clears on its own was not a demanding bar.",
    ]


def levels_table(training: list[dict], key: str | None = None) -> list[str]:
    """The per-pair constant levels in MW, so the WAPEs above can be checked.

    Skips rows with no recorded level, so a record written before ABL-389 (or a
    test fixture that predates it) renders without them rather than raising.
    """
    rows = [row for row in training if row.get("constant_reference_mw")]
    if not rows:
        return []
    heading = "| type | country |" if key else "| country |"
    rule = "|---|---|" if key else "|---|"
    lines = ["", "Constant levels used, from the same ABL-188-filtered target series the gate actuals and the "
             "D-7/persistence baselines come from — no refit, no second read, no additional upstream fetch:", "",
             f"{heading} fit-window mean (causal) | gate-window median (oracle) |",
             f"{rule}---:|---:|"]
    for row in rows:
        levels = row["constant_reference_mw"]
        prefix = f"| {row[key]} | {row['country']} |" if key else f"| {row['country']} |"
        causal, oracle = levels.get("constant_causal"), levels.get("constant_oracle")
        lines.append(f"{prefix} {'Not measured' if causal is None else f'{causal:.2f} MW'} | "
                     f"{'Not measured' if oracle is None else f'{oracle:.2f} MW'} |")
    return lines


def lost_to_the_oracle_constant(rows: list[dict], label) -> list[str]:
    """Name every cell whose challenger is beaten by the hindsight constant.

    Stated in prose for the same reason the harness already states "the TSO
    forecast is better than the challenger": a reader who sees PASS in the gate
    column will not otherwise compare two numbers in a fourteen-column table,
    and ABL-380 is the proof — 6/6 PASS was reported, correctly, on a pair whose
    fitted model was worse than a flat line. This changes no verdict.
    """
    beaten = []
    for row in rows:
        challenger = row["scores"]["challenger"]["wape_pct"]
        oracle = comparator_wape(row["scores"], "constant_oracle")
        if challenger is not None and oracle is not None and oracle < challenger:
            beaten.append((label(row), challenger, oracle))
    if not beaten:
        return []
    lines = ["- **The challenger loses to a constant chosen with hindsight in "
             f"{len(beaten)} cell(s).** A flat line at the gate-window median scores better than the fitted model there, so "
             "whatever those cells earn over the D-7 bar they earn by predicting close to the level and varying little around "
             "it — and not even at the best level. This does not change any verdict above; it bounds what the verdict means:"]
    lines.extend(f"  - {name}: challenger {challenger:.2f}% vs oracle constant {oracle:.2f}% "
                 f"({challenger - oracle:+.2f}pp)" for name, challenger, oracle in beaten)
    return lines
