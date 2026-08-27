#!/usr/bin/env python3
"""Run the ABL-439 source-table ratio screen on the pairs it could not reach.

ABL-439 screened 37 of the ABL-316 ledger's 41 pair-records for the revision
vintage that `energy_generation` carries: a stored row's value depends on how old
its target instant was when we fetched it, so a pair fitted before the vintage
boundary and scored after it is fitted and scored on two different publications
of the same series. Four records were left unscreened, and ledger section 5.6
records LV `solar`'s ratio of **1.1143** as a fifth, unexplained outlier.

This script screens all five, read-only, with no refit and no re-grade.

Why the four were missed -- two different causes, neither of them the data
--------------------------------------------------------------------------
Both are properties of ABL-439's `_programme_context`, not of the pairs:

1. **DE / NL `wind_offshore`** -- its record is `experiments/ABL322/
   results_abl436_offshore_reread.json`, and the screen's default glob is
   `experiments/ABL348/results_*.json`. A directory the glob does not name.
   Ledger section 5.6 attributes this to the record being gitignored (ABL-440);
   that is not what happened to *this* file -- it is tracked, committed by
   ABL-436 in `c2126b8`, and matched by no `.gitignore` rule. ABL-440 is about
   the path the `abl322-pilot` scope is *registered to write*
   (`experiments/ABL322/results.json`, which `experiments/*/results.json` does
   swallow); ABL-436 sidestepped it by writing a differently-named file. The
   screen was reachable all along.

2. **EE / FI `solar`** -- the screen takes one horizon band (`--programme-band`,
   default `24-36h`) and these two pairs have no cell in it. They are the ABL-434
   coverage cases: of three bands only `48-64h` gates at all, so a band-filtered
   sweep drops them silently rather than reporting them as unscreened.

Both are the ABL-431 lesson again: a sweep's blind spots live in its scope
declaration, so this script names its pairs explicitly instead of discovering
them from a glob.

Which window the ratio is taken over, and why the script reports three
---------------------------------------------------------------------
This matters more than it looks, and the two authorities disagree.

ABL-439 computed its 37 ratios over **2026-05-01 -> 2026-07-01**, a window chosen
to sit wholly before the vintage boundary; the field it stored is honestly named
`source_table_ratio_pre_step`. Ledger section 5.6 tabulates those same numbers
under the heading "ratio over fit window", and ABL-471 asks for the ratio over
ABL-348's registered fit window, **2026-01-14 -> 2026-07-11**. Those are not the
same measurement, and on the pair that decides a shipment they do not agree: NL
`wind_offshore` reads 0.9648 on the first and 0.9922 on the second.

So all three windows are reported per pair:

* `abl439_comparator` -- the window the other 37 ratios were taken over. Use this
  one, and only this one, to compare a pair against them.
* `abl348_fit_window` -- what ABL-471 asked for and what section 5.6's heading
  claims. Spans the vintage boundary, so on an affected pair it is a blend of
  both bases rather than a measurement of either.
* `abl348_gate_window` -- 2026-07-11 -> 2026-08-10, which lies wholly *after* the
  boundary.

and with them the quantity that is actually the harm:

    fit_gate_discontinuity = ratio(fit window) - ratio(gate window)

A ratio far from 1.0 is not itself a defect -- two tables can carry the same
series at a steady offset forever and every comparison still holds. What voids a
gate read is the offset **changing between the window a model was fitted on and
the window it is scored on**. That is what ABL-439 found on NL `wind_onshore` and
it is what this column measures directly.

The discriminator, when a pair does disagree
--------------------------------------------
Three signatures, and they are distinguishable:

* **Revision vintage** -- the two tables disagree before an instant and are then
  *bit-identical*, and only one of them tracks the TSO's own day-ahead forecast
  at a steady ratio. The TSO series is published by the same TSO for the same
  fleet and is not derived from the actuals, so dividing by it removes the
  weather: a real fleet or output change leaves `actual / TSO` flat, a change of
  publication moves it by the factor the basis moved.
* **Resolution** -- the two tables carry different numbers of rows per hour, so
  an average over raw rows is cadence-weighted (ABL-332). Reported per pair as
  `rows_per_hour`; note both sides are averaged to hours first, so a cadence
  difference alone cannot move these ratios.
* **Genuine content difference** -- a persistent, bidirectional, never-converging
  disagreement, with both tables moving together against the TSO.

Contamination note: `energy_renewable` zero-fills a type ENTSO-E did not return
(ABL-188), which reads as a spurious level difference at the head of a table's
coverage. Per-month `zero_fraction` and hour counts are reported on both tables
so that a partial first month is visible as coverage rather than as a level.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Reuse ABL-439's readers rather than restating them. This is the point: a screen
# that "extends" ABL-439 has to read rows the way ABL-439 read them, and the only
# way to guarantee that is to call the same functions. `_hourly` in particular
# carries the ABL-332 contract (average raw rows into hours before comparing).
import abl439_reporting_basis_probe as abl439  # noqa: E402
from src import db  # noqa: E402

#: The four records ABL-439's sweep could not reach, plus the outlier it found
#: and could not explain. Named explicitly -- see the module docstring on why a
#: glob is what created these gaps in the first place.
SCREEN_PAIRS = [
    ("DE", "wind_offshore", "unscreened: record outside the screen's glob directory"),
    ("NL", "wind_offshore", "unscreened: record outside the screen's glob directory"),
    ("EE", "solar", "unscreened: no cell in the screen's 24-36h band (ABL-434 coverage)"),
    ("FI", "solar", "unscreened: no cell in the screen's 24-36h band (ABL-434 coverage)"),
    ("LV", "solar", "screened at 1.1143 and recorded unexplained (ledger 5.6)"),
    # Not unscreened -- the reference. ABL-439 diagnosed this pair, so carrying it
    # through the same three windows is what makes "LV is the same defect, 17.1%
    # where NL was X%" a comparison rather than an assertion.
    ("NL", "wind_onshore", "reference: the pair ABL-439 diagnosed (ratio 2.4647)"),
]

#: `(name, start, end_exclusive)`. See the module docstring: these are three
#: different measurements and only the first is comparable to ABL-439's 37.
WINDOWS = [
    ("abl439_comparator", "2026-05-01", "2026-07-01"),
    ("abl348_fit_window", "2026-01-14", "2026-07-11"),
    ("abl348_gate_window", "2026-07-11", "2026-08-10"),
]

#: The range ledger section 5.6 reports the unaffected pairs as occupying. It is
#: a **description of where the clean pairs landed**, not the screen's rule --
#: ABL-439 decides `level_move_explained_by_basis` on
#: `abs(ratio - 1) > SWEEP_MATERIAL_RATIO`, i.e. +/-15%. Both verdicts are
#: reported per pair because they can disagree, and conflating them would let a
#: descriptive range silently become a promotion criterion.
DESCRIPTIVE_BAND = (0.99, 1.07)

#: An hour counts as agreeing when the two tables are within this of each other,
#: relatively. Only applied where the level clears `AGREEMENT_MIN_LEVEL_MW`: at
#: 0.1 MW against 0.2 MW a 1% test is measuring float noise on a dark solar hour.
AGREEMENT_TOLERANCE = 0.01
AGREEMENT_MIN_LEVEL_MW = 1.0

#: Mechanism thresholds, on the fit-to-gate ratio move.
#:
#: `NO_DISCONTINUITY` is set from the measured clean pairs, which come in at
#: 0.0000-0.0017; `VINTAGE_DISCONTINUITY` from the affected ones, the smallest of
#: which is 0.1706. The gap between 0.002 and 0.17 is two orders of magnitude
#: wide and empty, so nothing here is a close call and no pair is decided by
#: where in that gap the cut sits.
NO_DISCONTINUITY = 0.02
VINTAGE_DISCONTINUITY = 0.05
#: How near 1.0 the *gate*-window ratio must sit to count as converged.
VINTAGE_CONVERGED = 0.02

#: Ratios quoted in ledger section 5.6 and in ABL-439's committed record, pinned
#: here so a run that reads a different replica, a different window or a
#: different table says so instead of quietly reporting new numbers. The screen's
#: own reading is only worth as much as its agreement with the record it extends.
REPRODUCTION_PINS = [
    ("NL", "wind_onshore", 2.4647),
    ("NL", "solar", 1.6269),
    ("GR", "solar", 0.7945),
    ("CH", "wind_onshore", 1.0747),
    ("LV", "solar", 1.1143),
]


def _ratio_over(conn, column: str, country: str, start: str, end: str) -> dict:
    """Hourly-mean ratio `energy_generation / energy_renewable` over co-observed hours.

    Restricted to the intersection because the two tables' coverage differs at
    the head of `energy_renewable` (ABL-188), and a ratio of two means taken over
    different hour sets measures the coverage, not the level.
    """
    generation = abl439._hourly(conn, "energy_generation", column, country, start, end)
    renewable = abl439._hourly(conn, "energy_renewable", column, country, start, end)
    common = sorted(set(generation) & set(renewable))
    entry = {
        "start": start, "end_exclusive": end,
        "n_hours_generation": len(generation),
        "n_hours_renewable": len(renewable),
        "n_hours_common": len(common),
    }
    if not common:
        entry["ratio"] = None
        entry["note"] = "no co-observed hours"
        return entry
    mean_generation = sum(generation[h] for h in common) / len(common)
    mean_renewable = sum(renewable[h] for h in common) / len(common)
    entry["generation_mean_mw"] = round(mean_generation, 4)
    entry["renewable_mean_mw"] = round(mean_renewable, 4)
    entry["ratio"] = round(mean_generation / mean_renewable, 4) if mean_renewable else None
    agreeing = sum(
        1 for h in common
        if max(abs(generation[h]), abs(renewable[h])) > AGREEMENT_MIN_LEVEL_MW
        and abs(generation[h] - renewable[h]) / max(abs(generation[h]), abs(renewable[h]))
        <= AGREEMENT_TOLERANCE)
    entry["hours_agreeing_within_1pct"] = agreeing
    entry["pct_hours_agreeing_within_1pct"] = round(100.0 * agreeing / len(common), 2)
    return entry


def _vintage_share(conn, column: str, country: str, boundary: str,
                   start: str, end: str) -> dict:
    """Share of a window's co-observed hours that sits before the vintage boundary.

    ABL-439's harm is stated as "fitted ~95% on one vintage and scored 100% on the
    other", so the share is measured rather than inferred from the calendar --
    coverage gaps make the two differ.
    """
    generation = abl439._hourly(conn, "energy_generation", column, country, start, end)
    renewable = abl439._hourly(conn, "energy_renewable", column, country, start, end)
    common = sorted(set(generation) & set(renewable))
    if not common:
        return {"n_hours": 0, "pct_before_boundary": None}
    cut = abl439._norm(boundary)
    before = sum(1 for h in common if h < cut)
    return {"n_hours": len(common), "n_hours_before_boundary": before,
            "pct_before_boundary": round(100.0 * before / len(common), 2)}


def _cadence(conn, table: str, column: str, country: str, start: str, end: str) -> float:
    """Stored rows per co-observed hour. The ABL-332 resolution check."""
    counts = defaultdict(int)
    sql = (f"SELECT timestamp_utc, {column} FROM {table} "
           f"WHERE country_code = ? AND timestamp_utc >= ? AND timestamp_utc < ?")
    for stamp, value in conn.execute(sql, (country, start, end)):
        if value is not None:
            counts[abl439._norm(stamp).replace(minute=0, second=0)] += 1
    if not counts:
        return None
    return round(sum(counts.values()) / len(counts), 3)


def _convergence(conn, column: str, country: str, start: str, end: str) -> dict:
    """Last hour the two tables disagree, and how far apart they stay after it.

    This is the vintage tell, and it is deliberately *not* ABL-439's
    `_locate_step`. That function dates a changepoint on a **relative** daily
    tolerance of 15%, which is right for wind and wrong for solar: a constant
    absolute inflation falls under a relative threshold as soon as the seasonal
    level rises, so on LV `solar` `_locate_step` dates the step at 2026-04-12
    (purity 0.81, and it still reports `ratio_after` 1.0749 -- i.e. it knows the
    tables have not actually converged). The two series in fact become
    bit-identical on 2026-06-30. Dating the *last* disagreement has no seasonal
    failure mode: identical is identical at any level.
    """
    generation = abl439._hourly(conn, "energy_generation", column, country, start, end)
    renewable = abl439._hourly(conn, "energy_renewable", column, country, start, end)
    common = sorted(set(generation) & set(renewable))
    if not common:
        return {"dated": False, "reason": "no co-observed hours"}
    last = None
    for hour in common:
        scale = max(abs(generation[hour]), abs(renewable[hour]))
        if scale > AGREEMENT_MIN_LEVEL_MW and (
                abs(generation[hour] - renewable[hour]) / scale > AGREEMENT_TOLERANCE):
            last = hour
    after = [h for h in common if last is None or h > last]
    return {
        "dated": last is not None,
        "window": {"start": start, "end_exclusive": end},
        "last_disagreeing_hour_utc": last.isoformat(sep=" ") if last else None,
        "n_hours_after": len(after),
        "max_abs_difference_after_mw": round(
            max((abs(generation[h] - renewable[h]) for h in after), default=0.0), 4),
        "bit_identical_after": bool(
            after and max((abs(generation[h] - renewable[h]) for h in after),
                          default=0.0) == 0.0),
    }


def _monthly_ratio(conn, column: str, country: str, start: str, end: str) -> list:
    """Per-month level on both tables and their ratio, over co-observed hours.

    A vintage shows as a ratio that is one thing then exactly 1.0000; a content
    difference shows as a ratio that wanders and never settles.
    """
    generation = abl439._hourly(conn, "energy_generation", column, country, start, end)
    renewable = abl439._hourly(conn, "energy_renewable", column, country, start, end)
    months = defaultdict(lambda: [0.0, 0.0, 0, 0, 0])
    for hour in sorted(set(generation) & set(renewable)):
        bucket = months[hour.strftime("%Y-%m")]
        bucket[0] += generation[hour]
        bucket[1] += renewable[hour]
        bucket[2] += 1
        bucket[3] += 1 if generation[hour] == 0.0 else 0
        bucket[4] += 1 if renewable[hour] == 0.0 else 0
    out = []
    for month in sorted(months):
        gen, ren, n, zero_gen, zero_ren = months[month]
        out.append({
            "month": month, "n_hours_common": n,
            "generation_mean_mw": round(gen / n, 2),
            "renewable_mean_mw": round(ren / n, 2),
            "ratio": round(gen / ren, 4) if ren else None,
            "generation_zero_fraction": round(zero_gen / n, 4),
            "renewable_zero_fraction": round(zero_ren / n, 4),
        })
    return out


def _fetch_age(conn, column: str, country: str, boundary: str,
               start: str, end: str) -> dict:
    """Age-at-fetch of the stored rows on each table, split at the boundary.

    This is what turns "the two tables disagree" into "and here is why". ABL-439
    established that `energy_generation` carries two vintages of the same series
    selected by a row's age when we fetched it, with the cut between 28.03 and
    28.05 days. What that implies, and what this measures, is a difference in how
    the two tables are *written*: if one is only ever appended to at a fixed short
    lag while the other is periodically backfilled, then the backfilled one holds
    the revised publication for old instants and the first publication for recent
    ones -- and the boundary between them is (last backfill - 28 days), which is a
    fact about our fetch schedule and nothing about the country.
    """
    out = {}
    for table in ("energy_generation", "energy_renewable"):
        sql = (f"SELECT timestamp_utc, fetched_at FROM {table} "
               f"WHERE country_code = ? AND timestamp_utc >= ? AND timestamp_utc < ? "
               f"AND {column} IS NOT NULL")
        before, after = [], []
        cut = abl439._norm(boundary)
        for stamp, fetched in conn.execute(sql, (country, start, end)):
            if not fetched:
                continue
            target = abl439._norm(stamp)
            age = (abl439._norm(fetched) - target).total_seconds() / 86400.0
            (before if target < cut else after).append(age)

        def _describe(ages):
            if not ages:
                return None
            ages = sorted(ages)
            return {"n_rows": len(ages), "min_days": round(ages[0], 2),
                    "median_days": round(ages[len(ages) // 2], 2),
                    "max_days": round(ages[-1], 2)}

        out[table] = {"before_boundary": _describe(before),
                      "after_boundary": _describe(after)}
    out["boundary_utc"] = boundary
    return out


def _verdict(pair_result: dict) -> dict:
    """Band verdicts and the mechanism call, from the measurements already taken."""
    windows = pair_result["windows"]
    comparator = windows["abl439_comparator"].get("ratio")
    fit = windows["abl348_fit_window"].get("ratio")
    gate = windows["abl348_gate_window"].get("ratio")
    convergence = pair_result["convergence"]

    discontinuity = round(fit - gate, 4) if (fit is not None and gate is not None) else None
    out = {
        "ratio_abl439_comparator_window": comparator,
        "ratio_abl348_fit_window": fit,
        "ratio_abl348_gate_window": gate,
        "fit_gate_discontinuity": discontinuity,
    }
    for label, ratio in (("comparator", comparator), ("fit_window", fit)):
        out[f"in_descriptive_band_{label}"] = (
            None if ratio is None
            else DESCRIPTIVE_BAND[0] <= ratio <= DESCRIPTIVE_BAND[1])
        out[f"basis_affected_by_abl439_rule_{label}"] = (
            None if ratio is None
            else abs(ratio - 1.0) > abl439.SWEEP_MATERIAL_RATIO)

    # The mechanism call: a large fit-to-gate move whose *gate* end sits on
    # agreement is the vintage -- the two tables are the same series after the
    # boundary and were not before it.
    #
    # Convergence is tested as "gate ratio within VINTAGE_CONVERGED of 1.0", not
    # as bit-identical. Bit-identical is the stronger evidence and is reported
    # (`convergence.bit_identical_after`), but it is too strict to *classify* on:
    # NL `wind_onshore` -- the pair ABL-439 diagnosed as a vintage, and the
    # reference row here -- converges to 0.9933, not to 1.0000, and a classifier
    # keyed on bit-identical labels it a persistent disagreement instead.
    if (discontinuity is not None and abs(discontinuity) > VINTAGE_DISCONTINUITY
            and gate is not None and abs(gate - 1.0) <= VINTAGE_CONVERGED):
        mechanism = "revision_vintage"
    elif discontinuity is not None and abs(discontinuity) <= NO_DISCONTINUITY:
        mechanism = "no_fit_gate_discontinuity"
    else:
        mechanism = "persistent_disagreement"
    out["mechanism"] = mechanism
    out["convergence_evidence"] = {
        "gate_ratio_within_converged_band": (
            None if gate is None else abs(gate - 1.0) <= VINTAGE_CONVERGED),
        "bit_identical_after_last_disagreement":
            convergence.get("bit_identical_after"),
        "last_disagreeing_hour_utc": convergence.get("last_disagreeing_hour_utc"),
    }
    return out


def _reproduce_pins(conn, start: str, end: str) -> list:
    """Re-measure ratios the ledger and ABL-439's record already quote.

    Anything that moves here means this run is not reading what ABL-439 read, and
    every other number below should be discarded rather than reconciled.
    """
    out = []
    for country, forecast_type, expected in REPRODUCTION_PINS:
        column = abl439.TYPE_COLUMN[forecast_type]
        measured = _ratio_over(conn, column, country, start, end).get("ratio")
        out.append({
            "country": country, "forecast_type": forecast_type,
            "recorded": expected, "measured": measured,
            "reproduces": measured is not None and abs(measured - expected) < 5e-4,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replica-db", required=True,
                        help="path to the read-only replica; opened mode=ro")
    parser.add_argument("--history-start", default="2025-11-01",
                        help="start of the per-month context table")
    parser.add_argument("--history-end", default="2026-08-22",
                        help="end, exclusive, of the per-month context table")
    parser.add_argument("--vintage-boundary", default="2026-06-30 21:00:00",
                        help="instant to split the age-at-fetch report on; the "
                             "measured convergence of the two Baltic solar pairs, "
                             "which is the 2026-07-29 backfill less ~28.9 days")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    result = {
        "issue": "ABL-471",
        "extends": "ABL-439",
        "replica_db": str(args.replica_db),
        "replica_bytes": Path(args.replica_db).stat().st_size,
        "descriptive_band": {"low": DESCRIPTIVE_BAND[0], "high": DESCRIPTIVE_BAND[1],
                             "note": "ledger 5.6's description of where clean pairs "
                                     "landed; ABL-439's own rule is abs(ratio-1) > "
                                     f"{abl439.SWEEP_MATERIAL_RATIO}"},
        "windows_registered": {name: {"start": s, "end_exclusive": e}
                               for name, s, e in WINDOWS},
        "pairs": [],
    }

    with db.get_connection(readonly=True, db_path=str(args.replica_db)) as conn:
        result["reproduction_pins"] = _reproduce_pins(conn, "2026-05-01", "2026-07-01")

        for country, forecast_type, why in SCREEN_PAIRS:
            column = abl439.TYPE_COLUMN[forecast_type]
            entry = {
                "country": country, "forecast_type": forecast_type, "column": column,
                "why_unscreened": why,
                "windows": {name: _ratio_over(conn, column, country, start, end)
                            for name, start, end in WINDOWS},
                "rows_per_hour": {
                    source: _cadence(conn, source, column, country,
                                     "2026-01-14", "2026-07-11")
                    for source in ("energy_generation", "energy_renewable")},
                # Over the full history, not just the recent window: a pair whose
                # two tables agree from May onward may still have disagreed in
                # the winter, and "when did they last differ" is only meaningful
                # if the scan can see that far back.
                "convergence": _convergence(conn, column, country,
                                            args.history_start, args.history_end),
                "monthly": _monthly_ratio(conn, column, country,
                                          args.history_start, args.history_end),
                "tso_reference": abl439._tso_reference(
                    conn, column, country, args.history_start, args.history_end),
                "fetch_age": _fetch_age(conn, column, country, args.vintage_boundary,
                                        "2026-01-14", "2026-08-10"),
                "vintage_share": {
                    name: _vintage_share(conn, column, country,
                                         args.vintage_boundary, start, end)
                    for name, start, end in WINDOWS
                    if name != "abl439_comparator"},
            }
            entry["verdict"] = _verdict(entry)
            result["pairs"].append(entry)

    text = json.dumps(result, indent=2, allow_nan=False)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
