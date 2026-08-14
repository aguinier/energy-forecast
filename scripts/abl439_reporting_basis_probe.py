#!/usr/bin/env python3
"""Decide whether a fit-to-gate level shift is a fleet change or a reporting change.

ABL-439 asks this of NL `wind_onshore`. On tranche 2e's committed record
(`experiments/ABL348/results_abl417_tranche2e.json`) its causal constant -- a flat
line at the ABL-348 fit-window mean -- scores **225.54% WAPE** against an oracle
constant at **73.85%**, the largest such inflation in the ABL-316 table. A
constant at the fit-window mean cannot score 225% unless that constant is roughly
three times the gate-window level, so something moved by 3x inside seven months.

That is a question about the *series*, not about the gate harness, and it has two
very different answers. A fleet change is a fact about the Netherlands and the
gate read is simply measuring a hard window. A **reporting-basis change** is a
fact about the publication, and then the fit window and the gate window are not
measuring the same quantity at all -- which voids every comparison that spans
them. NL is where that question has already been answered the second way once:
ABL-325 found the published NL *solar* series to be a ~2% grid-metered subset
rather than national generation.

This script separates the two, read-only, with no refit and no re-gate.

What it measures
----------------
1. **Monthly levels on both source tables.** `energy_generation` and
   `energy_renewable` are the pre- and post-mapping sides of the same ENTSO-E
   fetch (ABL-331/ABL-345 make the source a property of the artifact, so a pair
   can be read from either). The ABL-396 attribution pattern applies: a shift
   present in *both* tables is upstream of the source mapping, and a shift in
   *one* is not.

2. **Where the two tables stop disagreeing, to the sampling interval.** This is
   the discriminator the monthly table cannot give you. A fleet change moves one
   series and leaves the other alone only if the two tables read different
   fields; a *vintage* change makes one table converge onto the other at an
   instant, and that instant is a property of when we fetched, not of the wind.

3. **An independent reference: the TSO's own day-ahead forecast**
   (`energy_generation_forecast`). It is published by the same TSO for the same
   fleet, it is not derived from the actuals, and it is not touched by whatever
   moved the actuals. The ratio `actual / TSO forecast` is therefore a
   basis-drift meter with the weather divided out -- a low-wind month lowers both
   terms. A fleet change moves that ratio not at all; a reporting change moves it
   by exactly the factor the basis changed by. Run it on a country with no
   suspected defect to see what "no change" looks like (DE is the control below).

4. **Ingest provenance.** `fetched_at` per session says *when* each stored value
   was retrieved. If one backfill session wrote values on both sides of the step,
   our mapper cannot be the author -- the same code produced both -- and the
   discontinuity is in what the upstream returned for those target instants.

5. **Window composition and reconstruction.** The share of the registered fit
   window that sits on each basis, and the model-free references rebuilt from
   the raw series. Reconstructing the published `constant_causal` and
   `constant_oracle` to the digit is the check that this script and the harness
   are reading the same rows; a reconstruction that misses means the diagnosis
   below is about some other series.

6. **A programme-wide sweep** over every (country, type) the two tables share,
   so "is this pair special" is answered by measurement rather than by assuming
   the pair someone happened to look at is the only one.

What it does not do
-------------------
It does not grade, refit, re-gate, or write anything but its own JSON. Whether a
level shift should change a ladder outcome is ABL-437's question, and this script
deliberately reports the inputs to it rather than pre-empting it.

Contamination note: `energy_renewable` can zero-fill a type ENTSO-E did not
return (ABL-188), which would read here as a spurious level difference. The
per-month zero fraction is reported on both tables for exactly that reason -- a
table that is zero-filling says so with a zero fraction, not with a ratio.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src import db

#: Column carrying each forecast type in both source tables.
TYPE_COLUMN = {
    "solar": "solar_mw",
    "wind_onshore": "wind_onshore_mw",
    "wind_offshore": "wind_offshore_mw",
    "biomass": "biomass_mw",
    "hydro_run": "hydro_run_mw",
}

#: Two tables count as carrying the same series over a **day** when their daily
#: means differ by less than this, relatively.
#:
#: The step is dated on daily aggregates rather than on single instants, and that
#: is a correctness point, not a smoothing preference. A per-instant relative test
#: is the wrong scale for a series that spends time near zero: measured on NL
#: `wind_onshore`, the largest absolute post-step disagreement between the two
#: tables over 1,021 hours is **13.40 MW**, but 164 of those hours sit at single-
#: digit MW where 13 MW reads as a 40-140% relative difference. Such a test dates
#: the step five weeks late, on a calm night rather than on the transition. Daily
#: means separate the two states by roughly 20x (pre-step 117%, post-step under
#: 5%) and have no such failure mode.
AGREEMENT_TOLERANCE = 0.15

#: Within the changepoint day the exact interval is found with an **absolute**
#: tolerance, set to this fraction of that day's own mean level. Same reasoning
#: one level down: the crossing instant on NL falls in the evening ramp, where
#: the level is a fraction of the daily mean.
INTRADAY_TOLERANCE_FRACTION = 0.10

#: A pair is worth reporting in the sweep when the two tables disagree about its
#: level by more than this before the step. Set from the measured separation:
#: the ABL-316 pairs that are genuinely the same series sit at 0.99-1.01, and
#: the ones that are not start at 1.15.
SWEEP_MATERIAL_RATIO = 0.15

#: In the sweep, a pair counts as having converged when the two tables' levels
#: land within this of each other afterwards.
SWEEP_CONVERGED_RATIO = 0.05

#: Fleets smaller than this are excluded from the sweep: a 1 MW against a 2 MW is
#: a 100% disagreement about nothing, and no ABL-316 pair is this small.
SWEEP_MIN_LEVEL_MW = 5.0


def _norm(stamp: str) -> datetime:
    """Parse either stored timestamp format without touching the index.

    The replica carries both `YYYY-MM-DD HH:MM:SS` and ISO-8601-with-offset in
    the same column (measured on NL `energy_renewable`: 26,912 of the first form
    and 668 of the second). Normalising inside a JOIN or WHERE would full-scan
    9.4 GB, so every query below bounds on date-only string literals -- which are
    a common prefix of both forms and use the index -- and normalises here.
    """
    return datetime.strptime(stamp.replace("T", " ")[:19], "%Y-%m-%d %H:%M:%S")


def _hourly(conn, table: str, column: str, country: str, start: str, end: str,
            time_column: str = "timestamp_utc") -> dict:
    """Hourly mean of the raw rows, which is the frame everything downstream uses.

    Not a `:00` sub-sample. Neither table is hourly and most countries are both
    (ABL-332: 22 of 24 carry sub-hourly rows in `energy_renewable`), so an
    average over raw rows is cadence-weighted and changes when the resolution
    does -- which is a second discontinuity that would be confused for this one.
    """
    buckets = defaultdict(list)
    sql = (f"SELECT {time_column}, {column} FROM {table} "
           f"WHERE country_code = ? AND {time_column} >= ? AND {time_column} < ?")
    for stamp, value in conn.execute(sql, (country, start, end)):
        if value is None:
            continue
        buckets[_norm(stamp).replace(minute=0, second=0)].append(value)
    return {hour: sum(vs) / len(vs) for hour, vs in buckets.items()}


def _wape_pct(prediction: float, actuals) -> float:
    denominator = sum(abs(v) for v in actuals)
    if not denominator:
        return float("nan")
    return 100.0 * sum(abs(prediction - v) for v in actuals) / denominator


def _monthly(series: dict) -> list:
    months = defaultdict(list)
    for hour, value in series.items():
        months[hour.strftime("%Y-%m")].append(value)
    out = []
    for month in sorted(months):
        values = sorted(months[month])
        n = len(values)
        out.append({
            "month": month,
            "n_hours": n,
            "mean_mw": round(sum(values) / n, 2),
            "median_mw": round(values[n // 2], 2),
            "p95_mw": round(values[int(0.95 * n)], 2),
            "max_mw": round(values[-1], 2),
            # ABL-188 tell: a zero-filled type reads as a measured zero.
            "zero_fraction": round(sum(1 for v in values if v == 0.0) / n, 4),
        })
    return out


def _raw(conn, table: str, column: str, country: str, start: str, end: str) -> dict:
    """Raw stored rows, at whatever cadence the source published them.

    Used only to refine the crossing inside the changepoint day. Everything that
    is *averaged* goes through `_hourly` instead, per the ABL-332 contract.
    """
    sql = (f"SELECT timestamp_utc, {column} FROM {table} "
           f"WHERE country_code = ? AND timestamp_utc >= ? AND timestamp_utc < ?")
    return {_norm(stamp): value
            for stamp, value in conn.execute(sql, (country, start, end))
            if value is not None}


def _locate_step(conn, column: str, country: str, start: str, end: str) -> dict:
    """Date the convergence of the two tables, and say whether it is a step.

    Two stages. First a changepoint over **daily** relative differences, chosen to
    maximise (days disagreeing before) + (days agreeing after) -- a proper
    changepoint on a binary series, so an isolated flip on either side cannot move
    it. Then, inside the changepoint day, the exact crossing at the source's own
    cadence, using an absolute tolerance set from that day's level.

    Step vs ramp is then decided by the width of the refined transition rather
    than by eye. One sampling interval is a step.
    """
    generation = _hourly(conn, "energy_generation", column, country, start, end)
    renewable = _hourly(conn, "energy_renewable", column, country, start, end)
    common = sorted(set(generation) & set(renewable))
    if not common:
        return {"dated": False, "reason": "no co-observed hours"}

    daily = defaultdict(lambda: [0.0, 0.0, 0])
    for hour in common:
        bucket = daily[hour.date()]
        bucket[0] += generation[hour]
        bucket[1] += renewable[hour]
        bucket[2] += 1
    days = sorted(daily)
    disagrees = []
    for day in days:
        gen, ren, n = daily[day]
        scale = max(abs(gen), abs(ren))
        disagrees.append(bool(scale) and abs(gen - ren) / scale > AGREEMENT_TOLERANCE)
    if not any(disagrees):
        return {"dated": False, "reason": "tables agree throughout"}
    if all(disagrees):
        return {"dated": False, "reason": "tables disagree throughout"}

    # Changepoint: index i splitting into [0, i) disagreeing and [i, n) agreeing.
    before_disagree = 0
    after_agree = sum(1 for d in disagrees if not d)
    best, best_i = -1, 0
    for i in range(len(days) + 1):
        if i > 0:
            before_disagree += 1 if disagrees[i - 1] else 0
            after_agree -= 0 if disagrees[i - 1] else 1
        score = before_disagree + after_agree
        if score > best:
            best, best_i = score, i
    changepoint_day = days[best_i] if best_i < len(days) else days[-1]

    # Refine inside the changepoint day at the source's own cadence.
    # The changepoint day is the first *fully* agreeing day, so the crossing
    # itself can sit in the tail of the day before -- on NL it is at 17:00 on the
    # preceding evening. Refine across both days or the answer is always midnight.
    #
    # Date-only bounds on both ends: they are a common prefix of the two stored
    # timestamp formats, so they cannot silently drop the ISO-with-offset rows
    # the way a bound carrying a time-of-day would ('T' sorts above ' ').
    day_start = (changepoint_day - timedelta(days=1)).isoformat()
    day_end = (changepoint_day + timedelta(days=1)).isoformat()
    gen_raw = _raw(conn, "energy_generation", column, country, day_start, day_end)
    ren_raw = _raw(conn, "energy_renewable", column, country, day_start, day_end)
    stamps = sorted(set(gen_raw) & set(ren_raw))
    level = (sum(abs(ren_raw[s]) for s in stamps) / len(stamps)) if stamps else 0.0
    tolerance_mw = max(INTRADAY_TOLERANCE_FRACTION * level, 1.0)
    crossing_from = crossing_to = None
    for index, stamp in enumerate(stamps):
        if abs(gen_raw[stamp] - ren_raw[stamp]) <= tolerance_mw and all(
                abs(gen_raw[s] - ren_raw[s]) <= tolerance_mw for s in stamps[index:]):
            crossing_to = stamp
            crossing_from = stamps[index - 1] if index else None
            break

    before = [h for h in common if h.date() < changepoint_day]
    after = [h for h in common if h.date() > changepoint_day]

    def _ratio(hours):
        gen = sum(generation[h] for h in hours)
        ren = sum(renewable[h] for h in hours)
        return round(gen / ren, 4) if ren else None

    out = {
        "dated": True,
        "changepoint_day": changepoint_day.isoformat(),
        "changepoint_day_purity": round(best / len(days), 4),
        "intraday_tolerance_mw": round(tolerance_mw, 2),
        "n_days_before": len(days[:best_i]),
        "n_days_after": len(days[best_i:]),
        "n_hours_before": len(before),
        "n_hours_after": len(after),
        "ratio_before": _ratio(before),
        "ratio_after": _ratio(after),
        "max_abs_difference_after_mw": round(
            max((abs(generation[h] - renewable[h]) for h in after), default=0.0), 2),
    }
    if crossing_to is not None:
        out["last_disagreeing_instant_utc"] = (
            crossing_from.isoformat(sep=" ") if crossing_from else None)
        out["first_agreeing_instant_utc"] = crossing_to.isoformat(sep=" ")
        width = ((crossing_to - crossing_from).total_seconds() / 3600.0
                 if crossing_from else 0.0)
        out["transition_width_hours"] = round(width, 4)
        # One sampling interval wide is as sharp as the source can resolve.
        out["shape"] = "step" if width <= 1.0 else "ramp"
        if crossing_from is not None:
            out["last_disagreeing_values_mw"] = [round(gen_raw[crossing_from], 2),
                                                 round(ren_raw[crossing_from], 2)]
        out["first_agreeing_values_mw"] = [round(gen_raw[crossing_to], 2),
                                           round(ren_raw[crossing_to], 2)]
    return out


def _tso_reference(conn, column: str, country: str, start: str, end: str) -> list:
    """Monthly `actual / TSO day-ahead forecast` on each source table.

    The TSO forecast is the independent series: same TSO, same fleet, published
    day-ahead, and not derived from the actuals. Dividing by it removes the
    weather, so a month that was simply calm leaves the ratio flat.
    """
    forecast = _hourly(conn, "energy_generation_forecast", column, country, start, end,
                       time_column="target_timestamp_utc")
    generation = _hourly(conn, "energy_generation", column, country, start, end)
    renewable = _hourly(conn, "energy_renewable", column, country, start, end)
    months = defaultdict(lambda: {"n": 0, "fc": 0.0, "gen": 0.0, "ren": 0.0})
    for hour in sorted(set(forecast) & set(generation) & set(renewable)):
        bucket = months[hour.strftime("%Y-%m")]
        bucket["n"] += 1
        bucket["fc"] += forecast[hour]
        bucket["gen"] += generation[hour]
        bucket["ren"] += renewable[hour]
    out = []
    for month in sorted(months):
        bucket = months[month]
        n, fc = bucket["n"], bucket["fc"]
        out.append({
            "month": month,
            "n_hours": n,
            "tso_forecast_mean_mw": round(fc / n, 2),
            "energy_generation_mean_mw": round(bucket["gen"] / n, 2),
            "energy_renewable_mean_mw": round(bucket["ren"] / n, 2),
            "generation_over_tso": round(bucket["gen"] / fc, 4) if fc else None,
            "renewable_over_tso": round(bucket["ren"] / fc, 4) if fc else None,
        })
    return out


def _provenance(conn, table: str, country: str, min_rows: int) -> list:
    """Fetch sessions, largest first. Names the code version that wrote a value.

    A single session spanning both sides of the step is the finding: one mapper
    run produced both bases, so the mapper is not what changed.
    """
    sql = (f"SELECT substr(fetched_at, 1, 10) AS day, COUNT(*), "
           f"MIN(timestamp_utc), MAX(timestamp_utc) FROM {table} "
           f"WHERE country_code = ? GROUP BY 1 HAVING COUNT(*) >= ? ORDER BY 2 DESC")
    return [{"fetch_day": row[0], "rows": row[1],
             "covers_from": row[2], "covers_to": row[3]}
            for row in conn.execute(sql, (country, min_rows))]


def _composition(conn, column: str, country: str, source: str, step_hour,
                 fit_start: str, gate_start: str, gate_end: str) -> dict:
    """Fit/gate split by basis, and the model-free references rebuilt from raw rows.

    `constant_causal` is a flat line at the fit-window mean and `constant_oracle`
    at the gate-window median (ABL-389). Both are rebuilt here so the numbers can
    be checked against the committed record rather than quoted from it.
    """
    fit = _hourly(conn, source, column, country, fit_start, gate_start)
    gate = _hourly(conn, source, column, country, gate_start, gate_end)
    if not fit or not gate:
        return {"source": source, "note": "no rows in one of the windows"}
    gate_values = sorted(gate.values())
    fit_mean = sum(fit.values()) / len(fit)
    gate_mean = sum(gate_values) / len(gate_values)
    gate_median = gate_values[len(gate_values) // 2]

    entry = {
        "source": source,
        "fit_hours": len(fit),
        "fit_mean_mw": round(fit_mean, 2),
        "gate_hours": len(gate),
        "gate_mean_mw": round(gate_mean, 2),
        "gate_median_mw": round(gate_median, 2),
        "fit_over_gate_mean": round(fit_mean / gate_mean, 4) if gate_mean else None,
        "constant_causal_wape_pct": round(_wape_pct(fit_mean, gate.values()), 2),
        "constant_oracle_wape_pct": round(_wape_pct(gate_median, gate.values()), 2),
    }
    if step_hour is not None:
        pre = [v for h, v in fit.items() if h < step_hour]
        post = [v for h, v in fit.items() if h >= step_hour]
        entry["fit_pre_step_hours"] = len(pre)
        entry["fit_post_step_hours"] = len(post)
        entry["fit_pre_step_pct"] = round(100.0 * len(pre) / len(fit), 2)
        entry["fit_pre_step_mean_mw"] = round(sum(pre) / len(pre), 2) if pre else None
        entry["fit_post_step_mean_mw"] = round(sum(post) / len(post), 2) if post else None
        entry["gate_pre_step_hours"] = sum(1 for h in gate if h < step_hour)
        if post:
            # What the causal constant would have been had the fit window seen
            # only the basis the gate window is scored on. Not a proposal to
            # re-register the window -- a sizing of how much of the 225% is the
            # basis rather than the model.
            post_mean = sum(post) / len(post)
            entry["constant_causal_wape_pct_post_step_only"] = round(
                _wape_pct(post_mean, gate.values()), 2)
    return entry


def _long_run_tso(conn, column: str, country: str, first_year: int,
                  last_year: int) -> list:
    """Yearly `energy_generation / TSO forecast`, back before `energy_renewable` exists.

    Intersects only the two series it names. `_tso_reference` intersects all
    three, so it cannot reach back past the second table's first row (NL
    `energy_renewable` starts 2025-11-09) -- and "is the pre-step level the
    long-run norm or itself recent" is a question about exactly that period.
    """
    out = []
    for year in range(first_year, last_year + 1):
        start, end = f"{year}-01-01", f"{year + 1}-01-01"
        forecast = _hourly(conn, "energy_generation_forecast", column, country,
                           start, end, time_column="target_timestamp_utc")
        generation = _hourly(conn, "energy_generation", column, country, start, end)
        common = sorted(set(forecast) & set(generation))
        if not common:
            continue
        fc = sum(forecast[h] for h in common)
        gen = sum(generation[h] for h in common)
        out.append({
            "year": year,
            "n_hours": len(common),
            "tso_forecast_mean_mw": round(fc / len(common), 2),
            "energy_generation_mean_mw": round(gen / len(common), 2),
            "generation_over_tso": round(gen / fc, 4) if fc else None,
        })
    return out


def _programme_context(conn, results_glob: str, band: str) -> list:
    """Rank every committed ABL-348 pair by its fit-to-gate level mismatch, and
    say for each whether its two source tables agreed before the step.

    This is what turns "NL is the only one" into a measurement. `constant_causal`
    bias is large whenever the fit window sat at a different level from the gate
    window, and most of the time that is simply weather -- a windy winter fit
    window against a calm summer gate window. The discriminator is the second
    column: a pair whose two tables already agreed cannot have a basis problem,
    however large its level move.
    """
    rows = []
    for path in sorted(Path(__file__).parent.parent.glob(results_glob)):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        meta = record.get("meta") or {}
        # The two harnesses record their scope differently and only one of them
        # puts a type on the cell. The wind harness registers `(type, country)`
        # pairs; the solar harness registers bare countries, because its whole
        # scope is one type. Resolve in that order rather than assuming either.
        pairs = {country: ftype for ftype, country in (meta.get("registered_pairs") or [])}
        for cell in record.get("gate_cells", []):
            if cell.get("horizon_band") != band:
                continue
            constant = (cell.get("scores") or {}).get("constant_causal") or {}
            if constant.get("bias_pct") is None:
                continue
            country = cell.get("country")
            ftype = (cell.get("forecast_type") or pairs.get(country)
                     or ("solar" if meta.get("registered_countries") else None))
            column = TYPE_COLUMN.get(ftype)
            ratio = None
            if column:
                gen = _hourly(conn, "energy_generation", column, country,
                              "2026-05-01", "2026-07-01")
                ren = _hourly(conn, "energy_renewable", column, country,
                              "2026-05-01", "2026-07-01")
                common = sorted(set(gen) & set(ren))
                if common:
                    mr = sum(ren[h] for h in common) / len(common)
                    if mr > SWEEP_MIN_LEVEL_MW:
                        ratio = round((sum(gen[h] for h in common) / len(common)) / mr, 4)
            rows.append({
                "scope": meta.get("scope") or path.stem.replace("results_", ""),
                "country": country,
                "forecast_type": ftype,
                "training_source": (record.get("meta") or {}).get("training_source"),
                "constant_causal_wape_pct": round(constant["wape_pct"], 2),
                "constant_causal_bias_pct": round(constant["bias_pct"], 2),
                "grade": (cell.get("grade") or {}).get("label"),
                "source_table_ratio_pre_step": ratio,
                # The discriminator: a level move with the two tables in
                # agreement is weather, not a change of basis.
                "level_move_explained_by_basis": bool(
                    ratio is not None and abs(ratio - 1.0) > SWEEP_MATERIAL_RATIO),
            })
    rows.sort(key=lambda r: -abs(r["constant_causal_bias_pct"]))
    return rows


def _sweep(conn, countries, types, pre_start: str, pre_end: str,
           post_start: str, post_end: str) -> list:
    """Every (country, type): do the two tables disagree before and agree after?

    This is what makes "NL is the only one" a measurement. Pairs whose two tables
    already agreed before the step are unaffected by a convergence, however large
    their fit-to-gate level move is -- that move is weather.
    """
    hits = []
    for country in countries:
        for ftype in types:
            column = TYPE_COLUMN.get(ftype)
            if not column:
                continue
            windows = []
            for start, end in ((pre_start, pre_end), (post_start, post_end)):
                gen = _hourly(conn, "energy_generation", column, country, start, end)
                ren = _hourly(conn, "energy_renewable", column, country, start, end)
                common = sorted(set(gen) & set(ren))
                if not common:
                    windows.append(None)
                    continue
                mg = sum(gen[h] for h in common) / len(common)
                mr = sum(ren[h] for h in common) / len(common)
                windows.append((mg, mr))
            if windows[0] is None or windows[1] is None:
                continue
            (pre_gen, pre_ren), (post_gen, post_ren) = windows
            if pre_ren <= SWEEP_MIN_LEVEL_MW or post_ren <= SWEEP_MIN_LEVEL_MW:
                continue
            pre_ratio, post_ratio = pre_gen / pre_ren, post_gen / post_ren
            if abs(pre_ratio - 1.0) <= SWEEP_MATERIAL_RATIO:
                continue
            hits.append({
                "country": country,
                "forecast_type": ftype,
                "pre_generation_mean_mw": round(pre_gen, 2),
                "pre_renewable_mean_mw": round(pre_ren, 2),
                "pre_ratio": round(pre_ratio, 4),
                "post_generation_mean_mw": round(post_gen, 2),
                "post_renewable_mean_mw": round(post_ren, 2),
                "post_ratio": round(post_ratio, 4),
                "converged": abs(post_ratio - 1.0) <= SWEEP_CONVERGED_RATIO,
            })
    return sorted(hits, key=lambda h: -abs(h["pre_ratio"] - 1.0))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--country", default="NL",
                        help="country to diagnose (default: NL)")
    parser.add_argument("--forecast-type", default="wind_onshore",
                        choices=sorted(TYPE_COLUMN),
                        help="type to diagnose (default: wind_onshore)")
    parser.add_argument("--control-country", default="DE",
                        help="country with no suspected defect, for the TSO ratio "
                             "(default: DE)")
    parser.add_argument("--replica-db", required=True,
                        help="path to the read-only replica; opened mode=ro")
    parser.add_argument("--fit-start", default="2026-01-14",
                        help="ABL-348 registered fit-window start")
    parser.add_argument("--gate-start", default="2026-07-11",
                        help="ABL-348 registered gate-window start")
    parser.add_argument("--gate-end", default="2026-08-10",
                        help="ABL-348 registered gate-window end, exclusive")
    parser.add_argument("--history-start", default="2024-01-01",
                        help="start of the monthly context table")
    parser.add_argument("--sweep", action="store_true",
                        help="also sweep every country and type for the same signature")
    parser.add_argument("--programme-results",
                        default="experiments/ABL348/results_*.json",
                        help="glob of committed tranche records to rank for context; "
                             "pass an empty string to skip")
    parser.add_argument("--programme-band", default="24-36h",
                        help="horizon band to take the ranking from")
    parser.add_argument("--long-run-from", type=int, default=2021,
                        help="first year of the yearly actual-over-TSO table")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    column = TYPE_COLUMN[args.forecast_type]
    country = args.country.upper()
    result = {
        "issue": "ABL-439",
        "country": country,
        "forecast_type": args.forecast_type,
        "column": column,
        "replica_db": str(args.replica_db),
        "windows": {"fit_start": args.fit_start, "gate_start": args.gate_start,
                    "gate_end": args.gate_end},
    }

    with db.get_connection(readonly=True, db_path=str(args.replica_db)) as conn:
        result["monthly_levels"] = {
            source: _monthly(_hourly(conn, source, column, country,
                                     args.history_start, args.gate_end))
            for source in ("energy_generation", "energy_renewable")
        }
        step = _locate_step(conn, column, country, args.history_start, args.gate_end)
        result["step"] = step
        step_hour = (_norm(step["first_agreeing_instant_utc"])
                     if step.get("first_agreeing_instant_utc") else None)

        result["tso_reference"] = {
            country: _tso_reference(conn, column, country,
                                    args.history_start, args.gate_end),
            args.control_country.upper(): _tso_reference(
                conn, column, args.control_country.upper(),
                args.history_start, args.gate_end),
        }
        result["provenance"] = {
            source: _provenance(conn, source, country, min_rows=500)
            for source in ("energy_generation", "energy_renewable")
        }
        if step_hour is not None:
            backfill = result["provenance"]["energy_generation"]
            if backfill:
                lag = (_norm(backfill[0]["fetch_day"] + " 00:00:00") - step_hour)
                result["step"]["days_from_step_to_largest_backfill"] = round(
                    lag.total_seconds() / 86400.0, 2)
        result["window_composition"] = [
            _composition(conn, column, country, source, step_hour,
                         args.fit_start, args.gate_start, args.gate_end)
            for source in ("energy_generation", "energy_renewable")
        ]
        result["long_run_tso_reference"] = _long_run_tso(
            conn, column, country, args.long_run_from,
            int(args.gate_end[:4]))
        if args.sweep:
            countries = [row[0] for row in conn.execute(
                "SELECT DISTINCT country_code FROM energy_renewable ORDER BY 1")]
            result["sweep"] = _sweep(conn, countries, sorted(TYPE_COLUMN),
                                     "2026-05-01", args.gate_start,
                                     args.gate_start, args.gate_end)
        if args.programme_results.strip():
            result["programme_context"] = _programme_context(
                conn, args.programme_results.strip(), args.programme_band)

    text = json.dumps(result, indent=2, allow_nan=False)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
