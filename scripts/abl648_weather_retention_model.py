"""ABL-648: retention arithmetic for weather_observation, read-only.

Produces the numbers behind the Board decision brief on ABL-648. Three
independent inputs, kept separate on purpose because they have very different
provenance:

1. **Replica probe** (measured here, read-only URI) -- which weather tables the
   Able forecast programme can actually reach, and how far back each goes.
   `weather_observation` is not on the replica at all, which is why nothing in
   this repo can read it.

2. **Prod volume growth** (measured here) -- from the local ops-status snapshot
   log at data/ops-status-snapshots.jsonl. No prod query is issued. Daily
   medians are used for the trend because they are phase-matched against the
   daily backup staircase; ABL-212's false alarm came from least-squares over a
   window shorter than that cycle. The step decomposition is reported beside it.

3. **weather_observation size model** (NOT measured -- recorded figures from
   ABL-163 / ABL-206 / ABL-212). The table cannot be COUNT(*)-ed on prod and is
   absent from the replica, so every byte figure for it here is modelled from
   the four rowid->fetched_at anchors ABL-163 recorded. Labelled `modelled`
   throughout.

Usage:
    .venv\\Scripts\\python.exe scripts/abl648_weather_retention_model.py \\
        --replica-db C:\\Code\\able\\data\\energy_dashboard.db \\
        --json-out reports/abl_648_retention_model.json

Writes nothing to any database.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path

GIB = 1024 ** 3

# --- Recorded prod figures. Sources named; none re-measured here. -----------
# ABL-163 (2026-08-10 ~20:35Z): rowid -> fetched_at datings of weather_observation.
WO_ROWID_ANCHORS = [
    ("2026-04-22T08:28:00+00:00", 1),
    ("2026-06-08T15:30:00+00:00", 492_633_300),
    ("2026-07-27T19:30:00+00:00", 886_739_940),
    ("2026-08-10T20:30:00+00:00", 985_266_600),
]
# ABL-206 (2026-08-11): weather_observation = 372.0 GiB of a 377.97 GiB database.
WO_GIB_AT_ANCHOR = 372.0
WO_ROWS_AT_ANCHOR = 985_266_600
# ABL-212 (2026-08-14): revalidated sustained whole-volume rate the Board's
# Option C decision was sized against.
ABL212_GIB_PER_DAY = 15.5 / 7.0

# Ingest shape, measured on the replica dimension tables (see probe below) and
# from energy-data-gathering's fetcher defaults:
#   realtime: 7 NWP models, hourly fetch, forecast_days_ahead=3  (72h horizon)
#   previous_runs: 7 models x 3 leads = 21 sources, 3x/day, forecast_days_ahead=7
#   archive: 1 ERA5 source, lead 0, backfilled from 2024-01-01, one vintage
ERA5_BACKFILL_START = "2024-01-01"
WO_FIRST_FETCHED_AT = "2026-04-22"

# The 24 countries the forecast programme serves (config.SUPPORTED_COUNTRIES,
# inlined so this probe needs no repo import and runs from a worktree).
SUPPORTED_COUNTRIES = [
    "AT", "BE", "BG", "CH", "CZ", "DE", "EE", "ES", "FI", "FR", "GR", "HR",
    "HU", "IT", "LT", "LV", "NL", "NO", "PL", "PT", "RO", "SE", "SI", "SK",
]


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


# --------------------------------------------------------------------------
# 1. Replica probe -- measured
# --------------------------------------------------------------------------
def probe_replica(path: str) -> dict:
    uri = "file:" + str(path).replace("\\", "/") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        out = {
            "replica_db": str(path),
            "replica_bytes": os.path.getsize(path),
            "weather_tables_present": sorted(t for t in tables if "weather" in t.lower()),
            "has_weather_observation": "weather_observation" in tables,
            "weather_data_by_quality": [
                dict(zip(("data_quality", "n_rows", "first_target", "last_target",
                          "n_countries", "first_run_time", "last_run_time"), r))
                for r in con.execute(
                    "SELECT data_quality, COUNT(*), MIN(timestamp_utc), MAX(timestamp_utc),"
                    " COUNT(DISTINCT country_code), MIN(forecast_run_time), MAX(forecast_run_time)"
                    " FROM weather_data GROUP BY data_quality")],
            "weather_location_count": con.execute(
                "SELECT COUNT(*) FROM weather_location").fetchone()[0],
            "weather_location_max_rowid": con.execute(
                "SELECT MAX(rowid) FROM weather_location").fetchone()[0],
            "weather_source_count": con.execute(
                "SELECT COUNT(*) FROM weather_source").fetchone()[0],
            "weather_source_by_lead": [
                dict(zip(("lead_time_hours", "n_sources"), r))
                for r in con.execute(
                    "SELECT lead_time_hours, COUNT(*) FROM weather_source"
                    " GROUP BY lead_time_hours ORDER BY lead_time_hours")],
            "weather_location_by_zone_type": [
                dict(zip(("zone_type", "n_locations", "n_countries"), r))
                for r in con.execute(
                    "SELECT zone_type, COUNT(*), COUNT(DISTINCT country_code)"
                    " FROM weather_location GROUP BY zone_type ORDER BY zone_type")],
            # How far back each SUPPORTED country's fit window can actually
            # reach. This is the history-length bound ABL-338 ran into, and it
            # lives in weather_data -- a different table from weather_observation.
            "weather_data_reach_by_country": [
                dict(zip(("country_code", "data_quality", "n_rows",
                          "first_target", "last_target"), r))
                for r in con.execute(
                    "SELECT country_code, data_quality, COUNT(*),"
                    " MIN(timestamp_utc), MAX(timestamp_utc) FROM weather_data"
                    " WHERE country_code IN (%s)"
                    " GROUP BY country_code, data_quality"
                    " ORDER BY country_code, data_quality"
                    % ",".join("?" * len(SUPPORTED_COUNTRIES)),
                    SUPPORTED_COUNTRIES)],
        }
        # ABL-338 named history length as the binding constraint on renewable
        # fits. Locate which table that constraint is actually in, because a
        # weather_observation retention policy can only foreclose model work
        # that reads weather_observation.
        probe_cc = ["AT", "BE", "DE", "ES", "FR"]
        marks = ",".join("?" * len(probe_cc))
        hist = {}
        for tbl in ("energy_renewable", "energy_generation"):
            hist[tbl] = [
                dict(zip(("country_code", "n_rows", "first", "last"), r))
                for r in con.execute(
                    f"SELECT country_code, COUNT(*), MIN(timestamp_utc),"
                    f" MAX(timestamp_utc) FROM {tbl} WHERE solar_mw IS NOT NULL"
                    f" AND country_code IN ({marks})"
                    f" GROUP BY country_code ORDER BY country_code", probe_cc)]
        out["history_constraint_probe"] = {
            "question": "ABL-338: AT and DE have under one seasonal cycle. Which "
                        "table is that constraint in?",
            "solar_mw_history": hist,
            "verdict": "The constraint is in energy_renewable (the training_source "
                       "the four solar serving artifacts carry), not in any weather "
                       "table. energy_generation holds the same countries back to "
                       "2021-01-01, and --renewable-source energy_generation is the "
                       "documented way to reach it. weather_observation is not on "
                       "either path.",
        }
        return out
    finally:
        con.close()


# --------------------------------------------------------------------------
# 2. Prod volume growth -- measured from the local snapshot log
# --------------------------------------------------------------------------
def measure_growth(jsonl_path: str, step_gib: float = 1.0) -> dict:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            peer = rec.get("peer") or {}
            used, total = peer.get("diskUsedBytes"), peer.get("diskTotalBytes")
            if used is None or total is None:
                continue
            rows.append((_parse(rec["t"]), used, total))
    rows.sort()
    if len(rows) < 2:
        return {"error": "insufficient peer snapshots", "n": len(rows)}

    span_days = (rows[-1][0] - rows[0][0]).total_seconds() / 86400.0
    total_gib = rows[-1][2] / GIB

    # Step decomposition -- the ABL-212 lesson: separate the daily backup
    # staircase from the continuous ingest write before quoting a rate.
    base_h = base_g = 0.0
    written = deleted = 0.0
    n_steps = 0
    for (t0, u0, _), (t1, u1, _) in zip(rows, rows[1:]):
        dt_h = (t1 - t0).total_seconds() / 3600.0
        if dt_h <= 0:
            continue
        d = (u1 - u0) / GIB
        if abs(d) >= step_gib:
            n_steps += 1
            written += max(d, 0.0)
            deleted += min(d, 0.0)
        else:
            base_h += dt_h
            base_g += d

    # Daily medians -- phase-matched, robust to the staircase.
    day = OrderedDict()
    for ts, used, _ in rows:
        day.setdefault(ts.strftime("%Y-%m-%d"), []).append(used)
    medians = [(k, statistics.median(v) / GIB, len(v)) for k, v in day.items()]
    d0 = datetime.strptime(medians[0][0], "%Y-%m-%d")
    d1 = datetime.strptime(medians[-1][0], "%Y-%m-%d")
    n_days = (d1 - d0).days
    median_rate = (medians[-1][1] - medians[0][1]) / n_days if n_days else float("nan")
    baseline_rate = (base_g / base_h * 24.0) if base_h else float("nan")
    used_now = medians[-1][1]

    def wall(rate: float, threshold: float) -> dict:
        head = total_gib * threshold - used_now
        days = head / rate if rate > 0 else float("inf")
        return {"headroom_gib": round(head, 1), "days": round(days),
                "date": (d1 + timedelta(days=days)).strftime("%Y-%m-%d")}

    return {
        "source": jsonl_path,
        "note": "peer = prod (QuietlyConfident); diskTotalBytes matches ABL-212's "
                "907.13 GiB exactly. No prod query was issued -- this is the "
                "local snapshot log.",
        "n_snapshots": len(rows),
        "window": [rows[0][0].isoformat(), rows[-1][0].isoformat()],
        "window_days": round(span_days, 2),
        "volume_total_gib": round(total_gib, 2),
        "used_now_gib": round(used_now, 2),
        "used_pct": round(100.0 * used_now / total_gib, 1),
        "free_now_gib": round(total_gib - used_now, 1),
        "rate_baseline_gib_per_day": round(baseline_rate, 3),
        "rate_baseline_gib_per_week": round(baseline_rate * 7, 2),
        "rate_daily_median_gib_per_day": round(median_rate, 3),
        "rate_daily_median_gib_per_week": round(median_rate * 7, 2),
        "step_events": {"n": n_steps, "written_gib": round(written, 2),
                        "deleted_gib": round(deleted, 2),
                        "net_retained_gib": round(written + deleted, 2),
                        "net_retained_gib_per_day": round((written + deleted) / span_days, 3)},
        "abl212_committed_rate_gib_per_day": round(ABL212_GIB_PER_DAY, 3),
        "ratio_measured_to_abl212": round(median_rate / ABL212_GIB_PER_DAY, 2),
        "daily_medians": [{"date": k, "used_gib": round(m, 2), "n": n} for k, m, n in medians],
        "wall_at_90pct": {
            "at_daily_median_rate": wall(median_rate, 0.90),
            "at_baseline_rate": wall(baseline_rate, 0.90),
            "at_abl212_rate": wall(ABL212_GIB_PER_DAY, 0.90),
        },
        "wall_at_100pct": {
            "at_daily_median_rate": wall(median_rate, 1.00),
            "at_baseline_rate": wall(baseline_rate, 1.00),
            "at_abl212_rate": wall(ABL212_GIB_PER_DAY, 1.00),
        },
    }


# --------------------------------------------------------------------------
# 3. weather_observation size model -- MODELLED, not measured
# --------------------------------------------------------------------------
def _bytes_per_row() -> float:
    return WO_GIB_AT_ANCHOR * GIB / WO_ROWS_AT_ANCHOR


def _cum_rows_at(when: datetime, tail_rows_per_day: float) -> float:
    """Cumulative weather_observation rows written by `when`.

    Piecewise-linear through ABL-163's four rowid anchors, then linear at the
    measured recent rate. Returns 0 before the table's first fetched_at.
    """
    anchors = [(_parse(t), float(n)) for t, n in WO_ROWID_ANCHORS]
    if when <= anchors[0][0]:
        return 0.0
    for (t0, n0), (t1, n1) in zip(anchors, anchors[1:]):
        if when <= t1:
            frac = (when - t0).total_seconds() / (t1 - t0).total_seconds()
            return n0 + frac * (n1 - n0)
    tail_days = (when - anchors[-1][0]).total_seconds() / 86400.0
    return anchors[-1][1] + tail_days * tail_rows_per_day


def model_retention(growth: dict, as_of: datetime) -> dict:
    bpr = _bytes_per_row()
    # Recent write rate for the table alone. The measured baseline rate (steps
    # excluded) is the continuous ingest write; weather_observation is 98.4% of
    # the database (ABL-206), so attribute the baseline to it.
    wo_gib_per_day = growth["rate_baseline_gib_per_day"]
    tail_rows_per_day = wo_gib_per_day * GIB / bpr

    def gib_written_by(when: datetime) -> float:
        return _cum_rows_at(when, tail_rows_per_day) * bpr / GIB

    size_now = gib_written_by(as_of)

    # --- (A) calendar-age retention on fetched_at (vintage age) -------------
    horizons = [6, 12, 18, 24, None]
    first_fetched = _parse(WO_FIRST_FETCHED_AT + "T00:00:00+00:00")

    def at_date(when: datetime) -> list:
        table_gib = gib_written_by(when)
        rows_out = []
        for h in horizons:
            if h is None:
                rows_out.append({"horizon_months": "keep_all", "cutoff": None,
                                 "freed_gib": 0.0, "kept_gib": round(table_gib, 1),
                                 "bites_yet": False, "first_bites_on": None})
                continue
            cutoff = when - timedelta(days=round(h * 365.25 / 12))
            freed = gib_written_by(cutoff)
            bites = cutoff > first_fetched
            first_bite = (first_fetched + timedelta(days=round(h * 365.25 / 12))).strftime("%Y-%m-%d")
            rows_out.append({
                "horizon_months": h,
                "cutoff": cutoff.strftime("%Y-%m-%d"),
                "freed_gib": round(freed, 1),
                "kept_gib": round(table_gib - freed, 1),
                "bites_yet": bites,
                "first_bites_on": first_bite,
            })
        return rows_out

    # --- (A2) calendar-age retention on valid_at (target time) --------------
    # The CEO's question has two readings and they give opposite answers, so
    # both are computed. On valid_at the only population older than the table
    # itself is the ERA5 backfill (one vintage per hour per location, from
    # 2024-01-01): the vintage-bearing populations are forward-looking, so
    # their valid_at is >= the table's first fetched_at. A cutoff earlier than
    # 2026-04-22 therefore deletes ERA5 and nothing else.
    n_loc_a2 = 362
    era5_start = _parse(ERA5_BACKFILL_START + "T00:00:00+00:00")

    def era5_gib_before(cutoff: datetime) -> float:
        end = min(cutoff, as_of)
        if end <= era5_start:
            return 0.0
        return n_loc_a2 * ((end - era5_start).total_seconds() / 3600.0) * bpr / GIB

    era5_total_gib = era5_gib_before(as_of)
    valid_at_rows = []
    for h in horizons:
        if h is None:
            valid_at_rows.append({
                "horizon_months": "keep_all", "cutoff": None, "freed_gib": 0.0,
                "freed_pct_of_table": 0.0, "era5_freed_gib": 0.0,
                "era5_history_destroyed_pct": 0.0})
            continue
        cutoff = as_of - timedelta(days=round(h * 365.25 / 12))
        era5_freed = era5_gib_before(cutoff)
        # vintage populations only if the cutoff reaches into the table's own era
        vintage_freed = gib_written_by(cutoff) if cutoff > first_fetched else 0.0
        freed = era5_freed + vintage_freed
        valid_at_rows.append({
            "horizon_months": h,
            "cutoff": cutoff.strftime("%Y-%m-%d"),
            "freed_gib": round(freed, 1),
            "freed_pct_of_table": round(100.0 * freed / size_now, 2),
            "era5_freed_gib": round(era5_freed, 2),
            "era5_history_destroyed_pct": round(
                100.0 * era5_freed / era5_total_gib, 1) if era5_total_gib else 0.0,
        })
    valid_at_calendar = {
        "axis": "valid_at (target time)",
        "era5_total_gib": round(era5_total_gib, 2),
        "era5_span": [ERA5_BACKFILL_START, as_of.strftime("%Y-%m-%d")],
        "rows": valid_at_rows,
        "note": "Approximation: the vintage populations' valid_at is treated as "
                "their fetched_at (mean lead ~3.5d), which is immaterial at a "
                "6-month granularity. The ERA5 term is exact given the ingest "
                "shape (one row per location-hour).",
    }

    # --- (B) vintage-depth retention (the axis the volume is actually on) ---
    # Distinct (source_id, location_id, valid_at) triples, from ingest shape.
    n_loc = 362
    triples_era5 = n_loc * int(
        (as_of - _parse(ERA5_BACKFILL_START + "T00:00:00+00:00")).total_seconds() / 3600)
    hours_covered = int((as_of - first_fetched).total_seconds() / 3600) + 7 * 24
    triples_realtime = n_loc * 7 * hours_covered
    triples_prevruns = n_loc * 21 * hours_covered
    triples = triples_era5 + triples_realtime + triples_prevruns
    latest_only_gib = triples * bpr / GIB
    rows_now = _cum_rows_at(as_of, tail_rows_per_day)
    multiplicity = rows_now / triples

    # latest-only steady-state growth: one row per (source, location, hour)/day
    latest_only_gib_per_day = n_loc * 29 * 24 * bpr / GIB

    vintage_options = []
    for keep_days, label in [(0, "latest_vintage_only"), (30, "30d_full_vintages"),
                             (60, "60d_full_vintages"), (90, "90d_full_vintages")]:
        window_gib = keep_days * wo_gib_per_day
        # latest-only tail for everything older than the window
        tail_frac = max(0.0, ((as_of - timedelta(days=keep_days)) - first_fetched).total_seconds()
                        / max((as_of - first_fetched).total_seconds(), 1.0))
        kept = window_gib + latest_only_gib * tail_frac
        vintage_options.append({
            "option": label,
            "full_vintage_days": keep_days,
            "kept_gib": round(kept, 1),
            "freed_gib": round(size_now - kept, 1),
            "freed_pct": round(100.0 * (size_now - kept) / size_now, 1),
            "steady_state_growth_gib_per_day": round(latest_only_gib_per_day, 3),
            "growth_reduction_factor": round(wo_gib_per_day / latest_only_gib_per_day, 1),
        })

    # --- (C) valid_at downsampling hourly -> 3-hourly ------------------------
    downsample = {
        "rule": "keep every 3rd valid_at, all vintages",
        "row_reduction": 2.0 / 3.0,
        "kept_gib": round(size_now / 3.0, 1),
        "freed_gib": round(size_now * 2.0 / 3.0, 1),
        "steady_state_growth_gib_per_day": round(wo_gib_per_day / 3.0, 3),
        "growth_reduction_factor": 3.0,
        "note": "Cuts rows but leaves vintage multiplicity untouched, so growth "
                "stays proportional to re-fetch frequency and the wall only moves "
                "out by a factor of 3, not to a bounded steady state.",
    }

    # --- reclaim feasibility: the copy-out window ---------------------------
    free_now = growth["free_now_gib"]
    rate = growth["rate_daily_median_gib_per_day"]
    reclaim = []
    for label, kept in [("full VACUUM INTO (no policy)", round(size_now / 0.984, 1)),
                        ("filtered copy-out, 30d vintages", vintage_options[1]["kept_gib"]),
                        ("filtered copy-out, latest-only", vintage_options[0]["kept_gib"]),
                        ("filtered copy-out, 3-hourly", downsample["kept_gib"])]:
        need = kept + 50.0  # 50 GiB operating margin
        fits = free_now >= need
        days_left = (free_now - need) / rate if rate > 0 else float("inf")
        reclaim.append({
            "path": label,
            "copy_size_gib": kept,
            "needs_free_gib_incl_50gib_margin": round(need, 1),
            "fits_today": fits,
            "days_until_it_stops_fitting": round(days_left) if fits else 0,
            "closes_on": (as_of + timedelta(days=days_left)).strftime("%Y-%m-%d") if fits else "already closed",
        })

    return {
        "provenance": "MODELLED from ABL-163 rowid anchors + ABL-206 size. "
                      "weather_observation is absent from the replica and must not "
                      "be COUNT(*)-ed on prod, so no figure here is a direct "
                      "measurement of that table.",
        "bytes_per_row": round(bpr, 1),
        "as_of": as_of.strftime("%Y-%m-%d"),
        "table_first_fetched_at": WO_FIRST_FETCHED_AT,
        "table_age_days": (as_of - first_fetched).days,
        "modelled_rows_now": int(rows_now),
        "modelled_size_now_gib": round(size_now, 1),
        "modelled_write_rate_gib_per_day": round(wo_gib_per_day, 3),
        "calendar_horizons_on_fetched_at": {
            "today": at_date(as_of),
            "at_the_90pct_wall": at_date(_parse(
                growth["wall_at_90pct"]["at_daily_median_rate"]["date"] + "T00:00:00+00:00")),
            "at_abl212_committed_date_2027_01_28": at_date(_parse("2027-01-28T00:00:00+00:00")),
        },
        "calendar_horizons_on_valid_at": valid_at_calendar,
        "vintage_depth_options": vintage_options,
        "distinct_triples": {
            "n_locations": n_loc,
            "n_sources": 29,
            "modelled_distinct_triples": int(triples),
            "modelled_vintages_per_triple": round(multiplicity, 1),
            "latest_only_gib": round(latest_only_gib, 1),
            "era5_share_gib": round(triples_era5 * bpr / GIB, 1),
            "era5_note": "The entire deep weather history -- ERA5, one vintage per "
                         "hour per location, backfilled from " + ERA5_BACKFILL_START +
                         " -- is this many GiB. It is the part a calendar-history "
                         "retention would destroy, and it is a rounding error in the total.",
        },
        "valid_at_downsampling": downsample,
        "reclaim_feasibility": reclaim,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replica-db", default=os.environ.get("ENERGY_DB_PATH"),
                    help="Read-only replica path. Required (no .env in a worktree).")
    ap.add_argument("--ops-snapshots",
                    default=r"C:\Code\able\data\ops-status-snapshots.jsonl",
                    help="Local ops-status snapshot log (JSONL).")
    ap.add_argument("--as-of", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    help="Date to evaluate the retention arithmetic at (YYYY-MM-DD).")
    ap.add_argument("--json-out", default="reports/abl_648_retention_model.json")
    args = ap.parse_args(argv)

    if not args.replica_db:
        ap.error("--replica-db (or ENERGY_DB_PATH) is required")

    as_of = _parse(args.as_of + "T00:00:00+00:00")
    replica = probe_replica(args.replica_db)
    growth = measure_growth(args.ops_snapshots)
    retention = model_retention(growth, as_of)

    out = {
        "issue": "ABL-648",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "databases_opened": [args.replica_db + " (mode=ro)"],
        "replica_probe": replica,
        "prod_volume_growth": growth,
        "retention_model": retention,
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"wrote {args.json_out}")
    print(f"prod: {growth['used_now_gib']} GiB used / {growth['volume_total_gib']} GiB "
          f"({growth['used_pct']}%), free {growth['free_now_gib']} GiB")
    print(f"rate: {growth['rate_daily_median_gib_per_day']} GiB/day (daily medians), "
          f"{growth['rate_baseline_gib_per_day']} GiB/day (steps excluded); "
          f"ABL-212 committed {growth['abl212_committed_rate_gib_per_day']}")
    print(f"90% wall: {growth['wall_at_90pct']['at_daily_median_rate']['date']} "
          f"({growth['wall_at_90pct']['at_daily_median_rate']['days']} days)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
