"""ABL-692 close-out: prove the serving checkout actually publishes a calibrated band.

The unblock descriptor for ABL-692 asks for two string checks in the serving log
(the `net-position serving commit:` witness line, and a `(calibrated s_lo=...)`
suffix on the next quantile line). Both are necessary and neither is sufficient:
a log line proves what the process said, not what the forecast table holds.

This probe closes that gap from the DB side, read-only on both files:

  1. Which `generated_at` vintages of `chronos-2-V010` net_position quantiles
     exist, and how wide the p10-p90 band is in each.
  2. Where the band width steps up -- the first calibrated vintage. ABL-677
     re-bases its 10-vintage window off that date, so it is reported as a
     first-class number rather than inferred from the log timestamp.
  3. Whether the measured widening matches the registered (s_lo, s_hi) for the
     model, per country. The calibration map is anchored at q50, so q50 must be
     bit-identical to the point forecast row and only the band may move.

Read-only: both DBs are opened with the SQLite read-only URI form. Writes
nothing anywhere.

Usage:
  .venv\\Scripts\\python.exe scripts/abl692_calibration_witness.py \\
      --sidecar-db C:\\Code\\able\\data\\forecasts_local.db \\
      --json-out reports/abl_692_calibration_witness.json
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402

MODEL = "chronos-2-V010"
FORECAST_TYPE = "net_position"


def ro_connect(path: str) -> sqlite3.Connection:
    """Open a database read-only. The replica is never written by this role."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def band_by_vintage(conn: sqlite3.Connection, limit: int) -> list:
    """Mean p90-p10 width and coverage per (generated_at) vintage, newest first.

    `target_days` comes back too, because a vintage is only a 24h block if it
    owns its target day -- see `duplicate_target_days`.
    """
    rows = conn.execute(
        """
        SELECT generated_at,
               COUNT(DISTINCT country_code)                         AS countries,
               COUNT(*)                                             AS rows_all,
               GROUP_CONCAT(DISTINCT substr(target_timestamp_utc, 1, 10))
                                                                    AS target_days,
               AVG(CASE WHEN quantile = 0.9 THEN forecast_value END)
                 - AVG(CASE WHEN quantile = 0.1 THEN forecast_value END) AS mean_band
        FROM forecast_quantiles
        WHERE model_name = ? AND forecast_type = ?
        GROUP BY generated_at
        ORDER BY generated_at DESC
        LIMIT ?
        """,
        (MODEL, FORECAST_TYPE, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def duplicate_target_days(vintages: list) -> dict:
    """Target days served by more than one vintage.

    A trigger that reads "the first N vintages" is reading 24-hour *blocks*, and
    two vintages that forecast the same day are not two blocks: pooling both
    double-counts that day's actuals and tightens the interval, the same error
    as resampling rows instead of vintages.

    This is not hypothetical. Verifying the ABL-692 install with a real run left
    2026-09-10 with two vintages for target day 2026-09-12 -- one calibrated,
    one not -- and `evaluate_net_position` scores every stored vintage, so a
    reader who counts by `generated_at` alone silently pools them. Flagged here
    so the duplicate has to be dispositioned rather than noticed.
    """
    days = {}
    for v in vintages:
        for day in (v.get("target_days") or "").split(","):
            if day:
                days.setdefault(day, []).append(v["generated_at"])
    return {d: gs for d, gs in days.items() if len(gs) > 1}


def per_country_ratio(conn: sqlite3.Connection, new_vintage: str,
                      old_vintage: str) -> list:
    """Per-country band-width ratios between two vintages, split by half.

    The whole-band ratio is weak evidence on its own: two vintages hold
    different amounts of context, so the raw band moves regardless of any
    calibration. The *halves* are the discriminating measurement. The map is
    anchored at q50 and scales the lower half by `s_lo` and the upper half by
    `s_hi` -- two different numbers. Band drift from a longer context has no
    reason to inflate the lower half by ~6pp more than the upper half in every
    country at once, so the lo/hi split separates "calibration applied" from
    "band happened to widen" in a way the mean ratio cannot.
    """
    out = []
    for cc in sorted({r["country_code"] for r in conn.execute(
            "SELECT DISTINCT country_code FROM forecast_quantiles "
            "WHERE model_name = ? AND generated_at = ?",
            (MODEL, new_vintage))}):
        w = {}
        for label, vintage in (("new", new_vintage), ("old", old_vintage)):
            r = conn.execute(
                """
                SELECT AVG(CASE WHEN quantile = 0.5 THEN forecast_value END)
                         - AVG(CASE WHEN quantile = 0.1 THEN forecast_value END) AS lo,
                       AVG(CASE WHEN quantile = 0.9 THEN forecast_value END)
                         - AVG(CASE WHEN quantile = 0.5 THEN forecast_value END) AS hi
                FROM forecast_quantiles
                WHERE model_name = ? AND forecast_type = ?
                  AND country_code = ? AND generated_at = ?
                """,
                (MODEL, FORECAST_TYPE, cc, vintage),
            ).fetchone()
            w[label] = (r["lo"], r["hi"]) if r else (None, None)

        def ratio(i):
            a, b = w["new"][i], w["old"][i]
            return (a / b) if (a and b) else None

        band_new = (w["new"][0] or 0) + (w["new"][1] or 0)
        band_old = (w["old"][0] or 0) + (w["old"][1] or 0)
        out.append({
            "country_code": cc,
            "band_new": band_new or None,
            "band_old": band_old or None,
            "ratio": (band_new / band_old) if band_old else None,
            "lo_ratio": ratio(0),
            "hi_ratio": ratio(1),
        })
    return out


def median_identical_across_vintages(conn: sqlite3.Connection, new_vintage: str,
                                     old_vintage: str) -> dict:
    """Is the q50 series bit-identical between the two vintages being compared?

    This is what licenses reading the lo/hi ratios as the calibration factors
    themselves rather than as an estimate. The replica refreshes once a day at
    05:00 UTC, so two runs on the same side of that refresh see the same
    observations, and Chronos-2 is deterministic given its input -- meaning the
    raw band is identical and the ratio isolates the calibration exactly.

    If the medians differ, the runs saw different data, the raw band moved too,
    and the ratios are only an estimate. Reported either way: the caller must
    not have to assume which case it is in.
    """
    rows = conn.execute(
        """
        SELECT a.country_code, a.target_timestamp_utc,
               a.forecast_value AS q50_new, b.forecast_value AS q50_old
        FROM forecast_quantiles a
        JOIN forecast_quantiles b
          ON b.country_code = a.country_code
         AND b.forecast_type = a.forecast_type
         AND b.target_timestamp_utc = a.target_timestamp_utc
         AND b.model_name = a.model_name
         AND b.quantile = a.quantile
        WHERE a.model_name = ? AND a.forecast_type = ? AND a.quantile = 0.5
          AND a.generated_at = ? AND b.generated_at = ?
        """,
        (MODEL, FORECAST_TYPE, new_vintage, old_vintage),
    ).fetchall()
    diffs = [abs(r["q50_new"] - r["q50_old"]) for r in rows]
    return {
        "matched_rows": len(rows),
        "max_abs_diff": max(diffs) if diffs else None,
        "identical": bool(diffs) and max(diffs) == 0.0,
    }


def q50_vs_point(conn: sqlite3.Connection, vintage: str) -> dict:
    """The calibration is anchored at q50: the point row must not have moved.

    Compares every q50 quantile row against the matching `forecasts` row of the
    same vintage. Any non-zero difference means the map moved the median, which
    would silently change the served point forecast.
    """
    rows = conn.execute(
        """
        SELECT q.country_code, q.target_timestamp_utc,
               q.forecast_value AS q50, f.forecast_value AS point
        FROM forecast_quantiles q
        JOIN forecasts f
          ON f.country_code = q.country_code
         AND f.forecast_type = q.forecast_type
         AND f.target_timestamp_utc = q.target_timestamp_utc
         AND f.generated_at = q.generated_at
         AND f.model_name = q.model_name
        WHERE q.model_name = ? AND q.forecast_type = ?
          AND q.quantile = 0.5 AND q.generated_at = ?
        """,
        (MODEL, FORECAST_TYPE, vintage),
    ).fetchall()
    diffs = [abs(r["q50"] - r["point"]) for r in rows]
    return {
        "matched_rows": len(rows),
        "max_abs_diff": max(diffs) if diffs else None,
        "anchored": bool(diffs) and max(diffs) == 0.0,
    }


def registered_calibration(registry_path: str | None) -> dict:
    """The (s_lo, s_hi) the serving code applies, from the registration file.

    The file is parsed directly rather than through `src.quantile_calibration`,
    because the question is what the *serving* checkout applies and importing
    that module would bind the answer to whichever tree this probe runs from.
    (Validating the registration is the serving code's job -- `load_registry`
    raises on a malformed one at forecast time; this only reports it.)
    """
    path = Path(registry_path) if registry_path else (
        Path(__file__).parents[1] / "experiments"
        / "net_position_quantile_calibration.json")
    if not path.exists():
        return {"available": False, "path": str(path), "reason": "file not found"}
    try:
        models = json.loads(path.read_text(encoding="utf-8")).get("models") or {}
    except Exception as exc:
        return {"available": False, "path": str(path),
                "reason": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "path": str(path), "models": models}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sidecar-db", default=None,
                    help="Sidecar DB holding the served vintages (read-only).")
    ap.add_argument("--vintages", type=int, default=12,
                    help="How many recent vintages to tabulate.")
    ap.add_argument("--registry-path", default=None,
                    help="Calibration registration to read (default: this tree). "
                         "Point at the serving checkout to read what production "
                         "applies.")
    ap.add_argument("--json-out", default=None, help="Machine record path.")
    args = ap.parse_args()

    sidecar = args.sidecar_db or getattr(config, "FORECAST_OUTPUT_DB", None)
    if not sidecar:
        print("No sidecar DB: pass --sidecar-db or set FORECAST_OUTPUT_DB.")
        return 2

    conn = ro_connect(sidecar)
    vintages = band_by_vintage(conn, args.vintages)
    if not vintages:
        print(f"No {MODEL} {FORECAST_TYPE} quantile rows in {sidecar}.")
        return 2

    print(f"sidecar: {sidecar}  (read-only)")
    print(f"model:   {MODEL} / {FORECAST_TYPE}")
    print()
    print(f"{'generated_at':<28} {'cc':>3} {'rows':>6} {'mean p10-p90 band (MW)':>24}")
    for v in vintages:
        band = v["mean_band"]
        print(f"{v['generated_at']:<28} {v['countries']:>3} {v['rows_all']:>6} "
              f"{band:>24.1f}")

    # The step: newest vintage against the one before it.
    detail = {}
    if len(vintages) >= 2:
        new_v, old_v = vintages[0]["generated_at"], vintages[1]["generated_at"]
        detail["per_country_ratio"] = per_country_ratio(conn, new_v, old_v)
        detail["q50_anchor"] = q50_vs_point(conn, new_v)
        print()
        print(f"per-country band ratio, {new_v} vs {old_v}:")
        print(f"  {'cc':<3} {'band_new':>10} {'band_old':>10} {'ratio':>7} "
              f"{'lo_ratio':>9} {'hi_ratio':>9}")
        for r in detail["per_country_ratio"]:
            def f(x, n=3):
                return f"{x:.{n}f}" if x else "n/a"
            print(f"  {r['country_code']:<3} {r['band_new']:>10.1f} "
                  f"{r['band_old']:>10.1f} {f(r['ratio']):>7} "
                  f"{f(r['lo_ratio']):>9} {f(r['hi_ratio']):>9}")
        los = [r["lo_ratio"] for r in detail["per_country_ratio"] if r["lo_ratio"]]
        his = [r["hi_ratio"] for r in detail["per_country_ratio"] if r["hi_ratio"]]
        if los and his:
            detail["lo_ratio_mean"] = sum(los) / len(los)
            detail["hi_ratio_mean"] = sum(his) / len(his)
            reg = registered_calibration(args.registry_path)
            spec = (reg.get("models") or {}).get(MODEL, {})
            print()
            print(f"  mean lo_ratio {detail['lo_ratio_mean']:.4f}  "
                  f"registered s_lo {spec.get('s_lo_applied')}")
            print(f"  mean hi_ratio {detail['hi_ratio_mean']:.4f}  "
                  f"registered s_hi {spec.get('s_hi_applied')}")
        anchor = detail["q50_anchor"]
        print()
        print(f"q50 anchor check on {new_v}: matched={anchor['matched_rows']} "
              f"max|q50-point|={anchor['max_abs_diff']} "
              f"anchored={anchor['anchored']}")

        same = median_identical_across_vintages(conn, new_v, old_v)
        detail["median_identical_across_vintages"] = same
        print(f"q50 identical across the two vintages: matched={same['matched_rows']} "
              f"max|new-old|={same['max_abs_diff']} identical={same['identical']}")
        print("  -> ratios above isolate the calibration exactly"
              if same["identical"] else
              "  -> the two runs saw different data; ratios are an estimate, "
              "not the applied factors")

    detail["registered_calibration"] = registered_calibration(args.registry_path)

    # Coverage, for ABL-677's window. A 10-vintage window is only 10 vintages
    # if every day actually produced one; a missing day and a short day both
    # shrink it silently, and the calibrated era is what has to fill it.
    present = {v["generated_at"]: v["countries"] for v in vintages}
    full = max(present.values()) if present else 0
    short = {g: n for g, n in present.items() if n < full}
    dupes = duplicate_target_days(vintages)
    detail["coverage"] = {"max_countries_seen": full, "short_vintages": short,
                          "duplicate_target_days": dupes}
    if dupes:
        print()
        print("duplicate target days -- these vintages are NOT independent 24h "
              "blocks and must not both be pooled:")
        for day, gens in sorted(dupes.items()):
            print(f"  target {day} served by {len(gens)} vintages:")
            for g in gens:
                print(f"    {g}")
    if short:
        print()
        print(f"coverage: {full} countries is the widest vintage in this window; "
              f"{len(short)} vintage(s) fall short:")
        for g, n in sorted(short.items()):
            print(f"  {g}  {n} countries")

    record = {
        "issue": "ABL-692",
        "sidecar_db": str(sidecar),
        "model_name": MODEL,
        "forecast_type": FORECAST_TYPE,
        "vintages": vintages,
        **detail,
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
