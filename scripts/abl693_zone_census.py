"""ABL-693 served-zone census: did the scored population change mid-window?

A pooled gate figure is only comparable across vintages if the same zones are
behind every vintage. Nothing checked that. `build_gate_scope` derives
`countries_measured` from whatever rows exist, and `GATE_EXCLUDED_COUNTRIES` is
a static list, so a zone that stops being served part-way through a window does
not appear as excluded -- it silently contributes fewer pairs and the pooled
number reweights toward the zones that stayed.

That is not hypothetical. `chronos-2-V010` and `chronos-2-V016` served 19 zones
through the 2026-09-05 vintage and 18 from 2026-09-06 on: PT's net-position
actuals stop at 2026-09-04 21:00 UTC upstream, and the ABL-650 context guard
refuses a stale context rather than forward-filling one. PT is one of the three
zones in ABL-650's own defect statement and one of the worst coverage cells, so
dropping it *raises* pooled coverage -- the direction that hides a miss.

This census is read-only on the sidecar. It answers one question per model:
was the served zone set constant across the window, and if not, which zones
entered or left, on which vintage, and how unbalanced is the resulting panel?

Exit code is the finding, not the health of the run:
  0 -- the population was constant across the window for every model read
  1 -- at least one model's population moved: a zone left, a zone entered, or
       a calendar day inside the span carries no vintage at all
  2 -- nothing matched the selection (no rows: unknown, which is not 'constant')

Usage:
  python scripts/abl693_zone_census.py --model chronos-2-V010
  python scripts/abl693_zone_census.py --since 2026-09-01 --table forecast_quantiles
  python scripts/abl693_zone_census.py --expect-zones AT,BE,BG --json-out out.json
"""
import argparse
import datetime
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_SIDECAR = "C:/Code/able/data/forecasts_local.db"

#: Only these two carry a (country_code, generated_at) population in the
#: sidecar. `forecast_quantiles` is the one a coverage read pools over, so it
#: is selectable rather than assumed identical to `forecasts`.
TABLES = ("forecasts", "forecast_quantiles")


def _connect_readonly(path):
    """Open the sidecar read-only. A census that can write is a census that can
    be blamed for the thing it measured."""
    return sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro",
                           uri=True, timeout=120.0)


def read_population(conn, table, forecast_type, model=None,
                    since=None, until=None):
    """Zone set per (model_name, vintage day), plus the row count behind each.

    The vintage key is the UTC *day* of `generated_at`, not the raw timestamp:
    the criterion ABL-677 reads is written in daily vintages, and a same-day
    re-run would otherwise split one day of evidence into two populations.
    """
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}; expected one of {TABLES}")
    where = ["forecast_type = ?"]
    params = [forecast_type]
    if model:
        where.append("model_name = ?")
        params.append(model)
    if since:
        where.append("generated_at >= ?")
        params.append(since)
    if until:
        where.append("generated_at < ?")
        params.append(until)
    query = f"""
        SELECT model_name,
               substr(generated_at, 1, 10) AS vintage_day,
               country_code,
               COUNT(*) AS rows_present
        FROM {table}
        WHERE {' AND '.join(where)}
        GROUP BY model_name, vintage_day, country_code
    """
    population = {}
    for model_name, day, country, rows in conn.execute(query, params):
        population.setdefault(model_name, {}).setdefault(day, {})[country] = rows
    return population


def census_for_model(by_day, expect_zones=None):
    """Compare every vintage's zone set against the reference population.

    The reference is `expect_zones` when the caller registered one, and
    otherwise the union over the window. The union is deliberate: taking the
    *first* vintage as the reference would report a zone that was missing from
    vintage 1 and present afterwards as an addition rather than as a vintage-1
    hole, and both directions matter to a pooled read.
    """
    days = sorted(by_day)
    if not days:
        return None
    union = set()
    for day in days:
        union |= set(by_day[day])
    reference = set(expect_zones) if expect_zones else union

    per_vintage = []
    for day in days:
        served = set(by_day[day])
        per_vintage.append({
            "vintage_day": day,
            "zones_served": len(served),
            "missing_vs_reference": sorted(reference - served),
            "extra_vs_reference": sorted(served - reference),
            "rows": int(sum(by_day[day].values())),
        })

    # A zone "left" if its last served vintage is before the window's last, and
    # "entered" if its first served vintage is after the window's first. Both
    # are reported with the boundary vintage, because a gate read needs to know
    # *when* the panel changed, not only that it did.
    first_day, last_day = days[0], days[-1]
    left, entered = [], []
    for zone in sorted(union):
        served_days = [d for d in days if zone in by_day[d]]
        if served_days[-1] != last_day:
            left.append({"zone": zone, "last_served": served_days[-1],
                         "first_absent": days[days.index(served_days[-1]) + 1]})
        if served_days[0] != first_day:
            entered.append({"zone": zone, "first_served": served_days[0]})

    # Panel balance. A zone present for every vintage contributes 1.0; the
    # pooled statistic is weighted by these shares whether or not it says so.
    balance = {}
    for zone in sorted(reference):
        served_days = [d for d in days if zone in by_day.get(d, {})]
        balance[zone] = {
            "vintages_served": len(served_days),
            "vintages_in_window": len(days),
            "completeness": round(len(served_days) / len(days), 4),
            "rows": int(sum(by_day[d].get(zone, 0) for d in days)),
        }
    total_rows = sum(v["rows"] for v in balance.values())
    for zone, entry in balance.items():
        entry["row_share"] = round(entry["rows"] / total_rows, 6) if total_rows else 0.0
        entry["balanced_share"] = round(1 / len(reference), 6) if reference else 0.0

    # A vintage that produced *nothing* has no rows, so it has no key here and
    # is invisible to every check above -- the 2026-09-02 run refused all 19
    # zones and simply is not in the data. Counting the calendar days inside
    # the span that carry no rows at all is the only way to see it.
    missing_days = _calendar_gaps(first_day, last_day, set(days))

    return {
        "vintage_days": days,
        "vintage_days_in_span": _span_length(first_day, last_day),
        "vintage_days_missing": missing_days,
        "vintage_calendar_complete": not missing_days,
        "reference_zones": sorted(reference),
        "reference_source": "expect_zones" if expect_zones else "window_union",
        "population_constant": not left and not entered
        and all(not v["missing_vs_reference"] and not v["extra_vs_reference"]
                for v in per_vintage),
        "zones_left": left,
        "zones_entered": entered,
        "per_vintage": per_vintage,
        "panel_balance": balance,
    }


def _span_length(first_day, last_day):
    return (_date(last_day) - _date(first_day)).days + 1


def _calendar_gaps(first_day, last_day, present):
    start, end = _date(first_day), _date(last_day)
    gaps, cursor = [], start
    while cursor <= end:
        stamp = cursor.isoformat()
        if stamp not in present:
            gaps.append(stamp)
        cursor += datetime.timedelta(days=1)
    return gaps


def _date(day):
    return datetime.date.fromisoformat(day)


def render(report):
    """Plain-text census. Report bodies may hold non-ASCII; this one does not
    need to, so it stays copy-pasteable into an issue comment."""
    lines = []
    for model in sorted(report["models"]):
        model_report = report["models"][model]
        days = model_report["vintage_days"]
        verdict = "CONSTANT" if model_report["population_constant"] else "CHANGED"
        lines.append(f"{model}: {verdict} over {len(days)} vintage days "
                     f"of {model_report['vintage_days_in_span']} in span "
                     f"[{days[0]} .. {days[-1]}], "
                     f"{len(model_report['reference_zones'])} reference zones "
                     f"({model_report['reference_source']})")
        for stamp in model_report["vintage_days_missing"]:
            lines.append(f"    NO VINTAGE {stamp}: the model wrote no row at "
                         f"all -- invisible to a per-zone check")
        for entry in model_report["zones_left"]:
            lines.append(f"    LEFT    {entry['zone']}: last served "
                         f"{entry['last_served']}, absent from "
                         f"{entry['first_absent']} onward")
        for entry in model_report["zones_entered"]:
            lines.append(f"    ENTERED {entry['zone']}: first served "
                         f"{entry['first_served']}")
        for zone, entry in sorted(model_report["panel_balance"].items()):
            if entry["completeness"] < 1.0:
                lines.append(
                    f"    PARTIAL {zone}: {entry['vintages_served']}/"
                    f"{entry['vintages_in_window']} vintages, row share "
                    f"{entry['row_share']:.4f} vs {entry['balanced_share']:.4f} "
                    f"balanced")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=("Census the served zone population per model per vintage, "
                     "so a pooled gate figure cannot silently pool over a "
                     "panel that changed mid-window. Read-only."))
    parser.add_argument('--sidecar-db', default=None,
                        help="sidecar path; defaults to FORECAST_OUTPUT_DB, "
                             "then the workstation sidecar")
    parser.add_argument('--table', default='forecasts', choices=TABLES,
                        help="which sidecar table to census (default: forecasts)")
    parser.add_argument('--forecast-type', default='net_position',
                        help="forecast_type to census (default: net_position)")
    parser.add_argument('--model', default=None,
                        help="restrict to one model_name (default: every model)")
    parser.add_argument('--since', default=None,
                        help="lower bound on generated_at, inclusive (YYYY-MM-DD)")
    parser.add_argument('--until', default=None,
                        help="upper bound on generated_at, exclusive (YYYY-MM-DD)")
    parser.add_argument('--expect-zones', default=None,
                        help="comma-separated registered zone list; without it "
                             "the reference is the union over the window")
    parser.add_argument('--json-out', default=None,
                        help="also write the full census as JSON")
    args = parser.parse_args()

    sidecar = (args.sidecar_db or os.environ.get('FORECAST_OUTPUT_DB')
               or DEFAULT_SIDECAR)
    expect = ([z.strip() for z in args.expect_zones.split(',') if z.strip()]
              if args.expect_zones else None)

    conn = _connect_readonly(sidecar)
    try:
        population = read_population(conn, args.table, args.forecast_type,
                                     args.model, args.since, args.until)
    finally:
        conn.close()

    if not population:
        print(f"no {args.forecast_type} rows in {args.table} for the selection "
              f"-- population unknown, which is not 'constant'")
        return 2

    report = {
        "sidecar_db": str(sidecar),
        "table": args.table,
        "forecast_type": args.forecast_type,
        "since": args.since,
        "until": args.until,
        "models": {model: census_for_model(by_day, expect)
                   for model, by_day in population.items()},
    }
    print(render(report))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"wrote {args.json_out}")

    changed = [m for m, r in report["models"].items()
               if not r["population_constant"]
               or not r["vintage_calendar_complete"]]
    if changed:
        print(f"population changed for: {', '.join(sorted(changed))}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
