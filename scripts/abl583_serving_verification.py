#!/usr/bin/env python
"""Record what `forecast_daily.py` actually served for CH solar, from its own output.

ABL-583 item 6: serving-verify end to end with
`forecast_daily.py --countries CH --types solar --horizons 1,2`.

This script does not generate anything. It reads the sidecar that run wrote and
turns it into a committed record, so section 6 of the pack rests on a machine
record rather than on a pasted log line. Two tables carry the whole answer:

    forecasts            the 48 rows that were served, POST-clamp
    forecast_clamp_log   what the ABL-337 clamp did to get there

WHY THE PROBE IS COMPARED AGAINST THIS
--------------------------------------
`scripts/abl583_ch_night_probe.py` reports the pre-clamp distribution, which the
served rows cannot show -- the clamp has already run by the time anything is
written. That makes the probe load-bearing for ABL-583's night question, so the
probe has to be shown to be measuring the same frame the runner served rather
than a reconstruction that merely resembles it. `forecast_clamp_log` is the
overlap: the probe recomputes every one of those fields independently, so
comparing them field by field is a real check and not a restatement.

That comparison is not decorative. The first probe run of this issue disagreed
with the serving run on every night field -- 9 of 16 hours zeroed against 16 of
16, -63.63 MW against -32.21. Same artifact, same replica, same reference date;
the two runs fired at different wall-clock hours. `predict_d2` anchors its
feature build on the clock as well as on the reference date, so the served night
series is not a function of (artifact, reference_date, replica) alone. The
agreement below is therefore a same-hour agreement, and it is recorded as such.

Read-only against the sidecar and the replica. Writes only its own JSON.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COUNTRY = "CH"
FORECAST_TYPE = "solar"

#: Every field `forecast_clamp_log` and the probe both compute. Compared exactly:
#: these are the same float arithmetic on both sides, so a tolerance would only
#: hide a disagreement.
SHARED_CLAMP_FIELDS = (
    "rows_total",
    "night_hours",
    "hours_zeroed_night",
    "hours_raised_floor",
    "mw_removed_night",
    "mw_added_floor",
    "mw_removed_total",
    "min_forecast_mw",
    "max_night_forecast_mw",
)


def read_sidecar(path: Path) -> tuple[list[dict], dict]:
    """The served rows and the clamp-log row the serving run wrote."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT target_timestamp_utc, forecast_value, horizon_hours, "
            "       model_name, forecast_type, renewable_type "
            "FROM forecasts "
            "WHERE country_code = ? AND renewable_type = ? "
            "ORDER BY target_timestamp_utc",
            (COUNTRY, FORECAST_TYPE),
        )]
        clamp = [dict(r) for r in conn.execute(
            "SELECT * FROM forecast_clamp_log "
            "WHERE country_code = ? AND renewable_type = ? "
            "ORDER BY id DESC LIMIT 1",
            (COUNTRY, FORECAST_TYPE),
        )]
    finally:
        conn.close()
    if not rows:
        raise SystemExit(f"no served rows for {COUNTRY}/{FORECAST_TYPE} in {path}")
    if not clamp:
        raise SystemExit(f"no clamp-log row for {COUNTRY}/{FORECAST_TYPE} in {path}")
    return rows, clamp[0]


def served_summary(rows: list[dict]) -> dict:
    """Post-clamp invariants, measured on the rows rather than assumed from the log."""
    values = [float(r["forecast_value"]) for r in rows]
    positive = [v for v in values if v > 0.0]
    return {
        "n_rows": len(rows),
        "target_window": [rows[0]["target_timestamp_utc"], rows[-1]["target_timestamp_utc"]],
        "horizon_hours_min": min(int(r["horizon_hours"]) for r in rows),
        "horizon_hours_max": max(int(r["horizon_hours"]) for r in rows),
        "model_names": sorted({r["model_name"] for r in rows}),
        "min_mw": round(min(values), 4),
        "max_mw": round(max(values), 4),
        "mean_mw": round(sum(values) / len(values), 4),
        "positive_hours_mean_mw": round(sum(positive) / len(positive), 4) if positive else None,
        "n_negative": sum(1 for v in values if v < 0.0),
        "n_exactly_zero": sum(1 for v in values if v == 0.0),
        "any_served_row_negative": any(v < 0.0 for v in values),
    }


def compare_to_probe(clamp: dict, probe: dict) -> dict:
    """Field-by-field against the probe's independently recomputed clamp fields."""
    probe_fields = probe.get("clamp_log_fields", {})
    per_field, mismatched = {}, []
    for name in SHARED_CLAMP_FIELDS:
        served = clamp.get(name)
        # The probe rounds to 4dp on the way into its record; compare on that basis
        # rather than declaring a mismatch that is only a printing difference.
        served_cmp = round(served, 4) if isinstance(served, float) else served
        probed = probe_fields.get(name)
        agrees = served_cmp == probed
        per_field[name] = {"served": served_cmp, "probe": probed, "agrees": agrees}
        if not agrees:
            mismatched.append(name)
    return {
        "fields_compared": list(SHARED_CLAMP_FIELDS),
        "per_field": per_field,
        "mismatched_fields": mismatched,
        "probe_reproduces_the_served_clamp": not mismatched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Turn the ABL-583 CH solar serving run's own output into a "
                     "committed record."))
    parser.add_argument("--sidecar", required=True,
                        help="The FORECAST_OUTPUT_DB the serving run wrote.")
    parser.add_argument("--probe", default="reports/abl_583_ch_night_probe.json",
                        help="The night probe's record, compared field by field.")
    parser.add_argument("--command", default=None,
                        help="The exact forecast_daily.py invocation, recorded verbatim.")
    parser.add_argument("--json-out", default="reports/abl_583_serving_verification.json")
    args = parser.parse_args()

    sidecar = Path(args.sidecar)
    if not sidecar.is_file():
        raise SystemExit(f"sidecar not found: {sidecar}")

    rows, clamp = read_sidecar(sidecar)
    probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))

    record = {
        "issue": "ABL-583",
        "check": "end-to-end serving verification (ABL-583 item 6)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "country": COUNTRY,
        "forecast_type": FORECAST_TYPE,
        "command": args.command,
        "wrote_to": {
            "sidecar": str(sidecar),
            "is_scratch": True,
            "replica_written": False,
            "note": ("FORECAST_OUTPUT_DB was pointed at a run-scoped scratch file, so the "
                     "replica was not written. Serving into production is a deploy and is "
                     "outside this issue."),
        },
        "served": served_summary(rows),
        "clamp_log": {k: v for k, v in clamp.items() if k != "id"},
        "agreement_with_night_probe": compare_to_probe(clamp, probe),
        "clock_dependence": {
            "observed": True,
            "note": ("The first probe run of this issue (16:36Z) disagreed with the serving "
                     "run (18:01Z) on every night field, at identical artifact, replica and "
                     "reference date. predict_d2 anchors its feature build on wall-clock time "
                     "as well as on the reference date. Re-running the probe in the same hour "
                     "as the serving run reproduces its clamp log exactly. The agreement "
                     "recorded here is therefore a same-hour agreement."),
            "superseded_probe_run": {
                "generated_at": "2026-08-27T16:36:37.063949+00:00",
                "hours_zeroed_night": 9,
                "hours_raised_floor": 2,
                "mw_removed_night": -63.6291,
                "min_forecast_mw": -21.4604,
                "max_night_forecast_mw": 2.4043,
                "pre_clamp_night_mean_mw": -3.9768,
                "pre_clamp_night_pct_negative": 43.75,
            },
        },
        "scored_or_graded": False,
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")

    s = record["served"]
    a = record["agreement_with_night_probe"]
    print(f"[{COUNTRY}/{FORECAST_TYPE}] {s['n_rows']} rows served, "
          f"{s['target_window'][0]} -> {s['target_window'][1]}")
    print(f"  post-clamp: min {s['min_mw']} MW, negative rows {s['n_negative']}, "
          f"exact zeros {s['n_exactly_zero']}")
    print(f"  probe reproduces the served clamp: {a['probe_reproduces_the_served_clamp']} "
          f"({len(a['fields_compared'])} fields)")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
