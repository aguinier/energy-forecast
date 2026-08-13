#!/usr/bin/env python3
"""ABL-419 -- why GR failed, and the two explanations that did not survive.

GR is the worst cell of tranche 2c by a distance: challenger 20.8% WAPE against
a **10.2%** seasonal-naive D-7 bar, skill -104%, grade C. Its error is a *level*
error and not a shape error -- correlation 1.00, slope 0.8, bias **-18.2%** --
so the model tracks GR's diurnal cycle almost perfectly and sits ~18% under it.

This probe tests the data explanations for that, because "the model is bad" and
"the series changed under the model" are different findings with different
owners, and the tranche read alone cannot tell them apart. It is read-only: it
opens the replica `mode=ro`, fits nothing, and writes only to `reports/`.

**What it found, in order of how much it changes the reading:**

1. **GR's `energy_generation` changes resolution inside the registered fit
   window.** It is hourly (1 row/hour) through 2026-04-30 and quarter-hourly
   (4 rows/hour) from mid-May. So 4,272 of the fit window's hours carry a `:00`
   row but only 1,611 carry `:15`/`:30`/`:45`, while **all 720** gate-window
   hours carry all four. The model is fitted across a resolution change and
   scored entirely on the far side of it. Real, and none of the other four
   countries does it -- ES/HR/IT are 4/hour throughout, PT is 1/hour throughout.

2. **That is not the mechanism.** Measured on the hours that carry all four
   samples, so the comparison is within one sample rather than across a season:
   the `:00` instant differs from the true 4-sample hourly mean by **+0.02%** of
   level in the fit window and **-0.02%** in the gate window. The hourly series
   the harness fits is not level-biased by the cadence change, and an 18.2% bias
   cannot come from a 0.02% one.

3. **Extrapolation is directionally right and not sufficient.** GR has the
   tranche's largest fit->gate level ratio at **1.857** (HR 1.62, ES 1.55,
   PT 1.55, IT 1.42), which matches the sign of the bias. But HR sits at 1.62
   and passed, so the ratio does not separate pass from fail on its own.

4. **The tree range ceiling is real and too small.** CatBoost is a tree
   ensemble, so its predictions cannot exceed the target range it was fitted on.
   GR has the most gate hours above its own fit-window maximum -- **29 of 720
   (4.0%)**, against 1 of 720 for HR, IT and PT -- and its gate max is 9.5%
   above its fit max. The largest in the tranche, and still far too few hours to
   produce an 18.2% bias by itself.

So: one real data defect, **excluded** as the cause by measurement; two partial
contributors, neither sufficient. GR's failure is not explained here and this
probe deliberately stops rather than asserting a cause it has not established.

A note on a trap this probe fell into first. Averaging *rows* rather than
*hours* over a mixed-cadence series over-weights the quarter-hourly era, which
for GR is the summer -- it reported a fit mean of 1720.67 MW against the
harness's own 1439.96 MW and made GR's extrapolation ratio look ordinary. Every
level here is therefore **hour-weighted**, one value per hour, and the fit means
are cross-checked against the `constant_causal` levels in the tranche's own
results file, which they reproduce exactly for all five countries.

Run: `.venv\\Scripts\\python.exe scripts/abl419_gr_resolution_probe.py`
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent.parent
REPLICA = r"C:\Code\able\data\energy_dashboard.db"
COUNTRIES = ("ES", "GR", "HR", "IT", "PT")
FIT = ("2026-01-14", "2026-07-11")
GATE = ("2026-07-11", "2026-08-10")
SOURCE = "energy_generation"

#: The harness's own `constant_causal` levels for this tranche -- the fit-window
#: mean of the series it actually fitted. Restated here only as a cross-check
#: target; a disagreement means this probe is reading a different series than the
#: gate did, which is the failure mode that makes every number below meaningless.
HARNESS_FIT_MEANS = {"ES": 7222.73, "GR": 1439.96, "HR": 97.72,
                     "IT": 4542.37, "PT": 770.03}

#: Date-only bounds throughout. `energy_generation` carries a mixed timestamp
#: spelling, and normalising it inside a WHERE full-scans a 9.4 GB table; a
#: date-only string bound is format-safe and uses the index.
_HOURLY = """
WITH h AS (SELECT substr(timestamp_utc,1,13) AS hr, AVG(solar_mw) AS v
           FROM energy_generation
           WHERE country_code=? AND timestamp_utc >= ? AND timestamp_utc < ?
             AND solar_mw IS NOT NULL
           GROUP BY hr)
SELECT COUNT(*), AVG(v), MAX(v) FROM h
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica", default=REPLICA)
    parser.add_argument("--json-out", default=str(ROOT / "reports" / "abl_419_gr_resolution.json"))
    args = parser.parse_args()

    replica = Path(args.replica)
    con = sqlite3.connect(f"file:{replica}?mode=ro", uri=True)
    record: dict = {
        "replica": str(replica),
        "replica_bytes": replica.stat().st_size,
        "source": SOURCE,
        "fit_window": FIT,
        "gate_window": GATE,
        "countries": {},
        "gr_cadence_by_month": {},
        "gr_minute_distribution": {},
        "top_of_hour_bias": {},
    }

    # 1. Cadence, level and range, per country, hour-weighted.
    for country in COUNTRIES:
        entry = {}
        for name, (low, high) in (("fit", FIT), ("gate", GATE)):
            hours, mean, peak = con.execute(_HOURLY, (country, low, high)).fetchone()
            rows = con.execute(
                "SELECT COUNT(*) FROM energy_generation WHERE country_code=? "
                "AND timestamp_utc >= ? AND timestamp_utc < ?", (country, low, high)).fetchone()[0]
            entry[name] = {"rows": rows, "hours": hours, "rows_per_hour": rows / hours,
                           "hour_weighted_mean_mw": mean, "hour_weighted_max_mw": peak}
        fit_max = entry["fit"]["hour_weighted_max_mw"]
        above = con.execute(
            "WITH h AS (SELECT substr(timestamp_utc,1,13) AS hr, AVG(solar_mw) AS v "
            "FROM energy_generation WHERE country_code=? AND timestamp_utc >= ? "
            "AND timestamp_utc < ? AND solar_mw IS NOT NULL GROUP BY hr) "
            "SELECT COUNT(*) FROM h WHERE v > ?", (country, *GATE, fit_max)).fetchone()[0]
        entry["level_ratio_gate_over_fit"] = (entry["gate"]["hour_weighted_mean_mw"]
                                              / entry["fit"]["hour_weighted_mean_mw"])
        entry["gate_hours_above_fit_max"] = above
        entry["gate_hours_above_fit_max_pct"] = 100.0 * above / entry["gate"]["hours"]
        entry["harness_constant_causal_mw"] = HARNESS_FIT_MEANS[country]
        entry["reproduces_harness_fit_mean"] = bool(
            abs(entry["fit"]["hour_weighted_mean_mw"] - HARNESS_FIT_MEANS[country]) < 0.01)
        record["countries"][country] = entry

    # 2. When GR's cadence changes.
    for month, rows, hours in con.execute(
            "SELECT substr(timestamp_utc,1,7) AS m, COUNT(*), "
            "COUNT(DISTINCT substr(timestamp_utc,1,13)) FROM energy_generation "
            "WHERE country_code='GR' AND timestamp_utc >= '2025-12-01' "
            "AND timestamp_utc < '2026-08-10' GROUP BY m ORDER BY m"):
        record["gr_cadence_by_month"][month] = {"rows": rows, "hours": hours,
                                                "rows_per_hour": rows / hours}

    # 3. Which minutes exist, per window.
    for name, (low, high) in (("fit", FIT), ("gate", GATE)):
        record["gr_minute_distribution"][name] = dict(con.execute(
            "SELECT substr(timestamp_utc,15,2) AS mm, COUNT(*) FROM energy_generation "
            "WHERE country_code='GR' AND timestamp_utc >= ? AND timestamp_utc < ? "
            "GROUP BY mm ORDER BY mm", (low, high)).fetchall())

    # 4. The control that excludes the cadence change as the mechanism: on hours
    #    carrying all four samples, is the `:00` instant the hourly mean or not?
    for name, (low, high) in (("fit", FIT), ("gate", GATE)):
        hours, top, mean, diff = con.execute(
            "WITH h AS (SELECT substr(timestamp_utc,1,13) AS hr, COUNT(*) AS n, "
            "AVG(solar_mw) AS hourly_mean, "
            "MAX(CASE WHEN substr(timestamp_utc,15,2)='00' THEN solar_mw END) AS at_top "
            "FROM energy_generation WHERE country_code='GR' AND timestamp_utc >= ? "
            "AND timestamp_utc < ? AND solar_mw IS NOT NULL GROUP BY hr HAVING COUNT(*)=4) "
            "SELECT COUNT(*), AVG(at_top), AVG(hourly_mean), AVG(at_top - hourly_mean) "
            "FROM h WHERE at_top IS NOT NULL", (low, high)).fetchone()
        record["top_of_hour_bias"][name] = {
            "n_four_sample_hours": hours, "mean_at_top_mw": top,
            "mean_hourly_mean_mw": mean, "bias_mw": diff,
            "bias_pct_of_level": 100.0 * diff / mean if mean else None}
    con.close()

    Path(args.json_out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"replica {record['replica']} ({record['replica_bytes']:,} bytes), mode=ro\n")
    print("%-4s %-6s %8s %8s %10s %14s %10s" % ("cc", "window", "rows", "hours",
                                                "rows/hr", "mean MW", "max MW"))
    for country, entry in record["countries"].items():
        for name in ("fit", "gate"):
            win = entry[name]
            print("%-4s %-6s %8d %8d %10.2f %14.2f %10.1f" % (
                country, name, win["rows"], win["hours"], win["rows_per_hour"],
                win["hour_weighted_mean_mw"], win["hour_weighted_max_mw"]))
    print()
    print("%-4s %10s %10s %26s %12s" % ("cc", "ratio", "fit mean", "gate hrs > fit max",
                                        "== harness?"))
    for country, entry in record["countries"].items():
        print("%-4s %10.3f %10.2f %18d / %d %12s" % (
            country, entry["level_ratio_gate_over_fit"],
            entry["fit"]["hour_weighted_mean_mw"], entry["gate_hours_above_fit_max"],
            entry["gate"]["hours"], entry["reproduces_harness_fit_mean"]))
    print()
    print("GR cadence by month:", {m: round(v["rows_per_hour"], 2)
                                   for m, v in record["gr_cadence_by_month"].items()})
    print("GR minutes present:", record["gr_minute_distribution"])
    print()
    for name, bias in record["top_of_hour_bias"].items():
        print(f"GR `:00` vs true hourly mean, {name} window, 4-sample hours only "
              f"(n={bias['n_four_sample_hours']:,}): "
              f"{bias['bias_mw']:+.2f} MW = {bias['bias_pct_of_level']:+.3f}% of level")
    print(f"\nwrote {args.json_out}")

    unmatched = [c for c, e in record["countries"].items() if not e["reproduces_harness_fit_mean"]]
    if unmatched:
        print(f"\nWARNING: fit mean does not reproduce the harness's constant_causal for "
              f"{unmatched} -- this probe is reading a different series than the gate did")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
