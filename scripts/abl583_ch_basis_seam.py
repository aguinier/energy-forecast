#!/usr/bin/env python
"""Where CH `solar`'s source basis actually changes, and it is not the gate boundary.

ABL-583 item 4 runs the contamination screens over the window this artifact was
fitted on. Two of them come back non-clean for CH, and read naively they both
point at the fit/gate boundary:

    ABL-439 vintage   fit-window ratio 1.0424, gate-window ratio 1.0000,
                      discontinuity 0.0424 against a 0.02 cut -> basis-INCONSISTENT
    night floor       674 of 1504 night hours above 1 MW before 2026-07-11,
                      0 of 270 after it

Both windows in that screen are ABL-348's *registered* fit/gate split, so the
screen can only ever report a number per window. It cannot say **when** inside a
window the basis moved, and the natural misreading -- that something happened at
2026-07-11, the boundary -- is wrong.

WHAT THE MONTHLY BREAKDOWN SHOWS
--------------------------------
Resolved to months, the two screens are one event on one date, and it is
**2026-06-01**:

    ratio energy_generation / energy_renewable   Jan..May: 0.65, 1.10, 1.17,
                                                 1.20, 0.96
                                                 Jun, Jul, Aug: 1.0000 exactly
    night hours above 1 MW                       Jan..May: 6% -> 98.5%, rising
                                                 Jun, Jul, Aug: 0.0%, mean 0.000

Before June the two source tables are two different series and the actuals carry
a small positive night floor; from June they are one identical series with a hard
zero at night. The 1.0424 the screen reports for the fit window is not a level:
it is a blend of 142 pre-seam days and 41 post-seam ones, and the gate window's
1.0000 is not a change, it is the post-seam regime the fit window is already 37%
made of.

WHY THIS MATTERS AND WHY IT IS NOT A BLOCKER
--------------------------------------------
It does **not** void ABL-581's read. That read is fitted and scored on the
registered `energy_generation` at both ends, so a divergence between that table
and `energy_renewable` is not in its path at all; and its gate window
(2026-07-11 -> 2026-08-10) lies wholly inside the post-seam regime, so the grades
are computed on one basis.

What it does say is that **this artifact is fitted across the seam** -- roughly
142 days of pre-seam rows against 81 post-seam -- while the regime it will serve
into is the post-seam one. That is a train/serve mismatch in the actuals, not in
the features, and it is a candidate explanation for a night level the served
series should not have. It is reported, not fixed: the fit window is the ship
set's module constant, shared with ABL-525's seven and ABL-580's three, and
moving it for one country would fork the batch and void
`abl525_repro_check.py`'s comparison for the other ten.

The magnitude is small and is stated so nobody over-reads it: the pre-seam night
mean is 0.21-2.93 MW against daylight means of 292-1526 MW, and the screen's own
`wape_floor_pct_if_clamped` for the whole fit window is 0.054%. This is a basis
change, not a BG-scale contamination.

Read-only against the replica. Writes only its own JSON.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from src.solar_geometry import (  # noqa: E402
    NIGHT_ELEVATION_THRESHOLD_DEG,
    is_night_hour,
)

COUNTRY = "CH"
FORECAST_TYPE = "solar"
COLUMN = "solar_mw"

#: The ship set's fit window; the screens and this probe cover the same rows.
FIT_START = "2026-01-11"
FIT_END = "2026-08-22"

#: ABL-338's night threshold, the one every published night screen uses.
NIGHT_THRESHOLD_MW = 1.0

#: The date the monthly breakdown puts the basis change on. Asserted rather than
#: discovered at run time so a later reader gets a claim that can fail.
EXPECTED_SEAM = "2026-06-01"


def monthly_series(conn, table, start, end):
    """`{timestamp -> value}` for one source table over the window."""
    rows = conn.execute(
        f"SELECT timestamp_utc, {COLUMN} FROM {table} "
        "WHERE country_code = ? AND timestamp_utc >= ? AND timestamp_utc < ? "
        f"AND {COLUMN} IS NOT NULL",
        (COUNTRY, start, end),
    ).fetchall()
    return dict(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Locate CH solar's source-basis seam by month (ABL-583 item 4)."))
    parser.add_argument("--replica-db", default=config.DATABASE_PATH,
                        help="Read-only replica (default: ENERGY_DB_PATH).")
    parser.add_argument("--json-out", default="reports/abl_583_ch_basis_seam.json")
    args = parser.parse_args()

    replica = Path(args.replica_db)
    if not replica.is_file():
        raise SystemExit(f"replica not found: {replica}")
    conn = sqlite3.connect(f"file:{replica.as_posix()}?mode=ro", uri=True, timeout=300)

    generation = monthly_series(conn, "energy_generation", FIT_START, FIT_END)
    renewable = monthly_series(conn, "energy_renewable", FIT_START, FIT_END)

    frame = pd.DataFrame(
        {"ts": pd.to_datetime(list(generation)), "mw": list(generation.values())}
    ).sort_values("ts")
    frame["night"] = is_night_hour(
        COUNTRY, pd.DatetimeIndex(frame["ts"]).floor("h"), NIGHT_ELEVATION_THRESHOLD_DEG)
    frame["month"] = frame["ts"].dt.to_period("M").astype(str)

    months = []
    for month, group in frame.groupby("month"):
        night = group[group["night"]]
        daylight = group[~group["night"]]
        above = int((night["mw"] > NIGHT_THRESHOLD_MW).sum())

        # The cross-table ratio on hours present in BOTH tables. Hours-present
        # differs between the two, and a ratio of two means over different hour
        # sets would mix a level difference with a coverage difference.
        keys = [k for k in generation if str(k)[:7] == month and k in renewable]
        ratio = None
        if keys:
            gen_mean = sum(generation[k] for k in keys) / len(keys)
            ren_mean = sum(renewable[k] for k in keys) / len(keys)
            ratio = round(gen_mean / ren_mean, 4) if ren_mean else None

        months.append({
            "month": month,
            "n_hours": int(len(group)),
            "n_night_hours": int(len(night)),
            "n_night_above_threshold": above,
            "pct_night_above_threshold": round(100.0 * above / max(len(night), 1), 2),
            "night_mean_mw": round(float(night["mw"].mean()), 4) if len(night) else None,
            "night_max_mw": round(float(night["mw"].max()), 4) if len(night) else None,
            "daylight_mean_mw": round(float(daylight["mw"].mean()), 2) if len(daylight) else None,
            "n_hours_in_both_tables": len(keys),
            "generation_over_renewable_ratio": ratio,
        })

    def regime(entry):
        """Post-seam iff the tables agree exactly AND no night hour clears 1 MW."""
        return (entry["generation_over_renewable_ratio"] == 1.0
                and entry["n_night_above_threshold"] == 0)

    post = [m["month"] for m in months if regime(m)]
    pre = [m["month"] for m in months if not regime(m)]
    seam = f"{min(post)}-01" if post else None

    payload = {
        "issue": "ABL-583",
        "check": "where CH solar's source basis changes, resolved to months",
        "country": COUNTRY,
        "forecast_type": FORECAST_TYPE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fit_window": [FIT_START, FIT_END],
        "replica_db": str(replica),
        "replica_bytes": replica.stat().st_size,
        "night_threshold_mw": NIGHT_THRESHOLD_MW,
        "months": months,
        "pre_seam_months": pre,
        "post_seam_months": post,
        "seam_date": seam,
        "seam_is_where_expected": seam == EXPECTED_SEAM,
        "seam_is_the_gate_boundary": seam == "2026-07-11",
        "reading": (
            "Two screens, one event, one date. Before the seam the two source "
            "tables are different series and the actuals carry a small positive "
            "night floor; from the seam they are one identical series with a hard "
            "zero at night. The ABL-439 screen's fit-window ratio of 1.0424 is a "
            "BLEND of the two regimes, not a level, and its gate-window 1.0000 is "
            "not a change -- it is the post-seam regime the fit window already "
            "contains. The seam is NOT the fit/gate boundary."),
        "consequence_for_this_artifact": (
            "ABL-581's read is unaffected: it fits and scores on the registered "
            "energy_generation at both ends, so the cross-table divergence is not "
            "in its path, and its gate window lies wholly post-seam. But THIS "
            "artifact is fitted across the seam while serving into the post-seam "
            "regime -- a train/serve mismatch in the actuals, and a candidate "
            "explanation for any night level in the served series. Reported, not "
            "fixed: the fit window is the ship set's shared module constant and "
            "moving it for one country would fork the batch."),
        "magnitude_caveat": (
            "Small. Pre-seam night means are 0.21-2.93 MW against daylight means "
            "of 292-1526 MW, and the contamination screen's own "
            "wape_floor_pct_if_clamped over the whole fit window is 0.054%. This "
            "is a basis change, not a BG-scale contamination."),
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"  {'month':9s} {'n_night':>8s} {'>1MW':>6s} {'pct':>7s} "
          f"{'night_mean':>10s} {'daylight':>9s} {'gen/ren':>8s}")
    for entry in months:
        print(f"  {entry['month']:9s} {entry['n_night_hours']:8d} "
              f"{entry['n_night_above_threshold']:6d} "
              f"{entry['pct_night_above_threshold']:6.1f}% "
              f"{entry['night_mean_mw']:10.3f} {entry['daylight_mean_mw']:9.1f} "
              f"{entry['generation_over_renewable_ratio']!s:>8s}")
    print(f"\n  seam: {seam}   as expected: {payload['seam_is_where_expected']}   "
          f"is the gate boundary: {payload['seam_is_the_gate_boundary']}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
