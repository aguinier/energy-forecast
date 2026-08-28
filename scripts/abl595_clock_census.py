#!/usr/bin/env python3
"""ABL-595 step 1: independently re-derive the vintage clock, read-only.

Does NOT trust the counts recorded on the issue. Counts vintages per model in
both databases over the post-fix cohort, and finds the actuals frontier per
zone. Nothing is written to either database.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.net_position import (  # noqa: E402
    FIX_DEPLOYED_UTC, GATE_COUNTRIES, GATE_EXCLUDED_COUNTRIES, _ro_connect,
)

REPLICA = r"C:\Code\able\data\energy_dashboard.db"
SIDECAR = r"C:\Code\able\data\forecasts_local.db"
MODELS = ["chronos-2-V010", "baseline-V012", "xgboost-V014", "chronos-2-V016"]
POST_FIX_START = "2026-08-07"


def census(path: str, label: str) -> pd.DataFrame:
    con = _ro_connect(path)
    try:
        df = pd.read_sql_query(
            """SELECT model_name, generated_at, country_code,
                      COUNT(*) AS rows
               FROM forecasts
               WHERE forecast_type = 'net_position'
               GROUP BY model_name, generated_at, country_code""", con)
    finally:
        con.close()
    df["db"] = label
    return df


def main() -> int:
    frames = [census(SIDECAR, "sidecar"), census(REPLICA, "replica")]
    allrows = pd.concat(frames, ignore_index=True)
    allrows["gen_ts"] = pd.to_datetime(allrows["generated_at"], format="mixed")
    allrows["gen_day"] = allrows["gen_ts"].dt.normalize()

    print("=== every model_name present, any vintage ===")
    print(allrows.groupby(["model_name", "db"])["rows"].sum().to_string())
    print()

    sub = allrows[allrows["model_name"].isin(MODELS)]
    print(f"=== vintages per model (union of both DBs), cohort_split={FIX_DEPLOYED_UTC} ===")
    for m in MODELS:
        g = sub[sub["model_name"] == m]
        post = g[g["gen_ts"] >= FIX_DEPLOYED_UTC]
        win = g[(g["gen_ts"] >= pd.Timestamp(POST_FIX_START))
                & (g["gen_ts"] < pd.Timestamp("2026-08-27"))]
        vint_all = sorted(g["gen_ts"].unique())
        vint_post = sorted(post["gen_ts"].unique())
        vint_win = sorted(win["gen_ts"].unique())
        print(f"\n-- {m}")
        print(f"   stored vintages (all):        {len(vint_all):3d}  "
              f"{str(vint_all[0])[:16] if vint_all else '-'} .. "
              f"{str(vint_all[-1])[:16] if vint_all else '-'}")
        print(f"   post cohort_split:            {len(vint_post):3d}  "
              f"days={len({v.normalize() for v in vint_post})}")
        print(f"   generated 08-07..08-26 incl.: {len(vint_win):3d}  "
              f"days={len({v.normalize() for v in vint_win})}")
        print(f"   vintage list in window: {[str(v)[:16] for v in vint_win]}")
        zones = win.groupby(win['gen_ts'])['country_code'].nunique()
        if len(zones):
            print(f"   zones per vintage in window: min={zones.min()} max={zones.max()}")

    # actuals frontier
    con = _ro_connect(REPLICA)
    try:
        act = pd.read_sql_query(
            """SELECT country_code, MAX(timestamp_utc) AS newest, COUNT(*) AS n
               FROM net_position WHERE net_position_mw IS NOT NULL
               GROUP BY country_code""", con)
    finally:
        con.close()
    act["newest_ts"] = pd.to_datetime(act["newest"], format="mixed")
    print("\n=== net_position actuals frontier per zone ===")
    gated = act[act["country_code"].isin(GATE_COUNTRIES)]
    print(gated.sort_values("newest_ts")[["country_code", "newest", "n"]].to_string(index=False))
    print(f"\nglobal frontier: {act['newest_ts'].max()}")
    print(f"gated zones present: {gated['country_code'].nunique()} / {len(GATE_COUNTRIES)}")
    missing = sorted(set(GATE_COUNTRIES) - set(gated["country_code"]))
    print(f"gated zones with NO actuals at all: {missing}")
    others = sorted(set(act['country_code']) - set(GATE_COUNTRIES))
    print(f"non-gated zones in actuals: {others}")
    print(f"excluded by name: {sorted(GATE_EXCLUDED_COUNTRIES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
