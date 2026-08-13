#!/usr/bin/env python3
"""ABL-65 addendum: the correction ceiling, a maximum-power holdout, and RO.

Three questions the rolling backtest cannot answer on its own:

1. What would a *perfect* correction recover? (in-sample oracle — an upper bound)
2. Is the rolling result just "the window was too short"? (fit on 99 days,
   score on the next 99 — the most estimation power this data can give)
3. Does RO's swinging level survive the two-day gap? (the named sub-question)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.net_position import (EvalConfig, GATE_COUNTRIES, _parse_ts,
                                         _ro_connect, load_actuals)

REPLICA = "C:/Code/able/data/energy_dashboard.db"
RECON = "C:/Code/able/data/forecasts_recon.db"

con = _ro_connect(RECON)
fc = pd.read_sql_query(
    """SELECT country_code, target_timestamp_utc, generated_at, forecast_value
       FROM forecasts WHERE forecast_type='net_position' AND model_name='chronos-2-V010'""", con)
con.close()
fc["target_ts"] = _parse_ts(fc["target_timestamp_utc"])
fc["generated_at"] = _parse_ts(fc["generated_at"])
fc = fc.drop(columns=["target_timestamp_utc"])
fc = fc[fc["country_code"].isin(GATE_COUNTRIES)]

cfg = EvalConfig(replica_db=REPLICA, start="2026-01-01", end="2026-08-13")
act = load_actuals(cfg)
p = fc.merge(act.rename(columns={"ts": "target_ts"}),
             on=["country_code", "target_ts"], how="inner")
p["err"] = p["forecast_value"] - p["actual"]
p["hour"] = p["target_ts"].dt.hour
p["day"] = p["target_ts"].dt.normalize()

pd.set_option("display.width", 260)

# --- 1 & 2: in-sample oracle vs a 99-day-fit / 99-day-score holdout ----------
rows = []
for cc, g in p.groupby("country_code"):
    g = g.sort_values("target_ts")
    days = np.sort(g["day"].unique())
    cut = days[len(days) // 2]
    tr, te = g[g["day"] < cut], g[g["day"] >= cut]

    mae0_te = np.mean(np.abs(te["err"]))

    # ORACLE (in-sample on the test half itself — an upper bound, never a result)
    orc_off = np.mean(np.abs(te["err"] - te["err"].mean()))
    prof_te = te.groupby("hour")["err"].transform("mean")
    orc_diu = np.mean(np.abs(te["err"] - prof_te))

    # HOLDOUT: fit on 99 days, apply unchanged to the next 99
    off = tr["err"].mean()
    ho_off = np.mean(np.abs(te["err"] - off))
    prof_tr = tr.groupby("hour")["err"].mean()
    ho_diu = np.mean(np.abs(te["err"] - te["hour"].map(prof_tr).to_numpy()))

    rows.append({
        "cc": cc, "n_test": len(te), "mae_uncorr": mae0_te,
        "oracle_offset_%": 100 * (1 - orc_off / mae0_te),
        "oracle_diurnal_%": 100 * (1 - orc_diu / mae0_te),
        "holdout_offset_%": 100 * (1 - ho_off / mae0_te),
        "holdout_diurnal_%": 100 * (1 - ho_diu / mae0_te),
        "fit_offset_mw": off,
    })
o = pd.DataFrame(rows).set_index("cc")
print("=== Ceiling vs maximum-power holdout (99 fit days / 99 score days) ===")
print("oracle_* = fitted ON the scored data (in-sample upper bound, NOT a result)")
print(o.round(2).to_string())
print("\nmedian:", o[["oracle_offset_%", "oracle_diurnal_%",
                      "holdout_offset_%", "holdout_diurnal_%"]].median().round(2).to_dict())
print("countries where the holdout offset helps:",
      sorted(o.index[o["holdout_offset_%"] > 0].tolist()))
print("countries where the holdout diurnal helps:",
      sorted(o.index[o["holdout_diurnal_%"] > 0].tolist()))

# --- 3: RO, the named sub-question -----------------------------------------
print("\n=== RO: day-level bias, and whether the two-day gap preserves it ===")
ro = p[p["country_code"] == "RO"].copy()
d = ro.groupby("day")["err"].mean().sort_index()
print(f"198 recon days: mean {d.mean():.1f} MW, sd {d.std():.1f} MW, "
      f"mean|actual| {ro['actual'].abs().mean():.1f} MW")
for lag in (1, 2, 3, 7):
    pr = pd.concat([d, d.shift(lag)], axis=1).dropna()
    print(f"  day-bias corr at lag {lag}d: {pr.corr().iloc[0,1]:+.3f}  (n={len(pr)})")
print("\nWhat a lag-2 level carry would have done to RO, applied to every day:")
pair = pd.concat([d.rename("today"), d.shift(2).rename("two_days_ago")], axis=1).dropna()
for phi in (0.25, 0.5, 1.0):
    resid = pair["today"] - phi * pair["two_days_ago"]
    print(f"  phi={phi:.2f}: day-bias sd {pair['today'].std():.1f} -> {resid.std():.1f} MW "
          f"({100*(1-resid.std()/pair['today'].std()):+.1f}%)")

print("\n=== Same, fleet-wide: sd of day-level bias after a lag-2 carry ===")
rows = []
for cc, g in p.groupby("country_code"):
    d = g.groupby("day")["err"].mean().sort_index()
    pair = pd.concat([d.rename("t"), d.shift(2).rename("t2")], axis=1).dropna()
    r = {"cc": cc, "sd_mw": pair["t"].std()}
    for phi in (0.25, 0.5):
        r[f"sd_after_phi{phi}"] = (pair["t"] - phi * pair["t2"]).std()
        r[f"delta_phi{phi}_%"] = 100 * (1 - r[f"sd_after_phi{phi}"] / r["sd_mw"])
    rows.append(r)
lv = pd.DataFrame(rows).set_index("cc")
print(lv.round(2).to_string())
print("\nmedian delta at phi=0.25: %.2f%%  at phi=0.50: %.2f%%"
      % (lv["delta_phi0.25_%"].median(), lv["delta_phi0.5_%"].median()))
print("countries improved at phi=0.25:", sorted(lv.index[lv["delta_phi0.25_%"] > 0].tolist()))
