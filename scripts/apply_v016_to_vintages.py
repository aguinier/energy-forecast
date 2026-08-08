#!/usr/bin/env python3
"""Apply a fitted V016 correction across stored champion vintages (ABL-68).

Turns a champion vintage set (normally the reconstruction) into the V016 series
it would have produced, so `evaluate_net_position.py` can score V016 with the
pre-registered metrics on the held-out window and on the backtest weeks —
before V016 has accumulated any live shadow vintages of its own.

The AR(1) term uses only residuals the run could have observed: for each
vintage, the latest champion forecast/actual pair strictly before that vintage's
serve-faithful `as_of`. Reading the residual at the target hour itself would be
reading the answer.

Usage:
    python scripts/apply_v016_to_vintages.py \
        --pairs-db C:\\Code\\able\\data\\forecasts_recon.db \
        --out-db  C:\\Code\\able\\data\\forecasts_v016_recon.db
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.challengers.correction import (CountryCorrection, apply_correction,
                                        latest_residual)
from src.challengers.registry import CHAMPION_MODEL_NAME, spec_for
from src.evaluation.net_position import _parse_ts, _ro_connect, as_of_for_vintage

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_code TEXT NOT NULL, forecast_type TEXT NOT NULL,
    target_timestamp_utc TIMESTAMP NOT NULL, generated_at TIMESTAMP NOT NULL,
    horizon_hours INTEGER NOT NULL, forecast_value REAL NOT NULL,
    model_name TEXT NOT NULL, model_version TEXT,
    UNIQUE(country_code, forecast_type, target_timestamp_utc, horizon_hours,
           model_name, generated_at));
CREATE TABLE IF NOT EXISTS forecast_quantiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_code TEXT NOT NULL, forecast_type TEXT NOT NULL,
    target_timestamp_utc TIMESTAMP NOT NULL, generated_at TIMESTAMP NOT NULL,
    quantile REAL NOT NULL, forecast_value REAL NOT NULL, model_name TEXT NOT NULL,
    UNIQUE(country_code, forecast_type, target_timestamp_utc, quantile,
           model_name, generated_at));
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs-db", required=True)
    p.add_argument("--out-db", required=True)
    p.add_argument("--replica-db", default=str(config.DATABASE_PATH))
    p.add_argument("--fit", default=None, help="default: experiments/V016/correction.json")
    p.add_argument("--model-name", default=CHAMPION_MODEL_NAME)
    args = p.parse_args()

    fit_path = Path(args.fit) if args.fit else \
        config.EXPERIMENTS_DIR / "V016" / "correction.json"
    doc = json.loads(fit_path.read_text())
    fits = {cc: CountryCorrection(**c) for cc, c in doc["corrections"].items()}
    out_name = spec_for("V016").model_name

    con = _ro_connect(args.pairs_db)
    try:
        champ = pd.read_sql_query(
            """SELECT country_code, target_timestamp_utc, generated_at,
                      horizon_hours, forecast_value FROM forecasts
               WHERE forecast_type = 'net_position' AND model_name = ?""",
            con, params=(args.model_name,))
        quant = pd.read_sql_query(
            """SELECT country_code, target_timestamp_utc, generated_at, quantile,
                      forecast_value FROM forecast_quantiles
               WHERE forecast_type = 'net_position' AND model_name = ?""",
            con, params=(args.model_name,))
    finally:
        con.close()
    champ["target_ts"] = _parse_ts(champ["target_timestamp_utc"])
    champ["generated_at"] = _parse_ts(champ["generated_at"])

    con = _ro_connect(args.replica_db)
    try:
        act = pd.read_sql_query(
            """SELECT country_code, timestamp_utc, net_position_mw FROM net_position
               WHERE net_position_mw IS NOT NULL AND timestamp_utc >= '2025-11-01'""",
            con)
    finally:
        con.close()
    act["target_ts"] = _parse_ts(act["timestamp_utc"]).dt.floor("h")
    act = (act.sort_values("target_ts").groupby(["country_code", "target_ts"]).tail(1)
             [["country_code", "target_ts", "net_position_mw"]]
             .rename(columns={"net_position_mw": "actual"}))

    # The freshest champion view of each past hour, paired with its actual --
    # the residual history the AR term reads from.
    history = (champ.sort_values("generated_at")
                    .drop_duplicates(["country_code", "target_ts"], keep="last")
                    .merge(act, on=["country_code", "target_ts"], how="left"))

    if not quant.empty:
        quant["target_ts"] = _parse_ts(quant["target_timestamp_utc"])
        quant["generated_at"] = _parse_ts(quant["generated_at"])

    out = Path(args.out_db)
    if out.exists():
        out.unlink()
    sink = sqlite3.connect(out)
    sink.executescript(SCHEMA)

    rows, qrows, n_identity = [], [], 0
    for (cc, gen), g in champ.groupby(["country_code", "generated_at"]):
        fit = fits.get(cc)
        if fit is None:
            continue
        if fit.is_identity:
            n_identity += 1
        as_of = as_of_for_vintage(gen)
        hist = history[(history["country_code"] == cc) &
                       (history["target_ts"] < as_of)]
        resid, resid_ts = latest_residual(hist, as_of, fit)
        g = g.sort_values("target_ts")
        targets = pd.DatetimeIndex(g["target_ts"])
        corrected = apply_correction(g["forecast_value"].to_numpy(), targets,
                                     fit, resid, resid_ts)
        version = pd.Timestamp(gen).strftime("%Y%m%d_%H%M%S")
        rows += [(cc, "net_position", str(ts), str(gen), int(h), float(v),
                  out_name, version)
                 for ts, h, v in zip(targets, g["horizon_hours"], corrected)]

        if not quant.empty:
            qg = quant[(quant["country_code"] == cc) & (quant["generated_at"] == gen)]
            for level, qq in qg.groupby("quantile"):
                qq = qq.sort_values("target_ts")
                shifted = apply_correction(qq["forecast_value"].to_numpy(),
                                           pd.DatetimeIndex(qq["target_ts"]),
                                           fit, resid, resid_ts)
                qrows += [(cc, "net_position", str(ts), str(gen), float(level),
                           float(v), out_name)
                          for ts, v in zip(qq["target_ts"], shifted) if np.isfinite(v)]

    sink.executemany("""INSERT OR REPLACE INTO forecasts (country_code, forecast_type,
        target_timestamp_utc, generated_at, horizon_hours, forecast_value,
        model_name, model_version) VALUES (?,?,?,?,?,?,?,?)""", rows)
    sink.executemany("""INSERT OR REPLACE INTO forecast_quantiles (country_code,
        forecast_type, target_timestamp_utc, generated_at, quantile, forecast_value,
        model_name) VALUES (?,?,?,?,?,?,?)""", qrows)
    sink.commit()
    sink.close()

    print(f"wrote {len(rows):,} point rows and {len(qrows):,} quantile rows "
          f"as {out_name} -> {out}")
    print(f"  {n_identity:,} country-vintages passed through uncorrected "
          f"(V016 == V010 there)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
