#!/usr/bin/env python3
"""Bit-reproduce stored live net-position vintages and write the gate artifact.

This is input parity, not sidecar/prod output parity. Every offline runner is
given the vintage's serve-time observation/publication bounds, then compared to
the stored sidecar median forecast by country and target hour. Databases are
opened read-only; the only write is ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from scripts.forecast_challengers import RUNNERS, load_actuals
from scripts.reconstruct_v010_vintages import build_engine, forecast_one
from src.challengers.registry import spec_for
from src.challengers.v014 import load_model as load_v014_model
from src.evaluation.net_position import _parse_ts, as_of_for_vintage

MODELS = {
    "chronos-2-V010": "V010",
    "baseline-V012": "V012",
    "xgboost-V014": "V014",
    "chronos-2-V016": "V016",
}
EXCLUDED_COUNTRIES = {"LU", "GR"}
DEFAULT_OUT = (Path(__file__).parent.parent / "experiments" /
               "net_position_serve_faithful_attestations.json")


def ro_connect(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def stored_vintage(sidecar_db: str, model_name: str,
                   generated_at: str) -> pd.DataFrame:
    con = ro_connect(sidecar_db)
    try:
        df = pd.read_sql_query(
            """SELECT country_code, target_timestamp_utc, generated_at,
                      forecast_value
                 FROM forecasts
                WHERE forecast_type = 'net_position' AND model_name = ?
                  AND generated_at = ?""",
            con, params=(model_name, generated_at))
    finally:
        con.close()
    if df.empty:
        raise ValueError(f"no {model_name} vintage at {generated_at}")
    df["target_ts"] = _parse_ts(df["target_timestamp_utc"])
    return df[~df["country_code"].isin(EXCLUDED_COUNTRIES)].copy()


def compare(stored: pd.DataFrame, offline: pd.DataFrame,
            tolerance_mw: float) -> dict:
    keys = ["country_code", "target_ts"]
    if stored.duplicated(keys).any() or offline.duplicated(keys).any():
        raise ValueError("duplicate country/target keys in parity comparison")
    paired = stored.merge(offline[keys + ["forecast_value"]], on=keys,
                          how="outer", suffixes=("_stored", "_offline"),
                          indicator=True)
    paired["abs_delta_mw"] = np.abs(
        paired["forecast_value_stored"] - paired["forecast_value_offline"])
    per_country = {}
    for cc, group in paired.groupby("country_code"):
        matched = group[group["_merge"] == "both"]
        worst = (float(matched["abs_delta_mw"].max())
                 if not matched.empty else None)
        per_country[cc] = {
            "stored_rows": int(group["forecast_value_stored"].notna().sum()),
            "offline_rows": int(group["forecast_value_offline"].notna().sum()),
            "rows_compared": int(len(matched)),
            "max_abs_delta_mw": worst,
            "verified": (len(matched) == len(group) and worst is not None and
                         worst <= tolerance_mw),
        }
    matched = paired[paired["_merge"] == "both"]
    worst = float(matched["abs_delta_mw"].max()) if not matched.empty else None
    return {
        "verified": (len(matched) == len(paired) and worst is not None and
                     worst <= tolerance_mw and
                     all(row["verified"] for row in per_country.values())),
        "stored_rows": int(stored.shape[0]),
        "offline_rows": int(offline.shape[0]),
        "rows_compared": int(matched.shape[0]),
        "max_abs_delta_mw": worst,
        "per_country": per_country,
    }


def input_spec(model_name: str, models_dir: str) -> dict:
    common = {
        "observation_bound": "net_position target_timestamp_utc < publication_cutoff_exclusive_utc",
        "publication_bound": "issued inputs have publication/run time <= generated_at_utc",
    }
    if model_name == "chronos-2-V010":
        return {**common, "runner": "scripts/forecast_chronos2.py (V010)",
                "target": "net_position.net_position_mw, 672 hourly observations",
                "past_covariates": [
                    "weather__temperature_2m_k", "weather__wind_speed_100m_ms",
                    "weather__shortwave_radiation_wm2", "cal__hour",
                    "cal__dayofweek", "cal__month", "cal__is_holiday",
                    "tso__load_forecast", "da__price", "flow__total_export_mw",
                    "flow__total_import_mw", "flow__net_mw"],
                "future_covariates": [
                    "weather__temperature_2m_k", "weather__wind_speed_100m_ms",
                    "weather__shortwave_radiation_wm2", "cal__hour",
                    "cal__dayofweek", "cal__month", "cal__is_holiday"],
                "prediction": "50 hours (26-hour gap plus target day's final 24 hours)"}
    if model_name == "baseline-V012":
        return {**common, "runner": "scripts/forecast_challengers.py::run_v012",
                "inputs": ["net_position.net_position_mw"],
                "prediction": "mean of D-7 persistence and 28-day same-hour climatology"}
    if model_name == "xgboost-V014":
        feature_columns = load_v014_model(models_dir, "BE").feature_columns
        return {**common, "runner": "scripts/forecast_challengers.py::run_v014",
                "source_cutoffs": {
                    "net_position, energy_price, energy_load_forecast, energy_generation_forecast":
                        "target run day D 21:00 UTC",
                    "crossborder_flows": "target_timestamp_utc - 72 hours",
                    "weather_data": "target hour; data_quality=forecast and forecast_run_time <= generated_at_utc",
                },
                "feature_columns_BE": feature_columns,
                "note": "neighbour-prefixed columns vary by country; every artifact supplies its own feature_columns"}
    return {**common, "runner": "scripts/forecast_challengers.py::run_v016",
            "inputs": ["stored co-run chronos-2-V010 median and quantiles",
                       "experiments/V016/correction.json",
                       "latest observable net_position residual strictly before the publication cutoff"],
            "prediction": "per-country affine map plus horizon-decayed AR(1); identity where fit refuses"}


def challenger_offline(experiment: str, stored: pd.DataFrame, target_date: str,
                       generated_at: pd.Timestamp, replica_db: str,
                       sidecar_db: str, models_dir: str,
                       actuals: dict[str, pd.Series]) -> pd.DataFrame:
    spec = spec_for(experiment)
    countries = sorted(stored["country_code"].unique())
    rows, _ = RUNNERS[experiment](
        spec, countries, target_date, generated_at, actuals,
        sidecar_db=sidecar_db, replica_db=replica_db, models_dir=models_dir)
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["country_code", "target_ts", "forecast_value"])
    return out[["country_code", "target_ts", "forecast_value"]]


def champion_offline(stored: pd.DataFrame, target_date: str,
                     generated_at: pd.Timestamp, device: str) -> pd.DataFrame:
    engine, builder = build_engine("V010", device)
    as_of = as_of_for_vintage(generated_at)
    rows = []
    for cc in sorted(stored["country_code"].unique()):
        median, _, index = forecast_one(engine, builder, cc, target_date,
                                        as_of, generated_at)
        rows.extend({"country_code": cc, "target_ts": ts,
                     "forecast_value": float(value)}
                    for ts, value in zip(index, median))
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica-db", default=str(config.DATABASE_PATH))
    parser.add_argument("--sidecar-db", default=config.FORECAST_OUTPUT_DB)
    parser.add_argument("--models-dir", default=str(config.MODELS_DIR))
    parser.add_argument("--champion-vintage", required=True,
                        help="exact stored V010 generated_at")
    parser.add_argument("--challenger-vintage", required=True,
                        help="exact stored V012/V014/V016 generated_at")
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tolerance-mw", type=float, default=0.0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    if not args.sidecar_db:
        parser.error("--sidecar-db is required")

    actuals = load_actuals(args.replica_db)
    artifacts = {}
    for model_name, experiment in MODELS.items():
        vintage_text = (args.champion_vintage if experiment == "V010"
                        else args.challenger_vintage)
        stored = stored_vintage(args.sidecar_db, model_name, vintage_text)
        generated_at = pd.Timestamp(stored["generated_at"].iloc[0])
        if experiment == "V010":
            offline = champion_offline(stored, args.target_date,
                                       generated_at, args.device)
        else:
            offline = challenger_offline(
                experiment, stored, args.target_date, generated_at,
                args.replica_db, args.sidecar_db, args.models_dir, actuals)
        result = compare(stored, offline, args.tolerance_mw)
        result.update({
            "model_name": model_name,
            "experiment": experiment,
            "baseline": "stored as-served live sidecar vintage",
            "vintage": {
                "generated_at_utc": str(generated_at),
                "target_start_utc": f"{args.target_date} 00:00:00",
                "target_end_utc": f"{args.target_date} 23:00:00",
                "publication_cutoff_exclusive_utc": str(as_of_for_vintage(generated_at)),
            },
            "input_spec": input_spec(model_name, args.models_dir),
        })
        artifacts[model_name] = result
        print(f"{model_name}: verified={result['verified']} rows="
              f"{result['rows_compared']} max|delta|={result['max_abs_delta_mw']} MW")

    replica = Path(args.replica_db).resolve()
    sidecar = Path(args.sidecar_db).resolve()
    con = ro_connect(str(replica))
    try:
        replica_range = con.execute(
            "SELECT COUNT(*), MIN(timestamp_utc), MAX(timestamp_utc) FROM net_position"
        ).fetchone()
    finally:
        con.close()
    doc = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim": "offline serve-time input reproduction against stored live median forecasts",
        "not_claimed": "sidecar versus prod-pushed output-row equality",
        "tolerance_mw": args.tolerance_mw,
        "databases": {
            "replica": {"path": str(replica), "bytes": replica.stat().st_size,
                        "net_position_rows": replica_range[0],
                        "net_position_min_utc": replica_range[1],
                        "net_position_max_utc": replica_range[2], "access": "read-only URI"},
            "sidecar": {"path": str(sidecar), "bytes": sidecar.stat().st_size,
                        "access": "read-only URI for attestation"},
        },
        "contamination": {
            "ABL-67": "touches scope: GR fabricated zero actuals; GR excluded by the pre-registered gate",
            "ABL-71": "touches provenance risk: ingest fixes are undeployed; replica currency was recorded, but row correctness is not thereby certified",
            "ABL-111_ABL-109": "do not touch these net-position inputs: actual-load rows are not consumed",
        },
        "models": artifacts,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0 if all(row["verified"] for row in artifacts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
