#!/usr/bin/env python3
"""ABL-65 — measure whether a residual/bias correction layer helps the champion.

Design + offline measurement only. Nothing here writes to the shared database,
touches a serving path, or promotes anything: it produces the evidence pack the
issue asks for.

Two cohorts, for two different jobs:

* **`--cohort recon`** — the 198-day serve-faithful reconstruction
  (`forecasts_recon.db`, built by `reconstruct_v010_vintages.py` with the
  post-1c5a24f context). ~4,752 residual hours per country. This is where the
  *structure* is measured, because 166 pairs per country cannot separate a real
  correction from a fitted one.
* **`--cohort live`** — the live post-fix vintages the gate reads. This is where
  the *verdict* is taken, per the acceptance criteria, against the same
  serve-faithful persistence+climatology ensemble the report uses.

The reconstruction is a reconstruction: LT, RO and BG reproduce the as-served
2026-08-06 vintage 38.8%, 5.9% and 1.4% away from it, so their recon numbers
describe a model close to but not identical with what production ran. Reported,
never silently pooled — `--flag-unverified` names them in the output.

Usage:
    python scripts/abl65_correction_study.py --cohort recon --out reports/...
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.net_position import (EvalConfig, FIX_DEPLOYED_UTC, GATE_COUNTRIES,
                                         _parse_ts, _ro_connect, as_of_for_vintage,
                                         baseline_predictions, load_actuals,
                                         load_forecasts)
from src.evaluation.residual_correction import (SERVE_LEADS_H, backtest_corrections,
                                                default_specs, score_corrections)

# Reconstruction serve-parity, measured 2026-08-07 by
# `reconstruct_v010_vintages.py --verify 2026-08-06T06:00:44` (tolerance 1%).
RECON_UNVERIFIED = {"LT": 38.8, "RO": 5.9, "BG": 1.4}

ACF_LAGS = (1, 2, 3, 6, 12, 24, 27, 36, 48, 50, 72, 96, 120, 168)


def load_recon_pairs(recon_db: str, replica_db: str, model: str) -> pd.DataFrame:
    con = _ro_connect(recon_db)
    try:
        f = pd.read_sql_query(
            """SELECT country_code, target_timestamp_utc, generated_at,
                      horizon_hours, forecast_value
               FROM forecasts
               WHERE forecast_type = 'net_position' AND model_name = ?""",
            con, params=(model,))
    finally:
        con.close()
    if f.empty:
        raise SystemExit(f"no '{model}' net_position rows in {recon_db}")
    f["target_ts"] = _parse_ts(f["target_timestamp_utc"])
    f["generated_at"] = _parse_ts(f["generated_at"])
    return f.drop(columns=["target_timestamp_utc"])


def attach_actuals_and_baselines(fc: pd.DataFrame, cfg: EvalConfig) -> pd.DataFrame:
    """Join actuals and build the serve-faithful baselines, exactly as the eval does.

    Reusing `baseline_predictions` rather than reimplementing it is the point:
    the correction is judged against the same ensemble the gate reads, so a
    disagreement between this study and the report cannot come from two
    different baselines.
    """
    act = load_actuals(cfg)
    paired = fc.merge(act.rename(columns={"ts": "target_ts"}),
                      on=["country_code", "target_ts"], how="left")
    lookup = {c: g.set_index("ts")["actual"].sort_index()
              for c, g in act.groupby("country_code")}
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    chunks = {"persistence": [], "climatology": []}
    for (country, gen), g in paired.groupby(["country_code", "generated_at"]):
        preds = baseline_predictions(lookup.get(country, empty),
                                     as_of_for_vintage(gen),
                                     pd.DatetimeIndex(g["target_ts"]),
                                     cfg.climatology_days)
        for k in chunks:
            chunks[k].append(pd.Series(preds[k].to_numpy(), index=g.index))
    for k, parts in chunks.items():
        paired[k] = pd.concat(parts).sort_index()
    paired["baseline_ensemble"] = paired[["persistence", "climatology"]].mean(axis=1)
    return paired


def residual_acf(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-country autocorrelation of the operational residual chain.

    The decision this table drives: rows `L27`..`L50` are the only lags a D+2
    correction can read. `L1` is the number the issue's premise quotes, and it
    is measured between adjacent hours the run never observes together.
    """
    rows = []
    for cc, g in pairs.dropna(subset=["actual"]).groupby("country_code"):
        s = (g.assign(err=g["forecast_value"] - g["actual"])
              .sort_values(["generated_at", "target_ts"])
              .drop_duplicates("target_ts", keep="last")
              .set_index("target_ts")["err"].asfreq("h"))
        row = {"country_code": cc, "n_hours": int(s.notna().sum())}
        for lag in ACF_LAGS:
            pair = pd.concat([s, s.shift(lag)], axis=1).dropna()
            row[f"lag{lag}"] = (float(pair.corr().iloc[0, 1])
                                if len(pair) >= 48 else None)
        if row.get("lag1") is not None:
            lo, hi = SERVE_LEADS_H
            row["ar1_extrapolated_to_min_lead"] = float(row["lag1"] ** lo)
            row["ar1_extrapolated_to_max_lead"] = float(row["lag1"] ** hi)
        rows.append(row)
    return pd.DataFrame(rows).set_index("country_code")


def day_level_persistence(pairs: pd.DataFrame) -> pd.DataFrame:
    """Does a day's mean residual predict the day two days later?

    This is the level question RO raises, asked at the real gap. A per-vintage
    level correction is only possible if this correlation is non-zero; a static
    per-country constant is only useful if the day-level bias barely moves.
    """
    rows = []
    for cc, g in pairs.dropna(subset=["actual"]).groupby("country_code"):
        d = (g.assign(err=g["forecast_value"] - g["actual"],
                      day=g["target_ts"].dt.normalize())
              .groupby("day")["err"].mean().sort_index())
        pair2 = pd.concat([d, d.shift(2)], axis=1).dropna()
        pair1 = pd.concat([d, d.shift(1)], axis=1).dropna()
        rows.append({
            "country_code": cc,
            "n_days": int(len(d)),
            "day_bias_mean_mw": float(d.mean()),
            "day_bias_sd_mw": float(d.std()),
            "day_bias_corr_lag1": float(pair1.corr().iloc[0, 1]) if len(pair1) >= 10 else None,
            "day_bias_corr_lag2": float(pair2.corr().iloc[0, 1]) if len(pair2) >= 10 else None,
            "mean_abs_actual_mw": float(g["actual"].abs().mean()),
        })
    return pd.DataFrame(rows).set_index("country_code")


def hour_profile_stability(pairs: pd.DataFrame, split_frac: float = 0.5) -> pd.DataFrame:
    """Is the hour-of-day error profile the same in the first and second half?

    The diurnal share of MSE is the largest correctable term in the corrected
    decomposition, but it is fitted in-sample. A profile that does not reproduce
    out of sample is 24 fitted parameters, not a correction.
    """
    rows = []
    for cc, g in pairs.dropna(subset=["actual"]).groupby("country_code"):
        g = g.assign(err=g["forecast_value"] - g["actual"]).sort_values("target_ts")
        cut = g["target_ts"].quantile(split_frac)
        a = g[g["target_ts"] <= cut].groupby(g["target_ts"].dt.hour)["err"].mean()
        b = g[g["target_ts"] > cut].groupby(g["target_ts"].dt.hour)["err"].mean()
        common = a.index.intersection(b.index)
        if len(common) < 12:
            rows.append({"country_code": cc, "profile_corr_halves": None})
            continue
        rows.append({
            "country_code": cc,
            "profile_corr_halves": float(np.corrcoef(a[common], b[common])[0, 1]),
            "profile_sd_first_mw": float(a[common].std()),
            "profile_sd_second_mw": float(b[common].std()),
        })
    return pd.DataFrame(rows).set_index("country_code")


def run(args) -> dict:
    cfg = EvalConfig(replica_db=args.replica_db, sidecar_db=args.sidecar_db,
                     model_name=args.model, start=args.start, end=args.end)

    if args.cohort == "recon":
        fc = load_recon_pairs(args.recon_db, args.replica_db, args.model)
        score_from = None
        cohort_note = (f"reconstruction ({Path(args.recon_db).name}), "
                       f"post-1c5a24f context replayed over historical days")
    else:
        fc = load_forecasts(cfg)
        score_from = FIX_DEPLOYED_UTC if args.score_from is None else pd.Timestamp(args.score_from)
        cohort_note = "live as-served vintages (sidecar + prod-pushed)"

    fc = fc[fc["country_code"].isin(GATE_COUNTRIES)]
    pairs = attach_actuals_and_baselines(fc, cfg)
    scored_pairs = pairs.dropna(subset=["actual"])

    specs = default_specs()
    applied = backtest_corrections(pairs, specs, score_from=score_from)
    applied = applied.dropna(subset=["actual"])
    table = score_corrections(
        applied, baselines=scored_pairs[["country_code", "generated_at", "target_ts",
                                         "baseline_ensemble"]])

    diag_pairs = scored_pairs if score_from is None else \
        scored_pairs[scored_pairs["generated_at"] >= score_from]

    result = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cohort": args.cohort,
            "cohort_note": cohort_note,
            "model": args.model,
            "replica_db": args.replica_db,
            "serve_leads_h": list(SERVE_LEADS_H),
            "score_from_vintage": str(score_from) if score_from is not None else None,
            "vintages_scored": int(applied["generated_at"].nunique()) if len(applied) else 0,
            "vintage_days_scored": int(pd.to_datetime(applied["generated_at"])
                                       .dt.normalize().nunique()) if len(applied) else 0,
            "countries": sorted(applied["country_code"].unique().tolist()) if len(applied) else [],
            "pairs_per_spec": int(len(applied) / applied["spec"].nunique()) if len(applied) else 0,
            "actuals_max_ts": str(load_actuals(cfg)["ts"].max()),
            "recon_serve_parity_unverified_pct": RECON_UNVERIFIED if args.cohort == "recon" else None,
        },
        "specs": {s.name: {"kind": s.kind, "describe": s.describe()} for s in specs},
        "residual_acf": residual_acf(diag_pairs).reset_index().to_dict("records"),
        "day_level_persistence": day_level_persistence(diag_pairs).reset_index().to_dict("records"),
        "hour_profile_stability": hour_profile_stability(diag_pairs).reset_index().to_dict("records"),
        "per_country_spec": table.to_dict("records"),
    }

    # The headline the acceptance criteria asks for: how many countries each
    # shape beats the ensemble in, against the uncorrected model's own count.
    summary = []
    for spec, g in table.groupby("spec"):
        ok = g.dropna(subset=["skill_vs_ensemble_pct"])
        beat = int((ok["skill_vs_ensemble_pct"] > 0).sum())
        summary.append({
            "spec": spec,
            "countries_evaluable": int(len(ok)),
            "countries_beating_ensemble": beat,
            "median_skill_vs_ensemble_pct": float(ok["skill_vs_ensemble_pct"].median())
            if len(ok) else None,
            "median_mae_delta_vs_uncorrected_pct": float(g["mae_delta_pct"].median()),
            "countries_mae_improved": int((g["mae_delta_pct"] > 0).sum()),
            "mean_applied_frac": float(g["applied_frac"].mean()),
        })
    result["summary"] = sorted(summary, key=lambda r: -(r["median_mae_delta_vs_uncorrected_pct"] or -99))
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", choices=("recon", "live"), default="live")
    p.add_argument("--replica-db", default="C:/Code/able/data/energy_dashboard.db")
    p.add_argument("--sidecar-db", default="C:/Code/able/data/forecasts_local.db")
    p.add_argument("--recon-db", default="C:/Code/able/data/forecasts_recon.db")
    p.add_argument("--model", default="chronos-2-V010")
    p.add_argument("--start", default=None, help="target-window start (UTC)")
    p.add_argument("--end", default=None)
    p.add_argument("--score-from", default=None,
                   help="vintage generated_at floor for scoring (live: defaults to the fix)")
    p.add_argument("--out", default=None, help="write the JSON result here")
    args = p.parse_args()

    result = run(args)

    s = pd.DataFrame(result["summary"])
    pd.set_option("display.width", 220)
    print(f"\n=== ABL-65 correction study — cohort {args.cohort} ===")
    m = result["meta"]
    print(f"{m['vintages_scored']} vintages / {m['vintage_days_scored']} run-days, "
          f"{len(m['countries'])} countries, {m['pairs_per_spec']:,} pairs per shape")
    print(s.to_string(index=False))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
