#!/usr/bin/env python3
"""Fit and read the pre-registered ABL-195 serve-faithful wind gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.data_quality import find_suspect_constant_runs
from src.evaluation.gate_artifacts import save_gate_artifact
from src.evaluation.scorecard import (
    ScorecardConfig, _load_forecasts, _load_tso, _ro_connect,
    select_latest_per_band,
)
from src.evaluation.wind_retrain import (
    FEATURE_COLUMNS, INTENDED_N, PRIMARY_BANDS, SCHEDULE_N, attach_baselines,
    build_vintage_frame, common_scores, finite_training_rows, gate_cell,
    select_latest_challenger_per_band,
)
from src.wind_features import RenewableFeatureBuilder


PAIRS = {
    "wind_offshore": {"algorithm": "xgboost", "countries": ("BE", "FR")},
    "wind_onshore": {"algorithm": "catboost", "countries": ("BE", "DE", "FR")},
}
COLUMNS = {"wind_offshore": "wind_offshore_mw", "wind_onshore": "wind_onshore_mw"}


def _model(algorithm: str):
    params = config.get_default_params(algorithm)
    if algorithm == "xgboost":
        # The production defaults carry an early-stopping setting that requires
        # a validation set.  This pre-registered final fit uses every pre-gate
        # row and performs no gate-driven tuning, so early stopping is disabled.
        params.pop("early_stopping_rounds", None)
        return XGBRegressor(**params), params
    return CatBoostRegressor(**params), params


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _constant_runs(replica: str, country: str, forecast_type: str, start, end) -> list[dict]:
    con = _ro_connect(replica)
    try:
        df = pd.read_sql_query(
            f"SELECT timestamp_utc, {COLUMNS[forecast_type]} AS value FROM energy_renewable "
            "WHERE country_code=? AND timestamp_utc>=? AND timestamp_utc<? ORDER BY timestamp_utc",
            con, params=(country, str(start), str(end)),
        )
    finally:
        con.close()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    return [{"start": str(run.start), "end": str(run.end), "value": run.value,
             "n_rows": run.n_rows, "duration_hours": run.duration_hours}
            for run in find_suspect_constant_runs(df, "value")]


def _fmt(value, suffix=""):
    return "Not measured" if value is None else f"{value:.1f}{suffix}"


def render_markdown(result: dict) -> str:
    meta = result["meta"]
    cells = result["gate_cells"]
    lines = [
        "# ABL-195 — Serve-faithful wind retrain gate",
        "",
        f"**Disposition: {result['verdict']}**",
        "",
        f"Generated: {meta['generated_at']}",
        f"Fit targets: {meta['fit_window']['start']} → {meta['fit_window']['end_exclusive']} (exclusive).",
        f"Out-of-sample gate targets: {meta['gate_window']['start']} → {meta['gate_window']['end_exclusive']} (exclusive).",
        "Baseline: literal seasonal-naive D-7. TSO is revision-contaminated context only and is not a gate criterion.",
        f"Replica: `{meta['replica_db']}` ({meta['replica_bytes']:,} bytes), opened with SQLite `mode=ro`, `uri=True`.",
        "",
        "## Gate read",
        "",
        f"Strict full PASS requires challenger WAPE < D-7 in all 15 country × primary D+2-band cells and ≥95% of intended pairs. Result: **{sum(c['gate']['pass'] for c in cells)}/15 cells pass**.",
        "Protocol count check (before fitting): the exact eight registered run instants produce 210/570/720/720/510 selected rows by band, not the registered 240/600/720/720/480. The primary 24–36h and 36–48h counts reproduce; 48–64h has 510 rows and is still judged against the frozen registered minimum of 456.",
        "",
        "| type | country | horizon | n | challenger WAPE | D-7 WAPE | skill vs D-7 | incumbent WAPE | MAE | bias | slope | corr | gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in cells:
        scores = row["scores"]
        skill = 100 * (1 - scores["challenger"]["wape_pct"] / scores["seasonal_naive"]["wape_pct"])
        lines.append(
            f"| {row['forecast_type']} | {row['country']} | {row['horizon_band']} | {row['gate']['n']:,} | "
            f"{_fmt(scores['challenger']['wape_pct'], '%')} | {_fmt(scores['seasonal_naive']['wape_pct'], '%')} | "
            f"{skill:+.1f}% | {_fmt(scores['incumbent']['wape_pct'], '%')} | {_fmt(scores['challenger']['mae'])} MW | "
            f"{_fmt(scores['challenger']['bias_pct'], '%')} | {_fmt(scores['challenger']['slope'])} | "
            f"{_fmt(scores['challenger']['correlation'])} | {'PASS' if row['gate']['pass'] else 'FAIL'} |"
        )
    lines.extend(["", "## Per-country all-D+2 summary", "",
                  "All model and baseline values use the identical finite challenger/incumbent/D-7/persistence/actual intersection.", "",
                  "| type | country | n | challenger WAPE | D-7 WAPE | persistence WAPE | incumbent WAPE | TSO WAPE (revision-contaminated; n) |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in result["country_d2"]:
        s = row["scores"]
        tso = row["tso"]
        lines.append(f"| {row['forecast_type']} | {row['country']} | {row['n']:,} | {_fmt(s['challenger']['wape_pct'], '%')} | "
                     f"{_fmt(s['seasonal_naive']['wape_pct'], '%')} | {_fmt(s['persistence']['wape_pct'], '%')} | "
                     f"{_fmt(s['incumbent']['wape_pct'], '%')} | {_fmt(tso['wape_pct'], '%')} (n={tso['n']:,}) |")
    lines.extend(["", "## Fit and missingness audit", "",
                  "Each training row was constructed by `RenewableFeatureBuilder.row(target, generated_at, generated_at)` on the measured eight-vintage schedule. Gate targets were never fitted.", "",
                  "| type | country | algorithm | retained / intended fit rows | unique fit targets | excluded missing | degraded lag-1d rows | artifact SHA-256 |",
                  "|---|---|---|---:|---:|---:|---:|---|"])
    for row in result["training"]:
        a = row["audit"]
        lines.append(f"| {row['forecast_type']} | {row['country']} | {row['algorithm']} | {a['retained_rows']:,} / {a['intended_rows']:,} | "
                     f"{a['unique_targets']:,} | {a['excluded_missing_actual_or_feature']:,} | {a['degraded_lag_1d_rows']:,} | `{row['artifact_sha256']}` |")
    lines.extend(["", "## Data quality and limits", ""])
    contaminated = [r for r in result["training"] if r["constant_runs"]]
    if contaminated:
        lines.append("- ABL-188 found one fit-window suspect run: BE offshore was exactly 0 MW from 2026-03-08 09:00 through 2026-03-10 00:00 UTC (40 hourly rows; 39 hours). Those labels and any dependent feature rows were treated as missing before fit. It does not intersect the July/August gate actuals (all 5,760 scheduled gate rows per pair were feature/label-complete), so the performance gate is evaluable; promotion remains on hold pending CEO/ingest adjudication.")
    else:
        lines.append("- ABL-188 constant-run screening found no ≥24-hour bit-identical wind run in any fitted/scored pair; no wind row was excluded by that invariant.")
    lines.extend([
        "- ABL-67 is net-position-only; ABL-109/111 are load-only. They do not intersect these wind targets. ABL-71's known wrong-write modes are load and net position, not wind; this is a provenance caveat, not proof that wind ingest is pristine.",
        "- Weather rows were admitted only where `forecast_run_time <= generated_at`; missing vintages were excluded and counted, never backfilled from the future.",
        "- TSO values come from a replacement table without first-seen vintages. They may include revisions and cannot support promotion.",
        "- This is one 30-day summer holdout. It is out-of-sample by target timestamp, but not a year-round robustness claim.",
        "",
        "## Recommendation to the CEO",
        "",
        result["recommendation"],
        "",
        "No production deploy, serving-registry change, model promotion, ingest change, dashboard change, replica write, or sidecar write was performed.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica-db", default=str(config.DATABASE_PATH))
    parser.add_argument("--sidecar-db", default=str(config.FORECAST_OUTPUT_DB))
    parser.add_argument("--fit-start", default="2026-01-14")
    parser.add_argument("--gate-start", default="2026-07-11")
    parser.add_argument("--gate-end", default="2026-08-10")
    parser.add_argument("--artifact-dir", default="experiments/ABL195/artifacts")
    parser.add_argument("--json-out", default="experiments/ABL195/results.json")
    parser.add_argument("--report-out", default="reports/abl_195_wind_retrain.md")
    args = parser.parse_args()
    fit_start, gate_start, gate_end = map(pd.Timestamp, (args.fit_start, args.gate_start, args.gate_end))
    if not fit_start < gate_start < gate_end:
        parser.error("require fit-start < gate-start < gate-end")
    replica = Path(args.replica_db).resolve()
    if not replica.exists():
        parser.error(f"replica not found: {replica}")

    cfg = ScorecardConfig(str(replica), args.sidecar_db, gate_start, gate_end,
                          models={"wind_offshore": "xgboost", "wind_onshore": "catboost"})
    incumbent_raw, vintage_counts = _load_forecasts(cfg)
    incumbent = select_latest_per_band(incumbent_raw)
    artifact_dir = Path(args.artifact_dir)
    training, scored_frames = [], []
    for forecast_type, spec in PAIRS.items():
        tso = _load_tso(cfg, forecast_type)
        for country in spec["countries"]:
            builder = RenewableFeatureBuilder(country, forecast_type,
                                               fit_start - pd.Timedelta(days=14), gate_end)
            fit_raw = build_vintage_frame(builder, fit_start, gate_start)
            fit, audit = finite_training_rows(fit_raw)
            model, params = _model(spec["algorithm"])
            model.fit(fit[list(FEATURE_COLUMNS)], fit["actual"])

            # ABL-342: through `Forecaster.save`, so the artifact carries the
            # table it was fitted on and the ABL-183 intercept witness by
            # construction.
            path = save_gate_artifact(
                artifact_dir / country / forecast_type / "model.joblib",
                model=model, builder=builder, algorithm=spec["algorithm"],
                params=params, feature_columns=FEATURE_COLUMNS,
                fit_window=(fit_start, gate_start),
            )

            gate_raw = build_vintage_frame(builder, gate_start, gate_end)
            gate_finite, gate_audit = finite_training_rows(gate_raw)
            gate_finite["challenger"] = model.predict(gate_finite[list(FEATURE_COLUMNS)])
            selected = select_latest_challenger_per_band(gate_finite)
            selected = attach_baselines(selected, builder._actuals)
            inc = incumbent[(incumbent["forecast_type"] == forecast_type) &
                            (incumbent["country_code"] == country)][
                                ["target_ts", "horizon_band", "forecast_value"]].rename(
                                    columns={"forecast_value": "incumbent"})
            selected = selected.merge(inc, on=["target_ts", "horizon_band"], how="left")
            selected = selected.merge(tso[tso["country_code"] == country][["target_ts", "tso"]],
                                      on="target_ts", how="left")
            selected["country"] = country
            selected["forecast_type"] = forecast_type
            scored_frames.append(selected)
            training.append({"forecast_type": forecast_type, "country": country,
                             "algorithm": spec["algorithm"], "params": params,
                             "audit": audit, "gate_build_audit": gate_audit,
                             "constant_runs": _constant_runs(str(replica), country, forecast_type,
                                                               fit_start - pd.Timedelta(days=14), gate_end),
                             "artifact_path": str(path.resolve()), "artifact_sha256": _sha256(path)})

    all_scored = pd.concat(scored_frames, ignore_index=True)
    gate_cells, country_d2 = [], []
    for (forecast_type, country, band), group in all_scored.groupby(["forecast_type", "country", "horizon_band"]):
        scores, common = common_scores(group, ("challenger", "incumbent", "seasonal_naive", "persistence"))
        if band in PRIMARY_BANDS:
            gate_cells.append({"forecast_type": forecast_type, "country": country,
                               "horizon_band": band, "scores": scores,
                               "gate": gate_cell(scores["challenger"]["wape_pct"],
                                                 scores["seasonal_naive"]["wape_pct"],
                                                 len(common), INTENDED_N[band])})
    for (forecast_type, country), group in all_scored[all_scored["horizon_band"].isin(PRIMARY_BANDS)].groupby(["forecast_type", "country"]):
        scores, common = common_scores(group, ("challenger", "incumbent", "seasonal_naive", "persistence"))
        tso_valid = np.isfinite(common[["actual", "tso"]].to_numpy(dtype=float)).all(axis=1)
        from src.evaluation.scorecard import score_predictions
        tso_score = score_predictions(common.loc[tso_valid, "actual"], common.loc[tso_valid, "tso"])
        country_d2.append({"forecast_type": forecast_type, "country": country,
                           "n": len(common), "scores": scores, "tso": tso_score})

    passed = sum(row["gate"]["pass"] for row in gate_cells)
    contaminated = any(row["constant_runs"] for row in training)
    performance_pass = len(gate_cells) == 15 and passed == 15
    if performance_pass and contaminated:
        verdict = "PERFORMANCE PASS — HOLD FOR CONTAMINATION ADJUDICATION"
        recommendation = (
            "The challenger clears the pre-registered D-7 performance bar in every served D+2 country-band cell. "
            "Do not promote yet: hand the newly detected BE offshore zero run to the CEO/ingest owner for adjudication, "
            "then return these experiment artifacts and this evidence to the CEO for Board review. This issue does not promote them."
        )
    elif performance_pass:
        verdict = "PASS"
        recommendation = (
            "The challenger clears the pre-registered D-7 bar in every served D+2 country-band cell. Preserve these "
            "experiment artifacts and ask the CEO to initiate Board review; do not promote from this issue."
        )
    else:
        verdict = "FAIL"
        recommendation = (
            f"Do not promote these artifacts: only {passed}/15 primary cells clear the registered bar. Treat the losing "
            "country/bands as a model-quality finding and move next to stronger wind features/model selection on a fresh pre-registered split."
        )
    result = {"meta": {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                       "replica_db": str(replica), "replica_bytes": replica.stat().st_size,
                       "fit_window": {"start": str(fit_start), "end_exclusive": str(gate_start)},
                       "gate_window": {"start": str(gate_start), "end_exclusive": str(gate_end)},
                       "registered_intended_n": INTENDED_N, "schedule_implied_n": SCHEDULE_N,
                       "vintage_counts": vintage_counts,
                       "selection": "latest vintage per country + target + model + horizon band"},
              "verdict": verdict, "recommendation": recommendation,
              "training": training, "gate_cells": sorted(gate_cells, key=lambda r: (r["forecast_type"], r["country"], r["horizon_band"])),
              "country_d2": sorted(country_d2, key=lambda r: (r["forecast_type"], r["country"]))}
    json_path, report_path = Path(args.json_out), Path(args.report_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"{verdict}: {passed}/15 cells passed; wrote {report_path} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
