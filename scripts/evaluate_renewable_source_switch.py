#!/usr/bin/env python3
"""ABL-321 -- before/after backtest for the individual-renewable training source.

One variable changes between the two arms: which table
`load_renewable_type_data` reads. Everything else -- feature builder, the eight
pre-registered vintage instants per target hour, the fit/gate split, the
algorithm, the baseline -- is held fixed, and is the same protocol ABL-195 and
ABL-253 registered.

    arm A ("before")  target + every lag/rolling feature from `energy_renewable`
    arm B ("after")   target + every lag/rolling feature from `energy_generation`

**Truth is not a free variable.** Scoring each arm against its own source table
would be circular: a model trained on zero-filled data would be scored against
the same zero-fill and look excellent. Every cell here is therefore scored
against *both* candidate truths, on the identical rows:

    truth=energy_generation   the NULL-preserving table, zero duplicate
                              instants, and the table ABL-188 used to
                              adjudicate `energy_renewable`'s zeros as wrong.
                              Primary.
    truth=energy_renewable    what `src/evaluation/scorecard.py` scores the
                              live models against today. Secondary, and
                              reported precisely so the switch cannot be
                              accused of grading its own homework.

A conclusion is only reported as robust where the two truths agree.

Scoring is restricted to **common rows**: (target, horizon band) pairs where
both arms produced a finite prediction and the truth is finite. Arm B loses
coverage for FR (`energy_generation` is missing 2026-06-30 23:45 -> 2026-07-22
14:15, ABL-318 section 3), and letting each arm score on its own row set would
different holdouts. The coverage loss is reported separately as its own
consequence of the switch rather than being absorbed into the metric.

Writes nothing but report files. No replica write, no sidecar write, no
serving change, no promotion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src import db
from src.baselines import aligned_point_baselines
from src.evaluation.scorecard import _ro_connect, horizon_band, score_predictions
from src.evaluation.wind_retrain import (
    INTENDED_N, PRIMARY_BANDS, build_vintage_frame, schedule_vintages,
)
from src.wind_features import RenewableFeatureBuilder

SOLAR_FEATURES = (
    "hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos",
    "day_sin", "day_cos", "month_sin", "month_cos",
    "target_value_lag_1d", "target_value_lag_7d", "target_value_lag_14d",
    "target_value_roll_24h_mean", "target_value_roll_24h_std",
    "target_value_roll_24h_min", "target_value_roll_24h_max",
    "target_value_roll_168h_mean", "target_value_roll_168h_std",
    "target_value_roll_168h_min", "target_value_roll_168h_max",
    "shortwave_radiation_wm2", "direct_radiation_wm2",
    "diffuse_radiation_wm2", "temperature_c",
)
WIND_FEATURES = SOLAR_FEATURES[:21] + (
    "wind_speed_100m_ms", "wind_speed_10m_ms", "temperature_c",
)
FEATURES = {"solar": SOLAR_FEATURES,
            "wind_onshore": WIND_FEATURES,
            "wind_offshore": WIND_FEATURES}

#: The country/stream pairs that already have a serving model -- the only ones
#: acceptance criterion 2 asks about. A switch that helps a country with no
#: model is a bonus; a switch that hurts one of these is disqualifying.
SERVED_PAIRS = (
    ("AT", "solar"), ("BE", "solar"), ("DE", "solar"), ("FR", "solar"),
    ("AT", "wind_onshore"), ("BE", "wind_onshore"),
    ("DE", "wind_onshore"), ("FR", "wind_onshore"),
    ("BE", "wind_offshore"), ("FR", "wind_offshore"),
)

ARMS = {"before": "energy_renewable", "after": "energy_generation"}

#: Raw truth column per stream, read straight from each table with no ABL-188
#: guard applied -- "what the table says the actual was", which is what a
#: scorecard would use.
TRUTH_COLUMNS = {"solar": "solar_mw", "wind_onshore": "wind_onshore_mw",
                 "wind_offshore": "wind_offshore_mw"}

#: A cell is called materially worse when arm B's WAPE exceeds arm A's by more
#: than this, relative. Set before any number was computed.
MATERIAL_PCT = 2.0


def _model(algorithm: str):
    params = config.get_default_params(algorithm)
    if algorithm == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**params), params
    if algorithm == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**params), params
    raise ValueError(f"unsupported algorithm: {algorithm}")


def _truth_series(replica: str, table: str, country: str, stream: str,
                  start, end) -> pd.Series:
    """Raw per-hour actual from `table`. Duplicate instants (which only
    `energy_renewable` has: 78,510 rows, 5,425 disagreeing) are collapsed by
    taking the last spelling, and the disagreement count is surfaced by the
    caller rather than being silently resolved."""
    con = _ro_connect(replica)
    try:
        frame = pd.read_sql_query(
            f"SELECT timestamp_utc, {TRUTH_COLUMNS[stream]} AS v FROM {table} "
            "WHERE country_code=? AND timestamp_utc>=? AND timestamp_utc<? "
            "AND data_quality='actual'",
            con, params=(country, str(start), str(end)),
        )
    finally:
        con.close()
    if frame.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    ts = pd.to_datetime(frame["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    frame = frame.assign(h=ts).dropna(subset=["v"])
    frame = frame[frame["h"].dt.minute == 0]
    series = frame.groupby("h")["v"].last().astype(float).sort_index()
    return series


def _duplicate_disagreements(replica: str, table: str, country: str, stream: str,
                             start, end) -> int:
    con = _ro_connect(replica)
    try:
        frame = pd.read_sql_query(
            f"SELECT timestamp_utc, {TRUTH_COLUMNS[stream]} AS v FROM {table} "
            "WHERE country_code=? AND timestamp_utc>=? AND timestamp_utc<?",
            con, params=(country, str(start), str(end)),
        )
    finally:
        con.close()
    if frame.empty:
        return 0
    ts = pd.to_datetime(frame["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    grouped = frame.assign(i=ts).groupby("i")["v"].nunique(dropna=False)
    return int((grouped > 1).sum())


def _fit_and_predict(country, stream, source, fit_start, gate_start, gate_end,
                     algorithm, features):
    """One arm, end to end. Returns (gate predictions frame, audit dict)."""
    builder = RenewableFeatureBuilder(
        country, stream, fit_start - pd.Timedelta(days=14), gate_end,
        actuals_source=source,
    )
    if builder._actuals.empty:
        return None, {"source": source, "empty_actuals": True,
                      "fit_rows": 0, "gate_rows": 0}

    fit_raw = build_vintage_frame(builder, fit_start, gate_start, features)
    required = ["actual", *features]
    ok = np.isfinite(fit_raw[required].to_numpy(dtype=float)).all(axis=1)
    fit = fit_raw.loc[ok]
    if len(fit) < 500:
        return None, {"source": source, "empty_actuals": False,
                      "fit_rows": int(len(fit)), "gate_rows": 0,
                      "too_few_fit_rows": True}

    model, params = _model(algorithm)
    model.fit(fit[list(features)], fit["actual"])

    gate_raw = build_vintage_frame(builder, gate_start, gate_end, features)
    feat_ok = np.isfinite(gate_raw[list(features)].to_numpy(dtype=float)).all(axis=1)
    gate = gate_raw.loc[feat_ok].copy()
    gate["prediction"] = model.predict(gate[list(features)]) if len(gate) else []
    gate = gate.dropna(subset=["horizon_band"])
    gate = (gate.sort_values(["target_ts", "horizon_band", "generated_at"])
                .drop_duplicates(["target_ts", "horizon_band"], keep="last"))
    audit = {
        "source": source, "empty_actuals": False,
        "fit_rows": int(len(fit)), "fit_rows_intended": int(len(fit_raw)),
        "fit_unique_targets": int(fit["target_ts"].nunique()),
        "gate_rows": int(len(gate)),
        "gate_unique_targets": int(gate["target_ts"].nunique()),
        "actuals_hours_in_span": int(builder._actuals.notna().sum()),
        "params": params,
    }
    return gate[["target_ts", "generated_at", "horizon_band", "prediction"]], audit


def _score_arm_pair(merged: pd.DataFrame, truth_col: str, bands) -> dict:
    """Score both arms plus the baseline on the identical finite rows."""
    cols = ["before", "after", "seasonal_naive", truth_col]
    sub = merged[merged["horizon_band"].isin(bands)] if bands else merged
    valid = np.isfinite(sub[cols].to_numpy(dtype=float)).all(axis=1)
    common = sub.loc[valid]
    if common.empty:
        return {"n": 0}
    out = {"n": int(len(common))}
    for name in ("before", "after", "seasonal_naive"):
        out[name] = score_predictions(common[truth_col], common[name])
    a, b = out["before"]["wape_pct"], out["after"]["wape_pct"]
    out["delta_wape_pct"] = None if (a is None or b is None) else round(b - a, 4)
    out["relative_change_pct"] = (
        None if (a in (None, 0) or b is None) else round(100.0 * (b - a) / a, 2)
    )
    for name in ("before", "after"):
        w, s = out[name]["wape_pct"], out["seasonal_naive"]["wape_pct"]
        out[f"{name}_skill_vs_d7_pct"] = (
            None if (w is None or s in (None, 0)) else round(100.0 * (1 - w / s), 2)
        )
    return out


def run(args) -> dict:
    replica = Path(args.replica_db).resolve()
    fit_start, gate_start, gate_end = map(
        pd.Timestamp, (args.fit_start, args.gate_start, args.gate_end))
    pairs = [p for p in SERVED_PAIRS
             if (not args.only_country or p[0] in args.only_country.split(","))
             and (not args.only_stream or p[1] in args.only_stream.split(","))]

    cells, per_pair, audits = [], [], []
    for country, stream in pairs:
        started = time.time()
        features = FEATURES[stream]
        algorithm = args.algorithm
        arms, arm_audit = {}, {}
        for arm, source in ARMS.items():
            frame, audit = _fit_and_predict(
                country, stream, source, fit_start, gate_start, gate_end,
                algorithm, features)
            arm_audit[arm] = audit
            arms[arm] = frame

        if arms["before"] is None or arms["after"] is None:
            audits.append({"country": country, "stream": stream,
                           "arms": arm_audit, "skipped": True,
                           "seconds": round(time.time() - started, 1)})
            print(f"  {country}/{stream}: SKIPPED "
                  f"(before={arm_audit['before']}, after={arm_audit['after']})")
            continue

        merged = arms["before"].rename(columns={"prediction": "before"}).merge(
            arms["after"].rename(columns={"prediction": "after"}),
            on=["target_ts", "generated_at", "horizon_band"], how="outer")

        truth = {}
        for label, table in (("truth_gen", "energy_generation"),
                             ("truth_ren", "energy_renewable")):
            series = _truth_series(replica, table, country, stream,
                                   gate_start - pd.Timedelta(days=14), gate_end)
            truth[label] = series
            merged[label] = pd.DatetimeIndex(merged["target_ts"]).map(series)

        # One baseline per truth definition, from that truth's own history --
        # a D-7 baseline scored against a different series than it was built
        # from is not the same baseline.
        for label in ("truth_gen", "truth_ren"):
            base = aligned_point_baselines(
                truth[label], pd.DatetimeIndex(merged["target_ts"]),
                pd.DatetimeIndex(merged["generated_at"]))
            merged[f"seasonal_naive_{label}"] = base["seasonal_naive"].to_numpy()

        for label in ("truth_gen", "truth_ren"):
            scoped = merged.rename(columns={f"seasonal_naive_{label}": "seasonal_naive"})
            for band in PRIMARY_BANDS:
                scores = _score_arm_pair(scoped, label, (band,))
                if scores["n"]:
                    cells.append({"country": country, "stream": stream,
                                  "truth": label, "horizon_band": band,
                                  "intended_n": INTENDED_N[band], **scores})
            agg = _score_arm_pair(scoped, label, PRIMARY_BANDS)
            if agg["n"]:
                per_pair.append({"country": country, "stream": stream,
                                 "truth": label, **agg})

        coverage = {
            "rows_before_only": int(merged["before"].notna().sum()
                                    - (merged["before"].notna() & merged["after"].notna()).sum()),
            "rows_after_only": int(merged["after"].notna().sum()
                                   - (merged["before"].notna() & merged["after"].notna()).sum()),
            "rows_both": int((merged["before"].notna() & merged["after"].notna()).sum()),
            "truth_gen_hours": int(merged["truth_gen"].notna().sum()),
            "truth_ren_hours": int(merged["truth_ren"].notna().sum()),
            "truth_disagreement_rows": int(
                (np.isfinite(merged[["truth_gen", "truth_ren"]].to_numpy(dtype=float)).all(axis=1)
                 & (merged["truth_gen"] - merged["truth_ren"]).abs().gt(1e-6)).sum()),
            "energy_renewable_duplicate_instants": _duplicate_disagreements(
                str(replica), "energy_renewable", country, stream, gate_start, gate_end),
            "energy_generation_duplicate_instants": _duplicate_disagreements(
                str(replica), "energy_generation", country, stream, gate_start, gate_end),
        }
        audits.append({"country": country, "stream": stream, "arms": arm_audit,
                       "coverage": coverage, "skipped": False,
                       "seconds": round(time.time() - started, 1)})
        gen = [c for c in per_pair if c["country"] == country
               and c["stream"] == stream and c["truth"] == "truth_gen"]
        if gen:
            print(f"  {country}/{stream}: n={gen[0]['n']} "
                  f"before={gen[0]['before']['wape_pct']:.2f}% "
                  f"after={gen[0]['after']['wape_pct']:.2f}% "
                  f"({gen[0]['relative_change_pct']:+.1f}% rel) "
                  f"[{audits[-1]['seconds']:.0f}s]")

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "replica_db": str(replica), "replica_bytes": replica.stat().st_size,
            "fit_window": {"start": str(fit_start), "end_exclusive": str(gate_start)},
            "gate_window": {"start": str(gate_start), "end_exclusive": str(gate_end)},
            "algorithm": args.algorithm, "arms": ARMS,
            "material_change_pct": MATERIAL_PCT,
            "vintages_per_target": len(schedule_vintages(gate_start)),
            "protocol": "ABL-195/ABL-253 serve-faithful vintage protocol, "
                        "unchanged; the only variable is the source table.",
            "scoring": "common rows only -- both arms finite and truth finite",
        },
        "cells": cells, "per_pair": per_pair, "audits": audits,
    }


def render_markdown(result: dict) -> str:
    meta = result["meta"]
    lines = [
        "# ABL-321 — before/after backtest: `energy_renewable` vs `energy_generation`", "",
        f"Generated: {meta['generated_at']}",
        f"Replica: `{meta['replica_db']}` ({meta['replica_bytes']:,} bytes), "
        "opened `mode=ro`, `uri=True`. No write of any kind.",
        f"Fit targets: {meta['fit_window']['start']} → {meta['fit_window']['end_exclusive']} (exclusive).",
        f"Out-of-sample scoring targets: {meta['gate_window']['start']} → "
        f"{meta['gate_window']['end_exclusive']} (exclusive). **Out-of-sample by target timestamp.**",
        f"Algorithm: {meta['algorithm']}, identical in both arms. "
        f"{meta['vintages_per_target']} pre-registered vintages per target hour.",
        f"Baseline: literal seasonal-naive D-7, rebuilt from each truth series.",
        f"Scoring: {meta['scoring']}.", "",
        "**Arms.** before = `energy_renewable`, after = `energy_generation`. "
        "The source table sets the training target *and* every lag and rolling "
        "feature; nothing else differs.", "",
        "## Per country/stream, D+2 primary bands (24-36h, 36-48h, 48-64h)", "",
    ]
    for truth, label in (("truth_gen", "energy_generation (primary)"),
                         ("truth_ren", "energy_renewable (what the live scorecard uses)")):
        rows = [r for r in result["per_pair"] if r["truth"] == truth]
        if not rows:
            continue
        lines += [f"### Scored against truth = `{label}`", "",
                  "| country | stream | n | before WAPE | after WAPE | Δ WAPE | relative | "
                  "D-7 WAPE | before skill | after skill | verdict |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|"]
        for r in sorted(rows, key=lambda x: (x["stream"], x["country"])):
            rel = r["relative_change_pct"]
            if rel is None:
                verdict = "not measured"
            elif rel < -MATERIAL_PCT:
                verdict = "**after better**"
            elif rel > MATERIAL_PCT:
                verdict = "**after WORSE**"
            else:
                verdict = "no material change"
            lines.append(
                f"| {r['country']} | {r['stream']} | {r['n']:,} | "
                f"{r['before']['wape_pct']:.2f}% | {r['after']['wape_pct']:.2f}% | "
                f"{r['delta_wape_pct']:+.2f} pp | {rel:+.1f}% | "
                f"{r['seasonal_naive']['wape_pct']:.2f}% | "
                f"{r['before_skill_vs_d7_pct']:+.1f}% | {r['after_skill_vs_d7_pct']:+.1f}% | "
                f"{verdict} |")
        lines.append("")

    lines += ["## Per horizon band", "",
              "| country | stream | truth | band | n | intended n | before WAPE | after WAPE | relative |",
              "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for c in sorted(result["cells"], key=lambda x: (x["stream"], x["country"], x["truth"], x["horizon_band"])):
        lines.append(
            f"| {c['country']} | {c['stream']} | `{c['truth'][6:]}` | {c['horizon_band']} | "
            f"{c['n']:,} | {c['intended_n']:,} | {c['before']['wape_pct']:.2f}% | "
            f"{c['after']['wape_pct']:.2f}% | {c['relative_change_pct']:+.1f}% |")

    lines += ["", "## Coverage and data audit", "",
              "| country | stream | rows both | before only | after only | truth hours (gen/ren) | "
              "truth disagreements | dup instants (ren/gen) | fit rows before/after |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in result["audits"]:
        if a.get("skipped"):
            lines.append(f"| {a['country']} | {a['stream']} | *skipped* | | | | | | |")
            continue
        cov = a["coverage"]
        lines.append(
            f"| {a['country']} | {a['stream']} | {cov['rows_both']:,} | "
            f"{cov['rows_before_only']:,} | {cov['rows_after_only']:,} | "
            f"{cov['truth_gen_hours']:,} / {cov['truth_ren_hours']:,} | "
            f"{cov['truth_disagreement_rows']:,} | "
            f"{cov['energy_renewable_duplicate_instants']:,} / "
            f"{cov['energy_generation_duplicate_instants']:,} | "
            f"{a['arms']['before']['fit_rows']:,} / {a['arms']['after']['fit_rows']:,} |")

    lines += [
        "", "## Caveats", "",
        "- One 30-day summer holdout. Out-of-sample by target timestamp, not year-round evidence.",
        "- FR `energy_generation` is missing 2026-06-30 23:45 → 2026-07-22 14:15 (518.5 h, "
        "ABL-318 §3, not covered by ABL-71/67/111/109). That eats the fit window's tail and "
        "the first 11.6 days of the scoring window for FR, so FR's `after` arm trains on less "
        "and scores on fewer rows. Common-row scoring keeps the comparison fair; the lost "
        "coverage is the separate finding.",
        "- ABL-67 is net-position-only; ABL-109/111 are load-only; ABL-71's known wrong-write "
        "modes are load and net position. None is a proof that solar/wind ingest is pristine.",
        "- TSO forecasts are not used here. They are revision-contaminated and cannot support promotion.",
        "- No production deploy, serving-registry change, model promotion, ingest change, "
        "dashboard change, replica write or sidecar write was performed.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica-db", default=str(config.DATABASE_PATH))
    parser.add_argument("--fit-start", default="2026-01-14")
    parser.add_argument("--gate-start", default="2026-07-11")
    parser.add_argument("--gate-end", default="2026-08-10")
    parser.add_argument("--algorithm", default="catboost")
    parser.add_argument("--only-country", default="")
    parser.add_argument("--only-stream", default="")
    parser.add_argument("--json-out", default="experiments/ABL321/results.json")
    parser.add_argument("--report-out", default="reports/abl_321_source_switch.md")
    args = parser.parse_args()

    print(f"replica: {args.replica_db}")
    print(f"source default in db.py: {db.RENEWABLE_TYPE_SOURCE_TABLE}")
    result = run(args)

    json_path, report_path = Path(args.json_out), Path(args.report_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    report_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"wrote {report_path} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
