#!/usr/bin/env python3
"""Run the pre-registered ABL-187 price-correction holdout experiment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.evaluation.price_correction import evaluate_country
from src.evaluation.scorecard import (
    ScorecardConfig,
    _attach_evidence,
    _load_forecasts,
    score_predictions,
    select_latest_per_band,
)


def _fmt(value: float | None) -> str:
    return "Not measured" if value is None else f"{value:.1f}%"


def render_markdown(result: dict) -> str:
    meta = result["meta"]
    summary = result["summary"]
    lines = [
        "# ABL-187 — Price bias/affine correction holdout",
        "",
        f"Generated: {meta['generated_at']}",
        f"Fit window: {meta['fit_window']['start']} → {meta['fit_window']['end_exclusive']} (exclusive)",
        f"Holdout: {meta['holdout_window']['start']} → {meta['holdout_window']['end_exclusive']} (exclusive)",
        f"Protocol: {meta['selection']}; fit and holdout are disjoint by target timestamp.",
        f"Sample: {summary['pooled']['n']:,} out-of-sample rows across {meta['observed_country_count']} countries; "
        f"{summary['pooled']['n_fit']:,} training rows.",
        "Baseline: literal seasonal-naive D−7, scored on the exact finite holdout intersection used by all variants.",
        "",
        "## Verdict",
        "",
        summary["verdict"],
        "",
        f"Bias-only helped raw CatBoost in **{summary['bias_only']['countries_beating_raw']}/{meta['observed_country_count']}** "
        f"measured countries and beat D−7 in **{summary['bias_only']['countries_beating_naive']}/{meta['observed_country_count']}**. "
        f"Slope+intercept helped raw in **{summary['affine']['countries_beating_raw']}/{meta['observed_country_count']}** "
        f"and beat D−7 in **{summary['affine']['countries_beating_naive']}/{meta['observed_country_count']}**.",
        "",
        "## Holdout results by country",
        "",
        "All WAPE values below are out-of-sample. A negative Δ is an improvement over raw.",
        "",
        "| country | n fit | n holdout | raw CatBoost | bias-only | Δ vs raw | beats D−7? | affine | Δ vs raw | beats D−7? | seasonal-naive D−7 |",
        "|---|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|---:|",
    ]
    for row in result["by_country"]:
        lines.append(
            f"| {row['country']} | {row['n_fit']:,} | {row['n']:,} | {_fmt(row['raw']['wape_pct'])} | "
            f"{_fmt(row['bias_only']['wape_pct'])} | {row['bias_only']['delta_vs_raw_points']:+.1f} pp | "
            f"{'yes' if row['bias_only']['beats_naive'] else 'no'} | {_fmt(row['affine']['wape_pct'])} | "
            f"{row['affine']['delta_vs_raw_points']:+.1f} pp | "
            f"{'yes' if row['affine']['beats_naive'] else 'no'} | {_fmt(row['seasonal_naive']['wape_pct'])} |"
        )
    pooled = summary["pooled"]
    lines.extend([
        f"| **pooled** | **{pooled['n_fit']:,}** | **{pooled['n']:,}** | **{_fmt(pooled['raw']['wape_pct'])}** | "
        f"**{_fmt(pooled['bias_only']['wape_pct'])}** | **{pooled['bias_only']['delta_vs_raw_points']:+.1f} pp** | "
        f"**{'yes' if pooled['bias_only']['beats_naive'] else 'no'}** | **{_fmt(pooled['affine']['wape_pct'])}** | "
        f"**{pooled['affine']['delta_vs_raw_points']:+.1f} pp** | "
        f"**{'yes' if pooled['affine']['beats_naive'] else 'no'}** | **{_fmt(pooled['seasonal_naive']['wape_pct'])}** |",
        "",
        "## Country-count discrepancy",
        "",
        meta["country_count_note"],
        "",
        "## Fit parameters",
        "",
        "These parameters were estimated only on the fit window; no holdout outcome selected or changed them.",
        "",
        "| country | bias-only intercept | affine slope | affine intercept |",
        "|---|---:|---:|---:|",
    ])
    for row in result["by_country"]:
        lines.append(
            f"| {row['country']} | {row['fit']['bias_only']['intercept']:.3f} | "
            f"{row['fit']['affine']['slope']:.4f} | {row['fit']['affine']['intercept']:.3f} |"
        )
    lines.extend([
        "",
        "## Data integrity and limits",
        "",
        "- **ABL-71 touches the period operationally:** the then-undeployed price-window fix delayed fetching tomorrow's day-ahead price. It does not identify fabricated price values, and no price rows were excluded here. Actuals are latest-replica values, while the scored CatBoost forecasts are stored issued rows; first-seen price-source vintages are not archived, so source revision uplift cannot be measured.",
        "- **ABL-67 does not touch this result:** its 216 fabricated rows are confined to `net_position`; this experiment reads `energy_price` actuals and `price` forecasts.",
        "- **ABL-111 does not touch this result:** its zero-as-missing contamination is confined to `energy_load`; no load actual is used here.",
        "- The issued-weather archive starts 2026-01-11. This fit starts after that date and the holdout is in July/August, so this is not one of the W01–W10 weather-blind backtests. The underlying served model can still receive zero-filled covariates after the 6-hour forward-fill limit; this correction experiment neither repairs nor reconstructs those inputs.",
        "- This is one 30-day summer holdout, not a year-round backtest. Stored forecasts begin only on 2026-02-03 for CatBoost price.",
        "- Forecast rows are selected exactly as ABL-129: latest vintage per country + target + model + horizon band. Thus one target can contribute once per horizon band, matching the cited 34.3% comparison.",
        "",
        "## Recommendation to the CEO",
        "",
        summary["recommendation"],
        "",
        "No model artifact, serving registry, dashboard code, ingest code, production container, replica row, or sidecar row was changed.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica-db", default=str(config.DATABASE_PATH))
    parser.add_argument("--train-start", default="2026-02-03")
    parser.add_argument("--holdout-start", default="2026-07-11")
    parser.add_argument("--holdout-end", default="2026-08-10")
    parser.add_argument("--json-out", default="experiments/ABL187/results.json")
    parser.add_argument("--report-out", default="reports/abl_187_price_affine.md")
    args = parser.parse_args()

    train_start = pd.Timestamp(args.train_start)
    holdout_start = pd.Timestamp(args.holdout_start)
    holdout_end = pd.Timestamp(args.holdout_end)
    if not train_start < holdout_start < holdout_end:
        parser.error("require train-start < holdout-start < holdout-end")
    if not Path(args.replica_db).exists():
        parser.error(f"replica DB not found: {args.replica_db}")

    cfg = ScorecardConfig(
        replica_db=args.replica_db,
        sidecar_db=None,
        start=train_start,
        end=holdout_end,
        models={"price": "catboost"},
    )
    forecasts, vintage_counts = _load_forecasts(cfg)
    selected = select_latest_per_band(forecasts)
    evidence = _attach_evidence(cfg, selected, "price")
    evidence = evidence[evidence["actual"].notna()].copy()
    train = evidence[(evidence["target_ts"] >= train_start) & (evidence["target_ts"] < holdout_start)]
    holdout = evidence[(evidence["target_ts"] >= holdout_start) & (evidence["target_ts"] < holdout_end)]

    by_country = []
    pooled_frames = []
    for country in sorted(set(train["country_code"]) & set(holdout["country_code"])):
        scores, scored = evaluate_country(
            train[train["country_code"] == country],
            holdout[holdout["country_code"] == country],
        )
        by_country.append({"country": country, **scores})
        pooled_frames.append(scored)
    pooled_rows = pd.concat(pooled_frames, ignore_index=True)
    pooled = {
        "n_fit": sum(row["n_fit"] for row in by_country),
        "n": int(len(pooled_rows)),
        **{name: score_predictions(pooled_rows["actual"], pooled_rows[column])
           for name, column in (("raw", "forecast_value"), ("bias_only", "bias_only"),
                                ("affine", "affine"), ("seasonal_naive", "seasonal_naive"))},
    }
    for variant in ("bias_only", "affine"):
        pooled[variant]["delta_vs_raw_points"] = pooled[variant]["wape_pct"] - pooled["raw"]["wape_pct"]
        pooled[variant]["beats_raw"] = pooled[variant]["wape_pct"] < pooled["raw"]["wape_pct"]
        pooled[variant]["beats_naive"] = pooled[variant]["wape_pct"] < pooled["seasonal_naive"]["wape_pct"]

    observed = len(by_country)
    summaries = {}
    for variant in ("bias_only", "affine"):
        summaries[variant] = {
            "countries_beating_raw": sum(row[variant]["beats_raw"] for row in by_country),
            "countries_beating_naive": sum(row[variant]["beats_naive"] for row in by_country),
        }
    best_variant = min(("bias_only", "affine"), key=lambda name: pooled[name]["wape_pct"])
    best = pooled[best_variant]
    if best["beats_naive"]:
        verdict = f"The best correction ({best_variant}) beats D−7 on the pooled holdout."
        recommendation = "Do not promote from this experiment alone. Pre-register a year-round backtest and a serving/shadow gate, then return any promotion recommendation to the CEO and Board."
    else:
        verdict = (f"**Do not ship this correction.** The best corrected variant ({best_variant}) is "
                   f"{best['wape_pct'] - pooled['seasonal_naive']['wape_pct']:.1f} WAPE points worse than free D−7 on the pooled holdout.")
        recommendation = "Do not ship or promote either correction. Use seasonal-naive D−7 as the minimum model-development bar and move to a better price model/features; an affine layer has not cleared that bar out-of-sample."

    result = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "replica_db": str(Path(args.replica_db).resolve()),
            "replica_bytes": Path(args.replica_db).stat().st_size,
            "fit_window": {"start": str(train_start), "end_exclusive": str(holdout_start)},
            "holdout_window": {"start": str(holdout_start), "end_exclusive": str(holdout_end)},
            "selection": "latest vintage per country + target + model + horizon band",
            "model": "catboost",
            "observed_country_count": observed,
            "expected_country_count_in_issue": 21,
            "observed_countries": [row["country"] for row in by_country],
            "vintage_counts": vintage_counts,
            "country_count_note": (f"The issue asks for 21 countries, but the cited CatBoost score contains only {observed}: "
                                   f"{', '.join(row['country'] for row in by_country)}. The replica stores price forecasts for five additional countries as `xgboost`, not `catboost`; adding them would change the model under test. The denominator is therefore {observed}, and the 21-country premise does not reproduce."),
        },
        "summary": {"pooled": pooled, **summaries, "verdict": verdict, "recommendation": recommendation},
        "by_country": by_country,
    }

    json_path = Path(args.json_out)
    report_path = Path(args.report_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"wrote {report_path} and {json_path} ({pooled['n']:,} holdout rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
