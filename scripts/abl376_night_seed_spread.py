#!/usr/bin/env python3
"""ABL-376 §5: is the daylight move a rule effect, or a seed effect?

The gate A/B in `evaluate_solar_retrain.py --scope abl376` reads the fit rule at
**one seed**, and a one-seed solar gap is not a measurement. ABL-375 put a number
on why: a DE CatBoost solar fit moves 4.6-13.8% of its daylight MAE across seeds
alone, which is several times the effect this issue is trying to see. So the
daylight axis gets a spread, and the night axis gets one beside it.

The design
----------
Two arms differing in exactly one thing -- whether the fit dropped the night rows
`solar_features.exclude_impossible_night_rows` calls impossible -- fitted at each
of `SEEDS`, on frames built **once** per country and shared by every fit. Same
frame, same rows, same hyperparameters, one integer apart.

That makes the comparison *paired*: at each seed the two arms differ only in the
rule, so the difference is taken within a seed and the across-seed variance never
enters it. Two numbers come out of the sweep and they answer different questions:

- **the paired effect** -- mean over seeds of (treatment - control). What the
  rule does.
- **the unpaired null** -- every control-vs-control seed pair, |control_i -
  control_j|. What a *single-seed* gap looks like when nothing changed at all.
  This is the number that says whether a one-seed read could have been quoted.

Both arms are scored on **identical, unfiltered** gate rows. The rule is fit-side
only: a contaminated night actual still scores against the challenger, or the
night number would measure the filter rather than the model.

What this is not
----------------
Not a gate read -- the gate is `evaluate_solar_retrain.py`, this is the
diagnostic beside it. It scores the rows where actual and the features are
finite, not the four-column gate basis, so its `n` is its own and is reported;
what matters is that both arms see the same rows.

Not comparable to ABL-338's headline numbers either. That holdout fits on the
whole history with training-time features and scores a spring window; this fits
the registered window with the serve-faithful builder and scores the registered
summer gate window. Same two axes, different frames -- see §1 of the findings on
how far that alone moves a night row count.

Usage
-----
    .venv\\Scripts\\python.exe scripts/abl376_night_seed_spread.py \\
        --replica-db C:\\Code\\able\\data\\energy_dashboard.db
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402
from src import db  # noqa: E402
from src.evaluation.solar_retrain import (  # noqa: E402
    ALGORITHM, FEATURE_COLUMNS, PRIMARY_BANDS,
    build_vintage_frame, finite_training_rows, select_latest_challenger_per_band,
)
from src.solar_features import (  # noqa: E402
    IMPOSSIBLE_NIGHT_THRESHOLD_MW, NIGHT_ELEVATION_THRESHOLD_DEG,
    SOLAR_BANDS, exclude_impossible_night_rows, solar_bands,
)
from src.wind_features import RenewableFeatureBuilder  # noqa: E402

logger = logging.getLogger("abl376.seed_spread")

#: Frozen before the first fit, and deliberately **disjoint from 42** -- the seed
#: the registered gate read in §4. A spread anchored on the arm that produced the
#: headline is not a spread; these eight are a fresh draw the reported effect was
#: not selected on. Eight is what makes the 28-pair null below dense enough to
#: quote a maximum from, at a cost of 16 fits per country.
SEEDS = (101, 103, 107, 109, 113, 127, 131, 137)

#: name -> whether the fit drops physically impossible night rows.
ARMS = {"control": False, "night_fit": True}

COUNTRIES = ("FR", "DE", "BE")


def _band_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Point metrics for one band. WAPE only where the denominator means something."""
    n = int(len(actual))
    if n == 0:
        return {"n": 0}
    error = predicted - actual
    out = {
        "n": n,
        "mean_actual_mw": float(actual.mean()),
        "mae_mw": float(np.abs(error).mean()),
        "rmse_mw": float(np.sqrt((error ** 2).mean())),
        "bias_mw": float(error.mean()),
        "mean_pred_mw": float(predicted.mean()),
        "max_pred_mw": float(predicted.max()),
        "min_pred_mw": float(predicted.min()),
        "n_negative_pred": int((predicted < 0).sum()),
    }
    # A band whose actuals are ~0 has no meaningful relative error: WAPE there
    # divides by nothing and reads as a huge percentage that says only that the
    # denominator is small. Night is exactly that band, so it gets MW only.
    total_actual = float(np.abs(actual).sum())
    if total_actual > 0 and actual.mean() > 1.0:
        out["wape_pct"] = 100.0 * float(np.abs(error).sum()) / total_actual
    return out


def _fit_predict(fit: pd.DataFrame, gate_x: pd.DataFrame, seed: int) -> np.ndarray:
    """One fit at one seed. Everything but `random_seed` is the gate's own config."""
    params = dict(config.get_default_params(ALGORITHM))
    params["random_seed"] = seed
    model = CatBoostRegressor(**params)
    model.fit(fit[list(FEATURE_COLUMNS)], fit["actual"])
    return np.asarray(model.predict(gate_x), dtype=float)


def sweep_country(country: str, replica: str, source: str,
                  fit_start: pd.Timestamp, gate_start: pd.Timestamp,
                  gate_end: pd.Timestamp, seeds: tuple) -> dict:
    """Build once, fit `2 * len(seeds)` times, score every fit on the same rows."""
    builder = RenewableFeatureBuilder(country, "solar", fit_start - pd.Timedelta(days=14),
                                      gate_end, actuals_source=source, db_path=replica)

    started = time.monotonic()
    fit_raw = build_vintage_frame(builder, fit_start, gate_start, FEATURE_COLUMNS)
    fit_all, fit_audit = finite_training_rows(fit_raw, FEATURE_COLUMNS)
    fit_clean, night_audit = exclude_impossible_night_rows(fit_all, country)

    gate_raw = build_vintage_frame(builder, gate_start, gate_end, FEATURE_COLUMNS)
    gate_finite, gate_audit = finite_training_rows(gate_raw, FEATURE_COLUMNS)
    # The same selection the gate makes, and it depends only on the schedule --
    # not on any prediction -- so it is taken once and every arm scores the rows
    # it picks. Unfiltered, on purpose: the rule is fit-side only.
    selected = select_latest_challenger_per_band(gate_finite)
    selected = selected[selected["horizon_band"].isin(PRIMARY_BANDS)].reset_index(drop=True)
    logger.info("%s: built %d fit rows, %d scored gate rows in %.1f min",
                country, len(fit_all), len(selected), (time.monotonic() - started) / 60)

    gate_x = selected[list(FEATURE_COLUMNS)]
    actual = selected["actual"].to_numpy(dtype=float)
    bands = solar_bands(country, selected["target_ts"]).to_numpy()
    frames = {"control": fit_all, "night_fit": fit_clean}

    runs = []
    for seed in seeds:
        for arm, drops in ARMS.items():
            began = time.monotonic()
            predicted = _fit_predict(frames[arm], gate_x, seed)
            runs.append({
                "arm": arm, "seed": seed, "drops_impossible_night": drops,
                "n_fit_rows": int(len(frames[arm])),
                "fit_seconds": round(time.monotonic() - began, 1),
                "bands": {b: _band_metrics(actual[bands == b], predicted[bands == b])
                          for b in SOLAR_BANDS},
                "all": _band_metrics(actual, predicted),
            })
            logger.info("%s seed=%d arm=%-9s daylight MAE %.1f MW, night mean pred %.2f MW (%.0fs)",
                        country, seed, arm, runs[-1]["bands"]["daylight"]["mae_mw"],
                        runs[-1]["bands"]["night"]["mean_pred_mw"], runs[-1]["fit_seconds"])

    return {
        "country": country,
        "fit_audit": fit_audit,
        "gate_build_audit": gate_audit,
        "night_fit_audit": night_audit,
        "scored_rows": int(len(selected)),
        "scored_band_n": {b: int((bands == b).sum()) for b in SOLAR_BANDS},
        "runs": runs,
    }


def _paired(result: dict, band: str, metric: str) -> dict:
    """Treatment - control at each seed, then summarised. Pairing is the point.

    Both arms saw the same rows at the same seed, so the across-seed variance
    that swamps an unpaired read cancels inside each difference.
    """
    by_arm = {(r["arm"], r["seed"]): r["bands"][band].get(metric) for r in result["runs"]}
    seeds = sorted({r["seed"] for r in result["runs"]})
    control = np.array([by_arm[("control", s)] for s in seeds], dtype=float)
    treatment = np.array([by_arm[("night_fit", s)] for s in seeds], dtype=float)
    difference = treatment - control

    # The null: what a gap between two *single-seed* reads of the same arm looks
    # like. Relative to the control mean, so it is comparable to the effect.
    scale = float(np.mean(np.abs(control))) or 1.0
    null = np.array([abs(a - b) for a, b in combinations(control, 2)], dtype=float)

    # A one-seed run has no spread and no pairs. It is a probe rather than the
    # registered read, and it reports `null_*` as None rather than crashing or,
    # worse, printing a zero that reads as "no seed sensitivity here".
    spread = {
        "control_sd": float(control.std(ddof=1)) if len(control) > 1 else None,
        "treatment_sd": float(treatment.std(ddof=1)) if len(treatment) > 1 else None,
        "paired_sd": float(difference.std(ddof=1)) if len(difference) > 1 else None,
        "null_max": float(null.max()) if null.size else None,
        "null_max_pct": 100.0 * float(null.max()) / scale if null.size else None,
        "null_mean_pct": 100.0 * float(null.mean()) / scale if null.size else None,
        "null_pairs": int(null.size),
    }

    return {
        "band": band, "metric": metric, "seeds": seeds,
        "control": [float(v) for v in control],
        "treatment": [float(v) for v in treatment],
        "control_mean": float(control.mean()),
        "treatment_mean": float(treatment.mean()),
        "paired_difference": [float(v) for v in difference],
        "paired_mean": float(difference.mean()),
        "paired_mean_pct": 100.0 * float(difference.mean()) / scale,
        "seeds_improved": int((difference < 0).sum()), "n_seeds": len(seeds),
        **spread,
    }


def summarise(payload: dict) -> dict:
    """The two axes, per country: daylight MAE against its null, and night level."""
    return {
        result["country"]: {
            "daylight_mae": _paired(result, "daylight", "mae_mw"),
            "shoulder_mae": _paired(result, "shoulder", "mae_mw"),
            "night_mean_pred": _paired(result, "night", "mean_pred_mw"),
            "night_max_pred": _paired(result, "night", "max_pred_mw"),
        }
        for result in payload["countries"]
    }


def _fmt(value, spec: str) -> str:
    """Format a number, or say `n/a` — a probe run has no spread to report."""
    return "n/a" if value is None else format(value, spec)


def _render_markdown(payload: dict) -> str:
    meta, summary = payload["meta"], payload["summary"]
    lines = [
        "# ABL-376 §5 — the fit rule over a seed spread", "",
        f"Generated: {meta['generated_at']}.",
        f"Seeds: `{', '.join(str(s) for s in meta['seeds'])}` — frozen in "
        "`scripts/abl376_night_seed_spread.py` before the first fit, and disjoint from the "
        "gate's seed 42.",
        f"Fit targets {meta['fit_window']['start']} → {meta['fit_window']['end_exclusive']} "
        f"(exclusive); scored on the registered gate window "
        f"{meta['gate_window']['start']} → {meta['gate_window']['end_exclusive']} (exclusive), "
        "out-of-sample by target timestamp.",
        f"Replica `{meta['replica_db']}` ({meta['replica_bytes']:,} bytes), source table "
        f"`{meta['training_source']}`, opened read-only.",
        f"Night is `solar_geometry.is_night_hour` (sun below {meta['night_threshold_deg']:g} deg "
        f"for the whole hour); the fit drops night rows above "
        f"{meta['impossible_night_threshold_mw']:g} MW and **the score drops nothing**.", "",
        "## Night level — the result", "",
        "Mean challenger prediction over the gate's night hours, MW. Both arms scored on "
        "identical unfiltered rows.", "",
        "| country | night rows | control (mean ± sd) | night-fit (mean ± sd) | paired change | seeds moved down |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for country, axes in summary.items():
        night = axes["night_mean_pred"]
        n = next(r["scored_band_n"]["night"] for r in payload["countries"] if r["country"] == country)
        lines.append(
            f"| {country} | {n:,} | {night['control_mean']:.2f} ± {_fmt(night['control_sd'], '.2f')} | "
            f"{night['treatment_mean']:.2f} ± {_fmt(night['treatment_sd'], '.2f')} | "
            f"{night['paired_mean']:+.2f} MW | {night['seeds_improved']}/{night['n_seeds']} |"
        )
    lines += [
        "", "## Daylight MAE — the effect against its own null", "",
        "`paired change` is the mean of (night-fit − control) taken **within** each seed. "
        "`single-seed null` is the largest gap between two control fits that differ only by "
        "seed — what a one-seed read could have reported with nothing changed at all.", "",
        "| country | daylight rows | control MAE | paired change | as % | single-seed null (max) | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for country, axes in summary.items():
        day = axes["daylight_mae"]
        n = next(r["scored_band_n"]["daylight"] for r in payload["countries"] if r["country"] == country)
        if day["null_max"] is None:
            verdict = "no null (single seed)"
        elif abs(day["paired_mean"]) <= day["null_max"]:
            verdict = "inside the null"
        else:
            verdict = "outside the null"
        lines.append(
            f"| {country} | {n:,} | {day['control_mean']:,.1f} MW | "
            f"{day['paired_mean']:+,.1f} MW | {day['paired_mean_pct']:+.2f}% | "
            f"{_fmt(day['null_max'], ',.1f')} MW ({_fmt(day['null_max_pct'], '.2f')}%) | {verdict} |"
        )
    lines += [
        "", "## What the rule removed from each fit", "",
        "| country | night fit rows | excluded rows | excluded hours | max excluded actual |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in payload["countries"]:
        audit = result["night_fit_audit"]
        max_mw = "n/a" if audit["max_excluded_mw"] is None else f"{audit['max_excluded_mw']:,.1f} MW"
        lines.append(
            f"| {result['country']} | {audit['night_rows']:,} | {audit['excluded_rows']:,} | "
            f"{audit['excluded_targets']:,} | {max_mw} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica-db", default=str(config.DATABASE_PATH))
    parser.add_argument("--fit-start", default="2026-01-14")
    parser.add_argument("--gate-start", default="2026-07-11")
    parser.add_argument("--gate-end", default="2026-08-10")
    parser.add_argument("--countries", default=",".join(COUNTRIES))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                        help="Comma-separated integer seeds. The default is the registered set; "
                             "overriding it makes the run a probe, not the registered read.")
    parser.add_argument("--renewable-source", default=None)
    parser.add_argument("--json-out", default="experiments/ABL376/seed_spread.json")
    parser.add_argument("--report-out", default="reports/abl_376_seed_spread.md")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    fit_start, gate_start, gate_end = map(pd.Timestamp, (args.fit_start, args.gate_start, args.gate_end))
    if not fit_start < gate_start < gate_end:
        parser.error("require fit-start < gate-start < gate-end")
    replica = Path(args.replica_db).resolve()
    if not replica.exists():
        parser.error(f"replica not found: {replica}")
    source = args.renewable_source or db.RENEWABLE_TYPE_SOURCE_TABLE
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "replica_db": str(replica), "replica_bytes": replica.stat().st_size,
            "training_source": source, "algorithm": ALGORITHM,
            "seeds": list(seeds), "registered_seeds": list(SEEDS),
            "seeds_are_registered": list(seeds) == list(SEEDS),
            "night_threshold_deg": NIGHT_ELEVATION_THRESHOLD_DEG,
            "impossible_night_threshold_mw": IMPOSSIBLE_NIGHT_THRESHOLD_MW,
            "fit_window": {"start": str(fit_start), "end_exclusive": str(gate_start)},
            "gate_window": {"start": str(gate_start), "end_exclusive": str(gate_end)},
        },
        "countries": [],
    }

    json_out, report_out = Path(args.json_out), Path(args.report_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    for country in countries:
        payload["countries"].append(
            sweep_country(country, str(replica), source, fit_start, gate_start, gate_end, seeds)
        )
        # Written after every country: the sweep is long enough that a run
        # interrupted at the third country should still leave the first two
        # readable rather than nothing at all.
        payload["summary"] = summarise(payload)
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        report_out.write_text(_render_markdown(payload), encoding="utf-8")
        logger.info("%s done; wrote %s", country, json_out)

    # The report is on disk in UTF-8; the console is not, and it is cp1252 here.
    # `evaluate_solar_retrain.py` prints its paths for the same reason.
    for country, axes in payload["summary"].items():
        day, night = axes["daylight_mae"], axes["night_mean_pred"]
        print(f"{country}: night mean pred {night['control_mean']:.2f} -> "
              f"{night['treatment_mean']:.2f} MW; daylight MAE {day['paired_mean']:+,.1f} MW "
              f"({day['paired_mean_pct']:+.2f}%) against a "
              f"{_fmt(day['null_max_pct'], '.2f')}% single-seed null")
    print(f"wrote {json_out} and {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
