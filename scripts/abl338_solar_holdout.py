"""ABL-338: does the solar fix cost daylight accuracy? A held-out A/B.

The CEO's stop condition on ABL-338 is "if daylight accuracy degrades, say so
and stop", and the night-hour error is guaranteed to look spectacular because
the incumbent was emitting garbage there. So this script scores **daylight,
shoulder and night hours separately** and never lets the night number into a
headline.

Four arms per country, differing in exactly one thing at a time:

  control            incumbent feature set (25 names), incumbent loss
  geometry           + ABL-338 solar geometry (27 names), incumbent loss
  geometry_tweedie   + geometry, Tweedie log-link loss (non-negativity)
  geometry_poisson   + geometry, Poisson log-link loss (non-negativity)

Every arm uses the live artifact's own algorithm and hyperparameters for that
country (AT xgboost, BE/DE/FR catboost), refitted on an identical truncated
window. **The control is a refit, not the live artifact**, and that is the
point: the live artifacts were fitted through roughly today, so scoring them on
any recent holdout would score them in-sample and flatter the incumbent into a
bar nothing could clear.

What this is not
----------------
Not a serve-faithful backtest. Features are built by the training-time pipeline
(`features.create_all_features`), whose lags and rolling windows are anchored at
the target hour -- at serving they are anchored at the generation instant
(`wind_features.RenewableFeatureBuilder`, ABL-183). Every arm carries that
identically, so the *comparison* is sound, but the absolute MW are optimistic
against what the rail would produce. The serve-faithful check is the clamp
counter in `scripts/abl338_retrain_solar.py`, which runs the real serve path.

The two ABL-338 geometry features are the exception and are the same number in
both paths by construction -- both call `solar_features.solar_geometry_frame`.

Usage
-----
    .venv\\Scripts\\python.exe scripts/abl338_solar_holdout.py \\
        --countries AT,BE,DE,FR --holdout 2026-04-14:2026-08-11 \\
        --out reports/abl_338_solar

ABL-375 reuses this script for a narrower question - which algorithm fits solar
better, with geometry on both sides - and added two flags for it. Both default to
today's behaviour, so a run without them produces the same arm keys as before:

``--arms``
    Fit only the named arms. ABL-375 needs `control` and `geometry` and nothing
    else: the log-link arms are a settled question (ABL-338 rejected them on DE,
    where CatBoost's Poisson collapsed to a constant 1.0 MW), and fitting all
    eight would spend four times the compute on arms no verdict reads.

``--seeds``
    Fit each arm once per seed, keyed ``arm@seed``. A cross-algorithm MAE gap is
    only a result if it is larger than the spread one arm shows against its own
    seed, and that spread is measurable rather than assumable - so ABL-375
    registers a seed set in advance and derives its noise floor from the observed
    spread instead of quoting a remembered percentage.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from src.db import load_training_data  # noqa: E402
from src.features import create_all_features, get_feature_columns  # noqa: E402
from src.forecaster import Forecaster  # noqa: E402
from src.solar_features import (  # noqa: E402
    NIGHT_ELEVATION_THRESHOLD_DEG,
    SOLAR_GEOMETRY_FEATURES,
    night_mask,
    solar_geometry_frame,
)

logger = logging.getLogger("energy_forecast")

#: `models/` is gitignored, so a worktree has none. The live artifacts live in
#: the primary checkout, which is also what the scheduled job serves from — the
#: arms take their algorithm and hyperparameters from there so "control" means
#: the incumbent's configuration rather than this repo's defaults.
LIVE_MODELS_DIR = Path(r"C:\Code\able\energy-forecast\models")

#: arm -> (use geometry features, nonneg objective, hyperparameter overrides).
#:
#: `*_deep` exists because of a measured failure mode rather than a hunch. A log
#: link starts the fit at `exp(0) = 1 MW` and has to climb ~10 in log space to
#: reach a 20-50 GW fleet; CatBoost's Poisson never made the climb at all (1 tree
#: kept, constant 1.0 MW output — see the report), and XGBoost's arms may be
#: under-converged for the same reason rather than genuinely worse at daylight.
#: Tripling the trees separates "this loss is wrong here" from "this fit had not
#: finished".
#: `daylight_fit*` is a hurdle structure rather than a loss change: the
#: regression is fitted on daylight and shoulder rows only, and night hours are
#: emitted as exact zero from the geometry. It is here because the two
#: requirements pull apart — the log link fixes night at a cost to daylight, and
#: a hurdle should fix night at *no* cost to daylight, since the fit stops
#: spending capacity on the ~40% of rows that are identically zero.
#:
#: It carries a caveat the loss arms do not, and the report says so: a model
#: that is zero at night by construction drives `forecast_clamp_log`'s night
#: counters to zero by construction too, which is the independent instrument the
#: CEO required. That makes it a decision for the CEO, not a selection this
#: script can make on MAE.
ARM_SPECS = {
    # name: (geometry features, nonneg objective, hyperparam overrides, fit daylight only)
    "control": (False, None, {}, False),
    "geometry": (True, None, {}, False),
    "geometry_tweedie": (True, "tweedie", {}, False),
    "geometry_poisson": (True, "poisson", {}, False),
    "geometry_tweedie_deep": (True, "tweedie", {"n_estimators": 1500, "iterations": 1500}, False),
    "daylight_fit": (True, None, {}, True),
    "daylight_fit_tweedie": (True, "tweedie", {}, True),
    "geometry_nightw100": (True, None, {}, False),
}

#: Sample weight applied to night rows, per arm. Squared error at these
#: magnitudes gives a night row almost no say: a 200 MW error at midnight
#: contributes 4e4 to the loss against 9e6 from a 3 GW error at noon, so the
#: ~40% of rows that are identically zero are worth well under 1% of the
#: objective and the ensemble has no reason to land on them exactly. Reweighting
#: buys that back without changing the loss family — so, unlike a log link, it
#: cannot re-weight the daylight hours against each other.
ARM_NIGHT_WEIGHT = {"geometry_nightw100": 100.0}

ARMS = tuple(ARM_SPECS)


def _legacy_feature_columns() -> list:
    """`get_feature_columns('solar')` minus the ABL-338 geometry pair.

    This used to be documented as "the 25 names every live solar artifact
    carries". Measured under ABL-375: it returns **29**. The four holiday
    features — `is_holiday`, `days_to_holiday`, `days_from_holiday`,
    `is_bridge_day` — are in the solar list and absent from all four serving
    artifacts, which were fitted before them.

    So the `control` arm is *this repo's current non-geometry solar feature set*,
    not the serving one, and no arm here is a stand-in for the live artifact's
    feature list. That is fine for an A/B — every arm carries the same 29 — but
    it means a result from this script cannot be phrased as "beats the serving
    artifact". Every committed run, ABL-338's included, used 29/31.
    """
    return [c for c in get_feature_columns("solar") if c not in SOLAR_GEOMETRY_FEATURES]


def _bands(country_code: str, timestamps: pd.Series) -> pd.Series:
    """Label each hour night / shoulder / daylight.

    `night` is the serving clamp's own predicate (sun below
    NIGHT_ELEVATION_THRESHOLD_DEG for the whole hour). `shoulder` is the band
    ABL-337 flagged as the clamp's blind spot: not dark enough to be zeroed, but
    the sun is still below the horizon at the hour's midpoint and the fleet
    should be at ~0. `daylight` is everything else.
    """
    geometry = solar_geometry_frame(country_code, timestamps)
    elevation = geometry["sun_elevation_deg"].to_numpy()
    night = night_mask(country_code, timestamps)
    labels = np.where(night, "night", np.where(elevation <= 0.0, "shoulder", "daylight"))
    return pd.Series(labels, index=pd.RangeIndex(len(labels)))


def _band_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Point metrics for one band. WAPE only where the denominator means something."""
    n = int(len(actual))
    if n == 0:
        return {"n": 0}
    error = predicted - actual
    total_actual = float(np.abs(actual).sum())
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
    if total_actual > 0 and actual.mean() > 1.0:
        out["wape_pct"] = 100.0 * float(np.abs(error).sum()) / total_actual
    return out


def _seasonal_naive(frame: pd.DataFrame) -> np.ndarray:
    """D-7 same-hour actual — the free baseline every metric here is quoted against."""
    return frame["target_value_lag_7d"].to_numpy(dtype=float)


#: Per-algorithm name for the seed knob. `--seeds` varies this and nothing else.
SEED_PARAM = {"xgboost": "random_state", "lightgbm": "random_state", "catboost": "random_seed"}


def evaluate_country(
    country_code: str,
    start_date: str,
    holdout_start: str,
    holdout_end: str,
    drop_impossible_night: bool,
    force_algorithm: str = None,
    arms: tuple = ARMS,
    seeds: tuple = (None,),
) -> dict:
    live_path = LIVE_MODELS_DIR / country_code / "solar" / "model.joblib"
    incumbent = Forecaster.load(country_code, "solar", path=str(live_path))
    algorithm = force_algorithm or incumbent.algorithm
    # Incumbent hyperparameters only transfer when the algorithm does; a forced
    # algorithm takes that algorithm's own defaults rather than a foreign dict.
    hyperparams = dict(incumbent.hyperparams) if force_algorithm is None else None
    training_source = incumbent.training_source
    logger.info(
        f"{country_code}: incumbent algorithm={algorithm} source={training_source} "
        f"version={incumbent.model_version}"
    )

    raw = load_training_data(
        country_code, "solar", start_date,
        (pd.Timestamp(holdout_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        source=training_source,
    )
    if raw.empty:
        raise ValueError(f"No solar training data for {country_code}")

    featured = create_all_features(raw, "solar", country_code=country_code)
    timestamps = pd.to_datetime(featured["timestamp_utc"])
    featured = featured.reset_index(drop=True)
    band = _bands(country_code, timestamps)
    featured["band"] = band.to_numpy()

    is_holdout = (timestamps >= pd.Timestamp(holdout_start)) & (
        timestamps <= pd.Timestamp(holdout_end) + pd.Timedelta(hours=23)
    )
    train_frame = featured.loc[~is_holdout.to_numpy()].reset_index(drop=True)
    holdout_frame = featured.loc[is_holdout.to_numpy()].reset_index(drop=True)

    # Physically impossible night actuals (ABL-337's FR finding: `energy_renewable`
    # carries 137-440 MW at sun elevations down to -65 deg). Counted always;
    # dropped from the fit only when asked, and never from the holdout — a
    # contaminated actual has to stay visible in the score or the night number
    # would be measured against a target nobody believes.
    night_rows = train_frame["band"] == "night"
    impossible = night_rows & (train_frame["target_value"] > 1.0)
    contamination = {
        "train_night_rows": int(night_rows.sum()),
        "train_night_rows_above_1mw": int(impossible.sum()),
        "train_night_max_mw": float(train_frame.loc[night_rows, "target_value"].max())
        if night_rows.any() else 0.0,
        "dropped_from_fit": bool(drop_impossible_night),
    }
    if drop_impossible_night and impossible.any():
        train_frame = train_frame.loc[~impossible.to_numpy()].reset_index(drop=True)
        logger.warning(
            f"{country_code}: dropped {int(impossible.sum())} physically impossible "
            f"night training rows (actual > 1 MW while the sun stays below "
            f"{NIGHT_ELEVATION_THRESHOLD_DEG} deg)"
        )

    negative_targets = int((train_frame["target_value"] < 0).sum())

    results = {
        "country_code": country_code,
        "algorithm": algorithm,
        "training_source": training_source,
        "incumbent_version": incumbent.model_version,
        "train_start": str(timestamps.min()),
        "train_end": str(pd.to_datetime(train_frame["timestamp_utc"]).max()),
        "n_train": int(len(train_frame)),
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "n_holdout": int(len(holdout_frame)),
        "negative_targets_in_train": negative_targets,
        "contamination": contamination,
        "bands": {b: int((holdout_frame["band"] == b).sum()) for b in ("daylight", "shoulder", "night")},
        "arms": {},
    }

    actual = holdout_frame["target_value"].to_numpy(dtype=float)

    # The free baseline, on exactly the holdout rows the arms are scored on.
    baseline_pred = _seasonal_naive(holdout_frame)
    results["baseline_seasonal_naive_d7"] = {
        b: _band_metrics(actual[(holdout_frame["band"] == b).to_numpy()],
                         baseline_pred[(holdout_frame["band"] == b).to_numpy()])
        for b in ("daylight", "shoulder", "night")
    }
    results["baseline_seasonal_naive_d7"]["all"] = _band_metrics(actual, baseline_pred)

    for arm, seed in ((a, s) for a in arms for s in seeds):
        use_geometry, nonneg, overrides, daylight_only = ARM_SPECS[arm]
        # One key per fit. `arm` alone when no seed set was asked for, so a run
        # without `--seeds` writes the same keys it always did.
        arm_key = arm if seed is None else f"{arm}@{seed}"
        feature_columns = _legacy_feature_columns()
        if use_geometry:
            feature_columns = feature_columns + list(SOLAR_GEOMETRY_FEATURES)
        feature_columns = [c for c in feature_columns if c in train_frame.columns]

        # `n_estimators` and `iterations` name the same knob in different
        # libraries; pass only the one this algorithm understands.
        arm_hyperparams = dict(hyperparams) if hyperparams else {}
        tree_count_key = "iterations" if algorithm == "catboost" else "n_estimators"
        for key, value in overrides.items():
            if key == tree_count_key:
                arm_hyperparams[key] = value
        if seed is not None:
            arm_hyperparams[SEED_PARAM[algorithm]] = seed

        forecaster = Forecaster(
            country_code, "solar", algorithm=algorithm,
            hyperparams=arm_hyperparams or None,
            training_source=training_source, nonneg_objective=nonneg,
        )
        forecaster.feature_columns = feature_columns

        # Same chronological early-stopping split `Forecaster.train` uses, on the
        # same real fit method — only the feature list differs between arms.
        val_size = config.VALIDATION_DAYS * 24
        fit_df = train_frame.iloc[:-val_size] if val_size < len(train_frame) else train_frame
        val_df = train_frame.iloc[-val_size:] if val_size < len(train_frame) else pd.DataFrame()

        # Hurdle: the regression never sees a night row, so it cannot spend
        # capacity on them and cannot be dragged by them. The split is taken
        # *after* the chronological cut so both arms hold out the same period.
        if daylight_only:
            fit_df = fit_df.loc[(fit_df["band"] != "night").to_numpy()]
            if not val_df.empty:
                val_df = val_df.loc[(val_df["band"] != "night").to_numpy()]

        y_fit = fit_df["target_value"]
        forecaster._assert_nonneg_target(y_fit)
        if not val_df.empty:
            forecaster._assert_nonneg_target(val_df["target_value"])

        night_weight = ARM_NIGHT_WEIGHT.get(arm)
        if night_weight is None:
            forecaster._train_simple(
                fit_df[feature_columns], y_fit,
                val_df[feature_columns] if not val_df.empty else None,
                val_df["target_value"] if not val_df.empty else None,
            )
        else:
            # Weighted fit, kept inside this script rather than in `Forecaster`:
            # sample weights are being measured here, not adopted, and the
            # production class should not grow a knob nothing serves.
            forecaster.model = forecaster._create_model()
            weights = np.where((fit_df["band"] == "night").to_numpy(), night_weight, 1.0)
            fit_kwargs = {"sample_weight": weights}
            if not val_df.empty:
                fit_kwargs["eval_set"] = [(val_df[feature_columns], val_df["target_value"])]
                if algorithm == "xgboost":
                    fit_kwargs["verbose"] = False
            forecaster.model.fit(fit_df[feature_columns], y_fit, **fit_kwargs)

        predicted = np.asarray(
            forecaster.model.predict(holdout_frame[feature_columns]), dtype=float
        )
        if daylight_only:
            # Exact zero at night, from the geometry rather than from the fit.
            predicted = np.where((holdout_frame["band"] == "night").to_numpy(), 0.0, predicted)

        model = forecaster.model
        # How many trees actually survived early stopping. A log-link arm that
        # collapses to a constant shows up here as ~0 trees, which distinguishes
        # "this loss is wrong for the data" from "this fit never got started".
        n_trees = getattr(model, "tree_count_", None) or getattr(model, "best_iteration", None)
        arm_result = {
            "arm": arm,
            "seed": seed,
            "n_features": len(feature_columns),
            "nonneg_objective": nonneg,
            "hyperparams_objective": forecaster.hyperparams.get("objective")
            or forecaster.hyperparams.get("loss_function"),
            "n_trees": int(n_trees) if n_trees is not None else None,
        }
        for b in ("daylight", "shoulder", "night"):
            in_band = (holdout_frame["band"] == b).to_numpy()
            arm_result[b] = _band_metrics(actual[in_band], predicted[in_band])
        arm_result["all"] = _band_metrics(actual, predicted)
        results["arms"][arm_key] = arm_result
        logger.info(
            f"{country_code}/{arm_key}: daylight MAE {arm_result['daylight']['mae_mw']:.1f} MW, "
            f"night mean pred {arm_result['night']['mean_pred_mw']:.2f} MW, "
            f"{arm_result['all']['n_negative_pred']} negative predictions"
        )

    return results


def _render_markdown(payload: dict) -> str:
    lines = [
        "# ABL-338 — solar non-negativity and solar geometry: held-out A/B",
        "",
        f"Generated {payload['generated_at']} against replica `{payload['replica_db']}`.",
        "",
        f"Holdout **{payload['holdout_start']} .. {payload['holdout_end']}**, "
        f"training from {payload['start_date']} up to the holdout start.",
        "",
        "Every arm is a **refit** on the identical truncated window — the live artifacts",
        "were fitted through roughly today, so scoring them here would be in-sample.",
        "Features come from the training-time pipeline, so these MW are optimistic",
        "against the serve path; the arms carry that identically. Night hours are",
        "reported in MW, never as a percentage: their denominator is ~0.",
        "",
    ]
    for country, result in payload["countries"].items():
        lines += [
            f"## {country} — {result['algorithm']}, source `{result['training_source']}`",
            "",
            f"n_train {result['n_train']:,} · n_holdout {result['n_holdout']:,} "
            f"(daylight {result['bands']['daylight']:,} / shoulder {result['bands']['shoulder']:,} "
            f"/ night {result['bands']['night']:,}) · incumbent version {result['incumbent_version']}",
            "",
            "| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        base = result["baseline_seasonal_naive_d7"]
        lines.append(
            f"| _seasonal-naive D-7_ | {base['daylight']['mae_mw']:,.1f} | "
            f"{base['daylight'].get('wape_pct', float('nan')):.1f}% | "
            f"{base['shoulder']['mae_mw']:,.1f} | {base['shoulder']['mean_pred_mw']:,.1f} | "
            f"{base['night']['mean_pred_mw']:,.2f} | {base['night']['max_pred_mw']:,.1f} | "
            f"{base['all']['n_negative_pred']} |"
        )
        # The arms this payload actually holds, in fit order — not the global
        # ARMS, which `--arms` and `--seeds` both make wrong.
        for arm in result["arms"]:
            a = result["arms"][arm]
            lines.append(
                f"| {arm} | {a['daylight']['mae_mw']:,.1f} | "
                f"{a['daylight'].get('wape_pct', float('nan')):.1f}% | "
                f"{a['shoulder']['mae_mw']:,.1f} | {a['shoulder']['mean_pred_mw']:,.1f} | "
                f"{a['night']['mean_pred_mw']:,.2f} | {a['night']['max_pred_mw']:,.1f} | "
                f"{a['all']['n_negative_pred']} |"
            )
        c = result["contamination"]
        lines += [
            "",
            f"Training-target contamination: {c['train_night_rows_above_1mw']:,} of "
            f"{c['train_night_rows']:,} night rows read above 1 MW "
            f"(max {c['train_night_max_mw']:,.1f} MW); dropped from fit: {c['dropped_from_fit']}.",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", default="AT,BE,DE,FR")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--holdout", required=True, help="START:END, both YYYY-MM-DD, inclusive")
    parser.add_argument("--drop-impossible-night", action="store_true",
                        help="Drop training rows whose actual exceeds 1 MW at an hour the "
                             "sun never clears the night threshold (ABL-337's FR finding)")
    parser.add_argument("--out", default="reports/abl_338_solar")
    parser.add_argument("--tag", default=None, help="Filename tag (default: the holdout window)")
    parser.add_argument("--force-algorithm", default=None,
                        choices=sorted(config.SUPPORTED_ALGORITHMS),
                        help="Refit every arm with this algorithm instead of the incumbent's")
    parser.add_argument("--arms", default=None,
                        help=f"Comma-separated subset of arms to fit (default: all). "
                             f"Known arms: {','.join(ARMS)}")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated integer seeds. Fits every arm once per seed, "
                             "keyed arm@seed, varying only random_state/random_seed. Omit for "
                             "one fit per arm at the configured seed.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    holdout_start, holdout_end = args.holdout.split(":")
    countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]

    arms = ARMS
    if args.arms:
        arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
        unknown = [a for a in arms if a not in ARM_SPECS]
        if unknown:
            parser.error(f"unknown arm(s): {unknown}. Known: {sorted(ARM_SPECS)}")
    seeds = (None,)
    if args.seeds:
        seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "replica_db": str(config.DATABASE_PATH),
        "start_date": args.start,
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "drop_impossible_night": args.drop_impossible_night,
        "force_algorithm": args.force_algorithm,
        "arms": list(arms),
        "seeds": list(seeds),
        "night_threshold_deg": NIGHT_ELEVATION_THRESHOLD_DEG,
        "countries": {},
    }
    for country in countries:
        payload["countries"][country] = evaluate_country(
            country, args.start, holdout_start, holdout_end, args.drop_impossible_night,
            force_algorithm=args.force_algorithm, arms=arms, seeds=seeds,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{holdout_start}_{holdout_end}"
    if args.force_algorithm:
        tag += f"_{args.force_algorithm}"
    if args.drop_impossible_night:
        tag += "_cleaned"
    (out_dir / f"holdout_{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = _render_markdown(payload)
    (out_dir / f"holdout_{tag}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nWrote {out_dir / f'holdout_{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
