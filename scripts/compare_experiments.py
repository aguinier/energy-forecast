#!/usr/bin/env python3
"""Compare experiment results across backtest weeks.

Runs backtests for specified experiments and generates comparison tables
with MAE, MAPE, RMSE, and skill scores.

Usage:
    # Compare XGBoost baseline vs Chronos-2 zero-shot on DE load
    python scripts/compare_experiments.py --experiments V001,V002 --weeks W01 --countries DE --types load

    # Full comparison across all backtest weeks
    python scripts/compare_experiments.py --experiments V001,V003 --weeks all --countries DE,FR,BE --types load,price,renewable

    # Quick single-week check
    python scripts/compare_experiments.py --experiments V001,V002 --weeks W01 --countries DE --types load --device cpu
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.db import load_energy_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("energy_forecast.compare")


def load_actuals(
    country_code: str,
    forecast_type: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Load actual values from DB, return hourly series."""
    # Handle net_position (not in the standard energy data tables)
    if forecast_type == "net_position":
        import sqlite3
        conn = sqlite3.connect(str(config.DATABASE_PATH))
        try:
            df = pd.read_sql_query(
                "SELECT timestamp_utc, net_position_mw as target_value FROM net_position WHERE country_code = ? AND timestamp_utc >= ? AND timestamp_utc < ? ORDER BY timestamp_utc",
                conn, params=(country_code, start_date, end_date)
            )
        finally:
            conn.close()
        if df.empty:
            return pd.Series(dtype=float)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
        return df.set_index("timestamp_utc")["target_value"].resample("h").mean()

    df = load_energy_data(country_code, forecast_type, start_date, end_date)
    if df.empty:
        return pd.Series(dtype=float)
    df = df.set_index("timestamp_utc")
    series = df["target_value"].resample("h").mean()
    return series


def compute_metrics(actuals: np.ndarray, forecasts: np.ndarray) -> dict:
    """Compute forecast quality metrics."""
    if len(actuals) == 0 or len(forecasts) == 0:
        return {"mae": np.nan, "mape": np.nan, "rmse": np.nan, "smape": np.nan}

    errors = actuals - forecasts
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # MAPE (avoid division by zero)
    nonzero = np.abs(actuals) > 1e-6
    if nonzero.sum() > 0:
        mape = float(np.mean(abs_errors[nonzero] / np.abs(actuals[nonzero])) * 100)
    else:
        mape = np.nan

    # SMAPE
    denom = (np.abs(actuals) + np.abs(forecasts)) / 2
    nonzero_s = denom > 1e-6
    if nonzero_s.sum() > 0:
        smape = float(np.mean(abs_errors[nonzero_s] / denom[nonzero_s]) * 100)
    else:
        smape = np.nan

    return {"mae": mae, "mape": mape, "rmse": rmse, "smape": smape}


def run_backtest_for_experiment(
    experiment_id: str,
    country_code: str,
    forecast_type: str,
    week_id: str,
    week_start: str,
    week_end: str,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run backtest for one experiment/country/type/week.

    Returns (actuals, forecasts) arrays, or None if failed.
    """
    # Handle persistence baseline (no config file needed)
    if experiment_id == "persistence":
        return _run_persistence_backtest(
            country_code, forecast_type, week_start, week_end,
        )

    from src.chronos2.engine import ChronosEngine
    from src.chronos2.input_builder import InputBuilder

    # Load experiment config
    exp_dir = config.EXPERIMENTS_DIR / experiment_id
    config_path = exp_dir / "config.json"

    with open(config_path) as f:
        exp_config = json.load(f)

    model_config = exp_config.get("model", {})
    training_config = exp_config.get("training", {})

    # Handle XGBoost experiments differently
    if model_config.get("type") == "xgboost":
        return _run_xgboost_backtest(
            experiment_id, country_code, forecast_type,
            week_start, week_end,
        )

    # Chronos-2 experiment
    if training_config.get("fine_tune", False):
        model_path = str(config.MODELS_DIR / "chronos2" / experiment_id / "finetuned-ckpt")
    else:
        model_path = None

    context_length = model_config.get("context_length", config.CHRONOS2_CONTEXT_LENGTH)

    engine = ChronosEngine(
        model_path=model_path,
        device=device,
        context_length=context_length,
    )
    input_builder = InputBuilder(context_length=context_length)

    all_actuals = []
    all_forecasts = []

    # For each day in the backtest week, generate D+2 forecast
    start_dt = pd.Timestamp(week_start)
    end_dt = pd.Timestamp(week_end)
    current = start_dt

    while current <= end_dt:
        target_date = current.strftime("%Y-%m-%d")

        try:
            # Build input and forecast
            inp = input_builder.build_for_country(
                country_code, forecast_type, target_date,
            )
            result = engine.forecast(
                target=inp["target"],
                past_covariates=inp.get("past_covariates"),
                future_covariates=inp.get("future_covariates"),
            )

            # Load actuals for this day
            next_day = (current + timedelta(days=1)).strftime("%Y-%m-%d")
            actuals_series = load_actuals(
                country_code, forecast_type, target_date, next_day,
            )

            if not actuals_series.empty:
                target_index = pd.date_range(current, periods=24, freq="h")
                actuals_aligned = actuals_series.reindex(target_index)

                # Only use hours where we have both forecast and actual
                valid = ~actuals_aligned.isna()
                if valid.sum() > 0:
                    all_actuals.extend(actuals_aligned[valid].values)
                    all_forecasts.extend(result["median"][valid.values])

        except Exception as e:
            logger.warning(f"  {target_date}: failed - {e}")

        current += timedelta(days=1)

    if not all_actuals:
        return None

    return np.array(all_actuals), np.array(all_forecasts)


def _run_xgboost_backtest(
    experiment_id: str,
    country_code: str,
    forecast_type: str,
    week_start: str,
    week_end: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run backtest using existing XGBoost models.

    Loads the trained XGBoost model and generates forecasts for each day
    in the backtest week.
    """
    try:
        from src.forecaster import Forecaster
    except ImportError:
        logger.warning("Could not import Forecaster for XGBoost backtest")
        return None

    all_actuals = []
    all_forecasts = []

    start_dt = pd.Timestamp(week_start)
    end_dt = pd.Timestamp(week_end)
    current = start_dt

    while current <= end_dt:
        target_date = current.strftime("%Y-%m-%d")

        try:
            forecaster = Forecaster.load(country_code, forecast_type)

            # Generate D+2 forecast for this day
            from datetime import date as date_cls
            ref_date = date_cls.fromisoformat(target_date) - timedelta(days=2)
            forecast_df = forecaster.predict_d2(
                reference_date=ref_date,
                horizon_days=2,
            )

            if forecast_df is not None and not forecast_df.empty:
                # Load actuals
                next_day = (current + timedelta(days=1)).strftime("%Y-%m-%d")
                actuals_series = load_actuals(
                    country_code, forecast_type, target_date, next_day,
                )

                if not actuals_series.empty and not forecast_df.empty:
                    pred_series = forecast_df.set_index("target_timestamp_utc")["forecast_value"]
                    common_idx = actuals_series.index.intersection(pred_series.index)
                    if len(common_idx) > 0:
                        all_actuals.extend(actuals_series.loc[common_idx].values)
                        all_forecasts.extend(pred_series.loc[common_idx].values)

        except Exception as e:
            logger.warning(f"  XGBoost {target_date}: failed - {e}")

        current += timedelta(days=1)

    if not all_actuals:
        return None

    return np.array(all_actuals), np.array(all_forecasts)


def _run_persistence_backtest(
    country_code: str,
    forecast_type: str,
    week_start: str,
    week_end: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run persistence baseline backtest (value at same hour 48h ago)."""
    all_actuals = []
    all_forecasts = []

    start_dt = pd.Timestamp(week_start)
    end_dt = pd.Timestamp(week_end)
    current = start_dt

    while current <= end_dt:
        target_date = current.strftime("%Y-%m-%d")
        next_day = (current + timedelta(days=1)).strftime("%Y-%m-%d")

        actuals_series = load_actuals(country_code, forecast_type, target_date, next_day)

        if not actuals_series.empty:
            history_start = (current - timedelta(days=3)).strftime("%Y-%m-%d")
            history_end = target_date
            history = load_actuals(country_code, forecast_type, history_start, history_end)

            if not history.empty:
                target_index = pd.date_range(current, periods=24, freq="h")
                persist_index = target_index - pd.Timedelta(hours=48)

                actuals_aligned = actuals_series.reindex(target_index)
                persist_values = history.reindex(persist_index)

                valid = ~actuals_aligned.isna() & ~persist_values.isna()
                if valid.sum() > 0:
                    all_actuals.extend(actuals_aligned[valid].values)
                    all_forecasts.extend(persist_values[valid].values)

        current += timedelta(days=1)

    if not all_actuals:
        return None

    return np.array(all_actuals), np.array(all_forecasts)


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--experiments", required=True, help="Comma-separated experiment IDs (e.g., V001,V003)")
    parser.add_argument("--weeks", default="all", help="Comma-separated week IDs (e.g., W01,W03) or 'all'")
    parser.add_argument("--countries", default="DE", help="Comma-separated country codes")
    parser.add_argument("--types", default="load", help="Comma-separated forecast types")
    parser.add_argument("--device", default="cuda", help="Device for Chronos-2 inference")
    parser.add_argument("--output", default=None, help="Output JSON file for results")
    args = parser.parse_args()

    experiments = args.experiments.split(",")
    countries = args.countries.split(",")
    forecast_types = args.types.split(",")

    if args.weeks == "all":
        weeks = config.BACKTEST_WEEKS
    else:
        week_ids = args.weeks.split(",")
        weeks = config.get_backtest_weeks(week_ids)

    logger.info(f"=== Comparing experiments: {experiments} ===")
    logger.info(f"Weeks: {[w[0] for w in weeks]}")
    logger.info(f"Countries: {countries}, Types: {forecast_types}")

    # Results structure: {exp_id: {country: {type: {week: metrics}}}}
    results = {}

    for exp_id in experiments:
        results[exp_id] = {}
        for cc in countries:
            results[exp_id][cc] = {}
            for ft in forecast_types:
                results[exp_id][cc][ft] = {}
                for week_id, w_start, w_end in weeks:
                    logger.info(f"  {exp_id} / {cc} / {ft} / {week_id}...")

                    bt_result = run_backtest_for_experiment(
                        exp_id, cc, ft, week_id, w_start, w_end,
                        device=args.device,
                    )

                    if bt_result is not None:
                        actuals, forecasts = bt_result
                        metrics = compute_metrics(actuals, forecasts)
                        results[exp_id][cc][ft][week_id] = metrics
                        logger.info(f"    MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.1f}%")
                    else:
                        results[exp_id][cc][ft][week_id] = None
                        logger.warning(f"    No results")

    # Print comparison table
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)

    baseline_id = experiments[0]

    for cc in countries:
        for ft in forecast_types:
            print(f"\n--- {cc} / {ft} ---")

            # Header
            header = f"{'Week':<8}"
            for exp_id in experiments:
                header += f"  {exp_id:>12} MAE  {exp_id:>8} MAPE"
            if len(experiments) > 1:
                header += "    Skill"
            print(header)
            print("-" * len(header))

            # Per-week rows
            all_week_metrics = {exp_id: [] for exp_id in experiments}

            for week_id, _, _ in weeks:
                row = f"{week_id:<8}"
                for exp_id in experiments:
                    m = results.get(exp_id, {}).get(cc, {}).get(ft, {}).get(week_id)
                    if m:
                        row += f"  {m['mae']:>12.2f}  {m['mape']:>8.1f}%"
                        all_week_metrics[exp_id].append(m)
                    else:
                        row += f"  {'N/A':>12}  {'N/A':>8} "

                # Skill score (last experiment vs first = baseline)
                if len(experiments) > 1:
                    m_base = results.get(baseline_id, {}).get(cc, {}).get(ft, {}).get(week_id)
                    m_test = results.get(experiments[-1], {}).get(cc, {}).get(ft, {}).get(week_id)
                    if m_base and m_test and m_base["mae"] > 0:
                        skill = 1 - (m_test["mae"] / m_base["mae"])
                        row += f"  {skill:>+.3f}"
                    else:
                        row += f"  {'N/A':>6}"

                print(row)

            # Average row
            row = f"{'AVG':<8}"
            for exp_id in experiments:
                wm = all_week_metrics[exp_id]
                if wm:
                    avg_mae = np.mean([m["mae"] for m in wm])
                    avg_mape = np.mean([m["mape"] for m in wm if not np.isnan(m["mape"])])
                    row += f"  {avg_mae:>12.2f}  {avg_mape:>8.1f}%"
                else:
                    row += f"  {'N/A':>12}  {'N/A':>8} "

            if len(experiments) > 1:
                wm_base = all_week_metrics[baseline_id]
                wm_test = all_week_metrics[experiments[-1]]
                if wm_base and wm_test:
                    avg_base = np.mean([m["mae"] for m in wm_base])
                    avg_test = np.mean([m["mae"] for m in wm_test])
                    if avg_base > 0:
                        skill = 1 - (avg_test / avg_base)
                        row += f"  {skill:>+.3f}"
                    else:
                        row += f"  {'N/A':>6}"

            print("-" * len(header))
            print(row)

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

    # Also save to experiment directories
    for exp_id in experiments:
        exp_dir = config.EXPERIMENTS_DIR / exp_id
        if exp_dir.exists():
            results_path = exp_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump(results[exp_id], f, indent=2, default=str)


if __name__ == "__main__":
    main()
