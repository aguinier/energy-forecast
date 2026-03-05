"""
Rolling-window backtesting framework for energy forecasting.

Simulates real-world forecasting by:
1. Training on data up to day T
2. Generating forecast for day T+horizon
3. Comparing against actual
4. Rolling forward and repeating

Supports:
- Expanding window (growing training set) — default
- Fixed window (sliding training set)
- Multiple models compared on identical data splits
- Aggregated metrics + per-day results for analysis
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

sys.path.insert(0, str(Path(__file__).parent.parent))
from db import load_training_data, load_energy_data, load_weather_data
from features import create_all_features, get_feature_columns
from evaluation.metrics import (
    calculate_point_metrics,
    diebold_mariano_test,
    skill_score,
    pinball_loss,
)

logger = logging.getLogger("energy_forecast.backtest")


class BacktestResult:
    """Container for backtest results from one model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.daily_results: List[Dict] = []  # per-day forecasts + actuals

    def add_day(self, date: str, actuals: np.ndarray, predictions: np.ndarray,
                hours: Optional[np.ndarray] = None):
        """Add one day of results (24 hourly values)."""
        self.daily_results.append({
            "date": date,
            "actuals": actuals,
            "predictions": predictions,
            "hours": hours,
        })

    @property
    def all_actuals(self) -> np.ndarray:
        return np.concatenate([d["actuals"] for d in self.daily_results])

    @property
    def all_predictions(self) -> np.ndarray:
        return np.concatenate([d["predictions"] for d in self.daily_results])

    @property
    def n_days(self) -> int:
        return len(self.daily_results)

    def aggregate_metrics(self) -> Dict[str, float]:
        """Compute overall metrics across all days."""
        return calculate_point_metrics(self.all_actuals, self.all_predictions)

    def daily_metrics(self) -> pd.DataFrame:
        """Compute per-day metrics for analysis."""
        rows = []
        for d in self.daily_results:
            m = calculate_point_metrics(d["actuals"], d["predictions"])
            m["date"] = d["date"]
            rows.append(m)
        return pd.DataFrame(rows).set_index("date")


def run_xgboost_backtest(
    country_code: str,
    forecast_type: str,
    start_date: str,
    end_date: str,
    horizon_days: int = 2,
    min_train_days: int = 365,
    window: str = "expanding",
    fixed_window_days: Optional[int] = None,
    algorithm: str = "xgboost",
    step_days: int = 1,
    verbose: bool = True,
) -> BacktestResult:
    """
    Run rolling-window backtest for XGBoost/LightGBM/CatBoost.

    For each test day T in [start_date, end_date]:
      1. Train on data before T (expanding or fixed window)
      2. Create features for T+horizon
      3. Predict 24 hours
      4. Compare vs actuals

    Args:
        country_code: e.g. "BE"
        forecast_type: e.g. "price", "load"
        start_date: First test day (YYYY-MM-DD) — needs min_train_days before it
        end_date: Last test day (YYYY-MM-DD)
        horizon_days: Forecast horizon (1=D+1, 2=D+2)
        min_train_days: Minimum training data in days
        window: "expanding" or "fixed"
        fixed_window_days: Window size for fixed window (required if window="fixed")
        algorithm: "xgboost", "lightgbm", or "catboost"
        step_days: Step between test days (1 = every day, 7 = weekly)
        verbose: Print progress

    Returns:
        BacktestResult with per-day forecasts and actuals
    """
    from forecaster import Forecaster

    result = BacktestResult(f"{algorithm}_D+{horizon_days}")

    # Determine full data range needed
    data_start = (pd.Timestamp(start_date) - timedelta(days=min_train_days + 30)).strftime("%Y-%m-%d")
    data_end = (pd.Timestamp(end_date) + timedelta(days=horizon_days + 1)).strftime("%Y-%m-%d")

    if verbose:
        print(f"Loading data {data_start} → {data_end} for {country_code}/{forecast_type}...")

    # Load all data once
    full_data = load_training_data(country_code, forecast_type, data_start, data_end)
    if full_data.empty:
        logger.error("No data loaded")
        return result

    # Create features on full dataset
    full_data = create_all_features(full_data, forecast_type, country_code=country_code)
    feature_cols = get_feature_columns(forecast_type)

    # Generate test dates
    test_dates = pd.date_range(start_date, end_date, freq=f"{step_days}D")

    if verbose:
        print(f"Backtesting {len(test_dates)} days: {start_date} → {end_date}")

    for i, test_date in enumerate(test_dates):
        target_date = test_date + timedelta(days=horizon_days)

        # Split: train on everything before test_date
        if window == "expanding":
            train_start = pd.Timestamp(data_start)
        else:
            train_start = test_date - timedelta(days=fixed_window_days or min_train_days)

        train_mask = full_data["timestamp_utc"] < test_date.strftime("%Y-%m-%d")
        if window == "fixed":
            train_mask &= full_data["timestamp_utc"] >= train_start.strftime("%Y-%m-%d")

        target_day_str = target_date.strftime("%Y-%m-%d")
        target_mask = (
            (full_data["timestamp_utc"] >= target_day_str)
            & (full_data["timestamp_utc"] < (target_date + timedelta(days=1)).strftime("%Y-%m-%d"))
        )

        train_df = full_data[train_mask]
        target_df = full_data[target_mask]

        if len(train_df) < min_train_days * 20:  # ~20 hours/day minimum
            if verbose:
                print(f"  [{i+1}/{len(test_dates)}] {test_date.date()} → skip (insufficient training data)")
            continue

        if len(target_df) < 20:  # Need most of 24 hours
            if verbose:
                print(f"  [{i+1}/{len(test_dates)}] {test_date.date()} → skip (no actuals for {target_day_str})")
            continue

        # Prepare train/target arrays
        available_features = [c for c in feature_cols if c in train_df.columns]
        X_train = train_df[available_features].values
        y_train = train_df["target_value"].values
        X_target = target_df[available_features].values
        y_actual = target_df["target_value"].values

        # Drop NaN rows
        train_valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]

        target_valid = ~np.isnan(X_target).any(axis=1)
        X_target = X_target[target_valid]
        y_actual = y_actual[target_valid]

        if len(X_train) < 1000 or len(X_target) < 20:
            continue

        # Train model
        try:
            params = config.get_default_params(algorithm)
            # Reduce estimators for speed during backtest
            if "n_estimators" in params:
                params["n_estimators"] = min(params["n_estimators"], 300)
            if "iterations" in params:
                params["iterations"] = min(params["iterations"], 300)
            # Remove early stopping for backtest (no validation set)
            params.pop("early_stopping_rounds", None)

            if algorithm == "xgboost":
                model = XGBRegressor(**params)
            elif algorithm == "lightgbm":
                model = LGBMRegressor(**params)
            elif algorithm == "catboost":
                model = CatBoostRegressor(**params)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            model.fit(X_train, y_train)
            predictions = model.predict(X_target)

            result.add_day(
                date=target_day_str,
                actuals=y_actual,
                predictions=predictions,
                hours=target_df[target_valid]["timestamp_utc"].values if "timestamp_utc" in target_df.columns else None,
            )

            if verbose and (i + 1) % 10 == 0:
                day_mae = np.mean(np.abs(y_actual - predictions))
                print(f"  [{i+1}/{len(test_dates)}] {test_date.date()} → MAE={day_mae:.2f}")

        except Exception as e:
            logger.error(f"Failed on {test_date.date()}: {e}")
            if verbose:
                print(f"  [{i+1}/{len(test_dates)}] {test_date.date()} → ERROR: {e}")

    if verbose:
        if result.n_days > 0:
            agg = result.aggregate_metrics()
            print(f"\n{'='*50}")
            print(f"Backtest complete: {result.n_days} days")
            print(f"  MAE:  {agg['mae']:.4f}")
            print(f"  RMSE: {agg['rmse']:.4f}")
            print(f"  MAPE: {agg['mape']:.2f}%")
            print(f"  Bias: {agg['bias']:.4f}")
        else:
            print("\nNo valid test days!")

    return result


def run_persistence_baseline(
    country_code: str,
    forecast_type: str,
    start_date: str,
    end_date: str,
    horizon_days: int = 2,
    step_days: int = 1,
) -> BacktestResult:
    """
    Persistence baseline: forecast = same hour from 7 days ago.
    (Weekly persistence is standard for energy — captures weekly patterns.)
    """
    result = BacktestResult(f"persistence_7d_D+{horizon_days}")

    data_start = (pd.Timestamp(start_date) - timedelta(days=14)).strftime("%Y-%m-%d")
    data_end = (pd.Timestamp(end_date) + timedelta(days=horizon_days + 1)).strftime("%Y-%m-%d")

    energy_df = load_energy_data(country_code, forecast_type, data_start, data_end)
    if energy_df.empty:
        return result

    energy_df = energy_df.set_index("timestamp_utc").resample("h").mean().reset_index()
    energy_df = energy_df.dropna()

    test_dates = pd.date_range(start_date, end_date, freq=f"{step_days}D")

    for test_date in test_dates:
        target_date = test_date + timedelta(days=horizon_days)
        persist_date = target_date - timedelta(days=7)

        target_day_str = target_date.strftime("%Y-%m-%d")
        persist_day_str = persist_date.strftime("%Y-%m-%d")

        target_mask = (
            (energy_df["timestamp_utc"] >= target_day_str)
            & (energy_df["timestamp_utc"] < (target_date + timedelta(days=1)).strftime("%Y-%m-%d"))
        )
        persist_mask = (
            (energy_df["timestamp_utc"] >= persist_day_str)
            & (energy_df["timestamp_utc"] < (persist_date + timedelta(days=1)).strftime("%Y-%m-%d"))
        )

        actuals = energy_df[target_mask]["target_value"].values
        persistence = energy_df[persist_mask]["target_value"].values

        if len(actuals) >= 20 and len(persistence) >= 20:
            # Align lengths
            n = min(len(actuals), len(persistence))
            result.add_day(target_day_str, actuals[:n], persistence[:n])

    return result


def compare_models(
    results: List[BacktestResult],
    baseline_name: Optional[str] = None,
) -> str:
    """
    Compare multiple backtest results and generate a summary report.

    Args:
        results: List of BacktestResult objects
        baseline_name: Name of baseline model for skill score computation

    Returns:
        Formatted comparison report (markdown)
    """
    if not results:
        return "No results to compare."

    lines = ["# Backtest Comparison Report", ""]

    # Summary table
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Model | Days | MAE | RMSE | MAPE (%) | Bias | Skill vs Baseline |")
    lines.append("|-------|------|-----|------|----------|------|-------------------|")

    baseline_mae = None
    if baseline_name:
        for r in results:
            if r.model_name == baseline_name:
                baseline_mae = r.aggregate_metrics()["mae"]
                break

    for r in results:
        if r.n_days == 0:
            lines.append(f"| {r.model_name} | 0 | - | - | - | - | - |")
            continue
        m = r.aggregate_metrics()
        sk = f"{skill_score(m['mae'], baseline_mae):.3f}" if baseline_mae else "-"
        lines.append(
            f"| {r.model_name} | {r.n_days} | {m['mae']:.2f} | {m['rmse']:.2f} | "
            f"{m['mape']:.1f} | {m['bias']:.2f} | {sk} |"
        )

    # Pairwise DM tests
    if len(results) >= 2:
        lines.extend(["", "## Diebold-Mariano Tests", ""])
        lines.append("| Model A | Model B | DM Stat | p-value | Winner |")
        lines.append("|---------|---------|---------|---------|--------|")

        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                ra, rb = results[i], results[j]
                if ra.n_days == 0 or rb.n_days == 0:
                    continue

                # Align on common dates
                dates_a = {d["date"] for d in ra.daily_results}
                dates_b = {d["date"] for d in rb.daily_results}
                common = sorted(dates_a & dates_b)

                if len(common) < 10:
                    lines.append(f"| {ra.model_name} | {rb.model_name} | - | - | too few common days |")
                    continue

                act_a, pred_a, pred_b_arr = [], [], []
                lookup_a = {d["date"]: d for d in ra.daily_results}
                lookup_b = {d["date"]: d for d in rb.daily_results}

                for dt in common:
                    da, db = lookup_a[dt], lookup_b[dt]
                    n = min(len(da["actuals"]), len(db["actuals"]))
                    act_a.extend(da["actuals"][:n])
                    pred_a.extend(da["predictions"][:n])
                    pred_b_arr.extend(db["predictions"][:n])

                act_a = np.array(act_a)
                pred_a = np.array(pred_a)
                pred_b_arr = np.array(pred_b_arr)

                dm_stat, p_val, interp = diebold_mariano_test(act_a, pred_a, pred_b_arr)
                if dm_stat < 0:
                    winner = ra.model_name
                elif dm_stat > 0:
                    winner = rb.model_name
                else:
                    winner = "tie"

                sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
                lines.append(
                    f"| {ra.model_name} | {rb.model_name} | {dm_stat:.3f} | {p_val:.4f} | {winner} {sig} |"
                )

    lines.extend(["", f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_"])
    return "\n".join(lines)


# Need these imports at module level for run_xgboost_backtest
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.WARNING, format=config.LOG_FORMAT)

    parser = argparse.ArgumentParser(description="Run backtest for energy forecasting models")
    parser.add_argument("--country", default="BE", help="Country code (default: BE)")
    parser.add_argument("--type", default="price", help="Forecast type (default: price)")
    parser.add_argument("--start", required=True, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=2, help="Forecast horizon in days (default: 2)")
    parser.add_argument("--algorithms", nargs="+", default=["xgboost"], help="Algorithms to test")
    parser.add_argument("--step", type=int, default=1, help="Step between test days")
    parser.add_argument("--window", default="expanding", choices=["expanding", "fixed"])
    parser.add_argument("--window-days", type=int, default=365, help="Fixed window size in days")
    parser.add_argument("--output", help="Output file for report (markdown)")
    parser.add_argument("--include-baseline", action="store_true", help="Include persistence baseline")

    args = parser.parse_args()

    results = []

    # Baseline
    if args.include_baseline:
        print("Running persistence baseline...")
        baseline = run_persistence_baseline(
            args.country, args.type, args.start, args.end,
            horizon_days=args.horizon, step_days=args.step,
        )
        results.append(baseline)

    # Models
    for algo in args.algorithms:
        print(f"\nRunning {algo} backtest...")
        r = run_xgboost_backtest(
            country_code=args.country,
            forecast_type=args.type,
            start_date=args.start,
            end_date=args.end,
            horizon_days=args.horizon,
            algorithm=algo,
            step_days=args.step,
            window=args.window,
            fixed_window_days=args.window_days if args.window == "fixed" else None,
        )
        results.append(r)

    # Report
    baseline_name = results[0].model_name if args.include_baseline else None
    report = compare_models(results, baseline_name=baseline_name)
    print("\n" + report)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\nReport saved to {args.output}")
