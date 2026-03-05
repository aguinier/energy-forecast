"""
Chronos-Bolt Fine-Tuning Script for Energy Price Prediction

Fine-tunes Chronos-Bolt-small on Belgian electricity price data with
TSO forecast covariates via AutoGluon TimeSeriesPredictor.

Usage (from chronos-venv):
    python chronos_train.py
    python chronos_train.py --steps 200 --eval-days 30
    python chronos_train.py --output-dir C:/path/to/save

Author: OpenClaw integration
Date: 2026-02-22
"""

import logging
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("energy_forecast.chronos_train")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATABASE_PATH = Path(os.getenv('ENERGY_DB_PATH', '/data/energy_dashboard.db'))

DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\guill\.openclaw\workspace\experiments\ag_chronos2_finetune"
)

KNOWN_COVARIATES = [
    "load_forecast_mw",
    "solar_forecast_mw",
    "wind_onshore_forecast_mw",
    "wind_offshore_forecast_mw",
    "total_gen_forecast_mw",
    "hour",
    "dayofweek",
    "month",
]

PREDICTION_LENGTH = 24
COUNTRY_CODE = "BE"


# ============================================================================
# DATA PREPARATION
# ============================================================================


def load_training_data(
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and merge price + covariate data for training.

    Returns a DataFrame with columns:
        item_id, timestamp, target, + KNOWN_COVARIATES
    """
    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    conn = sqlite3.connect(str(DATABASE_PATH), timeout=30.0)
    try:
        # Price data
        price_df = pd.read_sql_query(
            """
            SELECT timestamp_utc as timestamp, price_eur_mwh as target
            FROM energy_price
            WHERE country_code = ?
              AND timestamp_utc >= ? AND timestamp_utc < ?
              AND data_quality = 'actual'
            ORDER BY timestamp_utc
            """,
            conn,
            params=(COUNTRY_CODE, start_date, end_date),
        )

        # Load forecast
        load_df = pd.read_sql_query(
            """
            SELECT target_timestamp_utc as timestamp,
                   forecast_value_mw as load_forecast_mw
            FROM energy_load_forecast
            WHERE country_code = ?
              AND target_timestamp_utc >= ? AND target_timestamp_utc < ?
            ORDER BY target_timestamp_utc
            """,
            conn,
            params=(COUNTRY_CODE, start_date, end_date),
        )

        # Generation forecast
        gen_df = pd.read_sql_query(
            """
            SELECT target_timestamp_utc as timestamp,
                   solar_mw as solar_forecast_mw,
                   wind_onshore_mw as wind_onshore_forecast_mw,
                   wind_offshore_mw as wind_offshore_forecast_mw,
                   total_forecast_mw as total_gen_forecast_mw
            FROM energy_generation_forecast
            WHERE country_code = ?
              AND target_timestamp_utc >= ? AND target_timestamp_utc < ?
            ORDER BY target_timestamp_utc
            """,
            conn,
            params=(COUNTRY_CODE, start_date, end_date),
        )
    finally:
        conn.close()

    # Parse timestamps
    for df in [price_df, load_df, gen_df]:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Deduplicate
    price_df = price_df.drop_duplicates(subset="timestamp", keep="last")
    load_df = load_df.drop_duplicates(subset="timestamp", keep="last")
    gen_df = gen_df.drop_duplicates(subset="timestamp", keep="last")

    # Merge
    merged = price_df.copy()
    if not load_df.empty:
        merged = pd.merge(merged, load_df, on="timestamp", how="left")
    if not gen_df.empty:
        merged = pd.merge(merged, gen_df, on="timestamp", how="left")

    # Time features
    merged["hour"] = merged["timestamp"].dt.hour
    merged["dayofweek"] = merged["timestamp"].dt.dayofweek
    merged["month"] = merged["timestamp"].dt.month

    # Fill missing covariates
    for col in KNOWN_COVARIATES:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].ffill().fillna(0.0)

    # Ensure hourly frequency
    merged = merged.set_index("timestamp").asfreq("h")
    merged["target"] = merged["target"].interpolate(method="linear")
    for col in KNOWN_COVARIATES:
        merged[col] = merged[col].ffill().fillna(0.0)
    merged = merged.reset_index().rename(columns={"index": "timestamp"})

    # Add item_id
    merged["item_id"] = "BE_price"

    logger.info(f"Training data: {len(merged)} rows from {merged['timestamp'].min()} to {merged['timestamp'].max()}")
    return merged


# ============================================================================
# TRAINING
# ============================================================================


def train_chronos(
    output_dir: Optional[Path] = None,
    fine_tune_steps: int = 200,
    eval_days: int = 30,
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
) -> dict:
    """
    Fine-tune Chronos-Bolt-small on BE price data.

    Args:
        output_dir: Where to save the trained model
        fine_tune_steps: Number of fine-tuning steps
        eval_days: Days to hold out for evaluation
        start_date: Training start date
        end_date: Training end date

    Returns:
        Dictionary with training metrics
    """
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    # Load data
    data = load_training_data(start_date=start_date, end_date=end_date)

    # Split: hold out last eval_days for evaluation
    eval_hours = eval_days * 24
    train_data = data.iloc[:-eval_hours].copy()
    eval_data = data.copy()  # Full data (AutoGluon uses last prediction_length as test)

    logger.info(f"Train: {len(train_data)} rows, Eval holdout: {eval_hours} hours")

    # Build TimeSeriesDataFrame
    train_tsdf = TimeSeriesDataFrame.from_data_frame(
        train_data[["item_id", "timestamp", "target"] + KNOWN_COVARIATES],
        id_column="item_id",
        timestamp_column="timestamp",
    )

    eval_tsdf = TimeSeriesDataFrame.from_data_frame(
        eval_data[["item_id", "timestamp", "target"] + KNOWN_COVARIATES],
        id_column="item_id",
        timestamp_column="timestamp",
    )

    # Train
    logger.info(f"Fine-tuning Chronos-Bolt-small for {fine_tune_steps} steps...")
    predictor = TimeSeriesPredictor(
        prediction_length=PREDICTION_LENGTH,
        path=str(output_dir),
        target="target",
        known_covariates_names=KNOWN_COVARIATES,
        eval_metric="MAE",
        freq="h",
    )

    predictor.fit(
        train_tsdf,
        hyperparameters={
            "Chronos": {
                "model_path": "amazon/chronos-bolt-small",
                "fine_tune": True,
                "fine_tune_steps": fine_tune_steps,
                "context_length": 512,
                "ag_args_fit": {"num_gpus": 1},
            }
        },
        time_limit=None,
        enable_ensemble=False,
    )

    logger.info("Training complete!")

    # Evaluate on holdout
    metrics = evaluate_model(predictor, eval_data, eval_days)

    # Save metrics
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to {metrics_path}")

    return metrics


def evaluate_model(
    predictor,
    full_data: pd.DataFrame,
    eval_days: int = 30,
) -> dict:
    """
    Evaluate model on the last eval_days using rolling 24h forecasts.

    Returns dict with MAE, MAPE, RMSE.
    """
    from autogluon.timeseries import TimeSeriesDataFrame

    eval_hours = eval_days * 24
    n_total = len(full_data)

    all_actuals = []
    all_preds = []

    # Rolling evaluation: predict 24h at a time
    n_steps = eval_hours // PREDICTION_LENGTH
    for step in range(n_steps):
        # History ends at this point
        end_idx = n_total - eval_hours + step * PREDICTION_LENGTH
        if end_idx < PREDICTION_LENGTH * 2:
            continue

        history = full_data.iloc[:end_idx].copy()
        actuals_slice = full_data.iloc[end_idx : end_idx + PREDICTION_LENGTH].copy()

        if len(actuals_slice) < PREDICTION_LENGTH:
            break

        # Build TimeSeriesDataFrame for history
        ts_df = TimeSeriesDataFrame.from_data_frame(
            history[["item_id", "timestamp", "target"] + KNOWN_COVARIATES],
            id_column="item_id",
            timestamp_column="timestamp",
        )

        # Build known_covariates for forecast period
        future_cov = actuals_slice[["item_id", "timestamp"] + KNOWN_COVARIATES].copy()
        future_tsdf = TimeSeriesDataFrame.from_data_frame(
            future_cov,
            id_column="item_id",
            timestamp_column="timestamp",
        )

        try:
            predictions = predictor.predict(ts_df, known_covariates=future_tsdf)
            pred_values = predictions["mean"].values

            all_actuals.extend(actuals_slice["target"].values)
            all_preds.extend(pred_values)
        except Exception as e:
            logger.warning(f"Eval step {step} failed: {e}")
            continue

    if not all_actuals:
        logger.warning("No evaluation data collected")
        return {"mae": None, "mape": None, "rmse": None}

    actuals = np.array(all_actuals, dtype=float)
    preds = np.array(all_preds, dtype=float)

    mae = float(np.mean(np.abs(actuals - preds)))
    mape = float(np.mean(np.abs((actuals - preds) / np.where(actuals == 0, 1, actuals))) * 100)
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))

    metrics = {
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "rmse": round(rmse, 2),
        "eval_days": eval_days,
        "n_samples": len(actuals),
        "trained_at": datetime.now().isoformat(),
    }

    logger.info(f"Evaluation: MAE={mae:.2f}, MAPE={mape:.2f}%, RMSE={rmse:.2f}")
    return metrics


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fine-tune Chronos-Bolt for price forecasting")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--steps", type=int, default=200, help="Fine-tune steps")
    parser.add_argument("--eval-days", type=int, default=30, help="Evaluation holdout days")
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Training start date")
    parser.add_argument("--end-date", type=str, default=None, help="Training end date")
    args = parser.parse_args()

    output = Path(args.output_dir) if args.output_dir else None

    metrics = train_chronos(
        output_dir=output,
        fine_tune_steps=args.steps,
        eval_days=args.eval_days,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print(f"\nTraining Results:")
    print(json.dumps(metrics, indent=2))
