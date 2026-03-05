"""
TSO Error Correction Model for Renewable Forecasting.

Instead of replacing Elia's TSO forecasts, we learn their systematic biases
and correct them. The target is: actual - TSO_forecast (the residual).

Features:
- TSO forecast value itself (biases scale with forecast magnitude)
- Weather data (wind speed, radiation, temperature)
- Time features (hour, month, day of week)
- Lagged TSO errors (error persistence)

Supports:
- Training + evaluation on a single split
- Rolling-window backtesting via BacktestResult integration
- Model persistence (save/load via joblib)
- Production prediction: given TSO forecast + weather → corrected forecast
"""

import sys
import logging
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

sys.path.insert(0, str(Path(__file__).parent.parent))
from db import get_connection
from evaluation.metrics import calculate_point_metrics, skill_score, diebold_mariano_test
from evaluation.backtest import BacktestResult

logger = logging.getLogger("energy_forecast.tso_correction")

# Default model save directory
MODELS_DIR = Path(config.BASE_DIR) / "models" / "tso_correction"


# ============================================================================
# DATA LOADING
# ============================================================================

def load_tso_vs_actual(
    country_code: str,
    renewable_type: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Load aligned TSO forecast + actual data for a renewable type.

    Returns DataFrame with columns:
        timestamp_utc, actual_mw, tso_forecast_mw, tso_error
    """
    col_map = {
        "solar": "solar_mw",
        "wind_onshore": "wind_onshore_mw",
        "wind_offshore": "wind_offshore_mw",
    }
    if renewable_type not in col_map:
        raise ValueError(f"Unsupported: {renewable_type}. Use: {list(col_map.keys())}")

    col = col_map[renewable_type]

    with get_connection() as conn:
        actuals = pd.read_sql_query(f"""
            SELECT timestamp_utc, {col} as actual_mw
            FROM energy_renewable
            WHERE country_code = ? AND data_quality = 'actual'
              AND timestamp_utc >= ? AND timestamp_utc < ?
            ORDER BY timestamp_utc
        """, conn, params=(country_code, start_date, end_date))

        forecasts = pd.read_sql_query(f"""
            SELECT target_timestamp_utc as timestamp_utc, {col} as tso_forecast_mw
            FROM energy_generation_forecast
            WHERE country_code = ?
              AND target_timestamp_utc >= ? AND target_timestamp_utc < ?
            ORDER BY target_timestamp_utc
        """, conn, params=(country_code, start_date, end_date))

    if actuals.empty or forecasts.empty:
        logger.warning(f"No data for {country_code}/{renewable_type}")
        return pd.DataFrame()

    actuals["timestamp_utc"] = pd.to_datetime(actuals["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    forecasts["timestamp_utc"] = pd.to_datetime(forecasts["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)

    # Resample actuals to hourly (TSO forecasts are hourly)
    actuals = actuals.set_index("timestamp_utc").resample("h").mean().reset_index()

    df = pd.merge(actuals, forecasts, on="timestamp_utc", how="inner")
    df = df.dropna(subset=["actual_mw", "tso_forecast_mw"])
    df["tso_error"] = df["actual_mw"] - df["tso_forecast_mw"]

    logger.info(f"Loaded {len(df)} aligned hours for {country_code}/{renewable_type}")
    return df


def load_weather_for_correction(
    country_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load weather data for feature engineering."""
    with get_connection() as conn:
        df = pd.read_sql_query("""
            SELECT timestamp_utc,
                   temperature_2m_k, wind_speed_10m_ms, wind_speed_100m_ms,
                   shortwave_radiation_wm2, direct_radiation_wm2, diffuse_radiation_wm2
            FROM weather_data
            WHERE country_code = ? AND data_quality = 'actual'
              AND timestamp_utc >= ? AND timestamp_utc < ?
            ORDER BY timestamp_utc
        """, conn, params=(country_code, start_date, end_date))

    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)

    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

RADIATION_COLS = ["shortwave_radiation_wm2", "direct_radiation_wm2", "diffuse_radiation_wm2"]
WIND_COLS = ["wind_speed_10m_ms", "wind_speed_100m_ms"]

WEATHER_FEATURES = {
    "solar": RADIATION_COLS + ["temperature_2m_k"],
    "wind_onshore": WIND_COLS + ["temperature_2m_k"],
    "wind_offshore": WIND_COLS + ["temperature_2m_k"],
}


def create_correction_features(
    df: pd.DataFrame,
    renewable_type: str,
) -> pd.DataFrame:
    """
    Create features for the TSO error correction model.

    Handles NaN properly:
    - Radiation columns: fill NaN with 0 (nighttime is legitimately 0)
    - Wind/temperature: forward-fill then backward-fill
    - Lag features: NaN at start is expected (dropped later)
    """
    df = df.copy()
    ts = pd.to_datetime(df["timestamp_utc"])

    # --- Fix NaN in weather BEFORE creating features ---
    # Radiation at night is 0, not missing
    for col in RADIATION_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Wind and temperature: interpolate gaps
    for col in WIND_COLS + ["temperature_2m_k"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # --- Time features ---
    df["hour"] = ts.dt.hour
    df["month"] = ts.dt.month
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Lagged TSO errors (error persistence) ---
    for lag_h in [24, 48, 168]:
        df[f"tso_error_lag_{lag_h}h"] = df["tso_error"].shift(lag_h)

    # --- Rolling TSO error stats ---
    for window in [24, 168]:
        df[f"tso_error_roll_{window}h_mean"] = df["tso_error"].rolling(window, min_periods=1).mean()
        df[f"tso_error_roll_{window}h_std"] = df["tso_error"].rolling(window, min_periods=1).std().fillna(0)

    # --- TSO forecast rolling stats ---
    df["tso_forecast_roll_24h_mean"] = df["tso_forecast_mw"].rolling(24, min_periods=1).mean()

    return df


def get_correction_feature_cols(renewable_type: str) -> List[str]:
    """Get feature column names for correction model."""
    base_features = [
        "tso_forecast_mw",
        "hour", "month", "day_of_week", "is_weekend",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "tso_error_lag_24h", "tso_error_lag_48h", "tso_error_lag_168h",
        "tso_error_roll_24h_mean", "tso_error_roll_24h_std",
        "tso_error_roll_168h_mean", "tso_error_roll_168h_std",
        "tso_forecast_roll_24h_mean",
    ]

    weather = WEATHER_FEATURES.get(renewable_type, [])
    return base_features + weather


# ============================================================================
# MODEL CLASS
# ============================================================================

class TSOCorrectionModel:
    """
    Trainable, saveable TSO error correction model.

    Usage:
        model = TSOCorrectionModel("BE", "wind_onshore")
        model.train(train_df)
        corrected = model.predict(test_df)
        model.save()
        model = TSOCorrectionModel.load("BE", "wind_onshore")
    """

    def __init__(
        self,
        country_code: str,
        renewable_type: str,
        algorithm: str = "lightgbm",
    ):
        self.country_code = country_code
        self.renewable_type = renewable_type
        self.algorithm = algorithm
        self.feature_cols = get_correction_feature_cols(renewable_type)
        self.model = None
        self.trained_at = None
        self.train_samples = 0

    def _create_model(self):
        if self.algorithm == "lightgbm":
            return LGBMRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                verbose=-1, random_state=42,
            )
        else:
            return XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                verbosity=0, random_state=42,
            )

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Extract feature matrix, handling missing columns gracefully."""
        available = [c for c in self.feature_cols if c in df.columns]
        X = df[available].copy()
        return X, available

    def train(self, df: pd.DataFrame) -> int:
        """
        Train on prepared DataFrame (must have tso_error + feature columns).

        Args:
            df: DataFrame with features already created

        Returns:
            Number of valid training samples used
        """
        X, used_cols = self._prepare_features(df)
        y = df["tso_error"].values

        # Drop NaN rows
        valid = ~(X.isna().any(axis=1) | np.isnan(y))
        X, y = X[valid], y[valid]

        self.model = self._create_model()
        self.model.fit(X, y)
        self.trained_at = datetime.now()
        self.train_samples = len(X)
        self._used_cols = used_cols

        logger.info(f"Trained {self.renewable_type} correction on {len(X)} samples")
        return len(X)

    def predict_error(self, df: pd.DataFrame) -> np.ndarray:
        """Predict TSO error (actual - TSO). Add this to TSO to get corrected forecast."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X, _ = self._prepare_features(df)

        # Handle NaN in features for prediction (fill with 0 rather than dropping)
        X = X.fillna(0.0)

        return self.model.predict(X)

    def correct(self, df: pd.DataFrame) -> np.ndarray:
        """Return corrected forecast: TSO + predicted error."""
        predicted_error = self.predict_error(df)
        return df["tso_forecast_mw"].values + predicted_error

    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if self.model is None:
            return {}
        cols = getattr(self, "_used_cols", self.feature_cols)
        return dict(zip(cols, self.model.feature_importances_))

    def save(self, path: Optional[Path] = None):
        """Save model to disk."""
        if path is None:
            path = MODELS_DIR / self.country_code / self.renewable_type
        path.mkdir(parents=True, exist_ok=True)

        model_path = path / "model.joblib"
        meta = {
            "country_code": self.country_code,
            "renewable_type": self.renewable_type,
            "algorithm": self.algorithm,
            "feature_cols": self.feature_cols,
            "used_cols": getattr(self, "_used_cols", self.feature_cols),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "train_samples": self.train_samples,
        }

        joblib.dump({"model": self.model, "meta": meta}, model_path)
        logger.info(f"Saved model to {model_path}")

    @classmethod
    def load(cls, country_code: str, renewable_type: str, path: Optional[Path] = None) -> "TSOCorrectionModel":
        """Load model from disk."""
        if path is None:
            path = MODELS_DIR / country_code / renewable_type

        model_path = path / "model.joblib"
        data = joblib.load(model_path)

        meta = data["meta"]
        obj = cls(
            country_code=meta["country_code"],
            renewable_type=meta["renewable_type"],
            algorithm=meta["algorithm"],
        )
        obj.model = data["model"]
        obj.feature_cols = meta["feature_cols"]
        obj._used_cols = meta.get("used_cols", meta["feature_cols"])
        obj.trained_at = datetime.fromisoformat(meta["trained_at"]) if meta.get("trained_at") else None
        obj.train_samples = meta.get("train_samples", 0)

        logger.info(f"Loaded {renewable_type} correction model from {model_path}")
        return obj


# ============================================================================
# ROLLING BACKTEST
# ============================================================================

def run_tso_correction_backtest(
    country_code: str,
    renewable_type: str,
    start_date: str,
    end_date: str,
    min_train_days: int = 180,
    retrain_every_days: int = 30,
    algorithm: str = "lightgbm",
    verbose: bool = True,
) -> Tuple[BacktestResult, BacktestResult]:
    """
    Rolling-window backtest for TSO error correction.

    For each test day:
    1. Train correction model on data up to that day
    2. Predict corrected forecast for target day
    3. Compare TSO-only vs corrected vs actuals

    Retrains every `retrain_every_days` to simulate realistic retraining.

    Args:
        country_code: e.g. "BE"
        renewable_type: "solar", "wind_onshore", "wind_offshore"
        start_date: First test date
        end_date: Last test date
        min_train_days: Minimum training data before first test
        retrain_every_days: How often to retrain the model
        algorithm: "lightgbm" or "xgboost"
        verbose: Print progress

    Returns:
        (tso_result, corrected_result) — both BacktestResult for use with compare_models()
    """
    tso_result = BacktestResult(f"tso_raw_{renewable_type}")
    corrected_result = BacktestResult(f"tso_corrected_{renewable_type}")

    # Load all data
    data_start = (pd.Timestamp(start_date) - timedelta(days=min_train_days + 30)).strftime("%Y-%m-%d")
    data_end = (pd.Timestamp(end_date) + timedelta(days=2)).strftime("%Y-%m-%d")

    if verbose:
        print(f"Loading data {data_start} → {data_end} for {country_code}/{renewable_type}...")

    tso_data = load_tso_vs_actual(country_code, renewable_type, data_start, data_end)
    if tso_data.empty:
        logger.error("No TSO data")
        return tso_result, corrected_result

    weather = load_weather_for_correction(country_code, data_start, data_end)
    if not weather.empty:
        tso_data = pd.merge(tso_data, weather, on="timestamp_utc", how="left")

    tso_data = create_correction_features(tso_data, renewable_type)

    # Generate test dates
    test_dates = pd.date_range(start_date, end_date, freq="1D")
    model = None
    last_train_date = None
    days_since_train = retrain_every_days + 1  # Force train on first iteration

    if verbose:
        print(f"Backtesting {len(test_dates)} days: {start_date} → {end_date}")

    for i, test_date in enumerate(test_dates):
        test_day_str = test_date.strftime("%Y-%m-%d")
        next_day_str = (test_date + timedelta(days=1)).strftime("%Y-%m-%d")

        # Get test day data
        day_mask = (tso_data["timestamp_utc"] >= test_day_str) & (tso_data["timestamp_utc"] < next_day_str)
        day_df = tso_data[day_mask]

        if len(day_df) < 20:
            continue

        # Retrain if needed
        days_since_train += 1
        if days_since_train >= retrain_every_days or model is None:
            train_mask = tso_data["timestamp_utc"] < test_day_str
            train_df = tso_data[train_mask]

            if len(train_df) < min_train_days * 20:
                continue

            model = TSOCorrectionModel(country_code, renewable_type, algorithm)
            n_samples = model.train(train_df)

            if n_samples < 500:
                model = None
                continue

            days_since_train = 0
            if verbose:
                print(f"  [{i+1}/{len(test_dates)}] Retrained on {n_samples} samples")

        if model is None:
            continue

        # Predict
        actuals = day_df["actual_mw"].values
        tso_preds = day_df["tso_forecast_mw"].values
        corrected_preds = model.correct(day_df)

        tso_result.add_day(test_day_str, actuals, tso_preds)
        corrected_result.add_day(test_day_str, actuals, corrected_preds)

        if verbose and (i + 1) % 30 == 0:
            tso_mae = np.mean(np.abs(actuals - tso_preds))
            corr_mae = np.mean(np.abs(actuals - corrected_preds))
            print(f"  [{i+1}/{len(test_dates)}] {test_day_str}: TSO MAE={tso_mae:.1f}, Corrected MAE={corr_mae:.1f}")

    if verbose and corrected_result.n_days > 0:
        tso_agg = tso_result.aggregate_metrics()
        corr_agg = corrected_result.aggregate_metrics()
        sk = skill_score(corr_agg["mae"], tso_agg["mae"])
        print(f"\n{'='*50}")
        print(f"Backtest complete: {corrected_result.n_days} days")
        print(f"  TSO raw:   MAE={tso_agg['mae']:.1f} MW, RMSE={tso_agg['rmse']:.1f} MW")
        print(f"  Corrected: MAE={corr_agg['mae']:.1f} MW, RMSE={corr_agg['rmse']:.1f} MW")
        print(f"  Skill:     {sk*100:+.1f}%")

    return tso_result, corrected_result


# ============================================================================
# SINGLE-SPLIT EVALUATION (kept for quick testing)
# ============================================================================

def train_and_evaluate_correction(
    country_code: str,
    renewable_type: str,
    train_end: str,
    test_start: str,
    test_end: str,
    algorithm: str = "lightgbm",
    save_model: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Train TSO error correction model on a single split and evaluate.
    Optionally save the trained model for production use.
    """
    data_start = (pd.Timestamp(train_end) - timedelta(days=400)).strftime("%Y-%m-%d")

    if verbose:
        print(f"Loading TSO + actuals for {country_code}/{renewable_type}...")

    tso_data = load_tso_vs_actual(country_code, renewable_type, data_start, test_end)
    if tso_data.empty:
        return {"error": "No TSO data"}

    weather = load_weather_for_correction(country_code, data_start, test_end)
    if not weather.empty:
        tso_data = pd.merge(tso_data, weather, on="timestamp_utc", how="left")

    tso_data = create_correction_features(tso_data, renewable_type)

    # Split
    train_mask = tso_data["timestamp_utc"] < train_end
    test_mask = (tso_data["timestamp_utc"] >= test_start) & (tso_data["timestamp_utc"] < test_end)

    train_df = tso_data[train_mask].copy()
    test_df = tso_data[test_mask].copy()

    if verbose:
        print(f"  Train: {len(train_df)} hours, Test: {len(test_df)} hours")

    if len(train_df) < 500 or len(test_df) < 100:
        return {"error": f"Insufficient data: train={len(train_df)}, test={len(test_df)}"}

    # Train
    model = TSOCorrectionModel(country_code, renewable_type, algorithm)
    n_samples = model.train(train_df)

    if verbose:
        print(f"  Trained on {n_samples} samples, features: {len(model._used_cols)}")

    # Predict
    corrected_forecast = model.correct(test_df)
    y_actual = test_df["actual_mw"].values
    y_tso = test_df["tso_forecast_mw"].values

    # Metrics
    tso_metrics = calculate_point_metrics(y_actual, y_tso)
    corrected_metrics = calculate_point_metrics(y_actual, corrected_forecast)

    mae_improvement = skill_score(corrected_metrics["mae"], tso_metrics["mae"])
    rmse_improvement = skill_score(corrected_metrics["rmse"], tso_metrics["rmse"])

    dm_stat, dm_pval, dm_interp = diebold_mariano_test(y_actual, y_tso, corrected_forecast, loss="squared")

    top_features = sorted(model.feature_importance().items(), key=lambda x: -x[1])[:10]

    # Save if requested
    if save_model:
        model.save()

    result = {
        "renewable_type": renewable_type,
        "algorithm": algorithm,
        "train_hours": n_samples,
        "test_hours": len(y_actual),
        "tso_metrics": tso_metrics,
        "corrected_metrics": corrected_metrics,
        "mae_skill": mae_improvement,
        "rmse_skill": rmse_improvement,
        "dm_stat": dm_stat,
        "dm_pval": dm_pval,
        "dm_interpretation": dm_interp,
        "top_features": top_features,
    }

    if verbose:
        print(f"\n  --- {renewable_type} Results ---")
        print(f"  TSO alone:  MAE={tso_metrics['mae']:.1f} MW, RMSE={tso_metrics['rmse']:.1f} MW")
        print(f"  Corrected:  MAE={corrected_metrics['mae']:.1f} MW, RMSE={corrected_metrics['rmse']:.1f} MW")
        print(f"  MAE improvement: {mae_improvement*100:.1f}%")
        print(f"  RMSE improvement: {rmse_improvement*100:.1f}%")
        print(f"  DM test: {dm_interp}")
        print(f"  Top features: {[f[0] for f in top_features[:5]]}")

    return result


# ============================================================================
# FULL EVALUATION + REPORT
# ============================================================================

def run_full_tso_correction_eval(
    country_code: str = "BE",
    start_date: str = "2025-04-01",
    end_date: str = "2026-02-01",
    min_train_days: int = 180,
    retrain_every_days: int = 30,
    verbose: bool = True,
) -> str:
    """
    Run rolling backtest for all renewable types and generate comparison report.
    Uses the same BacktestResult + compare_models infrastructure as Phase 1.
    """
    from evaluation.backtest import compare_models

    types = ["solar", "wind_onshore", "wind_offshore"]
    all_results = []  # pairs of (tso, corrected)

    for rtype in types:
        print(f"\n{'='*60}")
        print(f"Rolling backtest: {rtype}")
        print(f"{'='*60}")

        tso_res, corr_res = run_tso_correction_backtest(
            country_code, rtype,
            start_date=start_date,
            end_date=end_date,
            min_train_days=min_train_days,
            retrain_every_days=retrain_every_days,
            verbose=verbose,
        )
        all_results.append((rtype, tso_res, corr_res))

    # Build report
    lines = [
        "# TSO Error Correction — Rolling Backtest Report",
        "",
        f"**Country:** {country_code}",
        f"**Period:** {start_date} → {end_date}",
        f"**Method:** LightGBM on TSO error residuals, retrained every {retrain_every_days} days",
        f"**Min training:** {min_train_days} days before first prediction",
        "",
    ]

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Type | Days | TSO MAE (MW) | Corrected MAE | MAE Skill | TSO RMSE | Corrected RMSE | RMSE Skill |")
    lines.append("|------|------|-------------|---------------|-----------|----------|---------------|-----------|")

    for rtype, tso_res, corr_res in all_results:
        if tso_res.n_days == 0:
            lines.append(f"| {rtype} | 0 | - | - | - | - | - | - |")
            continue

        tm = tso_res.aggregate_metrics()
        cm = corr_res.aggregate_metrics()
        mae_sk = skill_score(cm["mae"], tm["mae"])
        rmse_sk = skill_score(cm["rmse"], tm["rmse"])

        lines.append(
            f"| {rtype} | {corr_res.n_days} | {tm['mae']:.1f} | {cm['mae']:.1f} | "
            f"{mae_sk*100:+.1f}% | {tm['rmse']:.1f} | {cm['rmse']:.1f} | {rmse_sk*100:+.1f}% |"
        )

    # DM tests per type
    lines.extend(["", "## Statistical Significance (Diebold-Mariano)", ""])
    lines.append("| Type | DM Stat | p-value | Conclusion |")
    lines.append("|------|---------|---------|-----------|")

    for rtype, tso_res, corr_res in all_results:
        if tso_res.n_days < 10:
            continue
        # Align
        dates_t = {d["date"] for d in tso_res.daily_results}
        dates_c = {d["date"] for d in corr_res.daily_results}
        common = sorted(dates_t & dates_c)

        lookup_t = {d["date"]: d for d in tso_res.daily_results}
        lookup_c = {d["date"]: d for d in corr_res.daily_results}

        act, pred_t, pred_c = [], [], []
        for dt in common:
            dt_, dc = lookup_t[dt], lookup_c[dt]
            n = min(len(dt_["actuals"]), len(dc["actuals"]))
            act.extend(dt_["actuals"][:n])
            pred_t.extend(dt_["predictions"][:n])
            pred_c.extend(dc["predictions"][:n])

        dm_stat, dm_pval, dm_interp = diebold_mariano_test(
            np.array(act), np.array(pred_t), np.array(pred_c)
        )
        sig = "***" if dm_pval < 0.01 else ("**" if dm_pval < 0.05 else ("*" if dm_pval < 0.10 else "ns"))
        winner = "Corrected better" if dm_stat > 0 else "TSO better" if dm_stat < 0 else "No difference"
        lines.append(f"| {rtype} | {dm_stat:.3f} | {dm_pval:.4f} | {winner} {sig} |")

    lines.extend(["", f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_"])
    return "\n".join(lines)


# ============================================================================
# PRODUCTION: TRAIN + SAVE MODELS
# ============================================================================

def train_and_save_all(
    country_code: str = "BE",
    train_end: Optional[str] = None,
    algorithm: str = "lightgbm",
):
    """
    Train correction models for all renewable types and save to disk.
    Use this for deploying to production.

    Args:
        country_code: Country code
        train_end: Train on data up to this date. If None, uses yesterday.
        algorithm: "lightgbm" or "xgboost"
    """
    if train_end is None:
        train_end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    types = ["solar", "wind_onshore", "wind_offshore"]
    data_start = (pd.Timestamp(train_end) - timedelta(days=400)).strftime("%Y-%m-%d")

    weather = load_weather_for_correction(country_code, data_start, train_end)

    for rtype in types:
        print(f"\nTraining {rtype}...")
        tso_data = load_tso_vs_actual(country_code, rtype, data_start, train_end)
        if tso_data.empty:
            print(f"  No data, skipping")
            continue

        if not weather.empty:
            tso_data = pd.merge(tso_data, weather, on="timestamp_utc", how="left")

        tso_data = create_correction_features(tso_data, rtype)

        model = TSOCorrectionModel(country_code, rtype, algorithm)
        n = model.train(tso_data)
        model.save()
        print(f"  Saved. Trained on {n} samples.")

    print(f"\nAll models saved to {MODELS_DIR / country_code}/")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.WARNING, format=config.LOG_FORMAT)

    parser = argparse.ArgumentParser(description="TSO Error Correction")
    parser.add_argument("--mode", choices=["backtest", "train", "quick"], default="backtest",
                        help="backtest: rolling eval | train: train+save for production | quick: single-split eval")
    parser.add_argument("--country", default="BE")
    parser.add_argument("--start", default="2025-04-01", help="Backtest start date")
    parser.add_argument("--end", default="2026-02-01", help="Backtest end date")
    parser.add_argument("--retrain-days", type=int, default=30, help="Retrain every N days")
    parser.add_argument("--output", help="Save report to file")

    args = parser.parse_args()

    if args.mode == "train":
        train_and_save_all(args.country, train_end=args.end)

    elif args.mode == "quick":
        # Single-split quick evaluation
        for rtype in ["solar", "wind_onshore", "wind_offshore"]:
            train_and_evaluate_correction(
                args.country, rtype,
                train_end="2025-10-01", test_start="2025-10-01", test_end=args.end,
                save_model=True,
            )

    else:
        # Rolling backtest
        report = run_full_tso_correction_eval(
            country_code=args.country,
            start_date=args.start,
            end_date=args.end,
            retrain_every_days=args.retrain_days,
        )
        print("\n\n" + report)

        output = args.output or str(
            Path(r"C:\Users\guill\.openclaw\workspace\reports\tso_correction_backtest.md")
        )
        Path(output).write_text(report, encoding="utf-8")
        print(f"\nSaved: {output}")
