#!/usr/bin/env python3
"""
Train multipoint models for BE, FR, DE
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Set environment variable for DB path
os.environ['ENERGY_DB_PATH'] = r'C:\Code\energy-data-gathering\energy_dashboard.db'

from src.forecaster import Forecaster
from src.db import get_latest_data_timestamp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_multipoint_models():
    """
    Train multipoint models for BE, FR, DE (wind + solar only)
    """
    
    # Countries and forecast types to train
    countries = ['BE', 'FR', 'DE']
    forecast_types = ['wind_onshore', 'wind_offshore', 'solar']
    
    # Training parameters  
    start_date = '2025-12-15'  # Start of backfilled data
    end_date = '2026-03-14'    # Recent data
    
    logger.info("Starting multipoint model training")
    logger.info(f"Countries: {countries}")
    logger.info(f"Forecast types: {forecast_types}")
    logger.info(f"Training period: {start_date} to {end_date}")
    
    results = {}
    
    for country in countries:
        logger.info(f"\n=== Training models for {country} ===")
        results[country] = {}
        
        for forecast_type in forecast_types:
            try:
                logger.info(f"\nTraining {country}-{forecast_type} (multipoint)")
                
                # Create multipoint forecaster
                forecaster = Forecaster(
                    country_code=country,
                    forecast_type=forecast_type,
                    algorithm='xgboost',
                    weather_mode='multipoint'
                )
                
                # Train the model
                metrics = forecaster.train(
                    start_date=start_date,
                    end_date=end_date,
                    validation_days=7,
                    grid_search=False
                )
                
                # Save the model to multipoint subdirectory
                model_dir = Path(f'models/{country}/{forecast_type}/multipoint')
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / 'model.joblib'
                
                forecaster.save(str(model_path))
                
                # Store results
                results[country][forecast_type] = {
                    'multipoint': metrics,
                    'model_path': str(model_path)
                }
                
                logger.info(f"✓ {country}-{forecast_type} multipoint trained")
                if metrics:
                    logger.info(f"  MAE: {metrics.get('mae', 0):.3f}")
                    logger.info(f"  MAPE: {metrics.get('mape', 0):.1f}%")
                    logger.info(f"  RMSE: {metrics.get('rmse', 0):.3f}")
                
                # Train centroid model for comparison
                logger.info(f"Training {country}-{forecast_type} (centroid)")
                
                centroid_forecaster = Forecaster(
                    country_code=country,
                    forecast_type=forecast_type,
                    algorithm='xgboost',
                    weather_mode='centroid'
                )
                
                centroid_metrics = centroid_forecaster.train(
                    start_date=start_date,
                    end_date=end_date,
                    validation_days=7,
                    grid_search=False
                )
                
                results[country][forecast_type]['centroid'] = centroid_metrics
                
                logger.info(f"✓ {country}-{forecast_type} centroid trained")
                if centroid_metrics:
                    logger.info(f"  MAE: {centroid_metrics.get('mae', 0):.3f}")
                    logger.info(f"  MAPE: {centroid_metrics.get('mape', 0):.1f}%")
                    logger.info(f"  RMSE: {centroid_metrics.get('rmse', 0):.3f}")
                
                # Compare metrics
                if metrics and centroid_metrics:
                    mae_improvement = ((centroid_metrics.get('mae', 0) - metrics.get('mae', 0)) / centroid_metrics.get('mae', 1)) * 100
                    mape_improvement = centroid_metrics.get('mape', 0) - metrics.get('mape', 0)
                    rmse_improvement = ((centroid_metrics.get('rmse', 0) - metrics.get('rmse', 0)) / centroid_metrics.get('rmse', 1)) * 100
                    
                    logger.info(f"  Improvement: MAE {mae_improvement:+.1f}%, MAPE {mape_improvement:+.1f}pp, RMSE {rmse_improvement:+.1f}%")
                
            except Exception as e:
                logger.error(f"Failed to train {country}-{forecast_type}: {e}")
                results[country][forecast_type] = {'error': str(e)}
                continue
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    
    for country in countries:
        logger.info(f"\n{country}:")
        for forecast_type in forecast_types:
            if forecast_type in results[country]:
                result = results[country][forecast_type]
                if 'error' in result:
                    logger.info(f"  {forecast_type}: ❌ {result['error']}")
                else:
                    mp_metrics = result.get('multipoint', {})
                    ct_metrics = result.get('centroid', {})
                    
                    if mp_metrics and ct_metrics:
                        mp_mae = mp_metrics.get('mae', 0)
                        ct_mae = ct_metrics.get('mae', 0)
                        improvement = ((ct_mae - mp_mae) / ct_mae * 100) if ct_mae > 0 else 0
                        status = "🟢" if improvement > 0 else "🔴" if improvement < -1 else "🟡"
                        logger.info(f"  {forecast_type}: {status} {improvement:+.1f}% MAE improvement")
                    else:
                        logger.info(f"  {forecast_type}: ❓ Incomplete metrics")
            else:
                logger.info(f"  {forecast_type}: ❌ Not trained")
    
    return results

if __name__ == "__main__":
    print("Training multipoint models for BE, FR, DE...")
    results = train_multipoint_models()
    print("Training completed!")