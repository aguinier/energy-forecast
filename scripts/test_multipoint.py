#!/usr/bin/env python3
"""
Test multipoint training with a single model (BE wind_onshore)
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Set environment variable for DB path
os.environ['ENERGY_DB_PATH'] = r'C:\Code\energy-data-gathering\energy_dashboard.db'

# Direct imports to avoid complex module dependencies
import config
from forecaster import Forecaster

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_multipoint():
    """
    Test multipoint training with Belgium wind_onshore
    """
    
    logger.info("Testing multipoint training with BE wind_onshore")
    
    try:
        # Create multipoint forecaster
        logger.info("Creating multipoint forecaster...")
        forecaster = Forecaster(
            country_code='BE',
            forecast_type='wind_onshore',
            algorithm='xgboost',
            weather_mode='multipoint'
        )
        
        # Train the model
        logger.info("Training multipoint model...")
        metrics = forecaster.train(
            start_date='2025-12-15',
            end_date='2026-03-14',
            validation_days=7,
            grid_search=False
        )
        
        logger.info("✓ Multipoint training completed")
        if metrics:
            logger.info(f"  MAE: {metrics.get('mae', 0):.3f}")
            logger.info(f"  MAPE: {metrics.get('mape', 0):.1f}%")
            logger.info(f"  RMSE: {metrics.get('rmse', 0):.3f}")
        
        # Test centroid model for comparison
        logger.info("Creating centroid forecaster...")
        centroid_forecaster = Forecaster(
            country_code='BE',
            forecast_type='wind_onshore',
            algorithm='xgboost',
            weather_mode='centroid'
        )
        
        logger.info("Training centroid model...")
        centroid_metrics = centroid_forecaster.train(
            start_date='2025-12-15',
            end_date='2026-03-14',
            validation_days=7,
            grid_search=False
        )
        
        logger.info("✓ Centroid training completed")
        if centroid_metrics:
            logger.info(f"  MAE: {centroid_metrics.get('mae', 0):.3f}")
            logger.info(f"  MAPE: {centroid_metrics.get('mape', 0):.1f}%")
            logger.info(f"  RMSE: {centroid_metrics.get('rmse', 0):.3f}")
        
        # Compare
        if metrics and centroid_metrics:
            mae_improvement = ((centroid_metrics.get('mae', 0) - metrics.get('mae', 0)) / centroid_metrics.get('mae', 1)) * 100
            mape_improvement = centroid_metrics.get('mape', 0) - metrics.get('mape', 0)
            rmse_improvement = ((centroid_metrics.get('rmse', 0) - metrics.get('rmse', 0)) / centroid_metrics.get('rmse', 1)) * 100
            
            logger.info("Comparison:")
            logger.info(f"  MAE improvement: {mae_improvement:+.1f}%")
            logger.info(f"  MAPE improvement: {mape_improvement:+.1f} percentage points")
            logger.info(f"  RMSE improvement: {rmse_improvement:+.1f}%")
            
            if mae_improvement > 0:
                logger.info("🟢 Multipoint shows improvement!")
            elif mae_improvement > -1:
                logger.info("🟡 Multipoint similar to centroid")
            else:
                logger.info("🔴 Multipoint worse than centroid")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multipoint()