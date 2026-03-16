#!/usr/bin/env python3
"""
Compare forecasts from centroid vs multipoint weather modes
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Set environment variable for DB path
os.environ['ENERGY_DB_PATH'] = r'C:\Code\energy-data-gathering\energy_dashboard.db'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_forecasts(db_path, model_pattern, start_date, end_date):
    """Load forecasts from database"""
    
    try:
        with sqlite3.connect(db_path) as conn:
            query = """
                SELECT 
                    f.country_code,
                    f.forecast_type,
                    f.forecast_date,
                    f.target_hour,
                    f.forecast_value,
                    f.model_name,
                    f.created_at
                FROM forecasts f
                WHERE f.model_name LIKE ?
                  AND f.forecast_date >= ?
                  AND f.forecast_date <= ?
                ORDER BY f.country_code, f.forecast_type, f.forecast_date, f.target_hour
            """
            
            df = pd.read_sql_query(query, conn, params=[model_pattern, start_date, end_date])
            return df
            
    except Exception as e:
        logger.error(f"Error loading forecasts: {e}")
        return pd.DataFrame()

def load_actuals(db_path, start_date, end_date):
    """Load actual energy values for comparison"""
    
    tables_map = {
        'load': 'energy_load',
        'price': 'energy_price', 
        'renewable': 'energy_renewable'
    }
    
    all_actuals = []
    
    try:
        with sqlite3.connect(db_path) as conn:
            
            for forecast_type, table in tables_map.items():
                query = f"""
                    SELECT 
                        country_code,
                        '{forecast_type}' as forecast_type,
                        timestamp_utc,
                        value_mw as actual_value
                    FROM {table}
                    WHERE timestamp_utc >= ?
                      AND timestamp_utc <= ?
                """
                
                df = pd.read_sql_query(query, conn, params=[start_date, end_date])
                if not df.empty:
                    all_actuals.append(df)
            
            if all_actuals:
                return pd.concat(all_actuals, ignore_index=True)
            else:
                return pd.DataFrame()
                
    except Exception as e:
        logger.error(f"Error loading actuals: {e}")
        return pd.DataFrame()

def calculate_metrics(y_true, y_pred):
    """Calculate forecast accuracy metrics"""
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return {}
    
    import numpy as np
    
    # Remove any NaN values
    mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {}
    
    mae = np.mean(np.abs(y_pred_clean - y_true_clean))
    mape = np.mean(np.abs((y_pred_clean - y_true_clean) / y_true_clean)) * 100
    rmse = np.sqrt(np.mean((y_pred_clean - y_true_clean) ** 2))
    
    return {
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'n_samples': len(y_true_clean)
    }

def compare_weather_modes():
    """
    Compare forecasting performance between centroid and multipoint weather modes
    """
    
    db_path = Path(os.environ['ENERGY_DB_PATH'])
    
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return
    
    # Define comparison period
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    logger.info(f"Comparing forecasts from {start_date} to {end_date}")
    
    # Load forecasts from both models
    centroid_forecasts = load_forecasts(db_path, '%centroid%', start_date, end_date)
    multipoint_forecasts = load_forecasts(db_path, '%multipoint%', start_date, end_date)
    
    # Load actual values
    actuals = load_actuals(db_path, start_date, end_date)
    
    logger.info(f"Found {len(centroid_forecasts)} centroid forecasts")
    logger.info(f"Found {len(multipoint_forecasts)} multipoint forecasts")
    logger.info(f"Found {len(actuals)} actual values")
    
    if centroid_forecasts.empty and multipoint_forecasts.empty:
        logger.warning("No forecasts found. Models may not have been deployed yet.")
        
        # Create a simple report about available data instead
        report_content = f"""# Weather Mode Comparison Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Period:** {start_date} to {end_date}

## Status

❌ No forecasts available for comparison

### Available Data
- Multipoint weather data: Available for BE, FR, DE
- Energy data: Available for load, price, renewable
- Forecast data: No forecasts found in database

### Next Steps
1. Train and deploy centroid models
2. Train and deploy multipoint models  
3. Run forecasts for comparison period
4. Re-run this comparison script

### Multipoint Weather Data Summary
"""
        
        # Add multipoint data summary
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        country_code,
                        forecast_type,
                        COUNT(*) as records,
                        MIN(timestamp_utc) as min_date,
                        MAX(timestamp_utc) as max_date
                    FROM weather_data_multipoint 
                    GROUP BY country_code, forecast_type
                    ORDER BY country_code, forecast_type
                """)
                
                results = cursor.fetchall()
                
                report_content += "\n| Country | Forecast Type | Records | Date Range |\n"
                report_content += "|---------|---------------|---------|------------|\n"
                
                for country, ftype, count, min_date, max_date in results:
                    report_content += f"| {country} | {ftype} | {count:,} | {min_date} to {max_date} |\n"
        
        except Exception as e:
            report_content += f"\nError loading multipoint summary: {e}\n"
        
        # Save report
        report_path = Path(__file__).parent.parent / 'reports' / 'dev-validation-report.md'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to {report_path}")
        return
    
    # If we have forecasts, continue with comparison
    results = {}
    
    for country in ['BE', 'FR', 'DE']:
        for ftype in ['wind_onshore', 'wind_offshore', 'solar']:
            
            # Filter forecasts for this combination
            centroid_subset = centroid_forecasts[
                (centroid_forecasts['country_code'] == country) &
                (centroid_forecasts['forecast_type'] == ftype)
            ]
            
            multipoint_subset = multipoint_forecasts[
                (multipoint_forecasts['country_code'] == country) &
                (multipoint_forecasts['forecast_type'] == ftype)
            ]
            
            # Filter actuals
            actuals_subset = actuals[
                (actuals['country_code'] == country) &
                (actuals['forecast_type'] == ftype)
            ]
            
            if centroid_subset.empty or multipoint_subset.empty or actuals_subset.empty:
                continue
            
            # Merge with actuals and compare
            # This is a simplified comparison - in reality would need more sophisticated alignment
            
            results[f"{country}_{ftype}"] = {
                'centroid_count': len(centroid_subset),
                'multipoint_count': len(multipoint_subset),
                'actuals_count': len(actuals_subset)
            }
    
    logger.info(f"Comparison completed for {len(results)} combinations")
    
    # Generate report
    report_content = f"""# Weather Mode Comparison Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Period:** {start_date} to {end_date}

## Summary

Found forecasts for comparison - detailed analysis would be implemented here.

Results: {results}

"""
    
    # Save report
    report_path = Path(__file__).parent.parent / 'reports' / 'dev-validation-report.md'
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Comparison report saved to {report_path}")

if __name__ == "__main__":
    compare_weather_modes()