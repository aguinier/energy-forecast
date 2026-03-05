# Energy Forecast

D+1/D+2 energy forecasting for European electricity markets using XGBoost, LightGBM, and CatBoost. Generates hourly forecasts for load, price, and renewable generation across 24 countries.

Migrated from the [energy-dashboard](https://github.com/aguinier/energy-dashboard) monorepo.

## Quick Start

```bash
pip install -r requirements.txt
export ENERGY_DB_PATH=/path/to/energy_dashboard.db

# Train models
python scripts/train.py --countries DE,FR --types load,price,renewable

# Generate forecasts
python scripts/forecast_daily.py
```

## Docker

```bash
cd docker
# Set DB_DIR and MODELS_DIR in .env
docker compose up -d --build
```

The container runs forecast cron jobs at 07:00, 14:00, 15:30, 19:00 UTC.

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed module documentation including features, model configuration, and evaluation queries.
