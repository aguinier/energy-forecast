> **Archived from `CLAUDE.md` on 2026-08-27** (companion to the ABL-536
> energy-dashboard-frontend trim). Historical narrative, incident forensics
> and dated measurements; `file:line` references are frozen as of the archive
> date. The durable rules distilled from this material live in the repo-root
> `CLAUDE.md`; where they conflict, the root file wins.
# Model Storage

## Model Storage

Models are stored in a filesystem-based structure with embedded metadata:

```
models/
├── {country_code}/
│   ├── {forecast_type}/
│   │   └── model.joblib    # XGBoost model + metadata
```

**Example:**
```
models/
├── DE/
│   ├── load/model.joblib
│   ├── solar/model.joblib
│   └── wind_onshore/model.joblib
└── FR/
    ├── load/model.joblib
    ├── price/model.joblib
    └── renewable/model.joblib
```

**Metadata Structure:**
Each `.joblib` file contains a Python dictionary with:
- `model`: Trained XGBRegressor instance
- `feature_columns`: List of feature names used during training
- `country_code`: ISO 2-letter country code
- `forecast_type`: Type of forecast (load, price, renewable, solar, etc.)
- `model_version`: Timestamp of training (YYYYMMDD_HHMMSS)
- `training_metrics`: Dict with MAE, MAPE, RMSE, SMAPE
- `saved_at`: ISO timestamp of when model was saved

**Key Points:**
- No separate JSON metadata file - all metadata embedded in joblib
- Only latest model version kept per country/type (no historical versions)
- Models discovered via filesystem traversal
- Typical model size: 2-6 MB depending on number of features
