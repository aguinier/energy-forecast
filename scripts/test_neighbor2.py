#!/usr/bin/env python3
"""Test neighbor features with different configs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import config
config.ENABLE_NEIGHBOR_PRICE_FEATURES = True
import db
import logging
from features import create_all_features, get_feature_columns
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm
import numpy as np

logging.basicConfig(level=logging.WARNING)

df = db.load_training_data('BE', 'price', '2023-01-01', '2026-02-17')
df = create_all_features(df, 'price', country_code='BE')
all_cols = [c for c in get_feature_columns('price', country_code='BE') if c in df.columns]

# Baseline cols (no neighbor)
base_cols = [c for c in all_cols if not any(x in c for x in ['price_fr', 'price_nl', 'price_de', 'spread_be', 'neighbor_'])]

# Top neighbor features only
top_neighbor = ['price_fr_roll_24h_mean', 'price_de_roll_24h_mean', 'price_nl_roll_24h_mean', 'neighbor_price_mean_lag_1d']
selective_cols = base_cols + [c for c in top_neighbor if c in df.columns]

val_size = 720
train, val = df.iloc[:-val_size], df.iloc[-val_size:]

configs = {
    'Baseline (no neighbor)': (base_cols, dict(n_estimators=500, max_depth=8, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)),
    'All 13 neighbor (default)': (all_cols, dict(n_estimators=500, max_depth=8, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)),
    'All 13 neighbor (reg)': (all_cols, dict(n_estimators=1000, max_depth=6, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.7, colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=2.0, random_state=42, verbose=-1)),
    'Top 4 neighbor (default)': (selective_cols, dict(n_estimators=500, max_depth=8, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)),
    'Top 4 neighbor (reg)': (selective_cols, dict(n_estimators=1000, max_depth=6, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.7, colsample_bytree=0.6, reg_alpha=0.3, reg_lambda=1.0, random_state=42, verbose=-1)),
}

for name, (cols, params) in configs.items():
    m = LGBMRegressor(**params)
    m.fit(train[cols], train['target_value'],
          eval_set=[(val[cols], val['target_value'])],
          callbacks=[lightgbm.early_stopping(50, verbose=False)])
    pred = m.predict(val[cols])
    mae = mean_absolute_error(val['target_value'], pred)
    rmse = np.sqrt(np.mean((val['target_value'].values - pred)**2))
    print(f"{name:35s}  features={len(cols):2d}  MAE={mae:.2f}  RMSE={rmse:.2f}")
