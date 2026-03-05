#!/usr/bin/env python3
"""Test minimal neighbor features: only lag_1d per country + aggregate mean."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import config
config.ENABLE_NEIGHBOR_PRICE_FEATURES = True
import db, logging
from features import create_all_features, get_feature_columns
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm, numpy as np

logging.basicConfig(level=logging.WARNING)

df = db.load_training_data('BE', 'price', '2023-01-01', '2026-02-17')
df = create_all_features(df, 'price', country_code='BE')
all_cols = [c for c in get_feature_columns('price', country_code='BE') if c in df.columns]
base_cols = [c for c in all_cols if not any(x in c for x in ['price_fr', 'price_nl', 'price_de', 'spread_be', 'neighbor_'])]

# Only lag_1d features (most informative, least noisy)
lag1d_cols = base_cols + [c for c in ['price_fr_lag_1d', 'price_nl_lag_1d', 'price_de_lag_1d', 'neighbor_price_mean_lag_1d'] if c in df.columns]

# Only roll_24h_mean features  
roll_cols = base_cols + [c for c in ['price_fr_roll_24h_mean', 'price_nl_roll_24h_mean', 'price_de_roll_24h_mean', 'neighbor_price_mean_lag_1d'] if c in df.columns]

params = dict(n_estimators=500, max_depth=8, learning_rate=0.05, num_leaves=31,
              min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)

n_folds = 5; test_size = 720; total = len(df)

for label, cols in [("Baseline", base_cols), ("+ lag_1d only (4)", lag1d_cols), ("+ roll_24h only (4)", roll_cols), ("All 13 neighbor", all_cols)]:
    maes = []
    for fold in range(n_folds):
        te = total - fold * test_size; ts = te - test_size
        if ts < test_size: break
        tr, tt = df.iloc[:ts], df.iloc[ts:te]
        m = LGBMRegressor(**params)
        m.fit(tr[cols], tr['target_value'], eval_set=[(tt[cols], tt['target_value'])],
              callbacks=[lightgbm.early_stopping(50, verbose=False)])
        maes.append(mean_absolute_error(tt['target_value'], m.predict(tt[cols])))
    print(f"{label:25s}  f={len(cols):2d}  folds={['%.2f'%m for m in maes]}  AVG={np.mean(maes):.2f}")
