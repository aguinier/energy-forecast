#!/usr/bin/env python3
"""Quick A/B test: with vs without neighbor price features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import config
import db
import logging
from features import create_all_features, get_feature_columns
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm
import numpy as np

logging.basicConfig(level=logging.WARNING)

val_size = 720
lgbm_params = dict(n_estimators=500, max_depth=8, learning_rate=0.05,
                    num_leaves=31, min_child_samples=20, subsample=0.8,
                    colsample_bytree=0.8, random_state=42, verbose=-1)

# --- WITHOUT neighbor features ---
config.ENABLE_NEIGHBOR_PRICE_FEATURES = False
df1 = db.load_training_data('BE', 'price', '2023-01-01', '2026-02-17')
df1 = create_all_features(df1, 'price', country_code='BE')
cols1 = [c for c in get_feature_columns('price', country_code='BE') if c in df1.columns]
print(f"Features WITHOUT neighbor: {len(cols1)}")

train1, val1 = df1.iloc[:-val_size], df1.iloc[-val_size:]
m1 = LGBMRegressor(**lgbm_params)
m1.fit(train1[cols1], train1['target_value'],
       eval_set=[(val1[cols1], val1['target_value'])],
       callbacks=[lightgbm.early_stopping(50, verbose=False)])
pred1 = m1.predict(val1[cols1])
mae1 = mean_absolute_error(val1['target_value'], pred1)
rmse1 = np.sqrt(mean_squared_error(val1['target_value'], pred1))
print(f"  MAE:  {mae1:.2f}")
print(f"  RMSE: {rmse1:.2f}")

# --- WITH neighbor features ---
config.ENABLE_NEIGHBOR_PRICE_FEATURES = True
df2 = db.load_training_data('BE', 'price', '2023-01-01', '2026-02-17')
df2 = create_all_features(df2, 'price', country_code='BE')
cols2 = [c for c in get_feature_columns('price', country_code='BE') if c in df2.columns]
print(f"\nFeatures WITH neighbor: {len(cols2)}")

train2, val2 = df2.iloc[:-val_size], df2.iloc[-val_size:]
m2 = LGBMRegressor(**lgbm_params)
m2.fit(train2[cols2], train2['target_value'],
       eval_set=[(val2[cols2], val2['target_value'])],
       callbacks=[lightgbm.early_stopping(50, verbose=False)])
pred2 = m2.predict(val2[cols2])
mae2 = mean_absolute_error(val2['target_value'], pred2)
rmse2 = np.sqrt(mean_squared_error(val2['target_value'], pred2))
print(f"  MAE:  {mae2:.2f}")
print(f"  RMSE: {rmse2:.2f}")

# --- Comparison ---
print(f"\n--- COMPARISON ---")
print(f"MAE  change: {mae1:.2f} -> {mae2:.2f} ({(mae2-mae1)/mae1*100:+.1f}%)")
print(f"RMSE change: {rmse1:.2f} -> {rmse2:.2f} ({(rmse2-rmse1)/rmse1*100:+.1f}%)")

# Feature importance for neighbor features
print(f"\nTop 20 features (with neighbors):")
pairs = sorted(zip(cols2, m2.feature_importances_), key=lambda x: -x[1])
for n, i in pairs[:20]:
    marker = " <-- NEIGHBOR" if any(x in n for x in ['price_fr', 'price_nl', 'price_de', 'spread_be', 'neighbor_']) else ""
    print(f"  {i:5.0f}  {n}{marker}")

print(f"\nNeighbor features only:")
for n, i in pairs:
    if any(x in n for x in ['price_fr', 'price_nl', 'price_de', 'spread_be', 'neighbor_']):
        print(f"  {i:5.0f}  {n}")
