#!/usr/bin/env python3
"""Walk-forward validation: with vs without neighbor features."""
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
base_cols = [c for c in all_cols if not any(x in c for x in ['price_fr', 'price_nl', 'price_de', 'spread_be', 'neighbor_'])]

params = dict(n_estimators=500, max_depth=8, learning_rate=0.05, num_leaves=31,
              min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)

# Walk-forward: 5 folds, each 30 days test
n_folds = 5
test_days = 30
test_size = test_days * 24
total = len(df)

print(f"Total samples: {total}, Test size: {test_size}, Folds: {n_folds}")
print()

for label, cols in [("Without neighbor", base_cols), ("With neighbor", all_cols)]:
    maes = []
    for fold in range(n_folds):
        test_end = total - fold * test_size
        test_start = test_end - test_size
        if test_start < test_size:  # not enough train data
            break
        train_part = df.iloc[:test_start]
        test_part = df.iloc[test_start:test_end]
        
        m = LGBMRegressor(**params)
        m.fit(train_part[cols], train_part['target_value'],
              eval_set=[(test_part[cols], test_part['target_value'])],
              callbacks=[lightgbm.early_stopping(50, verbose=False)])
        pred = m.predict(test_part[cols])
        mae = mean_absolute_error(test_part['target_value'], pred)
        maes.append(mae)
    
    avg = np.mean(maes)
    print(f"{label:25s}  features={len(cols):2d}  MAE per fold: {['%.2f'%m for m in maes]}  AVG={avg:.2f}")
