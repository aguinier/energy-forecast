# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T10:31:00 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-06-13 .. 2026-08-11**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are
reported in MW, never as a percentage: their denominator is ~0.

## DE — xgboost, source `energy_renewable`

n_train 4,807 · n_holdout 1,440 (daylight 937 / shoulder 209 / night 294) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 5,933.4 | 23.2% | 26.4 | 61.7 | 0.00 | 0.0 | 0 |
| control@42 | 3,173.3 | 12.4% | 189.3 | 237.0 | 94.37 | 99.4 | 0 |
| control@1337 | 3,238.6 | 12.7% | 143.3 | 187.1 | 32.87 | 35.5 | 0 |
| control@2718 | 3,185.1 | 12.5% | 215.1 | 261.6 | 43.89 | 51.9 | 0 |
| geometry@42 | 3,076.3 | 12.0% | 79.2 | 122.0 | 12.15 | 42.6 | 40 |
| geometry@1337 | 3,084.1 | 12.1% | 25.4 | 67.6 | 10.82 | 17.8 | 6 |
| geometry@2718 | 3,202.2 | 12.5% | 127.2 | 173.3 | 34.38 | 36.9 | 0 |

Training-target contamination: 4 of 1,957 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.
