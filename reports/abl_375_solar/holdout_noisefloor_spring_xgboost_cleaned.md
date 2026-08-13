# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T10:31:20 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-03-01 .. 2026-04-29**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are
reported in MW, never as a percentage: their denominator is ~0.

## DE — xgboost, source `energy_renewable`

n_train 2,314 · n_holdout 1,440 (daylight 755 / shoulder 168 / night 517) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 8,420.3 | 38.3% | 40.6 | 16.4 | 0.04 | 1.7 | 0 |
| control@42 | 7,766.9 | 35.3% | 140.6 | 188.5 | 14.73 | 88.1 | 31 |
| control@1337 | 7,835.0 | 35.7% | 207.9 | 256.9 | 78.18 | 91.5 | 31 |
| control@2718 | 8,109.7 | 36.9% | 208.3 | 259.0 | 98.36 | 130.7 | 18 |
| geometry@42 | 7,634.9 | 34.7% | 99.0 | 151.9 | 52.66 | 111.3 | 18 |
| geometry@1337 | 7,954.9 | 36.2% | 243.4 | 299.5 | 197.91 | 244.7 | 3 |
| geometry@2718 | 7,721.9 | 35.1% | 132.2 | 188.1 | 64.16 | 113.2 | 9 |

Training-target contamination: 1 of 1,212 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.
