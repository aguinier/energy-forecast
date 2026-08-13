# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T10:30:46 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-06-13 .. 2026-08-11**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are
reported in MW, never as a percentage: their denominator is ~0.

## DE — catboost, source `energy_renewable`

n_train 4,807 · n_holdout 1,440 (daylight 937 / shoulder 209 / night 294) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 5,933.4 | 23.2% | 26.4 | 61.7 | 0.00 | 0.0 | 0 |
| control@42 | 3,852.2 | 15.1% | 284.9 | 304.3 | 220.12 | 669.6 | 52 |
| control@1337 | 3,542.4 | 13.9% | 157.4 | 162.3 | 26.94 | 293.6 | 136 |
| control@2718 | 3,504.5 | 13.7% | 203.3 | 216.6 | 23.62 | 320.2 | 138 |
| geometry@42 | 3,694.8 | 14.5% | 501.4 | 549.3 | 453.76 | 815.3 | 7 |
| geometry@1337 | 3,668.0 | 14.4% | 337.5 | 383.9 | 264.38 | 568.8 | 20 |
| geometry@2718 | 3,443.5 | 13.5% | 300.8 | 347.1 | 200.52 | 352.9 | 4 |

Training-target contamination: 4 of 1,957 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.
