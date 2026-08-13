# ABL-338 — solar non-negativity and solar geometry: held-out A/B

Generated 2026-08-13T10:31:05 against replica `C:\Code\able\data\energy_dashboard.db`.

Holdout **2026-03-01 .. 2026-04-29**, training from 2023-01-01 up to the holdout start.

Every arm is a **refit** on the identical truncated window — the live artifacts
were fitted through roughly today, so scoring them here would be in-sample.
Features come from the training-time pipeline, so these MW are optimistic
against the serve path; the arms carry that identically. Night hours are
reported in MW, never as a percentage: their denominator is ~0.

## DE — catboost, source `energy_renewable`

n_train 2,314 · n_holdout 1,440 (daylight 755 / shoulder 168 / night 517) · incumbent version 20260223_193822

| arm | daylight MAE | daylight WAPE | shoulder MAE | shoulder mean pred | night mean pred | night max pred | negative preds |
|---|---:|---:|---:|---:|---:|---:|---:|
| _seasonal-naive D-7_ | 8,420.3 | 38.3% | 40.6 | 16.4 | 0.04 | 1.7 | 0 |
| control@42 | 10,160.2 | 46.2% | 390.8 | 445.3 | 247.68 | 664.3 | 1 |
| control@1337 | 10,541.9 | 48.0% | 307.0 | 360.5 | 161.65 | 561.4 | 6 |
| control@2718 | 10,401.1 | 47.3% | 339.5 | 392.7 | 230.65 | 513.6 | 0 |
| geometry@42 | 8,866.6 | 40.4% | 376.1 | 432.3 | 190.18 | 633.4 | 12 |
| geometry@1337 | 9,285.2 | 42.3% | 416.3 | 472.6 | 293.29 | 608.3 | 12 |
| geometry@2718 | 9,055.4 | 41.2% | 233.3 | 284.6 | 154.63 | 525.2 | 7 |

Training-target contamination: 1 of 1,212 night rows read above 1 MW (max 1.7 MW); dropped from fit: True.
