# ABL-443 — DE/NL `wind_offshore` re-read at the trailing causal reference

Generated: 2026-08-14T02:09:36Z. Scope: **`abl443-offshore-trailing`** — a new scope, not an edit to `abl322-pilot`.
Registration: `experiments/ABL443/config.json`, committed before this read existed. Levelling inherited from `experiments/ABL437/config.json`.

Levelling: **`fit_window` → `trailing_28d`** (28-day window ending at each row's own `generated_at`). Arithmetic over ABL-436's committed record plus the two trailing references recomputed on the same rows — **no refit, no new model**, replica opened read-only.
Source record: `experiments/ABL322/results_abl436_offshore_reread.json` (SHA-256 `ef0f647449072fa0…`), source table `energy_generation`.
Replica: `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes).

**ABL-436's read is not edited, regenerated or withdrawn by this one.** It stands at its own path under its own scope, and its letters remain the letters decided on the fit-window references.

## 1. The row set is proved, not assumed

Each cell's rows are rebuilt from ABL-348's eight registered run instants and then checked by recomputing that cell's published `constant_causal` and `climatology_causal` WAPE *and* MAE from it, to 1e-09. A constant and a 24-bucket climatology agreeing on two statistics each is the row set; one agreeing alone would not be.

**6 of 6 cells reconstructed.** Every cell. Route: 6 on the schedule alone, 0 through the harness's own feature build.

## 2. The two grades, restated

**1 of 2 pairs move.** A pair grades on its worst band.

| pair | published (fit-window) | amended (trailing 28d) | what changed |
|---|:---:|:---:|---|
| DE wind_offshore | A | **B** | now fails G3 |
| NL wind_offshore | A | **A** | no condition changes outcome |

## 3. Every margin, in every case

The issue asks for the margin in every case, including where the verdict is *not readable*. `skill` is `(reference − challenger) / reference`, in percent — positive means the challenger is better. **G2 and G3 are sign tests**: ABL-437 declined to widen them to a floor test and this read does not either, so the `7.51%` k=1 wind floor below is a **diagnostic on the margin, never a ladder condition**. A letter that turns on a sub-floor margin is reported as the ladder computes it *and* flagged as not demonstrated at one seed.

| pair | band | n | reference | condition | challenger WAPE | reference WAPE | skill | vs floor |
|---|---|---:|---|---|---:|---:|---:|---|
| DE wind_offshore | 24-36h | 720 | `seasonal_naive` | G1 | 66.11% | 88.86% | +25.60pp | readable |
| DE wind_offshore | 24-36h | 720 | `constant_causal_28d` | G2 | 66.11% | 66.33% | +0.33pp | not readable at one seed |
| DE wind_offshore | 24-36h | 720 | `climatology_causal_28d` | G3 | 66.11% | 66.45% | +0.52pp | not readable at one seed |
| DE wind_offshore | 24-36h | 720 | `constant_causal` | G2 (as published) | 66.11% | 74.40% | +11.14pp | readable |
| DE wind_offshore | 24-36h | 720 | `climatology_causal` | G3 (as published) | 66.11% | 73.74% | +10.34pp | readable |
| DE wind_offshore | 24-36h | 720 | `constant_oracle` | reported only | 66.11% | 62.86% | -5.17pp | not readable at one seed |
| DE wind_offshore | 24-36h | 720 | `climatology_oracle` | reported only | 66.11% | 62.12% | -6.42pp | not readable at one seed |
| DE wind_offshore | 36-48h | 720 | `seasonal_naive` | G1 | 65.66% | 88.86% | +26.11pp | readable |
| DE wind_offshore | 36-48h | 720 | `constant_causal_28d` | G2 | 65.66% | 66.54% | +1.32pp | not readable at one seed |
| DE wind_offshore | 36-48h | 720 | `climatology_causal_28d` | G3 | 65.66% | 66.53% | +1.32pp | not readable at one seed |
| DE wind_offshore | 36-48h | 720 | `constant_causal` | G2 (as published) | 65.66% | 74.40% | +11.75pp | readable |
| DE wind_offshore | 36-48h | 720 | `climatology_causal` | G3 (as published) | 65.66% | 73.74% | +10.96pp | readable |
| DE wind_offshore | 36-48h | 720 | `constant_oracle` | reported only | 65.66% | 62.86% | -4.45pp | not readable at one seed |
| DE wind_offshore | 36-48h | 720 | `climatology_oracle` | reported only | 65.66% | 62.12% | -5.69pp | not readable at one seed |
| DE wind_offshore | 48-64h | 510 | `seasonal_naive` | G1 | 66.15% | 87.09% | +24.05pp | readable |
| DE wind_offshore | 48-64h | 510 | `constant_causal_28d` | G2 | 66.15% | 66.83% | +1.03pp | not readable at one seed |
| DE wind_offshore | 48-64h | 510 | `climatology_causal_28d` | G3 | 66.15% | 65.84% | -0.47pp | not readable at one seed |
| DE wind_offshore | 48-64h | 510 | `constant_causal` | G2 (as published) | 66.15% | 75.79% | +12.72pp | readable |
| DE wind_offshore | 48-64h | 510 | `climatology_causal` | G3 (as published) | 66.15% | 73.65% | +10.18pp | readable |
| DE wind_offshore | 48-64h | 510 | `constant_oracle` | reported only | 66.15% | 62.32% | -6.15pp | not readable at one seed |
| DE wind_offshore | 48-64h | 510 | `climatology_oracle` | reported only | 66.15% | 61.31% | -7.89pp | readable loss |
| NL wind_offshore | 24-36h | 720 | `seasonal_naive` | G1 | 60.46% | 81.79% | +26.08pp | readable |
| NL wind_offshore | 24-36h | 720 | `constant_causal_28d` | G2 | 60.46% | 77.29% | +21.78pp | readable |
| NL wind_offshore | 24-36h | 720 | `climatology_causal_28d` | G3 | 60.46% | 77.08% | +21.57pp | readable |
| NL wind_offshore | 24-36h | 720 | `constant_causal` | G2 (as published) | 60.46% | 89.97% | +32.80pp | readable |
| NL wind_offshore | 24-36h | 720 | `climatology_causal` | G3 (as published) | 60.46% | 87.28% | +30.73pp | readable |
| NL wind_offshore | 24-36h | 720 | `constant_oracle` | reported only | 60.46% | 71.74% | +15.73pp | readable |
| NL wind_offshore | 24-36h | 720 | `climatology_oracle` | reported only | 60.46% | 70.88% | +14.71pp | readable |
| NL wind_offshore | 36-48h | 720 | `seasonal_naive` | G1 | 61.26% | 81.79% | +25.10pp | readable |
| NL wind_offshore | 36-48h | 720 | `constant_causal_28d` | G2 | 61.26% | 77.32% | +20.77pp | readable |
| NL wind_offshore | 36-48h | 720 | `climatology_causal_28d` | G3 | 61.26% | 77.23% | +20.68pp | readable |
| NL wind_offshore | 36-48h | 720 | `constant_causal` | G2 (as published) | 61.26% | 89.97% | +31.91pp | readable |
| NL wind_offshore | 36-48h | 720 | `climatology_causal` | G3 (as published) | 61.26% | 87.28% | +29.81pp | readable |
| NL wind_offshore | 36-48h | 720 | `constant_oracle` | reported only | 61.26% | 71.74% | +14.62pp | readable |
| NL wind_offshore | 36-48h | 720 | `climatology_oracle` | reported only | 61.26% | 70.88% | +13.58pp | readable |
| NL wind_offshore | 48-64h | 510 | `seasonal_naive` | G1 | 63.75% | 88.51% | +27.97pp | readable |
| NL wind_offshore | 48-64h | 510 | `constant_causal_28d` | G2 | 63.75% | 79.16% | +19.46pp | readable |
| NL wind_offshore | 48-64h | 510 | `climatology_causal_28d` | G3 | 63.75% | 78.51% | +18.80pp | readable |
| NL wind_offshore | 48-64h | 510 | `constant_causal` | G2 (as published) | 63.75% | 92.99% | +31.44pp | readable |
| NL wind_offshore | 48-64h | 510 | `climatology_causal` | G3 (as published) | 63.75% | 88.41% | +27.89pp | readable |
| NL wind_offshore | 48-64h | 510 | `constant_oracle` | reported only | 63.75% | 73.27% | +12.99pp | readable |
| NL wind_offshore | 48-64h | 510 | `climatology_oracle` | reported only | 63.75% | 72.34% | +11.87pp | readable |

## 4. Every cell, both levellings

`c` = constant, `clim` = climatology. `inflation` is each causal reference's WAPE over the oracle constant's — the residual mis-levelling, which the trailing window **reduces rather than removes**. Do not quote the corrected reference as exact.

| pair | band | n | challenger | D-7 | c causal | c 28d | c oracle | clim causal | clim 28d | clim oracle | inflation causal / 28d | enough pairs | published | amended |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| DE wind_offshore | 24-36h | 720 | 66.11% | 88.86% | 74.40% | 66.33% | 62.86% | 73.74% | 66.45% | 62.12% | 18.36% / 5.52% | yes | A | A |
| DE wind_offshore | 36-48h | 720 | 65.66% | 88.86% | 74.40% | 66.54% | 62.86% | 73.74% | 66.53% | 62.12% | 18.36% / 5.85% | yes | A | A |
| DE wind_offshore | 48-64h | 510 | 66.15% | 87.09% | 75.79% | 66.83% | 62.32% | 73.65% | 65.84% | 61.31% | 21.62% / 7.25% | yes | A | B |
| NL wind_offshore | 24-36h | 720 | 60.46% | 81.79% | 89.97% | 77.29% | 71.74% | 87.28% | 77.08% | 70.88% | 25.40% / 7.73% | yes | A | A |
| NL wind_offshore | 36-48h | 720 | 61.26% | 81.79% | 89.97% | 77.32% | 71.74% | 87.28% | 77.23% | 70.88% | 25.40% / 7.77% | yes | A | A |
| NL wind_offshore | 48-64h | 510 | 63.75% | 88.51% | 92.99% | 79.16% | 73.27% | 88.41% | 78.51% | 72.34% | 26.91% / 8.03% | yes | A | A |

## 5. The trailing reference's own levels

The published fit-window constant beside the range the trailing window actually took across each cell's issue instants. A trailing window that never moves is a fit-window constant wearing a different name; one that moves a long way is carrying the level change the amendment exists to catch.

| pair | band | as-of instants | c causal (fixed) | c 28d min | c 28d mean | c 28d max | c oracle |
|---|---|---:|---:|---:|---:|---:|---:|
| DE wind_offshore | 24-36h | 121 | 3263.8 MW | 1780.9 MW | 2343.6 MW | 2865.9 MW | 2006.0 MW |
| DE wind_offshore | 36-48h | 121 | 3263.8 MW | 1780.9 MW | 2333.5 MW | 2865.9 MW | 2006.0 MW |
| DE wind_offshore | 48-64h | 120 | 3263.8 MW | 1780.9 MW | 2330.8 MW | 2865.9 MW | 2006.0 MW |
| NL wind_offshore | 24-36h | 121 | 1704.1 MW | 946.3 MW | 1129.4 MW | 1226.5 MW | 870.3 MW |
| NL wind_offshore | 36-48h | 121 | 1704.1 MW | 946.3 MW | 1128.3 MW | 1226.5 MW | 870.3 MW |
| NL wind_offshore | 48-64h | 120 | 1704.1 MW | 946.3 MW | 1127.8 MW | 1226.5 MW | 870.3 MW |

## 6. What this read did not touch

| path | why |
|---|---|
| `experiments/ABL322/results_abl436_offshore_reread.json` | ABL-436's committed record. Read, hashed, never written. |
| `reports/abl_436_offshore_reference_grade.md` | ABL-436's published gate report. |
| `reports/abl_436_offshore_evidence_pack.md` | ABL-436's published evidence pack. |
| `reports/abl_437_causal_levelling_reread.{json,md}` | ABL-437's re-read, which does not cover these two pairs. |
| `reports/abl_322_pilot_gate.md`, `experiments/ABL322/results.json` | the `abl322-pilot` scope's own outputs. |

The refusal is in the script (`PROTECTED`), not in the operator's memory.

Read-only on the replica. No model was loaded, fitted or scored again.
