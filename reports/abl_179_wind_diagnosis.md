# ABL-179 — wind forecast root-cause evidence pack

**Disposition:** root cause identified; no retraining, registry change, deploy, or
database write performed. Measurements use the live 5.90 GB replica at
`C:\Code\able\data\energy_dashboard.db`, opened only as a SQLite read-only URI.
Its BE/DE/FR renewable data reached 2026-08-10 23:00–23:45 UTC and the standard
BE/NL/AT/FR/DE currency probes reached 2026-08-11 21:00 UTC.

## Verdict

The xgboost interpreter hypothesis is **refuted for the two served
`wind_offshore` artifacts**. Both interpreters preserve the fitted intercepts
and produce bit-identical predictions on a fixed 336-row diagnostic slice.

The shared primary defect is a **train/serve feature-semantic mismatch**:

- training defines `lag_1d`, `lag_7d`, `lag_14d`, and rolling features relative
  to each target row (`src/features.py:251,285-298`);
- serving selects the most recent historical row having the target hour
  (`src/forecaster.py:636-640`) and only overrides calendar fields and the two
  configured wind-speed fields (`src/forecaster.py:642-666`) before prediction;
- both wind types are served at D+1 and D+2 (`config.py:57-58`). Thus a trained
  target-relative 1-day lag is served with an older, run-relative meaning
  (typically target-relative 2 or 3 days), and every rolling statistic is
  anchored to the proxy row rather than constructed under the serving cutoff;
- all five served artifacts include `temperature_c`, but wind inference does
  not replace it because the wind weather allow-list contains only 10 m and
  100 m wind speed (`config.py:263-270`). It therefore remains the proxy row's
  historical temperature rather than target-time forecast temperature.

This mismatch destroys temporal shape, but it is **not sufficient to explain
the level error**. On post-training target-aligned oracle features, correlation
improves materially, while WAPE and positive bias remain severe. The artifacts
are therefore weak/stale as well as incorrectly served; a one-line artifact
reload or affine correction is mathematically incapable of repairing both
level and shape.

## Evidence

### 1. Production scorecard reproduction

Out-of-sample stored-forecast window **2026-07-11 00:00 UTC to 2026-08-10
00:00 UTC exclusive**. The scorecard selects the latest vintage per country,
target, model, and horizon band. Stored vintages are not provably revision-safe
because ingest uses `INSERT OR REPLACE`.

| type | countries | n | model WAPE | D-7 WAPE | TSO WAPE | bias | corr |
|---|---|---:|---:|---:|---:|---:|---:|
| wind_offshore / xgboost | BE, FR | 5,520 | 115.7% | 89.7% | 25.8% | +74.9% | 0.026 |
| wind_onshore / catboost | BE, DE, FR | 8,280 | 76.9% | 71.6% | 13.1% | +38.3% | 0.549 |

Country decomposition on the same window and selection:

| type | country | n | WAPE | mean actual | mean forecast | forecast / actual | bias | corr |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| offshore | BE | 2,760 | 162.1% | 466.6 MW | 1,020.2 MW | 2.19x | +118.7% | 0.034 |
| offshore | FR | 2,760 | 77.7% | 570.0 MW | 792.6 MW | 1.39x | +39.1% | 0.106 |
| onshore | BE | 2,760 | 192.9% | 450.1 MW | 1,289.9 MW | 2.87x | +186.6% | 0.273 |
| onshore | DE | 2,760 | 61.5% | 8,419.0 MW | 9,293.1 MW | 1.10x | +10.4% | -0.202 |
| onshore | FR | 2,760 | 100.4% | 3,329.3 MW | 6,285.1 MW | 1.89x | +88.8% | 0.109 |

The country-specific ratios rule out a common MW↔kW or per-unit↔absolute unit
error. Forecast and actual p99 values are also of the same MW order. WAPE has
no capacity denominator, so a capacity-denominator mismatch cannot create the
reported figure. The score is computed only on the explicit model/country
intersections above, ruling out cross-country pooling of unmatched series.

### 2. Interpreter/artifact discriminating test

Artifacts tested: BE `20251226_155415` and FR `20251226_134328`. Fixed slice:
**2026-07-11 00:00 UTC to 2026-07-18 00:00 UTC exclusive, n=168 per country
(336 total)**. Inputs were a single actual-derived feature matrix used only to
discriminate artifact loading; the resulting WAPE is an oracle-input diagnostic,
not a serve-faithful forecast score.

| artifact | rail xgboost 3.3 intercept | conda xgboost 2.1.4 intercept | max prediction difference | rail / conda mean prediction |
|---|---:|---:|---:|---:|
| BE offshore | 750.23096 MW | 750.23096 MW | 0.0 MW | 1,360.1744 / 1,360.1744 MW |
| FR offshore | 391.4042 MW | 391.4042 MW | 0.0 MW | 991.8242 / 991.8242 MW |

The raw margin means also equal the prediction means under both interpreters.
Neither artifact resets to 0.5. The local artifact feature order exactly equals
the booster feature order (24/24 names) in both countries, ruling out a stored
feature-order mismatch. Stored rows name these exact model versions, so this is
the relevant artifact pair, not an adjacent registry copy.

### 3. Feature-semantic discrimination

No model was fit. Existing frozen artifacts were scored on the post-training
window **2026-07-11 to 2026-08-10 exclusive, n=720 target hours per
country/model** using target-aligned lags/rolling values and actual target-time
weather. These inputs leak information and are deliberately an oracle diagnostic,
not a forecast result and not a promotion number.

| type | country | stored corr (n=2,760 vintages) | oracle-aligned corr (n=720 targets) | oracle WAPE | oracle bias |
|---|---|---:|---:|---:|---:|
| offshore | BE | 0.034 | 0.703 | 123.1% | +117.8% |
| offshore | FR | 0.106 | 0.496 | 61.6% | +40.0% |
| onshore | BE | 0.273 | 0.652 | 182.9% | +182.8% |
| onshore | DE | -0.202 | 0.476 | 45.6% | +7.9% |
| onshore | FR | 0.109 | 0.493 | 96.5% | +92.2% |

Restoring the feature meanings recovers substantial shape in every country,
which discriminates the shared serve-feature defect. It does not recover the
level, which discriminates genuine artifact weakness/staleness as a second
cause. The artifact metadata cannot substitute for a backtest: it records no
validation window, pair count, baseline, or serve-faithful protocol.

### 4. Other hypotheses

- **Timezone/DST:** killed as a primary cause. On one latest row per target,
  shifts across ±6 hours found no common offset; the best shift varies by
  country and never restores useful accuracy. In particular BE onshore is best
  at 0 hours, while offshore BE/FR weakly peak at -6 hours, not the ±1/±2-hour
  signature of UTC/DST misalignment.
- **Feature order:** killed by exact 24-name/order equality in both offshore
  boosters. The serving call also reselects `features[self.feature_columns]` at
  prediction (`src/forecaster.py:690`).
- **Units/scale:** killed as a single mechanism by the non-constant 1.10x–2.87x
  mean ratios and same-order p99 values. Positive, country-specific seasonal
  level bias remains part of artifact weakness.
- **Country set/capacity denominator:** killed for the pooled metric. Only BE/FR
  enter offshore and BE/DE/FR enter onshore; all comparisons use identical
  pairs. WAPE divides by observed MWh-equivalent hourly MW sums, not installed
  capacity.
- **Capacity change:** no common step-change signature. Offshore observed maxima
  remain near historical order (BE about 2.1 GW and FR about 1.8 GW in both the
  artifact era and score window). This does not prove capacity metadata is
  perfect, but it cannot explain both wind types or the destroyed correlation.

## Contamination and vintage caveats

- **ABL-71:** its named live wrong-data paths are zero-filled load and fabricated
  net position, not wind generation. Wind uses the same ingest estate, so this
  is a provenance risk, but no ABL-71 wind predicate is known and no wind rows
  were silently filtered.
- **ABL-67:** affects fabricated GR/IE `net_position` rows only; it does not
  intersect either wind score.
- **ABL-111/ABL-109:** affects zero-valued `energy_load` rows only; it does not
  intersect the wind target table. It remains evidence that the shared actuals
  pipeline can encode missingness as zero.
- Stored forecast and TSO vintages are not provably the issued values because
  replacement ingest destroys first-seen history. The root-cause conclusion
  does not require TSO to be revision-safe: it follows from frozen artifacts,
  code semantics, and model-vs-actual pairs. TSO is retained only as the
  scorecard baseline and must not be treated as a D+2 drop-in.

## Recommended fix and owner

1. **Founding Engineer (serving-path owner):** replace the proxy-row inference
   construction with one explicit, shared train/serve feature builder parameterized
   by target timestamp, observation `as_of`, and weather `publication_as_of`.
   Remove target-relative features unavailable at the D+1/D+2 cutoff or redefine
   them identically in training. Ensure every saved feature is populated for the
   target meaning (including temperature), add a golden D+1/D+2 feature-vector
   test, and add the existing xgboost intercept witness/native-save guard to these
   legacy artifacts. This is an evidence handoff, not a patch or deploy request.
2. **Forecasting Scientist:** after that builder is reviewed, train new wind
   artifacts and run a pre-registered, serve-faithful out-of-sample gate against
   D-7/persistence and first-seen TSO vintages. Do not calibrate the current
   artifacts: their correlations show a shape failure, so affine correction
   cannot make them competitive.
3. **CEO:** decide any temporary registry/suspension action and escalate it to
   the Board. No alternative is recommended here: current TSO rows are revised
   D+1 products, not revision-safe D+2 substitutes.

