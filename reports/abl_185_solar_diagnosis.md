# ABL-185 — solar forecast root-cause evidence pack

**Disposition:** root cause identified; no retraining, deploy, registry change,
ingest/dashboard change, or database write performed. All database reads used the
SQLite read-only URI against the 5.90 GB live replica at
`C:\Code\able\data\energy_dashboard.db`; all artifact loads used the rail
interpreter (`.venv`, Python 3.14.3). The replica reached 2026-08-10 23:45 UTC
for DE/FR solar and 23:00 UTC for BE solar.

## Verdict

Solar **does share the ABL-179 train/serve feature-semantic defect, in part**.
The same serving path copies a historical same-hour feature row, then overrides
calendar and radiation fields. Target-relative lags and rolling statistics
therefore retain proxy-row meanings at D+1/D+2. This fix belongs in **ABL-183**;
no parallel serving implementation should be opened.

It is not the main explanation of pooled 48.4% WAPE. The principal cause is the
**DE artifact's severe level failure**:

- DE is 66.3% of the pooled observed-energy denominator but 85.5% of absolute
  model error. Its stored mean forecast is only 38.6% of mean actual generation.
- DE retains excellent shape (stored correlation 0.957 on one latest row per
  target), while target-aligned oracle inputs still score 61.7% WAPE with
  -60.6% bias and correlation 0.966. Correcting serve semantics cannot repair
  that level.
- The DE artifact was saved 2026-02-23 and CatBoost's first-tree leaf weights
  prove it was fit on only **3,208 rows**, despite `MIN_TRAINING_HOURS = 8,760`.
  The minimum is warning-only. The current replica has no DE solar history
  before 2025-09-08 and every one of the 5,096 quarter-hour rows from then
  through 2025-10-31 is exactly zero. Artifact metadata retains no training
  timestamps/data digest, so exact inclusion cannot be reconstructed, but this
  is a direct short-history/contaminated-training risk in the only possible
  pre-save source window.

BE and FR are materially better but still lose: their latest-target WAPEs are
23.2% and 20.0%. Restoring target-aligned meanings lowers those oracle-input
diagnostics to 18.4% and 12.5%, respectively. Thus ABL-183 matters, but a fresh
serve-faithful artifact and gate are still required afterward.

## Evidence

### 1. Production-number reproduction (out of sample)

Target window **2026-07-11 00:00 UTC to 2026-08-10 00:00 UTC exclusive**.
Artifacts were saved 2026-02-01 (BE/FR) and 2026-02-23 (DE), so this is
post-training/out-of-sample. The ABL-129 protocol selects the latest CatBoost
vintage per country, target and horizon band. Sample: **n=8,280 stored forecast
pairs** (2,760 per country; targets repeat across horizon bands).

| model/baseline | n | WAPE | MAE | bias | slope | corr | model skill |
|---|---:|---:|---:|---:|---:|---:|---:|
| CatBoost solar | 8,280 | **48.35%** | 4,283.6 MW | **-43.31%** | 0.380 | 0.853 | — |
| D-7 seasonal naive, identical pairs | 8,280 | **24.61%** | 2,180.4 MW | -1.19% | 0.912 | 0.944 | **-96.5%** |
| latest stored TSO D+1, identical pairs | 8,280 | **5.93%** | 525.0 MW | +1.94% | 1.005 | 0.997 | **-716.0%** |

This reproduces ABL-128 rev 2's 48.4% / ~5.9% claim. The earlier -50.7% bias
does **not** reproduce; the explicit harness measures -43.31%, which supersedes
it. TSO is a revised latest-stored D+1 series, not an as-issued D+2 substitute;
its unavailable vintage history makes 5.93% an optimistic reference, not a
serving recommendation.

Country decomposition on the same n=8,280 selection:

| country | n | WAPE | bias | corr | mean forecast / actual | denominator share | absolute-error share |
|---|---:|---:|---:|---:|---:|---:|---:|
| BE | 2,760 | 23.42% | -13.41% | 0.946 | 0.866x | 8.5% | 4.1% |
| **DE** | 2,760 | **62.37%** | **-61.36%** | **0.956** | **0.386x** | **66.3%** | **85.5%** |
| FR | 2,760 | 19.89% | -5.93% | 0.957 | 0.941x | 25.2% | 10.4% |

### 2. Oracle-input discriminator (diagnostic, deliberately leaky)

No model was fit. Existing frozen artifacts were scored on target-aligned
features over the same post-training target window, **n=720 unique target hours
per country (2,160 pooled)**. These rows use actual target-time weather and
target-relative lags/rolling features, including information unavailable at
serve time. They are oracle-input diagnostics, not forecast scores or promotion
evidence.

| country | stored latest-target WAPE / corr (n=720) | oracle WAPE | oracle bias | oracle corr |
|---|---:|---:|---:|---:|
| BE | 23.21% / 0.956 | **18.41%** | -10.68% | **0.975** |
| DE | 62.40% / 0.957 | **61.68%** | -60.59% | **0.966** |
| FR | 19.96% / 0.962 | **12.49%** | -5.03% | **0.983** |

Pooled oracle result: **45.58% WAPE, n=2,160, -42.32% bias, corr 0.865**, versus
the stored 48.35% headline (whose repeated horizon-band sample is not a direct
pairwise oracle comparison). The improvement in BE/FR proves the shared serve
defect. DE's near-unchanged WAPE with stronger correlation proves a genuinely
weak/underscaled artifact as the dominant second cause.

Replacing actual radiation in the aligned oracle with the latest stored
forecast radiation changes WAPE only modestly: BE 18.4→19.6%, DE 61.7→62.6%,
FR 12.5→12.3% (n=720 each). Latest stored weather is not revision-safe, but this
is enough to reject irradiance forecast error as the 48% primary mechanism.

### 3. Other hypotheses

- **Radiation/cloud inputs:** serving does refresh shortwave, direct and diffuse
  radiation. Latest-stored shortwave forecast/actual correlation is 0.950 BE,
  0.978 DE and 0.970 FR (n=720 each). No cloud-cover feature exists; forecast
  radiation already embodies the cloud effect available to this model. Direct
  and diffuse radiation have zero importance in BE/FR and only 1.77%/1.53% in
  DE; shortwave is the relevant weather input. Weather revisions prevent an
  as-served weather claim, but the oracle discriminator rejects this as primary.
- **Temperature sub-defect:** unlike the ABL-179 description, the generic
  inference block recomputes `temperature_c` whenever forecast temperature is
  present, even though temperature is absent from solar's weather allow-list.
  Solar therefore shares the proxy lag/rolling defect, **not** an ordinarily
  anchored-temperature defect.
- **Night hours/denominator:** all measured hours, including zeros, are scored.
  Zero actuals add error to the numerator and nothing to the WAPE denominator.
  Removing actual ≤10 MW hours changes latest-target WAPE only BE 23.21→22.57%,
  DE 62.40→61.98%, FR 19.96→19.80%; night rows contribute only 0.69% of pooled
  absolute error. Night handling cannot explain the result.
- **Clipping:** there is no clipping in the serving path. There are 999 negative
  predictions among 8,280 stored rows, concentrated at night in BE/FR, but DE
  has none and drives the pooled error. Zero-clipping could remove at most the
  small night contribution; it cannot repair the level failure.
- **Capacity normalization:** none exists in training or serving. Country mean
  ratios (0.386x DE, 0.866x BE, 0.941x FR) rule out a common unit/per-unit error.
  DE's forecast maximum is 20.4 GW against 54.7 GW observed in the scored rows;
  this is artifact extrapolation/level weakness, not a misapplied capacity
  denominator. A per-country scale correction might be testable later on a
  separate holdout, but no in-sample fit is offered here.
- **UTC/DST:** killed as primary. On one latest row per target, WAPE is minimized
  at zero-hour shift for BE, DE and FR across ±6 hours. The ±1/±2-hour signature
  is absent.
- **Artifact age/coverage:** BE/FR artifacts are six months old; DE is 5.5
  months old and fit on 3,208 rows. The configured minimum does not block a fit.
  This, the missing training provenance, and the zero-filled DE prehistory make
  a serve-faithful rebuild necessary after the shared builder is fixed.

## Contamination declaration

- **ABL-71:** its named active wrong-data paths are zero-filled load and
  fabricated net position, not solar. It does not directly predicate-filter the
  scored target window. The undeployed ingest estate remains a provenance risk.
- **ABL-67:** fabricated GR/IE `net_position` only; no intersection.
- **ABL-111/ABL-109:** `energy_load` zero handling only; no direct intersection.
- **New solar finding:** DE `energy_renewable.solar_mw` is identically zero for
  every stored quarter-hour from 2025-09-08 22:00 through 2025-10-31 23:45
  (**5,096 rows**), inside the only possible pre-artifact source window. These
  rows do not touch the July/August scoring target window, but they may
  contaminate training. Exact artifact inclusion is unprovable because the
  artifact stores neither training timestamps nor a source-data digest. This
  needs an ingest/data-quality owner before any retraining.

## Recommended owner and fix

1. **ABL-183 / Founding Engineer:** extend the already-owned shared
   train/serve feature builder and golden D+1/D+2 tests to solar. Define every
   lag/rolling feature identically under observation `as_of`; bound weather by
   `publication_as_of`. Do not open a parallel solar serving patch.
2. **Founding Engineer (separate data-quality issue):** adjudicate the 5,096 DE
   zero solar rows, trace their provenance, and provide a safe remediation plus
   a training-data invariant. No replica/prod write is authorized by this pack.
3. **Forecasting Scientist (only after 1–2):** build a fresh solar artifact with
   training-window/data-digest provenance and a hard minimum-data failure, then
   run a pre-registered serve-faithful out-of-sample gate per country against
   D-7 and first-seen TSO vintages. Calibration is considered only on a separate
   holdout. No current model is recommended for promotion.
4. **CEO:** decide any temporary registry/suspension action and escalate to the
   Board. The TSO D+1 score is not a D+2 replacement recommendation.
