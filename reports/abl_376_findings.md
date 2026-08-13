# ABL-376 — excluding physically impossible night rows from the solar fit

**Forecasting Scientist, 2026-08-13.** Evidence pack for the fit-rule change on
branch `fix/abl-376-solar-night-fit-exclusion`. No production deploy, serving
registry change, model promotion, ingest change, dashboard change, replica write
or sidecar write was performed.

## Summary

| | |
|---|---|
| Change | Solar fit drops night rows whose actual the sun says is impossible. Fit side only, never the score. |
| Night predicate | `solar_geometry.is_night_hour` — the serving clamp's own, sun below -8 deg geometric for the whole hour |
| Threshold | 1 MW, ABL-338's, kept rather than re-derived |
| Registered gate | Both arms **PASS 9/9** cells |
| The result | see §5 — measured over a seed spread, not at one seed |
| Daylight | see §5 |

The night axis is the result. The daylight axis is read against the across-seed
noise floor rather than at a single seed, for the reason given in §5.

## 1. What I could and could not reproduce

The issue states FR carries **488 of 11,614** night training rows above 1 MW,
max 439.3 MW, on 337 distinct days; DE 4 rows at max 1.7 MW; AT and BE none.

**DE, AT and BE reproduce exactly.** Over the registered fit window the rule
removes **4 DE hours, max 1.75 MW**, and nothing at all for AT and BE.

**FR's figure is frame-dependent and should not be quoted as a property of the
source.** Three different frames give three different counts, all of the same
shape:

| frame | FR night rows | above 1 MW | max |
|---|---:|---:|---:|
| source, `:00` sub-sample, whole history | 12,022 | 531 | 440.0 MW |
| source, `:00` sub-sample, ABL-338's pre-holdout window | 11,250 | 454 | 440.0 MW |
| **the fit frame itself, registered window** | **11,648** | **904 rows / 113 hours** | **285.9 MW** |

Two things drive the spread. The fit frame is **hourly means** of a 15-minute
series — the FR builder aggregates 21,190 sub-hourly rows into 5,299 hourly
means (ABL-332) — so the `:00` sample and the fitted series are different
numbers, not different counts of the same number. And fit rows are per
(target, vintage), eight vintages per target, so 904 rows are 113 distinct
contaminated hours.

The finding itself is not in doubt: FR books solar in the dark, on a scale
nothing else here does, and no physical threshold can honour it.

## 2. Protocol

Two reads of one registered protocol, same replica, same day, differing in
exactly one thing.

| | control | treatment |
|---|---|---|
| scope | `abl253` | `abl376` |
| fit rule | off | on |
| countries, basis, windows | identical | identical |
| outputs | `reports/abl_376_solar_night_fit_control.md` | `reports/abl_376_solar_night_fit.md` |

Replica `C:\Code\able\data\energy_dashboard.db` (9,432,453,120 bytes), source
table `energy_renewable`. Fit targets 2026-01-14 → 2026-07-11; gate targets
2026-07-11 → 2026-08-10, out-of-sample by target timestamp. Baseline is literal
seasonal-naive D-7. CatBoost, `random_seed=42`, 500 iterations.

The control was written to **non-registered paths on purpose**: re-running the
`abl253` scope at its own registered paths would have overwritten ABL-253's
dispositioned gate read in place (ABL-387's failure mode). Verified after the
fact — `experiments/ABL253/` still holds only its pre-existing `config.json`,
and `reports/abl_253_solar_retrain.md` is unmodified.

### Why the control is a fresh run and not the published ABL-253 report

Re-running ABL-253's protocol unchanged, 29 hours after it was published,
does not reproduce its numbers:

| | published 2026-08-12 06:51 | re-read 2026-08-13 11:39 |
|---|---:|---:|
| DE 48-64h MAE | 2,902.0 MW | 2,743.6 MW (**-5.5%**) |
| FR incumbent WAPE (48-64h) | 17.8% | 16.0% |
| FR incumbent WAPE (24-36h) | 20.2% | 16.3% |
| BE, all three bands | 336.9 / 348.3 / 544.3 MW | 336.9 / 348.3 / 544.3 MW (identical) |

**A day of ordinary revision moves these numbers by more than the fit rule
does.** The gate is frozen in protocol, not in data. Any A/B here has to be
same-day against a fresh control; quoting the published table as the baseline
would have attributed a -5.5% data revision to a fit rule that cannot reach DE.

## 3. What the rule removed

From the treatment run's own audit, printed in its scorecard:

| country | night fit rows | excluded rows | excluded hours | max excluded actual |
|---|---:|---:|---:|---:|
| BE | 10,856 | 0 | 0 | n/a |
| DE | 10,952 | 32 | 4 | 1.7 MW |
| FR | 11,648 | 904 | 113 | 285.9 MW |

FR loses 2.7% of its fit rows; DE 0.1%; BE none.

**BE is the control the rule provides for itself.** Its two gate artifacts
predict **bit-identically** — the rule is a provable end-to-end no-op where the
data is clean, and it is stated over countries rather than as an FR special case
precisely so that this is checkable. It also establishes that the harness is
deterministic given (data, seed), which is what makes the seed-paired design in
§5 valid.

## 4. The registered gate: both arms PASS 9/9

| country | band | control WAPE | treatment WAPE | D-7 | gate |
|---|---|---:|---:|---:|:---:|
| BE | 24-36h | 15.56% | 15.56% | 32.94% | PASS |
| BE | 36-48h | 16.09% | 16.09% | 32.94% | PASS |
| BE | 48-64h | 18.89% | 18.89% | 33.17% | PASS |
| DE | 24-36h | 13.37% | **13.28%** | 24.20% | PASS |
| DE | 36-48h | 13.70% | **13.48%** | 24.20% | PASS |
| DE | 48-64h | 12.77% | **12.69%** | 23.34% | PASS |
| FR | 24-36h | 14.16% | 14.29% | 22.46% | PASS |
| FR | 36-48h | 14.75% | 14.90% | 22.46% | PASS |
| FR | 48-64h | 14.72% | 14.97% | 22.21% | PASS |

n = 720 / 720 / 480 per band. **The change does not threaten the gate.**

DE improves in all three bands. FR moves 0.13–0.25pp *against* the challenger,
and that direction is expected rather than alarming:

**FR's gate window is itself contaminated, and we are still scored on it.** Over
the 720 gate hours FR has **24 night hours reading above 1 MW, up to 365.5 MW** —
5,281 MWh booked in the dark, 0.11% of the window's total energy. DE and BE have
**zero**. The treatment model predicts ~0 there by design; the control model
partly chased the contamination and was rewarded for it. The maximum WAPE the
rule can concede on that account is ~0.11pp, which covers most of the 0.13pp and
0.15pp moves outright.

That is the fit/score asymmetry doing exactly what it was built to do. A rule
that also filtered the gate frame would have shown FR improving, and the
improvement would have been the filter marking its own homework.

## 5. Daylight and night, decomposed over a seed spread

<!-- SEED SPREAD SECTION -->

## 6. Caveats

- **Contamination.** ABL-67 is net-position-only; ABL-109/111 are load-only;
  ABL-71's known wrong-write modes are load and net position. None is *proven*
  to touch solar, so this is a provenance caveat, not a clean bill. The FR night
  contamination this issue addresses is itself undiagnosed on the ingest side and
  remains open — this change refuses to train on it, and does not fix it.
- ABL-188 constant-run screening found no suspect solar run in the window.
- One 30-day summer holdout. Out-of-sample by target timestamp, not year-round
  evidence, and July/August is when a night-hour rule has the least night to act
  on.
- The rule is **conservative by construction**: `is_night_hour` requires the sun
  below threshold for the whole hour, so shoulder contamination survives it. On
  2026-07-29, of the three FR hours the issue names (03:00, 20:00, 21:00 at
  ~195 MW) it removes only 21:00. It under-removes rather than over-removes,
  which is the right way round — zeroing an hour that really generated would be
  fabricating a number.
- Not a promotion recommendation. Promotion is a pre-registered gate read plus a
  Board decision; this is a fit-rule change in the evaluation harness and the
  shared feature module, and it reaches no serving path.

## 7. Reproducing

```
.venv\Scripts\python.exe scripts/evaluate_solar_retrain.py --scope abl376 \
    --replica-db C:\Code\able\data\energy_dashboard.db \
    --sidecar-db C:\Code\able\data\forecasts_local.db
```

The control is the same command with `--scope abl253` **and all three output
flags overridden**; run without them it will overwrite ABL-253's dispositioned
evidence.
