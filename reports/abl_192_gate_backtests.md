# ABL-192 — W01-W12 gate backtests

**Run:** 2026-08-11 under `C:\Code\able\energy-forecast\.venv\Scripts\python.exe`  
**Replica:** `C:\Code\able\data\energy_dashboard.db`, opened read-only; 647,538
`net_position` rows, 2023-01-01 00:00 to 2026-08-11 21:00 UTC.  
**Window:** the twelve pre-registered weeks W01-W12, 2024-01-15 to
2026-02-22 (twelve disjoint seven-day windows). All figures below are
out-of-sample: V010 is zero-shot; V012 fits nothing; V014's fit excludes the
twelve weeks; V016's correction fit excludes the twelve weeks.

## Protocol

Each target day T is replayed as a 06:00 UTC run on T-2. Net-position
observations are bounded at run-day 22:00 UTC (`as_of`); weather and other
issued covariates are bounded at the actual 06:00 UTC run
(`publication_as_of`). The stored sidecar supports that schedule assumption:
18 of the champion's normal vintages were generated at 06 UTC and one at 07
UTC; the remaining three are manual runs at 10/16 UTC.

V012 calls the served baseline definition directly. V014 uses its existing
serve-window feature builder and held-out fitted artifacts. V016 reconstructs
V010, adds two warm-up target days per week, and applies its affine + AR(1)
layer using only the latest residual strictly before that vintage's `as_of`.
No database is written and nothing is promoted or deployed.

The prior V010 reference did **not reproduce** under this now-binding two-bound
protocol. It was committed 2026-07-25 before the `as_of` /
`publication_as_of` split: stored mean MAE was AT 880 / BE 889 / FR 1,573 / NL
1,560 MW, versus AT 1,214 / BE 1,140 / FR 2,130 / NL 1,988 MW on the current
reconstruction. Comparing current challengers with that earlier protocol would
not be valid, so `comparison_net_position_servefaithful.json` is regenerated
here on the same protocol as every challenger.

## Gate result on common reference coverage

MAE in MW; n=2,016 paired hourly points per displayed country and model (12 x
168). Baseline is the regenerated zero-shot V010 reference. A positive skill
means lower MAE than V010.

| model | AT MAE / skill | BE MAE / skill | FR MAE / skill | NL MAE / skill | `no_regression_W01_W12` |
|---|---:|---:|---:|---:|---|
| V010 reference | 1,214 / 0.0% | 1,140 / 0.0% | 2,130 / 0.0% | 1,988 / 0.0% | PASS (identity) |
| baseline-V012 | 1,122 / +7.6% | 1,117 / +2.0% | 2,053 / +3.6% | 2,010 / -1.1% | **FAIL — NL regresses** |
| xgboost-V014 | 1,070 / +11.9% | 1,044 / +8.4% | 1,927 / +9.5% | 1,769 / +11.0% | **PASS on common coverage** |
| chronos-2-V016 | 1,204 / +0.9% | 1,188 / -4.2% | 2,064 / +3.1% | 1,886 / +5.1% | **FAIL — BE regresses** |

These are real verdicts, not promotion recommendations. V014 wins this
criterion on the available reference scope; V012 and V016 lose it.

## Coverage and limits

- V012: 19 gated countries, 38,136 paired hours. Eighteen countries cover all
  twelve weeks; HU has 11/12 because one historical week has no pairable
  actuals and is omitted rather than represented as a zero-error week.
- V014: 19 gated countries, 38,136 paired hours, with the same HU limitation.
- V010 and V016: AT/BE/FR/NL only, 8,064 paired hours each. These are the four
  countries in the established champion reference and all have prior
  serve-parity evidence under the two-cutoff reconstruction.

The gate comparison is therefore explicitly **4/19**, even where the candidate
artifact contains 19 countries. It is meaningful as a credibility check on the
four priority majors, but it does **not** establish no-regression for the other
15 gated countries. The gate output now carries `coverage=4/19`, the compared
country list, and `coverage_complete=false`; a candidate missing even one of
the four reference countries fails rather than passing on an empty
intersection. Expanding the Chronos reference mechanically to 19 would be
misleading: suffix-1 covariate revisions cannot be bounded historically, and
serve parity is explicitly unverified for BG/LT/RO. Four-country honest
evidence is preferable to a nominal 19-country replay that may peek.

## Contamination

- ABL-67's confirmed fabricated net-position rows are GR and IE. GR is excluded
  from the pre-registered gate; IE is not one of the 19 gated zones. They do not
  enter any paired result above.
- ABL-71 leaves the sparse-A25 zero guard undeployed. Its confirmed historical
  contamination is the same ABL-67 population, so it does not enter these
  pairs; it remains a risk for future live scoring until deployed.
- ABL-109/ABL-111 concern actual-load zero-as-missing rows, not
  `net_position`; they do not touch this window.

Artifacts: `comparison_net_position_servefaithful.json`,
`experiments/V012/backtest_W01_W12.json`,
`experiments/V014/backtest_W01_W12.json`, and
`experiments/V016/backtest_W01_W12.json`.
