# ABL-239 — Wind shadow-candidate backfill

**Disposition: COMPLETE — evidence files only; nothing posted or deployed.**

Generated: 2026-08-11 20:20 UTC.
Target window: 2026-07-11 00:00 UTC through 2026-08-10 00:00 UTC exclusive (30 days; 720 hourly targets per served country).
Protocol: one serve-faithful D+2 prediction per country-hour, reconstructed at the pre-registered D-2 19:00 UTC run instant (horizons 29–52 hours). This is out-of-sample by target timestamp: the artifacts were fitted only through 2026-07-11 00:00 UTC exclusive.
Replica: `C:\Code\able\data\energy_dashboard.db` (5,958,455,296 bytes), opened with SQLite `mode=ro`, `uri=True`.

## Files

| forecast type | served countries | rows | model_name | model_version | payload SHA-256 |
|---|---|---:|---|---|---|
| wind_offshore | BE, FR | 1,440 | `xgboost-retrain-v1` | `abl195-df71d0314426` | `48d43c7b4460f070a5260c0c64777ebf250b0d75474b15001772bdf280f1da10` |
| wind_onshore | BE, DE, FR | 2,160 | `catboost-retrain-v1` | `abl195-c84911cc953d` | `352b032cdaf014099de6c80a257a7b270865b1fb3350cf1b0ed9ec11f05086b5` |

Payloads:

- `experiments/ABL239/wind_offshore_backfill.json`
- `experiments/ABL239/wind_onshore_backfill.json`
- `experiments/ABL239/manifest.json` records the artifact witnesses and file hashes.

The model names are visibly distinct from the live replica's wind model names. At export time, offshore held `xgboost`, `tso_raw`, and `tso_corrected`; onshore held `catboost`, `xgboost`, `tso_raw`, and `tso_corrected`. The exporter refuses a name collision.

## Artifact identity

The payload versions are deterministic digests over the per-country ABL-195 artifact paths and SHA-256 witnesses. The five artifact hashes reproduce `reports/abl_195_wind_retrain.md` exactly:

- offshore BE `304460cc…01e`, FR `ccb5cb51…95b`
- onshore BE `0dab6910…42d`, DE `195da7db…7ac`, FR `8b475ac2…778`

## Interpretation and caveats

These files contain predictions, not a new model-quality read. The relevant out-of-sample evidence remains ABL-195: challenger versus literal seasonal-naive D-7 on the identical 2026-07-11 → 2026-08-10 window, with 15/15 primary D+2 country-band cells passing. This export uses one unambiguous D+2 vintage per target so a downstream insert cannot create duplicate country/model/target rows; it does not repeat ABL-195's three-band scoring frame.

The top-level `generated_at` is the requested backfill production time. Each row's `horizon_hours` preserves its simulated serve-faithful D+2 horizon rather than the negative elapsed time from an August 11 backfill to a historical target.

Contamination: ABL-188/ABL-198's BE offshore zero-filled run is in the fit period, was screened out before ABL-195 fitting, and does not intersect this target window. ABL-67 is net-position-only; ABL-109/111 are load-only. ABL-71's confirmed wrong-write modes are load and net position, so it is a provenance caveat rather than a known wind-window intersection. This remains a single 30-day summer holdout, not year-round evidence.

No HTTP request, replica write, sidecar write, ingest change, dashboard change, registry change, production-default change, or model promotion was performed.
