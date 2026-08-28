#!/usr/bin/env python
"""Prove a ship-set refit is reproducible, by predictions rather than by hash.

ABL-525 item 7, ABL-580 item 4, ABL-583 item 3. The pairs are read from the
training record `--record` names, so this runs on any batch's record without an
edit; `--json-out` and the record's own `issue` field keep each batch's
reproducibility evidence in its own file.

An artifact sha256 cannot witness a refit: `Forecaster.save`
stamps `saved_at`, so three byte-identical fits give three different digests and
a hash comparison reports drift that is not there. The witness is what the two
artifacts *predict*.

Protocol, per pair:

  1. Refit into a scratch models directory with `abl525_train_ship_set.fit_one`
     -- the same function, the same window, the same seed, nothing re-specified
     here that could drift from the original run.
  2. Load the original and the refit through `Forecaster.load`, which is the
     entry point `forecast_daily.py` uses, so the comparison is between two
     things that were actually deserialised and not between two in-memory
     estimators that never round-tripped.
  3. Build ONE feature matrix from the shared builder and predict with both.
     A common matrix is the point: two matrices built separately could differ
     and hide a model difference, or agree and hide a builder difference.
  4. Assert max |a - b| < 1e-12, and report the value either way.

Also reports whether `feature_columns` and `training_source` round-tripped
identically, since a matching prediction vector on a mismatched column order
would be a coincidence worth failing on.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

import config
from src.forecaster import Forecaster
from src.wind_features import RenewableFeatureBuilder

from scripts.abl525_train_ship_set import (  # noqa: E402
    LOOKBACK_DAYS,
    RENEWABLE_SOURCE,
    columns_for,
    fit_one,
)

TOLERANCE = 1e-12

#: A fixed block of target hours to score both artifacts on. Inside the fit
#: window on purpose -- this measures artifact equality, not generalisation, and
#: an out-of-window hour would only add rows whose features are NaN.
PROBE_START = "2026-08-01"
PROBE_HOURS = 24 * 7


def probe_matrix(country, forecast_type, replica_db):
    """One feature matrix both artifacts are asked to predict on."""
    columns = list(columns_for(country, forecast_type))
    start = pd.Timestamp(PROBE_START)
    end = start + pd.Timedelta(hours=PROBE_HOURS)
    builder = RenewableFeatureBuilder(
        country,
        forecast_type,
        start - pd.Timedelta(days=LOOKBACK_DAYS),
        end,
        actuals_source=RENEWABLE_SOURCE,
        db_path=str(replica_db),
    )
    rows = []
    for target in pd.date_range(start, end, freq="h", inclusive="left"):
        # One vintage per target is enough; this is an artifact comparison.
        generated_at = target.normalize() - pd.Timedelta(days=2) + pd.Timedelta(hours=7)
        features = builder.row(target, generated_at, generated_at)
        rows.append({col: features[col].value for col in columns})
    frame = pd.DataFrame(rows, columns=columns)
    return frame[np.isfinite(frame.to_numpy(dtype=float)).all(axis=1)].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Refit a ship-set batch and prove prediction equality at 1e-12."
    )
    parser.add_argument("--replica-db", default=config.DATABASE_PATH)
    parser.add_argument("--models-dir", default=str(config.MODELS_DIR))
    parser.add_argument(
        "--record",
        default="reports/abl_525_ship_set_training.json",
        help="The training record naming the pairs to re-check.",
    )
    parser.add_argument("--json-out", default="reports/abl_525_reproducibility.json")
    args = parser.parse_args()

    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    pairs = [(p["country"], p["forecast_type"]) for p in record["pairs"]]

    results = []
    with tempfile.TemporaryDirectory(prefix="abl525_repro_") as scratch:
        for country, forecast_type in pairs:
            print(f"[REFIT] {country}/{forecast_type} ...", flush=True)
            refit = fit_one(country, forecast_type, Path(args.replica_db), scratch)

            original = Forecaster.load(
                country,
                forecast_type,
                path=str(Path(args.models_dir) / country / forecast_type / "model.joblib"),
            )
            replica_model = Forecaster.load(
                country, forecast_type, path=refit["artifact_path"]
            )

            frame = probe_matrix(country, forecast_type, args.replica_db)
            a = np.asarray(original.model.predict(frame), dtype=float)
            b = np.asarray(replica_model.model.predict(frame), dtype=float)
            max_abs = float(np.max(np.abs(a - b))) if len(a) else float("nan")

            entry = {
                "country": country,
                "forecast_type": forecast_type,
                "probe_rows": int(len(frame)),
                "max_abs_prediction_difference": max_abs,
                "tolerance": TOLERANCE,
                "identical_within_tolerance": bool(max_abs < TOLERANCE),
                "bit_identical": bool(np.array_equal(a, b)),
                "feature_columns_match": (
                    original.feature_columns == replica_model.feature_columns
                ),
                "training_source_match": (
                    original.training_source == replica_model.training_source
                ),
                "training_source": original.training_source,
                "n_features": len(original.feature_columns),
                # The two digests differ by construction; recorded so the record
                # itself shows why a hash is not the witness.
                "artifact_sha256_original": next(
                    p["artifact_sha256"] for p in record["pairs"]
                    if p["country"] == country and p["forecast_type"] == forecast_type
                ),
                "artifact_sha256_refit": refit["artifact_sha256"],
            }
            entry["artifact_sha256_differs"] = (
                entry["artifact_sha256_original"] != entry["artifact_sha256_refit"]
            )
            results.append(entry)
            print(
                f"[{'PASS' if entry['identical_within_tolerance'] else 'FAIL'}] "
                f"{country}/{forecast_type}: max|a-b| = {max_abs:.3e} over "
                f"{entry['probe_rows']} rows, bit-identical={entry['bit_identical']}, "
                f"sha256 differs={entry['artifact_sha256_differs']}",
                flush=True,
            )

    payload = {
        # From the record being re-checked, not from a constant here: this
        # script runs on whichever batch's record `--record` names, and a
        # hardcoded issue would mislabel every batch but one.
        "issue": record.get("issue", "ABL-316"),
        "batch": record.get("batch"),
        "training_record": str(Path(args.record).as_posix()),
        "check": "prediction equality across an independent refit",
        "tolerance": TOLERANCE,
        "probe_window": [PROBE_START, f"+{PROBE_HOURS}h"],
        # The refit reads the replica live, so the two arms are only comparable
        # if the replica did not move between the original fit and this run.
        # `able-db-sync` replaces every non-weather table inside one transaction,
        # so a sync landing in between would report a data change as a drift.
        # Recorded on both sides rather than assumed away.
        "replica": {
            "path": str(Path(args.replica_db)),
            "bytes_now": Path(args.replica_db).stat().st_size,
            "bytes_at_original_fit": record.get("environment", {}).get("replica_bytes"),
        },
        "all_pairs_reproducible": all(r["identical_within_tolerance"] for r in results),
        "every_artifact_sha256_differed": all(r["artifact_sha256_differs"] for r in results),
        "pairs": results,
    }
    payload["replica"]["unchanged_since_original_fit"] = (
        payload["replica"]["bytes_at_original_fit"] is not None
        and payload["replica"]["bytes_at_original_fit"] == payload["replica"]["bytes_now"]
    )
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")
    if not payload["replica"]["unchanged_since_original_fit"]:
        print("NOTE: the replica moved between the original fit and this refit "
              f"({payload['replica']['bytes_at_original_fit']} -> "
              f"{payload['replica']['bytes_now']} bytes). A prediction difference "
              "below may be a data change rather than a drift.")
    return 0 if payload["all_pairs_reproducible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
