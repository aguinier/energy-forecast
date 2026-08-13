#!/usr/bin/env python3
"""Export ABL-195 wind artifacts as read-only shadow backfill payloads.

This script never posts or writes to either database.  It reconstructs one
serve-faithful D+2 vintage per target hour (the D-2 19:00 UTC run) and writes
JSON files shaped like the dashboard's forecast-ingest payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.wind_features import RenewableFeatureBuilder, to_vector


PAIRS = {
    "wind_offshore": {
        "countries": ("BE", "FR"),
        "model_name": "xgboost-retrain-v1",
    },
    "wind_onshore": {
        "countries": ("BE", "DE", "FR"),
        "model_name": "catboost-retrain-v1",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ro_connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)


def _artifact_version(paths: list[Path]) -> str:
    witness = "\n".join(f"{path.as_posix()}:{_sha256(path)}" for path in paths)
    return f"abl195-{hashlib.sha256(witness.encode()).hexdigest()[:12]}"


def _production_names(replica: Path, forecast_type: str) -> list[str]:
    con = _ro_connect(replica)
    try:
        rows = con.execute(
            "SELECT DISTINCT model_name FROM forecasts WHERE forecast_type=? ORDER BY model_name",
            (forecast_type,),
        ).fetchall()
    finally:
        con.close()
    return [row[0] for row in rows]


def _simulated_run(target: pd.Timestamp) -> pd.Timestamp:
    """The last pre-registered D-2 run, yielding horizons 29 through 52h."""
    return target.normalize() - pd.Timedelta(days=2) + pd.Timedelta(hours=19)


def export_type(
    replica: Path,
    artifact_dir: Path,
    output_dir: Path,
    forecast_type: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    produced_at: str,
) -> dict:
    spec = PAIRS[forecast_type]
    artifact_paths = [artifact_dir / country / forecast_type / "model.joblib"
                      for country in spec["countries"]]
    missing = [str(path) for path in artifact_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing ABL-195 artifacts: {missing}")

    existing_names = _production_names(replica, forecast_type)
    if spec["model_name"] in existing_names:
        raise RuntimeError(
            f"refusing model_name collision for {forecast_type}: {spec['model_name']}"
        )

    rows = []
    artifact_witnesses = []
    for country, path in zip(spec["countries"], artifact_paths, strict=True):
        artifact = joblib.load(path)
        if artifact.get("country_code") != country or artifact.get("forecast_type") != forecast_type:
            raise RuntimeError(f"artifact identity mismatch: {path}")
        columns = artifact["feature_columns"]
        model = artifact["model"]
        builder = RenewableFeatureBuilder(
            country, forecast_type, start - pd.Timedelta(days=16), end
        )
        vectors = []
        provenance = []
        for target in pd.date_range(start, end, freq="h", inclusive="left"):
            simulated_generated_at = _simulated_run(target)
            vector = to_vector(
                builder.row(target, simulated_generated_at, simulated_generated_at),
                columns,
            )
            values = np.asarray(list(vector.values()), dtype=float)
            if not np.isfinite(values).all():
                raise RuntimeError(
                    f"non-finite serve-faithful features: {forecast_type}/{country}/{target}"
                )
            vectors.append(vector)
            provenance.append((target, simulated_generated_at))
        predictions = model.predict(pd.DataFrame(vectors, columns=columns))
        if not np.isfinite(predictions).all():
            raise RuntimeError(f"non-finite predictions: {forecast_type}/{country}")
        for (target, simulated_generated_at), prediction in zip(provenance, predictions, strict=True):
            rows.append({
                "country_code": country,
                "target_timestamp_utc": target.strftime("%Y-%m-%d %H:%M:%S"),
                "horizon_hours": int((target - simulated_generated_at).total_seconds() / 3600),
                "forecast_value": float(prediction),
            })
        artifact_witnesses.append({
            "country_code": country,
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        })

    payload = {
        "model": {
            "name": spec["model_name"],
            "version": _artifact_version(artifact_paths),
        },
        "generated_at": produced_at,
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{forecast_type}_backfill.json"
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "forecast_type": forecast_type,
        "output_path": str(output_path.resolve()),
        "payload_sha256": _sha256(output_path),
        "model_name": spec["model_name"],
        "model_version": payload["model"]["version"],
        "countries": list(spec["countries"]),
        "row_count": len(rows),
        "target_start": rows[0]["target_timestamp_utc"],
        "target_end": rows[-1]["target_timestamp_utc"],
        "horizon_min": min(row["horizon_hours"] for row in rows),
        "horizon_max": max(row["horizon_hours"] for row in rows),
        "existing_replica_model_names": existing_names,
        "artifacts": artifact_witnesses,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica-db", required=True)
    parser.add_argument("--artifact-dir", default="experiments/ABL195/artifacts")
    parser.add_argument("--output-dir", default="experiments/ABL239")
    parser.add_argument("--start", default="2026-07-11")
    parser.add_argument("--end", default="2026-08-10")
    parser.add_argument("--generated-at")
    args = parser.parse_args()

    replica = Path(args.replica_db)
    if not replica.exists():
        parser.error(f"replica not found: {replica}")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if start >= end:
        parser.error("require start < end")
    produced_at = args.generated_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    results = [
        export_type(
            replica, Path(args.artifact_dir), Path(args.output_dir), forecast_type,
            start, end, produced_at,
        )
        for forecast_type in PAIRS
    ]
    manifest = {
        "issue": "ABL-239",
        "generated_at": produced_at,
        "replica_db": str(replica.resolve()),
        "replica_bytes": replica.stat().st_size,
        "protocol": "one serve-faithful D-2 19:00 UTC vintage per target hour",
        "window": {"start": str(start), "end_exclusive": str(end)},
        "outputs": results,
        "writes": "JSON files only; no database or HTTP writes",
    }
    manifest_path = Path(args.output_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(
            f"{result['forecast_type']}: {result['row_count']} rows, "
            f"model_name={result['model_name']}, version={result['model_version']}"
        )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
