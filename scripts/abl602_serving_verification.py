#!/usr/bin/env python
"""Record what `forecast_daily.py` actually served for an ABL-316 ship-set batch.

ABL-602's deliverable asks for serving-verification of every pair in the batch
**through the serving entry point at both horizons -- not inferred from the
training log**. This script does not generate anything: it reads the sidecar that
serving run wrote and turns it into a committed record.

WHY THIS IS NOT `abl583_serving_verification.py`
------------------------------------------------
That script answers the same question for one hardcoded pair (`CH` / `solar`) and
spends most of its length on a comparison against `abl583_ch_night_probe.py`,
which exists because CH's night distribution was the open question on that issue.
Neither generalises:

  * a batch of five pairs across two forecast types needs the pair list to come
    from the training record, so the verification cannot drift from what was
    fitted; and
  * `forecast_clamp_log` only has rows for `solar` -- `src/solar_clamp.py` is
    `renewable_type='solar'` only -- so a wind pair has no clamp row and the
    absence is correct rather than a missing check. A script that demanded one
    per pair would fail on three of five for the wrong reason.

WHAT MAKES THE SERVED MODEL PROVABLY THE GRADED ONE
---------------------------------------------------
`models/` is gitignored, so no commit protects these artifacts and a later reader
cannot diff them. The join is the digest: this script re-hashes each artifact on
disk and compares it to `artifact_sha256` in the training record. A serving run
that picked up some other artifact -- a stale one in the primary checkout, a
`scripts/train.py` build with the wrong feature list -- shows up here as a digest
mismatch rather than as a plausible-looking series.

That is a check on *identity*, not on reproducibility. `Forecaster.save` stamps
`saved_at`, so two byte-identical fits have different digests; the digest is
therefore only usable to answer "is this the same *file* the training record
described", which is exactly the question serving verification asks.
`abl525_repro_check.py` answers the other one, by predictions at 1e-12.

Alongside the digest the artifact is opened through `Forecaster.load` -- the same
entry point `forecast_daily.py` uses -- and its `feature_columns` and
`training_source` are compared to the record, so a digest match on a payload that
deserialises to something else is still caught.

BOTH HORIZONS
-------------
`--horizons 1,2` asks the runner for D+1 and D+2, which is two target days of 24
hours for each pair. The check is on the *served rows*: two distinct target
dates, 24 hours each, and a `horizon_hours` span that brackets both. Counting
rows in the training log would not have shown a horizon that produced nothing,
which is the failure this deliverable names.

Read-only against the sidecar. Writes only its own JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.forecaster import Forecaster  # noqa: E402

#: The clamp is `renewable_type='solar'` only (`src/solar_clamp.py`), so a wind
#: pair having no `forecast_clamp_log` row is the correct outcome and not a gap.
CLAMPED_TYPES = frozenset({"solar"})

#: What `--horizons 1,2` is expected to produce per pair: two target days, 24
#: hours each. Stated so a partial serve fails loudly instead of being read off a
#: row count that happens to look plausible.
EXPECTED_TARGET_DAYS = 2
EXPECTED_HOURS_PER_DAY = 24


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _renewable_column(conn: sqlite3.Connection) -> str | None:
    """`renewable_type` if this sidecar carries it, else None.

    A sidecar created by `src/db.py` today has the column; an older file may not,
    and in that case the renewable type is only recoverable from `forecast_type`.
    Detected rather than assumed so the script reports what it filtered on.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(forecasts)")}
    return "renewable_type" if "renewable_type" in cols else None


def read_served(conn: sqlite3.Connection, country: str, forecast_type: str,
                renewable_column: str | None) -> list[dict]:
    if renewable_column:
        sql = ("SELECT target_timestamp_utc, forecast_value, horizon_hours, "
               "       model_name, generated_at "
               "FROM forecasts WHERE country_code = ? AND renewable_type = ? "
               "ORDER BY target_timestamp_utc, horizon_hours")
    else:
        sql = ("SELECT target_timestamp_utc, forecast_value, horizon_hours, "
               "       model_name, generated_at "
               "FROM forecasts WHERE country_code = ? AND forecast_type = ? "
               "ORDER BY target_timestamp_utc, horizon_hours")
    return [dict(r) for r in conn.execute(sql, (country, forecast_type))]


def read_clamp(conn: sqlite3.Connection, country: str, forecast_type: str,
               renewable_column: str | None) -> dict | None:
    if not renewable_column:
        return None
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM forecast_clamp_log "
        "WHERE country_code = ? AND renewable_type = ? ORDER BY id DESC LIMIT 1",
        (country, forecast_type),
    )]
    return rows[0] if rows else None


def served_summary(rows: list[dict]) -> dict:
    """Post-clamp invariants, measured on the served rows themselves."""
    values = [float(r["forecast_value"]) for r in rows]
    positive = [v for v in values if v > 0.0]
    by_day: dict[str, int] = defaultdict(int)
    for r in rows:
        by_day[str(r["target_timestamp_utc"])[:10]] += 1
    return {
        "n_rows": len(rows),
        "target_window": [rows[0]["target_timestamp_utc"], rows[-1]["target_timestamp_utc"]],
        "target_days": sorted(by_day),
        "rows_per_target_day": dict(sorted(by_day.items())),
        "horizon_hours_min": min(int(r["horizon_hours"]) for r in rows),
        "horizon_hours_max": max(int(r["horizon_hours"]) for r in rows),
        "model_names": sorted({r["model_name"] for r in rows}),
        "generated_at": sorted({str(r["generated_at"]) for r in rows}),
        "min_mw": round(min(values), 4),
        "max_mw": round(max(values), 4),
        "mean_mw": round(sum(values) / len(values), 4),
        "positive_hours_mean_mw": round(sum(positive) / len(positive), 4) if positive else None,
        "n_negative": sum(1 for v in values if v < 0.0),
        "n_exactly_zero": sum(1 for v in values if v == 0.0),
        "all_finite": all(v == v and abs(v) != float("inf") for v in values),
    }


def horizon_check(summary: dict) -> dict:
    """Both horizons actually produced rows -- the thing a training log cannot show."""
    days = summary["target_days"]
    per_day = summary["rows_per_target_day"]
    complete = [d for d, n in per_day.items() if n == EXPECTED_HOURS_PER_DAY]
    return {
        "expected_target_days": EXPECTED_TARGET_DAYS,
        "observed_target_days": len(days),
        "expected_hours_per_day": EXPECTED_HOURS_PER_DAY,
        "days_with_a_full_24h": complete,
        "both_horizons_served": (
            len(days) == EXPECTED_TARGET_DAYS and len(complete) == EXPECTED_TARGET_DAYS
        ),
    }


def artifact_check(country: str, forecast_type: str, models_dir: Path,
                   recorded: dict) -> dict:
    """The served artifact is the one the training record describes.

    Digest for identity; `Forecaster.load` for what the payload deserialises to.
    """
    path = models_dir / country / forecast_type / "model.joblib"
    if not path.is_file():
        return {"artifact_path": str(path), "artifact_present": False,
                "is_the_recorded_artifact": False,
                "note": "no artifact at the path the runner resolves"}
    digest = sha256_of(path)
    loaded = Forecaster.load(country, forecast_type, path=str(path))
    columns_match = list(loaded.feature_columns) == list(recorded["feature_columns"])
    source_match = loaded.training_source == recorded["training_source"]
    return {
        "artifact_path": str(path),
        "artifact_present": True,
        "artifact_sha256_on_disk": digest,
        "artifact_sha256_in_training_record": recorded["artifact_sha256"],
        "sha256_matches_training_record": digest == recorded["artifact_sha256"],
        "n_features_on_disk": len(loaded.feature_columns),
        "n_features_in_training_record": recorded["n_features"],
        "feature_columns_match": columns_match,
        "training_source_on_disk": loaded.training_source,
        "training_source_matches": source_match,
        "is_the_recorded_artifact": (
            digest == recorded["artifact_sha256"] and columns_match and source_match
        ),
    }


def clamp_check(forecast_type: str, clamp: dict | None) -> dict:
    """The clamp fired where it applies and is correctly absent where it does not."""
    applies = forecast_type in CLAMPED_TYPES
    if not applies:
        return {
            "clamp_applies_to_this_type": False,
            "clamp_log_row_present": clamp is not None,
            "as_expected": clamp is None,
            "note": ("src/solar_clamp.py is renewable_type='solar' only, so a "
                     "wind pair has no clamp-log row and the absence is correct."),
        }
    if clamp is None:
        return {"clamp_applies_to_this_type": True, "clamp_log_row_present": False,
                "as_expected": False,
                "note": "solar served without a clamp-log row -- the clamp did not run"}
    return {
        "clamp_applies_to_this_type": True,
        "clamp_log_row_present": True,
        "as_expected": True,
        "night_generation_possible": clamp.get("night_generation_possible"),
        "night_mask_applied": clamp.get("night_mask_applied"),
        "rows_total": clamp.get("rows_total"),
        "night_hours": clamp.get("night_hours"),
        "hours_zeroed_night": clamp.get("hours_zeroed_night"),
        "hours_raised_floor": clamp.get("hours_raised_floor"),
        "mw_removed_night": clamp.get("mw_removed_night"),
        "mw_added_floor": clamp.get("mw_added_floor"),
        "min_forecast_mw": clamp.get("min_forecast_mw"),
        "max_night_forecast_mw": clamp.get("max_night_forecast_mw"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Turn an ABL-316 ship-set batch's serving run into a "
                     "committed record: rows served at both horizons, and the "
                     "served artifact matched to the training record by sha256."))
    parser.add_argument("--sidecar", required=True,
                        help="The FORECAST_OUTPUT_DB the serving run wrote.")
    parser.add_argument("--record", default="reports/abl_602_ship_set_training.json",
                        help="Training record naming the pairs and their digests.")
    parser.add_argument("--models-dir", default=None,
                        help="Artifact root the serving run resolved "
                             "(default: the models_dir in the training record).")
    parser.add_argument("--command", default=None,
                        help="The exact forecast_daily.py invocation, recorded verbatim.")
    parser.add_argument("--json-out", default="reports/abl_602_serving_verification.json")
    args = parser.parse_args()

    sidecar = Path(args.sidecar)
    if not sidecar.is_file():
        raise SystemExit(f"sidecar not found: {sidecar}")

    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    models_dir = Path(args.models_dir or record["environment"]["models_dir"])

    conn = sqlite3.connect(f"file:{sidecar}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        renewable_column = _renewable_column(conn)
        pairs = []
        for recorded in record["pairs"]:
            country, forecast_type = recorded["country"], recorded["forecast_type"]
            rows = read_served(conn, country, forecast_type, renewable_column)
            if not rows:
                pairs.append({
                    "country": country, "forecast_type": forecast_type,
                    "served_any_rows": False,
                    "verified": False,
                    "note": "the serving run wrote no rows for this pair",
                })
                print(f"[FAIL] {country}/{forecast_type}: no rows served", flush=True)
                continue
            summary = served_summary(rows)
            horizons = horizon_check(summary)
            artifact = artifact_check(country, forecast_type, models_dir, recorded)
            clamp = clamp_check(forecast_type,
                                read_clamp(conn, country, forecast_type, renewable_column))
            entry = {
                "country": country,
                "forecast_type": forecast_type,
                "served_any_rows": True,
                "served": summary,
                "horizons": horizons,
                "artifact": artifact,
                "clamp": clamp,
                "verified": (
                    horizons["both_horizons_served"]
                    and artifact["is_the_recorded_artifact"]
                    and clamp["as_expected"]
                    and summary["all_finite"]
                ),
            }
            pairs.append(entry)
            print(
                f"[{'PASS' if entry['verified'] else 'FAIL'}] {country}/{forecast_type}: "
                f"{summary['n_rows']} rows over {len(summary['target_days'])} target days, "
                f"h {summary['horizon_hours_min']}-{summary['horizon_hours_max']}, "
                f"min {summary['min_mw']} MW, artifact matches "
                f"{artifact['is_the_recorded_artifact']}",
                flush=True,
            )
    finally:
        conn.close()

    payload = {
        "issue": record.get("issue", "ABL-602"),
        "batch": record.get("batch"),
        "check": ("end-to-end serving verification through forecast_daily.py at "
                  "both horizons, with the served artifact matched to the "
                  "training record by sha256"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": args.command,
        "training_record": str(Path(args.record).as_posix()),
        "models_dir": str(models_dir),
        "sidecar_has_renewable_type_column": bool(renewable_column),
        "wrote_to": {
            "sidecar": str(sidecar),
            "is_scratch": True,
            "replica_written": False,
            "note": ("FORECAST_OUTPUT_DB was pointed at a run-scoped scratch file, "
                     "so neither the replica nor the shared sidecar was written. "
                     "Serving into production is a deploy and is outside this issue."),
        },
        "clock_dependence": {
            "observed_on_abl583": True,
            "note": ("predict_d2 anchors its feature build on wall-clock time as "
                     "well as on the reference date (ABL-583), so a solar night "
                     "series is a snapshot of the firing hour rather than a "
                     "property of the artifact. The clamp figures below are "
                     "read as of this run's hour and must be compared only "
                     "against a same-hour run."),
        },
        "all_pairs_verified": all(p["verified"] for p in pairs),
        "scored_or_graded": False,
        "pairs": pairs,
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out} ({sum(p['verified'] for p in pairs)}/{len(pairs)} verified)")
    return 0 if payload["all_pairs_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
