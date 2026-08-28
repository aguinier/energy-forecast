"""ABL-602: the one file the deploy reads -- which of the fitted artifacts ship.

WHY THIS FILE EXISTS
--------------------
`reports/abl_602_ship_set_training.json` is the record of a *training run*. It
lists five pairs because five were fitted, at 2026-08-28T12:05:25Z, and not one
of its measured fields may be edited afterwards -- a record that is rewritten
when a decision changes is not evidence.

The CEO's ruling landed at 12:35Z, after that run: `HU` `wind_onshore` is
withdrawn. So the training record and the shipping set legitimately disagree,
and the disagreement is a hazard: `models/` is gitignored, HU's artifact exists
on disk beside the other four, and a deploy that reads the training record and
copies every row in `pairs` would serve a pair the CEO withdrew.

This script resolves that in the only direction that is safe -- it joins the
**membership table** (`SHIP_SET`, which carries the hold) onto the **training
record** (which carries the digests) and writes one flat list saying, per pair,
`deploy: true|false`. It re-derives no measurement: every number it emits is
copied from a committed record, and it refuses to emit a shipping row it cannot
find a fitted record for.

WHAT THE DEPLOY DOES WITH IT
----------------------------
For each `deploy: true` row: copy the artifact, re-hash it, and check the
digest against `artifact_sha256`. `in_sample_prediction_digest` is the stronger
follow-up -- a load-and-predict under the *pinned* environment that reproduces
it proves the copy is both intact and readable by the serving interpreter,
which a file hash does not (ABL-597: a wrong-interpreter load keeps the trees
and silently resets the intercept). For each `deploy: false` row: do not copy
it, and if it is already present, remove it.

This script scores nothing, grades nothing, opens no database and loads no
model.
"""
import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "abl525_train_ship_set", REPO / "scripts" / "abl525_train_ship_set.py"
)
_trainer = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _trainer
_spec.loader.exec_module(_trainer)

BATCH = "abl602"

#: Fields copied verbatim from the training record onto every row. Named rather
#: than copied wholesale so that adding a field to the training record cannot
#: silently widen what the deploy is handed.
CARRIED = (
    "algorithm",
    "training_source",
    "n_features",
    "artifact_path",
    "artifact_sha256",
    "in_sample_prediction_digest",
    "in_sample_prediction_mean",
)


def _fitted_index(record):
    return {(p["country"], p["forecast_type"]): p for p in record["pairs"]}


def _serving_index(record):
    return {(p["country"], p["forecast_type"]): p for p in record["pairs"]}


def build(training, serving, batch=BATCH):
    """Join SHIP_SET's holds onto the training record's digests.

    Raises if a shipping row has no fitted record -- the failure mode this file
    exists to prevent is a pair that ships without a digest behind it.
    """
    fitted = _fitted_index(training)
    served = _serving_index(serving) if serving else {}

    rows = []
    for entry in _trainer.SHIP_SET:
        if entry["batch"] != batch:
            continue
        key = (entry["country"], entry["forecast_type"])
        deploy = entry["hold"] is None
        fit = fitted.get(key)
        if deploy and fit is None:
            raise SystemExit(
                f"{key[0]}/{key[1]} ships but has no fitted record in the "
                f"training record -- refusing to write a shipping row without "
                f"a digest"
            )
        row = {
            "country": key[0],
            "forecast_type": key[1],
            "deploy": deploy,
            "tranche": entry["tranche"],
            "fitted": fit is not None,
        }
        if fit is not None:
            row.update({field: fit[field] for field in CARRIED})
        verification = served.get(key)
        row["serving_verified"] = (
            bool(verification["verified"]) if verification else None
        )
        if not deploy:
            row["withdrawn_reason"] = entry["hold"]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-record",
        default=str(REPO / "reports" / "abl_602_ship_set_training.json"),
        help="The training run's record. Read, never written.",
    )
    parser.add_argument(
        "--serving-record",
        default=str(REPO / "reports" / "abl_602_serving_verification.json"),
        help="The serving-verification record. Read, never written.",
    )
    parser.add_argument(
        "--json-out",
        default=str(REPO / "reports" / "abl_602_ship_disposition.json"),
    )
    args = parser.parse_args()

    training = json.loads(Path(args.training_record).read_text(encoding="utf-8"))
    serving_path = Path(args.serving_record)
    serving = (
        json.loads(serving_path.read_text(encoding="utf-8"))
        if serving_path.is_file() else None
    )

    rows = build(training, serving)
    ships = [row for row in rows if row["deploy"]]
    withdrawn = [row for row in rows if not row["deploy"]]

    payload = {
        "issue": "ABL-602",
        "batch": BATCH,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "The deploy's list. Five pairs were fitted; four ship. Derived by "
            "joining SHIP_SET (which carries the hold) onto the committed "
            "training and serving records (which carry the digests and the "
            "verification). Nothing here is re-measured."
        ),
        "decision": {
            "board_answer": "abl316 widen7, 2026-08-28 -- adopt the "
                            "causally-available standard for the widened set",
            "ceo_ruling": "2026-08-28T12:35Z on ABL-602: a pair ships unless a "
                          "causally-available screen shows a READABLE loss; an "
                          "abstention (margin inside the readability floor) "
                          "does not block. HU wind_onshore is the only pair in "
                          "the batch on the wrong side of that bar and is "
                          "withdrawn. NO and RO wind_onshore were already out "
                          "on G4 and are not in SHIP_SET at all.",
            "is_a_regrade": False,
            "registered_letters_moved": False,
        },
        "sources": {
            "membership": "scripts/abl525_train_ship_set.py -> SHIP_SET "
                          "(`hold` is the withdrawal)",
            "digests": str(Path(args.training_record).name),
            "serving": serving_path.name if serving else None,
            "note": "The training record still lists five pairs because five "
                    "were fitted at 2026-08-28T12:05:25Z. It is the record of "
                    "a run and is not edited when a disposition changes; this "
                    "file is the disposition.",
        },
        "counts": {
            "fitted": len(rows),
            "ships": len(ships),
            "withdrawn": len(withdrawn),
        },
        "deploy_check": (
            "Per shipping row: copy the artifact, re-hash, compare against "
            "artifact_sha256. Then load under the ABL-597 pinned environment "
            "and reproduce in_sample_prediction_digest -- a file hash cannot "
            "see a wrong-interpreter load, which keeps the trees and resets "
            "the intercept."
        ),
        "pairs": rows,
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        flag = "SHIP" if row["deploy"] else "HOLD"
        print(f"[{flag}] {row['country']}/{row['forecast_type']}")
    print(f"\nWrote {out} ({len(ships)} ship, {len(withdrawn)} withdrawn)")


if __name__ == "__main__":
    main()
