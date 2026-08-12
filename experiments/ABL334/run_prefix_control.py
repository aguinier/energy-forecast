#!/usr/bin/env python3
"""ABL-334 isolation control — the pre-ABL-332 builder, at the post-ABL-332 commit.

Registered in `experiments/ABL321/protocol.md`, Amendment 3, "Registered
addition 1", before any metric was read.

**What this exists to rule out.** The CEO's revert trigger differences arm A as
recorded in ABL-321 (`de369a6`) against arm A re-run on `origin/main`. Ten
commits separate those two points, and `b9ebb8a` / `1a133d6` (ABL-331),
`981e4d6` (ABL-337) and `ad98f53` (ABL-340) all touch `src/db.py`,
`src/baselines.py` or `scripts/`. That difference is therefore "ABL-332 plus
whatever else moved", and attributing a regression to ABL-332 that another
commit caused would be a wrongful revert recommendation.

This run holds the commit fixed and neutralises **only** ABL-332's two moving
parts, restoring the pre-fix builder exactly:

  1. `db.aggregate_renewable_to_hourly` -> identity, so the shared read returns
     the raw sub-hourly series it returned before ABL-332.
  2. `wind_features._assert_hourly` -> identity, so the guard ABL-332 added to
     make the old failure loud does not fire on that series.

With both neutralised, `RenewableFeatureBuilder` is once again reading
`series.loc[ts.floor("h")]` off a quarter-hourly index — the `:00` sub-sample
for lags and persistence, the raw ~96-sample day for rolling windows. That is
the instrument ABL-321 measured with.

Everything else is the unmodified `scripts/evaluate_renewable_source_switch.py`
on the identical CLI arguments: same window, same arms, same truths, same
common-row rule, same algorithm and seed. Both arms are run, so "common rows"
is defined the same way it is in the run this is compared against.

Reads the replica `mode=ro` and writes only report files. No replica write, no
sidecar write, no serving change, no promotion.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src import db  # noqa: E402
from src import wind_features  # noqa: E402


def _load_ab_module():
    """Import the unmodified A/B script by path, without copying any of it."""
    path = REPO / "scripts" / "evaluate_renewable_source_switch.py"
    spec = importlib.util.spec_from_file_location("_abl321_ab", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_abl321_ab"] = module
    spec.loader.exec_module(module)
    return module


def _neutralise_abl332() -> None:
    db.aggregate_renewable_to_hourly = lambda df, *a, **kw: df
    wind_features._assert_hourly = lambda series, context: series


def _prove_the_patch_took(source: str = "energy_renewable") -> dict:
    """Self-check: the control is worthless if it silently ran post-fix code.

    AT/solar is quarter-hourly in `energy_renewable` over this span, so a
    correctly neutralised build must come back with off-hour observations. If
    this comes back hourly the patch did not take and the run must not be
    reported as a control.
    """
    builder = wind_features.RenewableFeatureBuilder(
        "AT", "solar", pd.Timestamp("2025-12-31"), pd.Timestamp("2026-08-10"),
        actuals_source=source,
    )
    index = pd.DatetimeIndex(builder._actuals.index)
    off_hour = int((index != index.floor("h")).sum())
    check = {"rows": int(len(index)), "off_hour_rows": off_hour,
             "sub_hourly": off_hour > 0}
    if not check["sub_hourly"]:
        raise SystemExit(
            f"ABL-334 control aborted: AT/solar came back hourly ({check}). "
            "The ABL-332 neutralisation did not take effect, so this run would "
            "be a duplicate of the post-fix arm, not a control."
        )
    return check


def main() -> int:
    ab = _load_ab_module()

    _neutralise_abl332()
    check = _prove_the_patch_took()
    print(f"ABL-334 control: neutralisation verified, AT/solar {check}")

    argv = sys.argv[1:]
    parser_args = [
        "--replica-db", r"C:\Code\able\data\energy_dashboard.db",
        "--fit-start", "2026-01-14",
        "--gate-start", "2026-07-11",
        "--gate-end", "2026-08-10",
        "--algorithm", "catboost",
        "--json-out", "experiments/ABL334/results_prefix_control.json",
        "--report-out", "reports/abl_334_prefix_control.md",
    ]
    sys.argv = ["evaluate_renewable_source_switch.py", *parser_args, *argv]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica-db")
    parser.add_argument("--fit-start")
    parser.add_argument("--gate-start")
    parser.add_argument("--gate-end")
    parser.add_argument("--algorithm")
    parser.add_argument("--only-country", default="")
    parser.add_argument("--only-stream", default="")
    parser.add_argument("--json-out")
    parser.add_argument("--report-out")
    args = parser.parse_args(sys.argv[1:])

    result = ab.run(args)
    result["meta"]["abl334_control"] = {
        "what": "pre-ABL-332 feature builder, post-ABL-332 commit",
        "neutralised": ["db.aggregate_renewable_to_hourly -> identity",
                        "wind_features._assert_hourly -> identity"],
        "neutralisation_check": check,
        "registered": "experiments/ABL321/protocol.md, Amendment 3",
    }

    json_path, report_path = Path(args.json_out), Path(args.report_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    report_path.write_text(ab.render_markdown(result), encoding="utf-8")
    print(f"wrote {report_path} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
