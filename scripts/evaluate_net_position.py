#!/usr/bin/env python3
"""Evaluate the net-position forecast against actuals (ABL-30, Phase B).

Repeatable — every future net-position model claim goes through this script.
Joins the sidecar's as-served vintages (plus prod-pushed copies in the replica)
to `net_position` actuals; scores per country x horizon x hour-of-day x weekday
with the amplitude checks (slope, sd-ratio), pinball/coverage, serve-faithful
baselines, an error decomposition, and the pre-registered promotion gate.

Reads both databases strictly read-only. Writes only the report files.

Usage (workstation):
    .venv\\Scripts\\python.exe scripts\\evaluate_net_position.py
    # explicit paths / model:
    ... --replica-db C:\\Code\\able\\data\\energy_dashboard.db ^
        --sidecar-db C:\\Code\\able\\data\\forecasts_local.db ^
        --model chronos-2-V010 --out-dir reports\\net_position_eval

The weekly scheduled invocation (scripts/workstation/run-net-position.ps1)
writes reports keyed by ISO week, refreshed on every daily run.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.evaluation.net_position import (
    EvalConfig, compare_models, evaluate, render_comparison_markdown, render_markdown,
)

DEFAULT_REFERENCE_BACKTEST = Path(__file__).parent.parent / "comparison_net_position_servefaithful.json"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replica-db", default=str(config.DATABASE_PATH),
                   help="actuals + prod-pushed forecasts (default: ENERGY_DB_PATH)")
    p.add_argument("--sidecar-db", default=config.FORECAST_OUTPUT_DB,
                   help="as-served local vintages (default: FORECAST_OUTPUT_DB; "
                        "omit to score prod-pushed rows only)")
    p.add_argument("--model", nargs="+", default=["chronos-2-V010"],
                   help="one model to evaluate, or two or more to compare over an "
                        "identical vintage window (the C2c deliverable). This script "
                        "does NOT discover model versions — it scores exactly the "
                        "names given here.")
    p.add_argument("--start", help="target window start, UTC (default: all vintages)")
    p.add_argument("--end", help="target window end, UTC")
    p.add_argument("--gate-vintage-start",
                   help="earliest generated_at the GATE scores, UTC. Default: the "
                        "cohort split, so the gate never scores pre-fix vintages "
                        "(ABL-72 G1). The full report still covers every vintage.")
    p.add_argument("--gate-vintage-end",
                   help="exclusive generated_at upper bound for the gate, UTC")
    p.add_argument("--out-dir", default=str(Path(__file__).parent.parent / "reports" / "net_position_eval"))
    p.add_argument("--tag", help="report filename tag (default: ISO week, e.g. 2026-W32)")
    p.add_argument("--top-misses", type=int, default=10)
    p.add_argument("--climatology-days", type=int, default=28)
    p.add_argument("--cohort-split", default=None,
                   help="UTC timestamp separating pre/post-fix vintages "
                        "(default: 1c5a24f deploy time, 2026-08-04 14:29)")
    p.add_argument("--candidate-backtest",
                   help="candidate W01-W12 JSON for the no-regression gate check")
    p.add_argument("--reference-backtest", default=str(DEFAULT_REFERENCE_BACKTEST),
                   help="reference W01-W12 JSON (default: V010 serve-faithful)")
    p.add_argument("--serve-faithful-verified", action="store_true",
                   help="attest that serve-parity was manually verified for this "
                        "candidate (gate item; never inferred)")
    p.add_argument("--stdout", action="store_true", help="print markdown instead of writing files")
    args = p.parse_args()

    if not Path(args.replica_db).exists():
        print(f"replica DB not found: {args.replica_db}", file=sys.stderr)
        return 2
    if args.sidecar_db and not Path(args.sidecar_db).exists():
        print(f"sidecar DB not found: {args.sidecar_db} — scoring replica rows only",
              file=sys.stderr)
        args.sidecar_db = None

    cfg = EvalConfig(
        replica_db=args.replica_db, sidecar_db=args.sidecar_db, model_name=args.model[0],
        start=args.start, end=args.end, top_misses=args.top_misses,
        climatology_days=args.climatology_days,
        candidate_backtest=args.candidate_backtest,
        reference_backtest=args.reference_backtest,
        serve_faithful_verified=args.serve_faithful_verified,
        gate_vintage_start=(pd.Timestamp(args.gate_vintage_start)
                            if args.gate_vintage_start else None),
        gate_vintage_end=(pd.Timestamp(args.gate_vintage_end)
                          if args.gate_vintage_end else None))
    if args.cohort_split:
        cfg.cohort_split = pd.Timestamp(args.cohort_split)

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d %H:%M UTC")
    multi = len(args.model) > 1
    if multi:
        results = compare_models(cfg, args.model)
        md = render_comparison_markdown(results, stamp)
        prefix, summary = "net_position_compare", (
            "" if "error" in results else
            f"({len(results['per_model'])} models, "
            f"{results['window']['vintage_start'][:16]} → "
            f"{(results['window']['vintage_end'] or 'open')[:16]}, "
            + ", ".join(f"{m}: {v}" for m, v in results["verdict_per_model"].items()) + ")")
    else:
        results = evaluate(cfg)
        md = render_markdown(results, stamp)
        prefix, summary = "net_position_eval", (
            f"({results.get('meta', {}).get('pairs_scored', 0):,} pairs, "
            f"gate: {results.get('gate', {}).get('verdict', 'n/a')} over "
            f"{(results.get('gate_scope') or {}).get('vintages', 0)} vintages)")

    if args.stdout:
        # The report contains ·, → and ✅; a Windows console defaults to cp1252
        # and raises UnicodeEncodeError on all three, which made --stdout unusable
        # there. Re-encode the stream rather than degrading the report.
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
        print(md)
        return 0 if "error" not in results else 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"
    blob = json.dumps(results, indent=1, default=str)
    latest = "latest_compare" if multi else "latest"
    for name, content in [(f"{prefix}_{tag}.md", md), (f"{prefix}_{tag}.json", blob),
                          (f"{latest}.md", md), (f"{latest}.json", blob)]:
        (out_dir / name).write_text(content, encoding="utf-8")
    print(f"wrote {out_dir / f'{prefix}_{tag}.md'} {summary}")
    return 0 if "error" not in results else 1


if __name__ == "__main__":
    sys.exit(main())
