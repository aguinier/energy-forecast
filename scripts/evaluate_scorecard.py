#!/usr/bin/env python3
"""Write the recurring nine-type forecast-quality scorecard (ABL-129)."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.evaluation.scorecard import ScorecardConfig, evaluate_scorecard, render_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replica-db", default=str(config.DATABASE_PATH))
    parser.add_argument("--sidecar-db", default=config.FORECAST_OUTPUT_DB)
    parser.add_argument("--start", help="target window start UTC; default: 30 days before end")
    parser.add_argument("--end", help="exclusive target window end UTC; default: today 00:00 UTC")
    parser.add_argument("--out-dir", default=str(Path(__file__).parent.parent / "reports" / "forecast_scorecard"))
    parser.add_argument("--tag", help="versioned filename tag; default: end date")
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    start = pd.Timestamp(args.start) if args.start else end - pd.Timedelta(days=30)
    if start >= end:
        parser.error("--start must be earlier than --end")
    if not Path(args.replica_db).exists():
        print(f"replica DB not found: {args.replica_db}", file=sys.stderr)
        return 2
    sidecar = args.sidecar_db if args.sidecar_db and Path(args.sidecar_db).exists() else None

    cfg = ScorecardConfig(replica_db=args.replica_db, sidecar_db=sidecar,
                          start=start, end=end)
    results = evaluate_scorecard(cfg)
    now = datetime.now(timezone.utc)
    generated = now.strftime("%Y-%m-%d %H:%M UTC")
    results["meta"]["generated_at"] = generated
    markdown = render_markdown(results, generated)
    if args.stdout:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
        print(markdown)
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or end.strftime("%Y-%m-%d")
    blob = json.dumps(results, indent=1, allow_nan=False)
    for name, content in ((f"scorecard-{tag}.md", markdown),
                          (f"scorecard-{tag}.json", blob),
                          ("latest.md", markdown), ("latest.json", blob)):
        (out_dir / name).write_text(content, encoding="utf-8")
    print(f"wrote {out_dir / f'scorecard-{tag}.md'} "
          f"({results['meta']['paired_actual_rows']:,} paired rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
