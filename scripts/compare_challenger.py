#!/usr/bin/env python3
"""Exactly-paired head-to-head between a champion and a challenger (ABL-68).

This is the sanctioned way to answer "did the challenger beat the champion".
Do **not** answer it by putting two `evaluate_net_position.py` reports side by
side: those are scored on whatever rows each model happens to cover, and the
champion's set includes prod-pushed vintages a challenger's reconstruction never
had. Measured 2026-08-08 on V016's held-out window, the report-to-report read
said the challenger won in most countries while the paired read said it lost.

Usage:
    python scripts/compare_challenger.py \
        --champion-db C:/Code/able/data/forecasts_recon.db \
        --challenger-db C:/Code/able/data/forecasts_v016_holdout.db \
        --challenger chronos-2-V016 \
        --replica-db C:/Code/able/data/energy_dashboard.db \
        --start 2026-06-17 --end 2026-08-04T23:00 \
        --out-dir reports/head_to_head/V016

Write under `reports/head_to_head/`, not `reports/net_position_eval/`: the
latter is gitignored because the scheduled eval rewrites it every run, and a
promotion verdict has to survive in the repo that cites it.

Rows are paired per *run*, not per `generated_at` -- on the live sidecar the
champion and the challengers are separate processes and never share a stamp
(ABL-82). Exit status is **1 when nothing paired**, so a gate script cannot read
an empty comparison as "no difference".
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.evaluation.head_to_head import (MAX_RUN_SKEW, compare, pair,
                                         render_markdown)
from src.evaluation.net_position import (EvalConfig, _parse_ts, _ro_connect,
                                         load_actuals)
from src.challengers.registry import CHAMPION_MODEL_NAME


def load_series(db: str, model_name: str) -> pd.DataFrame:
    con = _ro_connect(db)
    try:
        df = pd.read_sql_query(
            """SELECT country_code, target_timestamp_utc, generated_at,
                      forecast_value FROM forecasts
               WHERE forecast_type = 'net_position' AND model_name = ?""",
            con, params=(model_name,))
    finally:
        con.close()
    if df.empty:
        raise SystemExit(f"no '{model_name}' net_position rows in {db}")
    df["target_ts"] = _parse_ts(df["target_timestamp_utc"])
    return df[["country_code", "target_ts", "generated_at", "forecast_value"]]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--champion-db", required=True)
    p.add_argument("--challenger-db", required=True)
    p.add_argument("--champion", default=CHAMPION_MODEL_NAME)
    p.add_argument("--challenger", required=True)
    p.add_argument("--replica-db", default=str(config.DATABASE_PATH))
    p.add_argument("--start", help="target window start, UTC")
    p.add_argument("--end", help="target window end, UTC")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--stdout", action="store_true")
    p.add_argument("--max-run-skew-hours", type=float,
                   default=MAX_RUN_SKEW.total_seconds() / 3600.0,
                   help="two vintages sharing an actuals cutoff but further "
                        "apart than this are different runs and are not paired. "
                        "It cannot widen the comparison past the cutoff: "
                        "vintages that saw different actuals are never "
                        "candidates at any value")
    args = p.parse_args()

    if not Path(args.replica_db).exists():
        print(f"replica DB not found: {args.replica_db}", file=sys.stderr)
        return 2

    a = load_series(args.champion_db, args.champion)
    b = load_series(args.challenger_db, args.challenger)

    def window(df):
        if args.start:
            df = df[df["target_ts"] >= pd.Timestamp(args.start)]
        if args.end:
            df = df[df["target_ts"] <= pd.Timestamp(args.end)]
        return df

    a, b = window(a), window(b)

    actuals = load_actuals(EvalConfig(
        replica_db=args.replica_db, sidecar_db=None,
        model_name=args.champion)).rename(columns={"ts": "target_ts"})

    # The join key lives in head_to_head, and the one-sided counts come back
    # from the same call. Recomputing them here is what let the caller and the
    # module disagree about what a vintage is (ABL-82).
    paired, scope = pair(a, b, actuals,
                         max_run_skew=pd.Timedelta(hours=args.max_run_skew_hours))

    h = compare(paired, args.champion, args.challenger, scope)
    now = datetime.now(timezone.utc)
    win = f"{args.start or 'all'} .. {args.end or 'all'}"
    md = render_markdown(h, win, now.strftime("%Y-%m-%d %H:%M UTC"))

    # An unmeasured comparison exits non-zero. A promotion gate that reads this
    # script must not mistake "nothing paired" for "no difference" — that is
    # exactly how the ABL-82 join defect stayed invisible, printing a full
    # report of 0.0 MW MAEs for three challengers at once.
    if not h.measured:
        print(f"NOT MEASURED: 0 rows paired between '{args.champion}' and "
              f"'{args.challenger}' ({scope.n_only_a:,} rows only in the "
              f"champion, {scope.n_only_b:,} only in the challenger, "
              f"{scope.n_rejected_skew:,} hours rejected as different runs). "
              f"No comparison was made.", file=sys.stderr)

    if args.stdout or not args.out_dir:
        # The report carries non-ASCII (Δ, ·) and a Windows console is cp1252,
        # so a bare print() raises UnicodeEncodeError and loses the whole run.
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
        sys.stdout.write(md)
        return 0 if h.measured else 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"
    (out_dir / f"head_to_head_{tag}.md").write_text(md, encoding="utf-8")
    (out_dir / f"head_to_head_{tag}.json").write_text(
        json.dumps(h.to_dict(), indent=1, default=str), encoding="utf-8")
    (out_dir / "latest.md").write_text(md, encoding="utf-8")
    if not h.measured:
        print(f"wrote {out_dir / f'head_to_head_{tag}.md'} (not measured)")
        return 1
    print(f"wrote {out_dir / f'head_to_head_{tag}.md'} "
          f"({h.n_paired:,} paired rows over {h.n_vintages} runs; challenger "
          f"{h.pooled_delta_pct:+.1f}% vs champion, materially better in "
          f"{h.n_materially_better}/{len(h.countries)} countries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
