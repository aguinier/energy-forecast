#!/usr/bin/env python
"""Re-run one country's net-position baseline read (ABL-280).

The confirmatory re-read ABL-280 is held open for. Reuses
`src/evaluation/net_position.py`'s loaders and serve-faithful baselines so the
numbers match the gate's conventions, and adds the zero-forecast baseline and
the level-vs-shape split (`src/evaluation/country_reread.py`).

    .venv\\Scripts\\python.exe scripts/reread_net_position_country.py \\
        --country RO \\
        --replica-db C:\\Code\\able\\data\\energy_dashboard.db \\
        --sidecar-db C:\\Code\\able\\data\\forecasts_local.db --stdout

Both databases are opened read-only; the only writes are the report files under
`reports/net_position_eval/country_reread/`. A run below the pre-registered
minimum scored-vintage count is labelled INTERIM in its own output rather than
being refused — an interim number is useful, an unlabelled one is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import net_position as npz  # noqa: E402
from src.evaluation.country_reread import (  # noqa: E402
    country_reread,
    fleet_summary,
    render_fleet_markdown,
    render_markdown,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "net_position_eval" / "country_reread"


def build_paired(cfg: npz.EvalConfig) -> pd.DataFrame:
    """Forecast rows left-joined to actuals, with serve-faithful baselines.

    Mirrors the first half of `net_position.evaluate` deliberately rather than
    calling it: `evaluate` returns rendered results, and this read needs the
    unpaired rows too (they are what distinguishes a vintage that exists from a
    vintage that carries evidence).
    """
    forecasts = npz.load_forecasts(cfg)
    if forecasts.empty:
        raise SystemExit(f"no '{cfg.model_name}' net_position forecasts found")
    actuals = npz.load_actuals(cfg)
    lookup = {c: g.set_index("ts")["actual"].sort_index()
              for c, g in actuals.groupby("country_code")}
    paired = forecasts.merge(actuals.rename(columns={"ts": "target_ts"}),
                             on=["country_code", "target_ts"], how="left")
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    chunks = {"persistence": [], "climatology": []}
    for (country, gen), g in paired.groupby(["country_code", "generated_at"]):
        preds = npz.baseline_predictions(lookup.get(country, empty),
                                         npz.as_of_for_vintage(gen),
                                         pd.DatetimeIndex(g["target_ts"]),
                                         cfg.climatology_days)
        for k in chunks:
            chunks[k].append(pd.Series(preds[k].to_numpy(), index=g.index))
    for k, ch in chunks.items():
        paired[k] = pd.concat(ch).sort_index()
    paired["baseline_ensemble"] = paired[["persistence", "climatology"]].mean(axis=1)
    paired.attrs["actuals_max_ts"] = str(actuals["ts"].max()) if len(actuals) else None
    return paired


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--country", default="RO", help="country code (default RO)")
    p.add_argument("--replica-db", required=True)
    p.add_argument("--sidecar-db", default=None)
    p.add_argument("--model", default="chronos-2-V010")
    p.add_argument("--climatology-days", type=int, default=28)
    p.add_argument("--min-scored-vintages", type=int,
                   default=npz.GATE_MIN_LIVE_VINTAGES,
                   help="pre-registered minimum, counted in SCORED vintages")
    p.add_argument("--cohort-split", default=str(npz.FIX_DEPLOYED_UTC))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--fleet", action="store_true",
                   help="also sweep every gate country — the context that "
                        "separates a zone-specific defect from a fleet-wide one")
    p.add_argument("--stdout", action="store_true", help="print the report")
    p.add_argument("--no-write", action="store_true", help="do not write files")
    args = p.parse_args()

    cfg = npz.EvalConfig(replica_db=args.replica_db, sidecar_db=args.sidecar_db,
                         model_name=args.model,
                         climatology_days=args.climatology_days)
    paired = build_paired(cfg)
    read = country_reread(paired, args.country.upper(),
                          cohort_split=pd.Timestamp(args.cohort_split),
                          min_scored_vintages=args.min_scored_vintages)
    read["meta"] = {
        "model": args.model, "replica_db": args.replica_db,
        "sidecar_db": args.sidecar_db,
        "climatology_days": args.climatology_days,
        "actuals_max_ts": paired.attrs.get("actuals_max_ts"),
        "sidecar_vs_pushed_max_abs_diff_mw":
            paired.attrs.get("overlap_max_abs_diff_mw"),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = render_markdown(read, stamp)
    fleet = fleet_md = None
    if args.fleet:
        fleet = fleet_summary(paired, npz.GATE_COUNTRIES,
                              cohort_split=pd.Timestamp(args.cohort_split),
                              min_scored_vintages=args.min_scored_vintages)
        fleet_md = render_fleet_markdown(fleet, stamp, args.min_scored_vintages)
        md = md + "\n" + fleet_md
    if args.stdout:
        print(md)

    if not args.no_write:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        base = f"{read['country']}_{day}"
        if fleet is not None:
            read["fleet"] = fleet
        (out / f"{base}.md").write_text(md, encoding="utf-8")
        (out / f"{base}.json").write_text(json.dumps(read, indent=2, default=str),
                                          encoding="utf-8")
        print(f"wrote {out / (base + '.md')}", file=sys.stderr)

    # Exit 0 always: an interim read is a legitimate outcome, not an error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
