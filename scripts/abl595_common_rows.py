#!/usr/bin/env python3
"""ABL-595: four-way comparison on the rows ALL FOUR models share.

Reported beside the gate read, never a criterion. The pre-registered gate is
per-model over the registered vintage window and is scored by
`evaluate_net_position.py`; this table exists only so the four columns are one
exam rather than three (the trap the CEO named in the 2026-08-11 early read).

Rows are matched exactly as `src/evaluation/head_to_head.pair` does -- on
(country, target hour, run), where a run is the actuals cutoff the vintage
could see, not its `generated_at` (ABL-82). A row enters only if all four
models produced a value from the same run and an actual exists.
"""
from __future__ import annotations

import json
import sys
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.head_to_head import MAX_RUN_SKEW  # noqa: E402
from src.evaluation.net_position import (  # noqa: E402
    EvalConfig, GATE_BIAS_FRAC, GATE_COVERAGE_RANGE, GATE_SLOPE_RANGE,
    _parse_ts, _ro_connect, as_of_for_vintage, load_actuals,
)

REPLICA = r"C:\Code\able\data\energy_dashboard.db"
SIDECAR = r"C:\Code\able\data\forecasts_local.db"
MODELS = {"chronos-2-V010": "V010", "baseline-V012": "V012",
          "xgboost-V014": "V014", "chronos-2-V016": "V016"}
VINTAGE_START = pd.Timestamp("2026-08-07")
VINTAGE_END = pd.Timestamp("2026-08-27")   # exclusive


def load(model: str) -> pd.DataFrame:
    frames = []
    for src, path in (("sidecar", SIDECAR), ("replica", REPLICA)):
        con = _ro_connect(path)
        try:
            df = pd.read_sql_query(
                """SELECT country_code, target_timestamp_utc, generated_at,
                          forecast_value FROM forecasts
                   WHERE forecast_type='net_position' AND model_name=?""",
                con, params=(model,))
            q = pd.read_sql_query(
                """SELECT country_code, target_timestamp_utc, generated_at,
                          quantile, forecast_value FROM forecast_quantiles
                   WHERE forecast_type='net_position' AND model_name=?""",
                con, params=(model,))
        finally:
            con.close()
        if df.empty:
            continue
        df["target_ts"] = _parse_ts(df["target_timestamp_utc"])
        df["generated_at"] = _parse_ts(df["generated_at"])
        df = df.drop(columns=["target_timestamp_utc"])
        if not q.empty:
            q["target_ts"] = _parse_ts(q["target_timestamp_utc"])
            q["generated_at"] = _parse_ts(q["generated_at"])
            w = q[q["quantile"].isin([0.1, 0.9])].pivot_table(
                index=["country_code", "target_ts", "generated_at"],
                columns="quantile", values="forecast_value")
            w.columns = [f"q{int(round(c * 100))}" for c in w.columns]
            df = df.merge(w.reset_index(),
                          on=["country_code", "target_ts", "generated_at"], how="left")
        frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True)
    key = ["country_code", "target_ts", "generated_at"]
    df = all_rows.drop_duplicates(subset=key, keep="first")
    df = df[(df["generated_at"] >= VINTAGE_START) & (df["generated_at"] < VINTAGE_END)]
    df["run_as_of"] = df["generated_at"].map(
        {ts: as_of_for_vintage(ts) for ts in df["generated_at"].unique()})
    for c in ("q10", "q90"):
        if c not in df.columns:
            df[c] = np.nan
    return df[["country_code", "target_ts", "run_as_of", "generated_at",
               "forecast_value", "q10", "q90"]]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    keys = ["country_code", "target_ts", "run_as_of"]
    per = {}
    for m, tag in MODELS.items():
        d = load(m)
        # One row per (country, hour, run): if a run bucket holds two vintages
        # for one target hour, keep the newest -- a re-run must not duplicate.
        d = (d.sort_values("generated_at", ascending=False, kind="stable")
               .drop_duplicates(keys, keep="first")
               .rename(columns={"forecast_value": f"f_{tag}",
                                "q10": f"q10_{tag}", "q90": f"q90_{tag}",
                                "generated_at": f"gen_{tag}"}))
        per[tag] = d
        print(f"{tag}: {len(d):,} rows, {d['run_as_of'].nunique()} runs, "
              f"{d['country_code'].nunique()} zones")

    joined = reduce(lambda a, b: a.merge(b, on=keys, how="inner"), per.values())
    # every pair of vintages inside one row must be the same run
    gens = joined[[f"gen_{t}" for t in MODELS.values()]]
    skew = (gens.max(axis=1) - gens.min(axis=1))
    print(f"\nwidest within-row vintage spread: {skew.max()}")
    joined = joined[skew <= MAX_RUN_SKEW]

    actuals = load_actuals(EvalConfig(replica_db=REPLICA, sidecar_db=None)
                           ).rename(columns={"ts": "target_ts"})
    joined = joined.merge(actuals, on=["country_code", "target_ts"], how="inner")
    joined = joined.dropna(subset=["actual"])
    print(f"common rows all four models + actual: {len(joined):,} over "
          f"{joined['run_as_of'].nunique()} runs, "
          f"{joined['country_code'].nunique()} zones, "
          f"targets {joined['target_ts'].min()} .. {joined['target_ts'].max()}")

    lo, hi = GATE_SLOPE_RANGE
    clo, chi = GATE_COVERAGE_RANGE
    rows = []
    for c, g in joined.groupby("country_code"):
        a = g["actual"].to_numpy()
        rec = {"country": c, "n": len(g), "mean_abs_actual": float(np.abs(a).mean())}
        for t in MODELS.values():
            f = g[f"f_{t}"].to_numpy()
            err = f - a
            rec[f"mae_{t}"] = float(np.abs(err).mean())
            rec[f"bias_{t}"] = float(err.mean())
            rec[f"slope_{t}"] = float(np.cov(a, f, bias=True)[0, 1] / np.var(a))
            rec[f"corr_{t}"] = float(np.corrcoef(a, f)[0, 1])
            band = g.dropna(subset=[f"q10_{t}", f"q90_{t}"])
            rec[f"cov_{t}"] = (100.0 * float(((band["actual"] >= band[f"q10_{t}"])
                                              & (band["actual"] <= band[f"q90_{t}"])).mean())
                               if len(band) else None)
        rows.append(rec)
    df = pd.DataFrame(rows).sort_values("country")

    print("\n### Four-way, common rows only (n identical across columns)\n")
    print("| country | n | mean abs NP | " + " | ".join(
        f"{t} MAE" for t in MODELS.values()) + " |")
    print("|---" * (len(MODELS) + 3) + "|")
    for r in df.itertuples():
        print(f"| {r.country} | {r.n:,} | {r.mean_abs_actual:,.0f} | "
              + " | ".join(f"{getattr(r, f'mae_{t}'):,.0f}" for t in MODELS.values()) + " |")

    tot = joined
    print("\n| model | pooled MAE MW | pooled bias MW | WAPE % | countries best-of-4 |")
    print("|---|---:|---:|---:|---:|")
    sumabs = float(np.abs(tot["actual"]).sum())
    best = df[[f"mae_{t}" for t in MODELS.values()]].idxmin(axis=1).value_counts()
    for t in MODELS.values():
        e = tot[f"f_{t}"] - tot["actual"]
        print(f"| {t} | {float(e.abs().mean()):,.1f} | {float(e.mean()):,.1f} | "
              f"{100.0 * float(e.abs().sum()) / sumabs:.1f} | "
              f"{int(best.get(f'mae_{t}', 0))} |")

    print("\n### Per-country screen counts on common rows (bar in brackets)\n")
    print("| screen | " + " | ".join(MODELS.values()) + " |")
    print("|---" * (len(MODELS) + 1) + "|")
    bias_pass = {t: int(sum(abs(r[f"bias_{t}"]) < GATE_BIAS_FRAC * r["mean_abs_actual"]
                            for _, r in df.iterrows())) for t in MODELS.values()}
    slope_pass = {t: int(sum(lo <= r[f"slope_{t}"] <= hi for _, r in df.iterrows()))
                  for t in MODELS.values()}
    cov_pass = {t: int(sum(r[f"cov_{t}"] is not None and clo <= r[f"cov_{t}"] <= chi
                           for _, r in df.iterrows())) for t in MODELS.values()}
    print("| |bias| < 5% of mean abs NP (19/19) | "
          + " | ".join(f"{bias_pass[t]}/19" for t in MODELS.values()) + " |")
    print("| slope in [0.8, 1.2] (19/19) | "
          + " | ".join(f"{slope_pass[t]}/19" for t in MODELS.values()) + " |")
    print("| 10-90 coverage in [75, 85]% (19/19) | "
          + " | ".join(f"{cov_pass[t]}/19" for t in MODELS.values()) + " |")

    out = Path(__file__).parents[1] / "_scratch" / "abl595" / "common_rows.json"
    out.write_text(json.dumps({
        "protocol": "common (country, target hour, run) rows across all four models",
        "vintage_window": [str(VINTAGE_START), str(VINTAGE_END)],
        "rows": int(len(joined)), "runs": int(joined["run_as_of"].nunique()),
        "per_country": df.to_dict(orient="records"),
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
