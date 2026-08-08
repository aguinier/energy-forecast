"""Exactly-paired comparison of two model versions (ABL-68).

**Why this exists.** `evaluate_net_position.py` scores one `model_name` per
invocation and writes one report per model. Reading two of those reports side by
side and comparing their MAE columns is wrong, and wrong in the direction that
flatters a challenger.

Measured 2026-08-08 on the V016 held-out window: the champion's report covers 57
vintages, the challenger's 49, because the champion's report also picks up
prod-pushed vintages that live in the replica and were never part of the
reconstruction a challenger is built from. Compared report-to-report, V016 looked
*better* in most countries (FR 2,464 -> 1,916 MW, DE 3,344 -> 3,014 MW). Compared
on the 22,344 rows both models actually cover, V016 is 1.4% *worse* (775.2 ->
786.1 MW). The difference is entirely which hours each model was scored on.

So a head-to-head joins on `(country_code, target_ts, generated_at)` and scores
only rows where both models produced a value and an actual exists. A model is
never credited or penalised for an hour its opponent never forecast.

`n_only_a` / `n_only_b` are reported rather than silently dropped: a large
one-sided remainder means the two series do not describe the same experiment,
and the comparison should be read as scoped to the overlap, not as complete.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

JOIN_KEYS = ["country_code", "target_ts", "generated_at"]

# Below this the two models differ by less than the rounding a reader would
# apply anyway. Calling a 0.1% gap a "win" invents a result out of noise.
MATERIAL_PCT = 0.5


@dataclass
class CountryHeadToHead:
    country: str
    n: int
    mae_a_mw: float
    mae_b_mw: float
    delta_mw: float          # b - a; negative means b (the challenger) is better
    delta_pct: float
    verdict: str             # "better" | "worse" | "tie" | "identical"


@dataclass
class HeadToHead:
    model_a: str
    model_b: str
    n_paired: int
    n_only_a: int
    n_only_b: int
    n_vintages: int
    pooled_mae_a_mw: float
    pooled_mae_b_mw: float
    pooled_delta_mw: float
    pooled_delta_pct: float
    countries: list[CountryHeadToHead] = field(default_factory=list)

    @property
    def n_better(self) -> int:
        return sum(1 for c in self.countries if c.verdict == "better")

    @property
    def n_materially_better(self) -> int:
        return sum(1 for c in self.countries
                   if c.verdict == "better" and abs(c.delta_pct) >= MATERIAL_PCT)

    @property
    def n_identical(self) -> int:
        return sum(1 for c in self.countries if c.verdict == "identical")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_better"] = self.n_better
        d["n_materially_better"] = self.n_materially_better
        d["n_identical"] = self.n_identical
        return d


def _verdict(mae_a: float, mae_b: float) -> str:
    if mae_a == mae_b:
        # A pass-through country: the challenger *is* the champion there. Saying
        # "tie" would hide that no correction was applied at all.
        return "identical"
    pct = 100.0 * (mae_b - mae_a) / mae_a if mae_a else float("nan")
    if abs(pct) < MATERIAL_PCT:
        return "tie"
    return "better" if mae_b < mae_a else "worse"


def pair(a: pd.DataFrame, b: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Inner-join two forecast series to each other and to actuals.

    Each frame needs `country_code`, `target_ts`, `generated_at`,
    `forecast_value`; `actuals` needs `country_code`, `target_ts`, `actual`.
    Rows missing a forecast from either model, or missing an actual, are
    dropped — an unmeasurable hour scores neither model.
    """
    a = a.rename(columns={"forecast_value": "forecast_a"})
    b = b.rename(columns={"forecast_value": "forecast_b"})
    joined = a.merge(b, on=JOIN_KEYS, how="inner")
    joined = joined.merge(actuals, on=["country_code", "target_ts"], how="inner")
    return joined.dropna(subset=["forecast_a", "forecast_b", "actual"])


def compare(paired: pd.DataFrame, model_a: str, model_b: str,
            n_only_a: int = 0, n_only_b: int = 0) -> HeadToHead:
    """Pooled and per-country MAE for two models over identical rows.

    Returns zeroed pooled metrics with no countries when `paired` is empty,
    rather than a NaN that reads as a measurement.
    """
    if paired.empty:
        return HeadToHead(model_a, model_b, 0, n_only_a, n_only_b, 0,
                          0.0, 0.0, 0.0, 0.0, [])

    err_a = (paired["forecast_a"] - paired["actual"]).abs()
    err_b = (paired["forecast_b"] - paired["actual"]).abs()
    mae_a, mae_b = float(err_a.mean()), float(err_b.mean())

    rows: list[CountryHeadToHead] = []
    for country, g in paired.groupby("country_code", sort=True):
        ca = float((g["forecast_a"] - g["actual"]).abs().mean())
        cb = float((g["forecast_b"] - g["actual"]).abs().mean())
        rows.append(CountryHeadToHead(
            country=str(country), n=int(len(g)), mae_a_mw=ca, mae_b_mw=cb,
            delta_mw=cb - ca,
            delta_pct=(100.0 * (cb - ca) / ca) if ca else float("nan"),
            verdict=_verdict(ca, cb)))

    return HeadToHead(
        model_a=model_a, model_b=model_b, n_paired=int(len(paired)),
        n_only_a=n_only_a, n_only_b=n_only_b,
        n_vintages=int(paired["generated_at"].nunique()),
        pooled_mae_a_mw=mae_a, pooled_mae_b_mw=mae_b,
        pooled_delta_mw=mae_b - mae_a,
        pooled_delta_pct=(100.0 * (mae_b - mae_a) / mae_a) if mae_a else float("nan"),
        countries=rows)


def render_markdown(h: HeadToHead, window: str, generated_at: str) -> str:
    """Report that leads with the scope of the comparison, not the winner."""
    better = "better" if h.pooled_delta_mw < 0 else "worse"
    lines = [
        f"# Head-to-head: `{h.model_b}` vs `{h.model_a}`", "",
        f"**Generated:** {generated_at} · **Window:** {window}",
        f"**Paired rows:** {h.n_paired:,} over {h.n_vintages} vintages · "
        f"rows only in `{h.model_a}`: {h.n_only_a:,} · "
        f"only in `{h.model_b}`: {h.n_only_b:,}", "",
        "Scored only on `(country, target hour, vintage)` rows where **both** "
        "models produced a value and an actual exists. Per-model reports are "
        "*not* comparable to each other — they cover different vintage sets.", "",
        "## Pooled", "",
        f"`{h.model_a}` **{h.pooled_mae_a_mw:,.1f} MW** MAE · "
        f"`{h.model_b}` **{h.pooled_mae_b_mw:,.1f} MW** MAE · "
        f"challenger is **{abs(h.pooled_delta_pct):.1f}% {better}** "
        f"({h.pooled_delta_mw:+,.1f} MW)", "",
        f"Materially better (>= {MATERIAL_PCT}%) in **{h.n_materially_better}/"
        f"{len(h.countries)}** countries; "
        f"identical (pass-through) in **{h.n_identical}**.", "",
        "Pooled MAE mixes countries of very different size — the per-country "
        "table is the one that gates.", "",
        "## Per country", "",
        f"| country | n | {h.model_a} MAE | {h.model_b} MAE | Δ MW | Δ % | verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for c in h.countries:
        lines.append(
            f"| {c.country} | {c.n:,} | {c.mae_a_mw:,.1f} | {c.mae_b_mw:,.1f} "
            f"| {c.delta_mw:+,.1f} | {c.delta_pct:+.1f}% | {c.verdict} |")
    lines += ["", "`identical` means the challenger passed that country through "
                  "uncorrected — it *is* the champion there, by design, not by tie."]
    return "\n".join(lines) + "\n"
