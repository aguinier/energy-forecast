"""Exactly-paired comparison of two model versions (ABL-68, ABL-82).

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

So a head-to-head scores only rows where both models produced a value *from the
same run* and an actual exists. A model is never credited or penalised for an
hour its opponent never forecast, and never for an hour its opponent forecast
with more information than it had.

**A run is not a `generated_at` (ABL-82).** The first cut joined on exact
`generated_at` equality, which is right for a reconstruction — one process
replays every vintage and stamps them all — and wrong for the live rail, where
the champion and the challengers are separate processes in
`run-net-position.ps1` and each stamps its own `datetime.now()`. Measured on the
live sidecar 2026-08-09, the three challengers ran 12.3 s after the champion and
the champion additionally carries microseconds, so the exact join paired **0**
rows for every challenger while 1,368 pairs were sitting there. An empty
head-to-head is the dangerous kind of empty: it reads as "the challenger has no
overlapping vintages", not as "the join key is wrong".

Two vintages therefore belong to the same run when they agree on **the actuals
they could see** — `net_position.as_of_for_vintage`, the same serve-faithful
cutoff the eval's baselines use — and their `generated_at` are within
`MAX_RUN_SKEW` of each other. The cutoff carries the meaning; the skew bound is
the guard, because one cutoff bucket is 24 h wide and two runs a day apart can
land in the same one.

Measured on the live sidecar 2026-08-09, the two rules agree exactly: the pairs
they admit are the 12.3 s co-runs, and the pairs they reject are the two
backfills (V012/V016 at 2026-08-07 21:25, 15 h 25 m after that day's champion;
V014 at 2026-08-08 11:36, 5 h 36 m after). Rejecting those is the point — a
challenger that ran at 21:25 saw a further day of actuals, so scoring it against
a 06:00 champion would flatter it for information the champion never had.

`n_only_a` / `n_only_b` are reported rather than silently dropped: a large
one-sided remainder means the two series do not describe the same experiment,
and the comparison should be read as scoped to the overlap, not as complete.
`n_rejected_skew` is reported for the same reason — it is the count of hours
both models covered on one cutoff but too far apart in time to be one run.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import pandas as pd

from src.evaluation.net_position import as_of_for_vintage

# `run_as_of` is derived, not stored: the actuals cutoff implied by a vintage's
# `generated_at`. See the module docstring for why it, and not `generated_at`,
# is what identifies a run.
JOIN_KEYS = ["country_code", "target_ts", "run_as_of"]

# Two vintages sharing a cutoff but further apart than this are different runs.
# Measured on the live sidecar 2026-08-09 the real champion->challenger gap is
# 3.8-12.3 s; the nearest thing to a legitimate long gap is a champion re-run
# (2026-08-06 06:00 and 10:52), and that is champion-to-champion, which the
# closest-pair reduction handles without needing the bound widened. Anything
# from a minute to five hours selects the same pairs on the live data, so this
# is not a tuned edge; 4 h is chosen to sit above any plausible pipeline
# duration and below the 24 h width of a cutoff bucket.
MAX_RUN_SKEW = pd.Timedelta(hours=4)

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
class PairScope:
    """What the pairing covered, and what it had to leave out.

    Every field here is a reason a reader should not treat the comparison as
    complete. They are returned beside the paired frame rather than recomputed
    by the caller, because the caller recomputing them means the join key is
    written down twice and drifts (it did — ABL-82).
    """
    n_only_a: int = 0
    n_only_b: int = 0
    n_rejected_skew: int = 0
    max_skew_seconds: float | None = None


@dataclass
class HeadToHead:
    model_a: str
    model_b: str
    n_paired: int
    n_only_a: int
    n_only_b: int
    n_vintages: int          # distinct paired runs, keyed on the actuals cutoff
    pooled_mae_a_mw: float | None
    pooled_mae_b_mw: float | None
    pooled_delta_mw: float | None
    pooled_delta_pct: float | None
    countries: list[CountryHeadToHead] = field(default_factory=list)
    n_rejected_skew: int = 0
    max_skew_seconds: float | None = None

    @property
    def measured(self) -> bool:
        return self.n_paired > 0

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
        d["measured"] = self.measured
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


def _with_run_as_of(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the actuals cutoff each row's vintage could see.

    `generated_at` arrives as text from SQLite in at least two shapes — the
    champion writes microseconds, the challengers truncate to the second — so it
    is parsed here rather than trusted. The cutoff is computed once per distinct
    vintage and mapped, because a reconstruction DB carries ~10^6 rows over a
    few dozen vintages and a per-row Python call would dominate the run.
    """
    out = df.copy()
    out["generated_at"] = pd.to_datetime(out["generated_at"], format="mixed")
    lut = {ts: as_of_for_vintage(ts)
           for ts in out["generated_at"].dropna().unique()}
    out["run_as_of"] = out["generated_at"].map(lut)
    return out


def _one_sided(src: pd.DataFrame, paired: pd.DataFrame, gen_col: str) -> int:
    """Rows of `src` that never made it into `paired`."""
    if paired.empty:
        return int(len(src))
    used = (paired[["country_code", "target_ts", gen_col]]
            .drop_duplicates()
            .rename(columns={gen_col: "generated_at"}))
    m = src.merge(used, on=["country_code", "target_ts", "generated_at"],
                  how="left", indicator=True)
    return int((m["_merge"] == "left_only").sum())


def pair(a: pd.DataFrame, b: pd.DataFrame, actuals: pd.DataFrame,
         max_run_skew: pd.Timedelta = MAX_RUN_SKEW
         ) -> tuple[pd.DataFrame, PairScope]:
    """Join two forecast series to each other, per run, and to actuals.

    Each frame needs `country_code`, `target_ts`, `generated_at`,
    `forecast_value`; `actuals` needs `country_code`, `target_ts`, `actual`.
    Rows missing a forecast from either model, or missing an actual, are
    dropped — an unmeasurable hour scores neither model.

    Rows are matched on `(country_code, target_ts, run_as_of)`. Where one cutoff
    holds more than one vintage per side — a champion re-run, as on 2026-08-06 —
    the pair closest in time wins and the rest fall to `n_only_*`, so a re-run
    duplicates nothing and does not silently replace a matched pair with a
    distant one.
    """
    a = _with_run_as_of(a)
    b = _with_run_as_of(b)
    ar = a.rename(columns={"forecast_value": "forecast_a",
                           "generated_at": "generated_at_a"})
    br = b.rename(columns={"forecast_value": "forecast_b",
                           "generated_at": "generated_at_b"})

    cand = ar.merge(br, on=JOIN_KEYS, how="inner")
    cand["vintage_skew"] = (cand["generated_at_b"] - cand["generated_at_a"]).abs()

    near = cand[cand["vintage_skew"] <= max_run_skew]
    n_rejected = int(len(cand[JOIN_KEYS].drop_duplicates())
                     - len(near[JOIN_KEYS].drop_duplicates()))

    # Closest pair per (country, hour, run). Ties break toward the newer
    # champion vintage so the reduction is deterministic across runs.
    joined = (near.sort_values(["vintage_skew", "generated_at_a"],
                               ascending=[True, False], kind="stable")
                  .drop_duplicates(JOIN_KEYS, keep="first"))

    joined = joined.merge(actuals, on=["country_code", "target_ts"], how="inner")
    paired = joined.dropna(subset=["forecast_a", "forecast_b", "actual"])

    scope = PairScope(
        n_only_a=_one_sided(a, paired, "generated_at_a"),
        n_only_b=_one_sided(b, paired, "generated_at_b"),
        n_rejected_skew=n_rejected,
        max_skew_seconds=(float(paired["vintage_skew"].max().total_seconds())
                          if len(paired) else None))
    return paired, scope


def compare(paired: pd.DataFrame, model_a: str, model_b: str,
            scope: PairScope | None = None) -> HeadToHead:
    """Pooled and per-country MAE for two models over identical rows.

    Returns `None` pooled metrics and no countries when `paired` is empty. Not
    `0.0`: a zeroed MAE renders as **0.0 MW**, which reads as a flawless
    forecast rather than as an unmeasured one, and that is how the ABL-82 join
    defect survived — it printed a full report of zeros instead of failing.
    """
    scope = scope or PairScope()
    if paired.empty:
        return HeadToHead(model_a, model_b, 0, scope.n_only_a, scope.n_only_b, 0,
                          None, None, None, None, [],
                          scope.n_rejected_skew, scope.max_skew_seconds)

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
        n_only_a=scope.n_only_a, n_only_b=scope.n_only_b,
        n_vintages=int(paired["run_as_of"].nunique()),
        pooled_mae_a_mw=mae_a, pooled_mae_b_mw=mae_b,
        pooled_delta_mw=mae_b - mae_a,
        pooled_delta_pct=(100.0 * (mae_b - mae_a) / mae_a) if mae_a else float("nan"),
        countries=rows,
        n_rejected_skew=scope.n_rejected_skew,
        max_skew_seconds=scope.max_skew_seconds)


def _skew_note(h: HeadToHead) -> str:
    bits = []
    if h.max_skew_seconds is not None:
        bits.append(f"widest paired vintage gap **{h.max_skew_seconds:,.1f} s**")
    if h.n_rejected_skew:
        bits.append(f"**{h.n_rejected_skew:,}** hours rejected as different runs "
                    f"(vintages more than {MAX_RUN_SKEW} apart)")
    return " · ".join(bits)


def render_markdown(h: HeadToHead, window: str, generated_at: str) -> str:
    """Report that leads with the scope of the comparison, not the winner."""
    head = [
        f"# Head-to-head: `{h.model_b}` vs `{h.model_a}`", "",
        f"**Generated:** {generated_at} · **Window:** {window}",
        f"**Paired rows:** {h.n_paired:,} over {h.n_vintages} runs · "
        f"rows only in `{h.model_a}`: {h.n_only_a:,} · "
        f"only in `{h.model_b}`: {h.n_only_b:,}", "",
        "Scored only on `(country, target hour, run)` rows where **both** "
        "models produced a value and an actual exists. A run is the actuals "
        "cutoff a vintage could see, not its `generated_at` — the champion and "
        "the challengers are separate processes and never share a stamp "
        "(ABL-82). Per-model reports are *not* comparable to each other — they "
        "cover different vintage sets.", "",
    ]
    note = _skew_note(h)
    if note:
        head += [note, ""]

    if not h.measured:
        # No number at all, rather than a zero that reads as a measurement.
        return "\n".join(head + [
            "## Not measured", "",
            "**No row was paired, so there is no comparison to report.** This is "
            "not a tie and not a 0 MW error — the two models were never scored "
            "on a common hour. Check, in order: whether actuals have arrived "
            "for the target hours at all (a D+2 vintage has none until the day "
            "lands, which is the ordinary reason a fresh shadow run is not yet "
            "measurable); whether both models have rows in the window; and "
            "whether their vintages saw the same actuals — a challenger "
            "backfilled hours after the champion is a different run by design, "
            "and shows up in the one-sided counts above.", ""]) + "\n"

    better = "better" if h.pooled_delta_mw < 0 else "worse"
    lines = head + [
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
