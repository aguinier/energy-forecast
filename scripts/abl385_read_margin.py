"""ABL-385: read the registered variance sweep and derive the decision margin.

This script never fits. It reads the JSON that `scripts/abl385_run_sweep.py`
wrote and computes every number that reaches the verdict, so the analysis can be
re-run and reviewed without touching a model. Every quantity it produces is
defined in `experiments/ABL385/config.json`, frozen before the first fit.

What it computes, in the order the registration defines them:

1. **Per-cell spread.** One cell is a (country, type, algorithm, arm, window)
   fitted at the 12 registered seeds. Mean, sample sd (ddof=1) and the seed
   spread as a coefficient of variation CV = s / mean. Solar is read on daylight
   MAE, every other type on all-hours MAE.

2. **Window variance against seed variance** (scope item 3). The six windows sit
   at very different levels - solar MAE in February is not solar MAE in July -
   so a raw pooled sd would be almost entirely seasonal. The decomposition is
   done on log MAE, where a between-window sd and a within-window (seed) sd are
   both in relative units and directly comparable. For a small spread,
   sd(log MAE) and CV agree to first order.

3. **Pooling.** Per (pair, algorithm, arm), the 6 window CVs pooled as a root
   mean square - variances add - plus the maximum across windows as the
   conservative value.

4. **The fleet margin.** delta_min(k) = z * sqrt(c_A^2 + c_B^2) / sqrt(k), the
   delta-method variance of a relative gap between two k-seed means. Reported at
   the 80th and 90th percentile of the pooled CV distribution, separately for
   solar and for wind, and per (pair, algorithm) where a pair-specific value
   exists.

5. **The independence assumption, measured rather than assumed.** The margin
   above treats the two arms' CVs as independent. That is exact across
   algorithms, whose RNG draws are unrelated by construction, but two arms of
   the same algorithm at matched seeds may share a draw. The solar
   control-vs-geometry cells were registered to measure that correlation, and
   the correlation-adjusted margin is reported next to the independent one.

6. **The three pre-specified predictions**, each against the refutation
   condition registered for it.

7. **The ABL-375 re-read** (scope item 5): what its 4.5% gap at k = 3 is worth
   under the margin registered here, and how many seeds it would have needed.
   It does not touch ABL-375's verdict on whether DE solar should move.

Usage
-----
    .venv\\Scripts\\python.exe scripts/abl385_read_margin.py \\
        --sweep reports/abl_385_sweep --out reports/abl_385_decision_margin
"""

import argparse
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
REGISTRATION = REPO_ROOT / "experiments" / "ABL385" / "config.json"

#: Two-sided 95%. Registered; not a knob.
Z = 1.96

#: Hartley's d2, the constant relating the expected range of n normal draws to
#: sigma. ABL-375 quoted max-minus-min over 3 seeds, and a range is only a
#: consistent estimator of sigma through this constant, so a 3-seed range and a
#: 12-seed range are not the same statistic. Used to convert ABL-375's published
#: numbers onto this issue's CV scale rather than comparing them directly.
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 10: 3.078, 12: 3.258}

#: Types whose renewable output has a band structure. Only solar: the bands
#: exist because the incumbent emitted garbage at night, and no other type has a
#: structurally-zero band to protect.
BANDED_TYPES = ("solar",)

KNOWN_TYPES = ("solar", "wind_onshore", "wind_offshore", "biomass", "hydro_total")
ALGORITHMS = ("catboost", "xgboost")

#: Which stream a type belongs to for the fleet percentile. Registered as
#: "separately for solar and for wind"; biomass and hydro_total are the
#: never-gated secondary and are reported but kept out of both streams, so a
#: pair that no gate has ever read cannot move the number a gate cites.
STREAM = {
    "solar": "solar",
    "wind_onshore": "wind",
    "wind_offshore": "wind",
    "biomass": "other",
    "hydro_total": "other",
}


def _mean(values):
    return sum(values) / len(values)


def _sd(values):
    """Sample sd, ddof=1. Registered."""
    if len(values) < 2:
        return float("nan")
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _percentile(values, pct):
    """Linear-interpolated percentile, numpy's default method.

    Written out rather than imported so this reader has no numpy dependency and
    the arithmetic behind a number a gate will cite is visible in one place.
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _fisher_mean(rhos):
    """Average correlations through Fisher's z, not arithmetically.

    Correlations are not additive; averaging them directly biases toward zero.
    """
    usable = [r for r in rhos if not math.isnan(r) and abs(r) < 0.9999]
    if not usable:
        return float("nan")
    z = _mean([0.5 * math.log((1 + r) / (1 - r)) for r in usable])
    return math.tanh(z)


def parse_filename(path):
    """Recover (tag, forecast_type, algorithm) from a sweep output name.

    The driver names its outputs `holdout_{tag}[_{type}]_{algorithm}[_cleaned]`.
    The tag has to come from the filename rather than the payload: the W6
    primary invocation and the `abl-2023-01-01` ablation point are the identical
    invocation - same countries, same fit start, same holdout - so their
    payloads are indistinguishable. That duplication is deliberate on the
    ablation's side and is used below as a reproducibility check.
    """
    stem = path.stem
    if not stem.startswith("holdout_"):
        return None
    stem = stem[len("holdout_"):]
    if stem.endswith("_cleaned"):
        stem = stem[: -len("_cleaned")]
    algorithm = None
    for candidate in ALGORITHMS:
        if stem.endswith("_" + candidate):
            algorithm = candidate
            stem = stem[: -(len(candidate) + 1)]
            break
    if algorithm is None:
        return None
    forecast_type = "solar"
    for candidate in KNOWN_TYPES:
        if candidate != "solar" and stem.endswith("_" + candidate):
            forecast_type = candidate
            stem = stem[: -(len(candidate) + 1)]
            break
    return stem, forecast_type, algorithm


def load_cells(sweep_dir):
    """Every (country, type, algorithm, arm, tag) cell -> {seed: metric_mw}.

    Also returns the per-cell context a metric is meaningless without: holdout
    bounds, n_holdout, n_train and the fit start.
    """
    cells = defaultdict(dict)
    context = {}
    files = sorted(Path(sweep_dir).glob("holdout_*.json"))
    for path in files:
        parsed = parse_filename(path)
        if parsed is None:
            continue
        tag, forecast_type, algorithm = parsed
        payload = json.loads(path.read_text(encoding="utf-8"))
        banded = forecast_type in BANDED_TYPES
        for country, result in payload["countries"].items():
            for arm_key, arm in result["arms"].items():
                if "@" not in arm_key:
                    # A run without --seeds. Not part of this registration.
                    continue
                arm_name, seed_text = arm_key.rsplit("@", 1)
                metric = arm["daylight"]["mae_mw"] if banded else arm["all"]["mae_mw"]
                key = (country, forecast_type, algorithm, arm_name, tag)
                cells[key][int(seed_text)] = float(metric)
                context[key] = {
                    "holdout_start": payload["holdout_start"],
                    "holdout_end": payload["holdout_end"],
                    "fit_start": payload["start_date"],
                    "n_holdout": result["n_holdout"],
                    "n_train": result["n_train"],
                    "metric": "daylight_mae_mw" if banded else "all_hours_mae_mw",
                    "source_file": path.name,
                }
    return cells, context, [p.name for p in files]


def cell_stats(cells, context):
    """Mean, sd and CV per cell. The registered per-cell estimator."""
    stats = {}
    for key, by_seed in cells.items():
        values = [by_seed[s] for s in sorted(by_seed)]
        mean = _mean(values)
        sd = _sd(values)
        stats[key] = {
            "n_seeds": len(values),
            "seeds": sorted(by_seed),
            "mean_mw": mean,
            "sd_mw": sd,
            "cv": sd / mean if mean else float("nan"),
            "min_mw": min(values),
            "max_mw": max(values),
            "range_pct_of_mean": (max(values) - min(values)) / mean * 100 if mean else float("nan"),
            **context[key],
        }
    return stats


def window_tags(registration):
    return [w["window"] for w in registration["scope"]["windows"]]


def pool_across_windows(stats, cells, windows):
    """Per (pair, algorithm, arm): RMS and max of the window CVs.

    Only the six registered rolling-origin windows are pooled. The ablation tags
    share W6's holdout and would otherwise enter the pool four extra times and
    weight one season.
    """
    grouped = defaultdict(dict)
    for (country, ftype, algorithm, arm, tag), s in stats.items():
        if tag in windows:
            grouped[(country, ftype, algorithm, arm)][tag] = s
    pooled = {}
    for key, by_window in grouped.items():
        country, ftype, algorithm, arm = key
        cvs = [by_window[t]["cv"] for t in sorted(by_window)]
        logs = {
            t: [math.log(v) for v in cells[(country, ftype, algorithm, arm, t)].values()]
            for t in by_window
        }
        pooled[key] = {
            "n_windows": len(cvs),
            "windows": sorted(by_window),
            "cv_by_window": {t: by_window[t]["cv"] for t in sorted(by_window)},
            "cv_rms": math.sqrt(_mean([c ** 2 for c in cvs])),
            "cv_max": max(cvs),
            "cv_min": min(cvs),
            "mean_mw_by_window": {t: by_window[t]["mean_mw"] for t in sorted(by_window)},
            "n_holdout_total": sum(by_window[t]["n_holdout"] for t in by_window),
            **_variance_split(logs),
        }
    return pooled


def _variance_split(logs_by_window):
    """Separate window variance from seed variance, on log MAE.

    Scope item 3 is "so window variance and seed variance are separable". They
    are not separable on the raw MW scale: the six windows differ by season, and
    a pooled sd there is dominated by the level shift. On log MAE both
    components are relative, so sd_seed_log is directly comparable to a CV and
    sd_window_log says how much of the total spread is the choice of window
    rather than the choice of seed.
    """
    if not logs_by_window:
        return {}
    per_window_means = [_mean(v) for v in logs_by_window.values()]
    within = []
    for values in logs_by_window.values():
        m = _mean(values)
        within.extend((v - m) ** 2 for v in values)
    dof = sum(len(v) for v in logs_by_window.values()) - len(logs_by_window)
    sd_seed = math.sqrt(sum(within) / dof) if dof > 0 else float("nan")
    sd_window = _sd(per_window_means) if len(per_window_means) > 1 else float("nan")
    total = math.sqrt(sd_seed ** 2 + sd_window ** 2)
    return {
        "sd_seed_log": sd_seed,
        "sd_window_log": sd_window,
        "seed_share_of_variance": (sd_seed ** 2 / total ** 2) if total else float("nan"),
    }


def delta_min(c_a, c_b, k, rho=0.0):
    """The registered minimum decision margin.

    Two arms each reported as the mean of k independent seeds. For the relative
    gap g = (M_A - M_B) / M_B the delta method gives
    Var(g) ~ (c_A^2 + c_B^2 - 2 rho c_A c_B) / k, so a gap is readable at 95%
    only if it exceeds z * sqrt(that). rho defaults to 0, the registered
    independent case: exact across algorithms, conservative for two arms of one
    algorithm if they are positively correlated.
    """
    var = c_a ** 2 + c_b ** 2 - 2 * rho * c_a * c_b
    return Z * math.sqrt(max(var, 0.0)) / math.sqrt(k)


def seeds_needed(c_a, c_b, target, rho=0.0):
    """Inverse of the above: seeds required to read a target relative gap."""
    if target <= 0:
        return float("inf")
    var = c_a ** 2 + c_b ** 2 - 2 * rho * c_a * c_b
    return Z ** 2 * max(var, 0.0) / target ** 2


def range_to_cv(range_fraction, n):
    """Convert a max-minus-min range over n draws to a CV.

    ABL-375, ABL-338 and ABL-253 all quote a range. A range over 3 seeds and a
    range over 12 are different statistics, so its published numbers are put on
    this scale before being compared to anything here.
    """
    return range_fraction / D2[n]


def arm_correlation(cells, windows):
    """Correlation between the solar arms at matched seeds.

    The margin's independence assumption is exact across algorithms and merely
    conservative within one, and the registration says to measure that rather
    than assume it. Per (country, algorithm, window), the Pearson correlation
    across the 12 matched seeds between the control and geometry MAE; pooled
    through Fisher's z.
    """
    per_cell = {}
    for (country, ftype, algorithm, arm, tag), by_seed in cells.items():
        if ftype != "solar" or arm != "control" or tag not in windows:
            continue
        other = cells.get((country, ftype, algorithm, "geometry", tag))
        if not other:
            continue
        shared = sorted(set(by_seed) & set(other))
        if len(shared) < 3:
            continue
        per_cell[f"{country}/{algorithm}/{tag}"] = _pearson(
            [by_seed[s] for s in shared], [other[s] for s in shared]
        )
    by_pair = defaultdict(list)
    for label, rho in per_cell.items():
        country, algorithm, _ = label.split("/")
        by_pair[f"{country}/{algorithm}"].append(rho)
    return {
        "per_cell": per_cell,
        "per_pair_fisher_mean": {k: _fisher_mean(v) for k, v in by_pair.items()},
        "fleet_fisher_mean": _fisher_mean(list(per_cell.values())),
        "interpretation": (
            "Positive correlation means the independent margin is conservative: "
            "matched seeds move the two arms together and part of the noise "
            "cancels out of the gap. Near-zero means the independent margin is "
            "about right even within one algorithm."
        ),
    }


def fleet_margin(pooled):
    """Per-stream CV percentiles and the margin table they imply."""
    streams = defaultdict(list)
    for (country, ftype, algorithm, arm), p in pooled.items():
        stream = STREAM[ftype]
        streams[stream].append({
            "unit": f"{country}/{ftype}/{algorithm}/{arm}",
            "cv_rms": p["cv_rms"],
            "cv_max": p["cv_max"],
        })
    out = {}
    for stream, units in streams.items():
        rms = [u["cv_rms"] for u in units]
        p80 = _percentile(rms, 80)
        p90 = _percentile(rms, 90)
        out[stream] = {
            "n_units": len(units),
            "units": sorted(units, key=lambda u: -u["cv_rms"]),
            "cv_rms_median": _percentile(rms, 50),
            "cv_rms_p80": p80,
            "cv_rms_p90": p90,
            "cv_rms_max": max(rms),
            # The headline. Two arms of comparable spread, which is the case a
            # gate read actually faces: challenger against incumbent, or one
            # algorithm against another, both drawn from this stream.
            "delta_min_pct_at_p90": {
                str(k): 100 * delta_min(p90, p90, k) for k in (1, 3, 5, 10, 12, 20)
            },
            "delta_min_pct_at_p80": {
                str(k): 100 * delta_min(p80, p80, k) for k in (1, 3, 5, 10, 12, 20)
            },
            "seeds_for_a_5pct_gap_at_p90": seeds_needed(p90, p90, 0.05),
            "seeds_for_a_10pct_gap_at_p90": seeds_needed(p90, p90, 0.10),
        }
    return out


def pair_margins(pooled):
    """delta_min per (pair, algorithm) using that pair's own measured CV.

    A pair-specific margin is the one to cite when it exists; the fleet
    percentile is for a pair this sweep did not measure.
    """
    out = {}
    for (country, ftype, algorithm, arm), p in pooled.items():
        out[f"{country}/{ftype}/{algorithm}/{arm}"] = {
            "cv_rms": p["cv_rms"],
            "cv_max": p["cv_max"],
            "delta_min_pct": {
                str(k): 100 * delta_min(p["cv_rms"], p["cv_rms"], k)
                for k in (1, 3, 5, 10, 12, 20)
            },
        }
    return out


def evaluate_predictions(stats, pooled, registration, windows):
    """The three pre-specified predictions, each against its refutation rule."""
    results = []

    # --- P1: DE CatBoost solar CV exceeds DE XGBoost solar CV on a majority of
    # the six windows. Read on the geometry arm, which is the arm ABL-375's
    # registered comparison used.
    per_window = {}
    for tag in windows:
        cb = stats.get(("DE", "solar", "catboost", "geometry", tag))
        xgb = stats.get(("DE", "solar", "xgboost", "geometry", tag))
        if cb and xgb:
            per_window[tag] = {
                "catboost_cv": cb["cv"],
                "xgboost_cv": xgb["cv"],
                "catboost_higher": cb["cv"] > xgb["cv"],
            }
    higher = sum(1 for v in per_window.values() if v["catboost_higher"])
    results.append({
        "id": "P1",
        "statement": registration["pre_specified_predictions"][0]["statement"],
        "per_window": per_window,
        "windows_with_catboost_higher": higher,
        "n_windows": len(per_window),
        "refuted": (len(per_window) - higher) >= 4,
        "verdict": _p1_verdict(higher, len(per_window)),
    })

    # --- P2: does CatBoost solar CV fall as the fit lengthens? Read on BE and
    # FR, where the fit-length span is widest.
    ablation = registration["scope"]["fit_length_ablation"]
    longest, shortest = ablation["fit_starts"][0], ablation["fit_starts"][-1]
    p2 = {}
    for country in ("BE", "FR"):
        long_cell = stats.get((country, "solar", "catboost", "geometry", f"abl-{longest}"))
        short_cell = stats.get((country, "solar", "catboost", "geometry", f"abl-{shortest}"))
        if long_cell and short_cell:
            ratio = long_cell["cv"] / short_cell["cv"] if short_cell["cv"] else float("nan")
            p2[country] = {
                "cv_at_longest_fit": long_cell["cv"],
                "n_train_longest": long_cell["n_train"],
                "cv_at_shortest_fit": short_cell["cv"],
                "n_train_shortest": short_cell["n_train"],
                "ratio_long_over_short": ratio,
                "halved": ratio < 0.5,
                "within_25pct": abs(ratio - 1.0) <= 0.25,
            }
    results.append({
        "id": "P2",
        "statement": registration["pre_specified_predictions"][1]["statement"],
        "per_country": p2,
        "short_fit_artefact": bool(p2) and all(v["halved"] for v in p2.values()),
        "refuted": bool(p2) and all(v["within_25pct"] for v in p2.values()),
        "verdict": _p2_verdict(p2),
    })

    # --- P3: at a matched short fit, is DE's CatBoost solar CV a factor of two
    # above the other three?
    p3 = {}
    for country in ("AT", "BE", "DE", "FR"):
        cell = stats.get((country, "solar", "catboost", "geometry", f"abl-{shortest}"))
        if cell:
            p3[country] = {"cv": cell["cv"], "n_train": cell["n_train"],
                           "mean_mw": cell["mean_mw"]}
    others = [v["cv"] for c, v in p3.items() if c != "DE"]
    de_cv = p3.get("DE", {}).get("cv")
    results.append({
        "id": "P3",
        "statement": registration["pre_specified_predictions"][2]["statement"],
        "matched_fit_start": shortest,
        "per_country": p3,
        "de_cv": de_cv,
        "median_other_cv": _percentile(others, 50) if others else None,
        "de_is_2x_the_others": (
            bool(others) and de_cv is not None
            and de_cv >= 2 * _percentile(others, 50)
        ),
        "verdict": _p3_verdict(de_cv, others),
    })
    return results


def _p1_verdict(higher, n):
    if n == 0:
        return "NOT READ - no DE geometry cells in the sweep"
    if (n - higher) >= 4:
        return f"REFUTED - XGBoost CV is at least CatBoost's on {n - higher} of {n} windows"
    if higher > n / 2:
        return f"HELD - CatBoost CV is higher on {higher} of {n} windows"
    return f"AMBIGUOUS - CatBoost higher on {higher} of {n}, short of a majority and short of the refutation rule"


def _p2_verdict(p2):
    if not p2:
        return "NOT READ - the ablation cells are missing"
    if all(v["halved"] for v in p2.values()):
        return "SHORT-FIT ARTEFACT - CV at the longest fit is below half its shortest-fit value on both BE and FR"
    if all(v["within_25pct"] for v in p2.values()):
        return "REFUTED - CV is within 25% of its shortest-fit value at the longest fit on both BE and FR, so the spread is not a fit-length effect"
    return "AMBIGUOUS - the two countries disagree, or the movement falls between the registered thresholds"


def _p3_verdict(de_cv, others):
    if de_cv is None or not others:
        return "NOT READ - the matched-fit cells are missing"
    median = _percentile(others, 50)
    if median == 0:
        return "NOT READ - a zero comparator CV"
    ratio = de_cv / median
    if ratio >= 2:
        return f"HELD - DE's CV is {ratio:.2f}x the median of the other three at a matched fit, so the instability is DE-specific rather than a fit-length effect"
    return f"NOT HELD - DE's CV is {ratio:.2f}x the median of the other three at a matched fit, short of the registered factor of 2"


def reread_abl375(pooled, stats):
    """Scope item 5: what ABL-375's 4.5% gap at k = 3 was worth.

    Supplies the error bar only. Whether DE solar should move to XGBoost is
    ABL-375's registered question and the CEO's decision, and nothing here
    answers it.
    """
    cb = pooled.get(("DE", "solar", "catboost", "geometry"))
    xgb = pooled.get(("DE", "solar", "xgboost", "geometry"))
    if not cb or not xgb:
        return {"status": "NOT READ - the DE geometry cells are missing"}
    c_a, c_b = cb["cv_rms"], xgb["cv_rms"]
    observed_gap = 0.045
    k = 3
    margin = delta_min(c_a, c_b, k)
    return {
        "abl375_holdout": "2026-04-30 .. 2026-06-12",
        "abl375_k": k,
        "abl375_observed_relative_gap": observed_gap,
        "de_catboost_geometry_cv_rms": c_a,
        "de_xgboost_geometry_cv_rms": c_b,
        "delta_min_at_k3_pct": 100 * margin,
        "gap_exceeds_margin": observed_gap > margin,
        "seeds_needed_for_that_gap": seeds_needed(c_a, c_b, observed_gap),
        "disposition_check": (
            "AMBIGUOUS was correct" if observed_gap <= margin
            else "the gap does clear the margin registered here"
        ),
        "abl375_published_ranges_converted": {
            "catboost_13.79pct_range_over_3_seeds_as_cv": range_to_cv(0.1379, 3),
            "catboost_6.98pct_range_over_3_seeds_as_cv": range_to_cv(0.0698, 3),
            "catboost_4.62pct_range_over_3_seeds_as_cv": range_to_cv(0.0462, 3),
            "xgboost_4.73pct_range_over_3_seeds_as_cv": range_to_cv(0.0473, 3),
            "note": (
                "ABL-375 quoted max-minus-min over 3 seeds. Divided by d2(3) = "
                "1.693 to reach a sigma-scale CV comparable to this issue's "
                "12-seed numbers."
            ),
        },
        "what_this_does_not_say": (
            "Nothing here is a verdict on whether DE solar should move to "
            "XGBoost. This supplies the error bar; the call is ABL-375's "
            "registered question and the CEO's decision."
        ),
    }


def duplicate_check(stats):
    """W6 and the abl-2023-01-01 ablation point are the identical invocation.

    Same countries, same fit start, same holdout, same seeds - run twice by
    construction. If the two disagree, something is non-deterministic that the
    whole CV estimate assumes is not, and every number in this report is
    suspect. Cheap, and it fails loudly.
    """
    checks = {}
    for (country, ftype, algorithm, arm, tag), s in stats.items():
        if tag != "W6" or ftype != "solar":
            continue
        twin = stats.get((country, ftype, algorithm, arm, "abl-2023-01-01"))
        if not twin:
            continue
        label = f"{country}/{algorithm}/{arm}"
        checks[label] = {
            "w6_mean_mw": s["mean_mw"],
            "ablation_twin_mean_mw": twin["mean_mw"],
            "abs_diff_mw": abs(s["mean_mw"] - twin["mean_mw"]),
            "identical": abs(s["mean_mw"] - twin["mean_mw"]) < 1e-6,
        }
    return {
        "checks": checks,
        "all_identical": bool(checks) and all(c["identical"] for c in checks.values()),
        "why": (
            "The W6 primary invocation and the abl-2023-01-01 ablation point "
            "carry identical arguments, so they are a free determinism check on "
            "the whole sweep."
        ),
    }


def render_markdown(payload):
    """The evidence pack. Every number carries its window, n and baseline."""
    reg = payload["registration"]
    prov = payload["registration_provenance"]
    lines = [
        "# ABL-385 - seed variance across the served renewable pairs, and a registered decision margin",
        "",
        f"Generated {payload['generated_at']}. Every number below comes from the "
        f"sweep the frozen registration defines.",
        "",
        "**Pre-registration provenance** (scope item 1, checked rather than asserted):",
        "",
        f"- Registration commit `{prov['commit'][:12] if prov['commit'] else '?'}`, "
        f"committed `{prov['committed_at']}`.",
        f"- Earliest fit in this sweep: `{prov['earliest_sweep_output']}` at "
        f"`{prov['earliest_sweep_output_at']}`.",
        f"- **{prov['ordering']}**",
        f"- Working tree: {prov['working_tree_note']}.",
        "",
        f"Replica `{payload['replica_db']}`, read-only. Interpreter: the rail "
        f"(`.venv`, Python 3.14.3, xgboost 3.3.0).",
        "",
        "## What was measured",
        "",
        f"- **{payload['n_cells']} cells**, each a (country, type, algorithm, arm, window) "
        f"fitted at the **{reg['scope']['n_seeds']} registered seeds** - "
        f"{payload['n_fits']} fits in total.",
        f"- **{payload['n_pairs']} served pairs** of the 14 on disk, over "
        f"**{payload['n_windows_read']} of the {len(reg['scope']['windows'])} registered "
        f"contiguous non-overlapping 30-day rolling-origin windows** "
        f"({reg['scope']['windows'][0]['start']} .. {reg['scope']['windows'][-1]['end']}). "
        "No holdout row is scored twice.",
        "- Solar is read on **daylight MAE**; every other type on **all-hours MAE**. "
        "Night is reported in MW and never as a percentage - its denominator is ~0.",
        "- Every arm is a **refit** on the identically truncated window. The live "
        "artifacts are fitted through roughly today, so scoring them on a recent "
        "holdout would be in-sample and would flatter the incumbent.",
        "",
        "**All numbers are out-of-sample** with respect to the fit frame, which ends "
        "strictly before each holdout starts.",
        "",
    ]

    dup = payload["determinism_check"]
    if dup["checks"]:
        state = "PASS" if dup["all_identical"] else "**FAIL**"
        worst = max(c["abs_diff_mw"] for c in dup["checks"].values())
        lines += [
            f"**Determinism check: {state}.** The W6 primary invocation and the "
            f"`abl-2023-01-01` ablation point carry identical arguments and were run "
            f"twice; {len(dup['checks'])} arms compared, largest disagreement "
            f"{worst:.3g} MW. The CV estimate assumes the only thing moving is the "
            f"seed, and this is what checks that.",
            "",
        ]

    lines += ["## 1. The headline - the registered decision margin", ""]
    lines += [
        "For two arms A and B scored on the same holdout, each reported as the mean "
        "of k seeds, the delta method gives Var(g) ~ (c_A^2 + c_B^2) / k for the "
        "relative gap g. So a gap is readable at two-sided 95% only if",
        "",
        "```",
        "delta_min(k) = 1.96 * sqrt(c_A^2 + c_B^2) / sqrt(k)",
        "```",
        "",
        "with c the per-fit CV of each arm. **This is the number a future "
        "registration cites instead of a remembered noise floor.**",
        "",
    ]
    for stream in ("solar", "wind", "other"):
        fleet = payload["fleet_margin"].get(stream)
        if not fleet:
            continue
        label = {"solar": "Solar", "wind": "Wind",
                 "other": "Biomass / hydro (served, never gated)"}[stream]
        lines += [
            f"### {label} - {fleet['n_units']} (pair, algorithm, arm) units",
            "",
            f"Pooled per-fit CV: median {100 * fleet['cv_rms_median']:.2f}%, "
            f"p80 {100 * fleet['cv_rms_p80']:.2f}%, "
            f"**p90 {100 * fleet['cv_rms_p90']:.2f}%**, "
            f"max {100 * fleet['cv_rms_max']:.2f}%.",
            "",
            "| seeds k | delta_min at p90 | delta_min at p80 |",
            "|---:|---:|---:|",
        ]
        for k in ("1", "3", "5", "10", "12", "20"):
            lines.append(
                f"| {k} | {fleet['delta_min_pct_at_p90'][k]:.2f}% | "
                f"{fleet['delta_min_pct_at_p80'][k]:.2f}% |"
            )
        lines += [
            "",
            f"To read a 5% gap on this stream takes "
            f"**{math.ceil(fleet['seeds_for_a_5pct_gap_at_p90']):d} seeds**; "
            f"a 10% gap takes "
            f"**{math.ceil(fleet['seeds_for_a_10pct_gap_at_p90']):d}**.",
            "",
        ]

    lines += ["## 2. Per-pair spread", "",
              "The pair-specific CV is the one to cite when it exists; the fleet "
              "percentile above is for a pair this sweep did not measure.",
              "",
              "| pair / algorithm / arm | CV (RMS over 6 windows) | CV (worst window) | "
              "delta_min at k=1 | at k=3 | at k=10 |",
              "|---|---:|---:|---:|---:|---:|"]
    for label in sorted(payload["pair_margins"],
                        key=lambda x: -payload["pair_margins"][x]["cv_rms"]):
        m = payload["pair_margins"][label]
        lines.append(
            f"| {label} | {100 * m['cv_rms']:.2f}% | {100 * m['cv_max']:.2f}% | "
            f"{m['delta_min_pct']['1']:.1f}% | {m['delta_min_pct']['3']:.1f}% | "
            f"{m['delta_min_pct']['10']:.1f}% |"
        )
    lines.append("")

    lines += ["## 3. Window variance against seed variance", "",
              "Scope item 3. The six windows sit at very different levels - solar MAE "
              "in February is not solar MAE in July - so the split is done on log MAE, "
              "where both components are relative and comparable. `sd_seed` is the "
              "spread from reseeding within one window; `sd_window` is the spread of "
              "the window means.",
              "",
              "| pair / algorithm / arm | sd_seed (log) | sd_window (log) | "
              "seed share of variance |",
              "|---|---:|---:|---:|"]
    ordered = sorted(payload["pooled"],
                     key=lambda x: -payload["pooled"][x].get("sd_seed_log", 0))
    for label in ordered:
        p = payload["pooled"][label]
        if "sd_seed_log" not in p:
            continue
        lines.append(
            f"| {label} | {p['sd_seed_log']:.4f} | {p['sd_window_log']:.4f} | "
            f"{100 * p['seed_share_of_variance']:.1f}% |"
        )
    lines.append("")

    corr = payload["arm_correlation"]
    lines += [
        "## 4. The independence assumption, measured",
        "",
        "delta_min treats c_A and c_B as independent. That is exact for two "
        "different algorithms, whose RNG draws are unrelated by construction. For "
        "two arms of the *same* algorithm at matched seeds it had to be measured, "
        "and the solar control-vs-geometry cells are what measures it.",
        "",
        f"Fleet correlation across {len(corr['per_cell'])} matched cells "
        f"(Fisher-z mean): **{corr['fleet_fisher_mean']:.3f}**.",
        "",
    ]
    if corr["per_pair_fisher_mean"]:
        lines += ["| pair / algorithm | correlation at matched seeds |", "|---|---:|"]
        for label in sorted(corr["per_pair_fisher_mean"]):
            lines.append(f"| {label} | {corr['per_pair_fisher_mean'][label]:.3f} |")
        lines.append("")
    lines += [corr["interpretation"], ""]

    lines += ["## 5. The three pre-specified predictions", ""]
    for pred in payload["predictions"]:
        lines += [f"### {pred['id']} - {pred['verdict'].split(' - ')[0]}", "",
                  f"*Registered statement:* {pred['statement']}", "",
                  pred["verdict"], ""]
        if pred["id"] == "P1" and pred["per_window"]:
            lines += ["| window | CatBoost CV | XGBoost CV | CatBoost higher |",
                      "|---|---:|---:|:--:|"]
            for tag in sorted(pred["per_window"]):
                v = pred["per_window"][tag]
                lines.append(
                    f"| {tag} | {100 * v['catboost_cv']:.2f}% | "
                    f"{100 * v['xgboost_cv']:.2f}% | "
                    f"{'yes' if v['catboost_higher'] else 'no'} |"
                )
            lines.append("")
        if pred["id"] == "P2" and pred["per_country"]:
            lines += ["| country | n_train longest | CV longest | n_train shortest | "
                      "CV shortest | ratio |", "|---|---:|---:|---:|---:|---:|"]
            for country in sorted(pred["per_country"]):
                v = pred["per_country"][country]
                lines.append(
                    f"| {country} | {v['n_train_longest']:,} | "
                    f"{100 * v['cv_at_longest_fit']:.2f}% | {v['n_train_shortest']:,} | "
                    f"{100 * v['cv_at_shortest_fit']:.2f}% | "
                    f"{v['ratio_long_over_short']:.2f} |"
                )
            lines.append("")
        if pred["id"] == "P3" and pred["per_country"]:
            lines += [f"At a matched fit start of {pred['matched_fit_start']}:", "",
                      "| country | n_train | CV | mean daylight MAE |",
                      "|---|---:|---:|---:|"]
            for country in sorted(pred["per_country"]):
                v = pred["per_country"][country]
                lines.append(
                    f"| {country} | {v['n_train']:,} | {100 * v['cv']:.2f}% | "
                    f"{v['mean_mw']:,.1f} MW |"
                )
            lines.append("")

    rr = payload["abl375_reread"]
    lines += ["## 6. Re-reading ABL-375's DE question under this margin", ""]
    if rr.get("status"):
        lines += [rr["status"], ""]
    else:
        lines += [
            f"ABL-375 read holdout {rr['abl375_holdout']} at k = {rr['abl375_k']} and "
            f"observed a {100 * rr['abl375_observed_relative_gap']:.1f}% relative gap "
            f"favouring XGBoost.",
            "",
            f"- DE CatBoost geometry CV (pooled, 6 windows, 12 seeds): "
            f"**{100 * rr['de_catboost_geometry_cv_rms']:.2f}%**",
            f"- DE XGBoost geometry CV: **{100 * rr['de_xgboost_geometry_cv_rms']:.2f}%**",
            f"- delta_min(k=3) = **{rr['delta_min_at_k3_pct']:.2f}%**",
            "",
            f"The observed gap "
            f"{'exceeds' if rr['gap_exceeds_margin'] else 'does not reach'} "
            f"that margin, so **{rr['disposition_check']}**. Reading a "
            f"{100 * rr['abl375_observed_relative_gap']:.1f}% gap at 95% would have "
            f"taken **{math.ceil(rr['seeds_needed_for_that_gap']):d} seeds**, not 3.",
            "",
            rr["what_this_does_not_say"],
            "",
        ]

    lines += ["## 7. Contamination", ""]
    for key, text in reg["contamination"].items():
        lines.append(f"- **{key}**: {text}")
    lines += ["", "## 8. Boundaries", ""]
    for text in reg["boundaries"]:
        lines.append(f"- {text}")
    lines.append("")
    return "\n".join(lines)


def _git(*argv):
    """A git read against the repo root. Returns None rather than raising."""
    try:
        done = subprocess.run(["git", "-C", str(REPO_ROOT), *argv],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def registration_provenance(sweep_dir, override=None):
    """Prove scope item 1 rather than asserting it in prose.

    The registration's first scope item is "registration committed before the
    first fit, git timestamp as the evidence". Prose cannot carry that claim -
    a reader has to be able to check it. So this reads the commit that froze
    `experiments/ABL385/config.json`, the mtime of the earliest sweep output,
    and compares them, and it reports whether the committed blob still matches
    what is on disk. A run that forgets to name the commit gets the real one
    instead of the string "(uncommitted)", and a run whose ordering is actually
    wrong says so in the pack rather than claiming the opposite.
    """
    rel = "experiments/ABL385/config.json"
    head = _git("log", "-1", "--format=%H%x09%cI", "--", rel)
    commit, committed_at = (head.split("\t", 1) if head and "\t" in head
                            else (None, None))
    dirty = _git("status", "--porcelain", "--", rel)

    fits = sorted(Path(sweep_dir).glob("holdout_*.json"),
                  key=lambda p: p.stat().st_mtime)
    first_fit_at = (datetime.fromtimestamp(fits[0].stat().st_mtime, timezone.utc)
                    if fits else None)

    verdict = "UNKNOWN - no registration commit found"
    if commit and first_fit_at is not None:
        frozen = datetime.fromisoformat(committed_at)
        if frozen < first_fit_at:
            delta = (first_fit_at - frozen).total_seconds() / 60.0
            verdict = (f"ORDERED - the registration was committed {delta:.1f} min "
                       f"before the earliest fit in this sweep ({fits[0].name})")
        else:
            verdict = ("VIOLATED - the earliest sweep output predates the "
                       "registration commit. This pack is not a pre-registered read.")
    return {
        "commit": override or commit or "(uncommitted)",
        "committed_at": committed_at,
        "registration_matches_commit": not dirty,
        "working_tree_note": (
            "clean - the config.json read here is byte-identical to the frozen commit"
            if not dirty else
            "MODIFIED since the freezing commit; the numbers below were computed "
            "from the working-tree copy, not the frozen one"
        ),
        "earliest_sweep_output": fits[0].name if fits else None,
        "earliest_sweep_output_at": (first_fit_at.isoformat() if first_fit_at else None),
        "ordering": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default="reports/abl_385_sweep")
    parser.add_argument("--out", default="reports/abl_385_decision_margin")
    parser.add_argument("--registration-commit", default=None,
                        help="Override the commit that froze "
                             "experiments/ABL385/config.json. Read from git when "
                             "omitted, which is the form that carries evidence.")
    args = parser.parse_args()

    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    windows = window_tags(registration)
    cells, context, files = load_cells(args.sweep)
    if not cells:
        raise SystemExit(f"no seeded cells found under {args.sweep}")
    stats = cell_stats(cells, context)
    pooled = pool_across_windows(stats, cells, windows)

    provenance = registration_provenance(args.sweep, args.registration_commit)

    payload = {
        "issue": "ABL-385",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "registration_commit": provenance["commit"],
        "registration_provenance": provenance,
        "replica_db": str(registration["protocol"]["replica"]).split(",")[0],
        "sweep_dir": str(args.sweep),
        "sweep_files": files,
        "n_cells": len(cells),
        "n_fits": sum(len(v) for v in cells.values()),
        "n_pairs": len({(c, t) for c, t, _, _, _ in cells}),
        # Measured, not assumed: a partial sweep must not describe itself as a
        # complete one, and an incomplete pack is still worth reading.
        "n_windows_read": len({tag for *_, tag in cells if tag in windows}),
        "windows_read": sorted({tag for *_, tag in cells if tag in windows}),
        "windows_registered": windows,
        "registration": registration,
        "cell_stats": {"/".join(k): v for k, v in sorted(stats.items())},
        "pooled": {"/".join(k): v for k, v in sorted(pooled.items())},
        "fleet_margin": fleet_margin(pooled),
        "pair_margins": pair_margins(pooled),
        "arm_correlation": arm_correlation(cells, windows),
        "predictions": evaluate_predictions(stats, pooled, registration, windows),
        "abl375_reread": reread_abl375(pooled, stats),
        "determinism_check": duplicate_check(stats),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    md_path = out.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"{payload['n_fits']} fits over {payload['n_cells']} cells -> {json_path}")
    print(f"evidence pack -> {md_path}")
    for stream, fleet in payload["fleet_margin"].items():
        print(f"  {stream:6s} p90 CV {100 * fleet['cv_rms_p90']:5.2f}%  "
              f"delta_min(k=1) {fleet['delta_min_pct_at_p90']['1']:5.2f}%  "
              f"delta_min(k=10) {fleet['delta_min_pct_at_p90']['10']:5.2f}%")
    for pred in payload["predictions"]:
        print(f"  {pred['id']}: {pred['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
