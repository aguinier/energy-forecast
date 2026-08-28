"""ABL-247: the pre-registered estimator, pinned where it can be checked.

The backtest's numbers depend on a replica and a 16-day archive, so they cannot
be asserted here. What *can* be asserted is the machinery the pre-registration
fixes -- leak-freeness of the feature lookup, the order of selection and
averaging, blocked leave-one-day-out CV, and the c = 0 decision rule -- against
synthetic series where the right answer is known by construction.

These are the tests that would have caught the three ways this analysis could
have produced a confident wrong number: a feature read after its cutoff, an
in-fold prediction scored as out-of-sample, and a cadence-weighted average.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.abl247_tso_feature_backtest import (  # noqa: E402
    BANDS,
    NOT_BACKTESTED_BANDS,
    PRIMARY_BAND,
    band_of,
    composition_term,
    coverage_table,
    feature_at_cutoffs,
    fit_affine,
    json_safe,
    normalize_ts,
    wape,
)


def _vintages(rows) -> pd.DataFrame:
    """`(country, target, value, first_seen)` tuples as the reader returns them."""
    frame = pd.DataFrame(rows, columns=["country_code", "target",
                                        "forecast_value", "first_seen"])
    frame["target"] = pd.to_datetime(frame["target"])
    frame["first_seen"] = pd.to_datetime(frame["first_seen"])
    return frame


# --------------------------------------------------------------------------
# Leak-freeness -- the property the whole issue rests on
# --------------------------------------------------------------------------


def test_feature_never_uses_a_vintage_first_seen_after_the_cutoff():
    """A revision published after our run must be invisible to that run.

    This is the defect the ABL-184 archive exists to make detectable: the
    retained pre-archive series carried the *last* vintage, so a feature built
    from it would have handed the model a number that did not exist yet on ~40%
    of load targets (prereg section 2).
    """
    tso = _vintages([
        ("DE", "2026-08-14 12:00", 100.0, "2026-08-13 06:00"),   # before cutoff
        ("DE", "2026-08-14 12:00", 999.0, "2026-08-13 18:00"),   # after cutoff
    ])
    out = feature_at_cutoffs(tso, pd.DatetimeIndex(["2026-08-13 07:00"]))

    assert len(out) == 1
    assert out["f_tso"].iloc[0] == 100.0
    assert out["tso_first_seen"].iloc[0] <= pd.Timestamp("2026-08-13 07:00")


def test_feature_takes_the_latest_vintage_visible_at_the_cutoff():
    """Leak-free is not the same as stale: the freshest *visible* value wins."""
    tso = _vintages([
        ("DE", "2026-08-14 12:00", 100.0, "2026-08-13 06:00"),
        ("DE", "2026-08-14 12:00", 150.0, "2026-08-13 09:00"),
        ("DE", "2026-08-14 12:00", 999.0, "2026-08-13 20:00"),
    ])
    out = feature_at_cutoffs(tso, pd.DatetimeIndex(["2026-08-13 12:00"]))

    assert out["f_tso"].iloc[0] == 150.0


def test_a_cutoff_before_any_vintage_yields_no_feature_rather_than_a_default():
    """Absent is absent. A zero here would be a fabricated forecast."""
    tso = _vintages([("DE", "2026-08-14 12:00", 100.0, "2026-08-13 06:00")])
    out = feature_at_cutoffs(tso, pd.DatetimeIndex(["2026-08-13 05:00"]))

    assert out.empty


# --------------------------------------------------------------------------
# Selection order -- select per instant, then average to the hour
# --------------------------------------------------------------------------


def test_selection_happens_per_instant_before_the_hourly_average():
    """Averaging first would blend a visible value with an invisible one.

    Two quarter-hourly instants inside one target hour, revised at different
    times. At a cutoff between the two revisions, the hour's feature is the
    *visible* value of each instant averaged -- (100 + 200) / 2 = 150. Averaging
    the whole vintage stack first and then applying the cutoff would let the
    unpublished 900 into the mean, which is a number nobody could have held.
    """
    tso = _vintages([
        ("DE", "2026-08-14 12:00", 100.0, "2026-08-13 06:00"),
        ("DE", "2026-08-14 12:15", 200.0, "2026-08-13 06:00"),
        ("DE", "2026-08-14 12:15", 900.0, "2026-08-13 20:00"),
    ])
    out = feature_at_cutoffs(tso, pd.DatetimeIndex(["2026-08-13 07:00"]))

    assert len(out) == 1
    assert out["target"].iloc[0] == pd.Timestamp("2026-08-14 12:00")
    assert out["f_tso"].iloc[0] == pytest.approx(150.0)


def test_each_cutoff_gets_its_own_feature_value():
    """One vintage stack, two runs, two different features -- not one shared one."""
    tso = _vintages([
        ("DE", "2026-08-14 12:00", 100.0, "2026-08-13 06:00"),
        ("DE", "2026-08-14 12:00", 180.0, "2026-08-13 18:00"),
    ])
    out = feature_at_cutoffs(
        tso, pd.DatetimeIndex(["2026-08-13 07:00", "2026-08-13 19:00"]))

    by_cutoff = dict(zip(out["generated_at"], out["f_tso"]))
    assert by_cutoff[pd.Timestamp("2026-08-13 07:00")] == 100.0
    assert by_cutoff[pd.Timestamp("2026-08-13 19:00")] == 180.0


# --------------------------------------------------------------------------
# The registered estimator
# --------------------------------------------------------------------------


def _panel(days: int, c_true: float, noise: float, seed: int = 0,
           countries=("DE",)) -> pd.DataFrame:
    """A panel where the truth really is `a + b*ours + c*tso` plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for country in countries:
        for day in range(days):
            base = pd.Timestamp("2026-08-13") + pd.Timedelta(days=day)
            for hour in range(24):
                target = base + pd.Timedelta(hours=hour)
                ours = 1000.0 + 200.0 * np.sin(hour / 24 * 2 * np.pi) + rng.normal(0, 20)
                tso = ours + rng.normal(0, 50)
                actual = (50.0 + (1.0 - c_true) * ours + c_true * tso
                          + rng.normal(0, noise))
                rows.append({"country_code": country, "target": target,
                             "target_day": base, "band": PRIMARY_BAND,
                             "f_ours": ours, "f_tso": tso, "actual": actual,
                             "available": True})
    return pd.DataFrame(rows)


def test_the_combiner_recovers_a_real_coefficient_and_excludes_zero():
    """When the TSO genuinely carries signal, `c` is found and the CI clears 0."""
    fit = fit_affine(_panel(days=14, c_true=0.5, noise=5.0, seed=1))

    assert fit.days == 14
    assert fit.c_hat == pytest.approx(0.5, abs=0.05)
    low, high = fit.c_ci
    assert low > 0.0
    assert fit.verdict == "c CI excludes 0"
    assert fit.wape_combiner_cv < fit.wape_null_cv


def test_a_useless_feature_returns_the_pre_committed_negative():
    """`c`'s CI covering 0 is the registered stopping rule, not a soft signal.

    Prereg section 5: "If c-hat's CI includes 0 at 0-24h for a series, the
    feature has not earned its place for that series. Report the negative; do
    not escalate to the retrain arm hoping for a win."
    """
    panel = _panel(days=14, c_true=0.0, noise=60.0, seed=2)
    rng = np.random.default_rng(3)
    panel["f_tso"] = rng.normal(1000.0, 200.0, len(panel))  # pure noise

    fit = fit_affine(panel)

    low, high = fit.c_ci
    assert low <= 0.0 <= high
    assert fit.verdict == "c CI includes 0 -- feature has NOT earned its place"


def test_cross_validation_blocks_on_the_day_not_the_row():
    """A day's own rows take no part in the fit that predicts them.

    The within-day autocorrelation prereg section 3 names is what makes
    row-level CV optimistic: neighbouring hours are near-duplicates, so a
    row-shuffled fold sees the held-out row's twin.

    The check is structural rather than a WAPE comparison, because a WAPE gap
    could come from anywhere. One day is given a relationship no other day
    shares. Under leave-one-*day*-out there is exactly one fold per day, and the
    fold that holds out the rogue day is the only one fitted on clean data --
    so its coefficient is the one that differs. Under any row-level split every
    fold would see the rogue rows and no such asymmetry could exist.
    """
    # The two arms are deliberately orthogonal here, unlike the real series and
    # unlike `_panel`. Where `f_ours` and `f_tso` are near-collinear the split
    # of a common effect between `b` and `c` is not identified, and this test
    # would be asserting on which of two exchangeable coefficients absorbed it.
    rng = np.random.default_rng(4)
    rows = []
    for day in range(10):
        base = pd.Timestamp("2026-08-13") + pd.Timedelta(days=day)
        for hour in range(24):
            ours = 1000.0 + rng.normal(0, 100)
            tso = 500.0 + rng.normal(0, 100)
            rows.append({"country_code": "DE", "target_day": base,
                         "target": base + pd.Timedelta(hours=hour),
                         "band": PRIMARY_BAND, "f_ours": ours, "f_tso": tso,
                         "actual": ours, "available": True})
    panel = pd.DataFrame(rows)

    days = sorted(panel["target_day"].unique())
    rogue_index = 4
    rogue = panel["target_day"] == days[rogue_index]
    # The rogue day alone is driven hard by the TSO series.
    panel.loc[rogue, "actual"] = (panel.loc[rogue, "actual"]
                                  + 4.0 * panel.loc[rogue, "f_tso"])

    fit = fit_affine(panel)

    assert fit.days == 10
    # One fold per day is the definition of the blocking.
    assert len(fit.c_fold_spread) == 10

    folds = np.asarray(fit.c_fold_spread, dtype=float)
    clean_fold = folds[rogue_index]          # the fit that excluded the rogue day
    contaminated = np.delete(folds, rogue_index)

    # Every fold that saw the rogue day is pulled towards it; the one that did
    # not is left near the true c = 0.
    assert abs(clean_fold) < 0.1
    assert contaminated.min() > 0.2
    assert abs(clean_fold - fit.c_hat) > abs(contaminated.mean() - fit.c_hat)


def test_a_single_day_cannot_be_cross_validated():
    """One block is not a blocked design, and the fit says so rather than guessing."""
    fit = fit_affine(_panel(days=1, c_true=0.5, noise=5.0, seed=5))

    assert fit.days == 1
    assert "single day" in fit.verdict


def test_an_empty_panel_is_reported_not_imputed():
    fit = fit_affine(pd.DataFrame(columns=["actual", "f_ours", "f_tso",
                                           "target_day", "country_code"]))
    assert fit.verdict == "no rows"
    assert fit.n == 0


# --------------------------------------------------------------------------
# Bands and the re-scope
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hours,expected", [
    (0.0, "0-24h"), (23.9, "0-24h"), (24.0, "24-48h"), (47.9, "24-48h"),
    (48.0, "48-64h"), (64.0, "48-64h"), (65.0, None), (None, None),
])
def test_band_boundaries_are_half_open_with_an_inclusive_64h_endpoint(hours, expected):
    assert band_of(hours) == expected


def test_the_rescoped_band_is_declared_not_merely_absent():
    """48-64h is out of the backtest by decision, and the decision is in the code.

    A band silently missing from a results table reads as "no data". This one is
    a CEO re-scope accepted on 2026-08-14 -- 0.0% coverage is a product property
    of a day-ahead series -- and it stays measured so a later reader can check
    the re-scope rather than inherit it.
    """
    assert "48-64h" in NOT_BACKTESTED_BANDS
    assert PRIMARY_BAND == "0-24h"
    assert [name for name, _lo, _hi in BANDS] == ["0-24h", "24-48h", "48-64h"]


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


def test_the_composition_term_isolates_the_coverage_effect():
    """Our own arm moves between all-rows and matched purely because rows differ.

    Here the feature happens to be present on the easy rows. Our forecast is
    unchanged, but its matched WAPE is better than its all-rows WAPE -- and that
    difference is the composition term, not a feature effect. Prereg section 5
    requires it named, because it is the one number that can make a routed
    design look like a modelling win.
    """
    panel = pd.DataFrame({
        "band": [PRIMARY_BAND] * 4,
        "country_code": ["DE"] * 4,
        "target_day": [pd.Timestamp("2026-08-13")] * 4,
        "f_ours": [100.0, 100.0, 100.0, 100.0],
        "actual": [100.0, 100.0, 50.0, 50.0],
        "available": [True, True, False, False],
    })

    row = composition_term(panel).iloc[0]

    assert row["n_all_rows"] == 4
    assert row["n_matched"] == 2
    assert row["wape_ours_matched"] == pytest.approx(0.0)
    assert row["wape_ours_all_rows"] > 0.0
    assert row["composition_term_pp"] < 0.0


def test_coverage_table_reports_days_and_the_backtested_flag_per_band():
    panel = pd.DataFrame({
        "band": ["0-24h", "0-24h", "48-64h"],
        "country_code": ["DE", "DE", "DE"],
        "target_day": [pd.Timestamp("2026-08-13"), pd.Timestamp("2026-08-14"),
                       pd.Timestamp("2026-08-13")],
        "available": [True, False, False],
        "feature_lead_h": [22.5, np.nan, np.nan],
        "feature_age_at_cutoff_h": [2.0, np.nan, np.nan],
    })

    table = coverage_table({"load": panel}).set_index("band")

    assert table.loc["0-24h", "coverage_pct"] == pytest.approx(50.0)
    assert table.loc["0-24h", "target_days"] == 2
    # Lead is reported over the rows that have the feature, not over all rows.
    assert table.loc["0-24h", "median_feature_lead_h"] == pytest.approx(22.5)
    # A band with no feature row has no lead to report -- unmeasured, not zero.
    assert pd.isna(table.loc["48-64h", "median_feature_lead_h"])
    assert bool(table.loc["0-24h", "backtested"]) is True
    assert bool(table.loc["48-64h", "backtested"]) is False


# --------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------


def test_wape_is_unmeasured_rather_than_infinite_on_a_zero_denominator():
    """A country whose truth sums to zero has no WAPE, and gets NaN not inf."""
    assert np.isnan(wape([1.0, -1.0], [0.0, 0.0]))
    assert wape([10.0], [100.0]) == pytest.approx(10.0)


def test_the_record_is_strict_json_with_unmeasured_cells_as_null():
    """An unmeasured cell must serialise as `null`, never as bare `NaN`.

    `json.dumps` emits `NaN` by default, which is not JSON. A reader in a strict
    parser rejects the file; a reader in a lenient one silently gets a float
    that is not a number. Both are worse than the honest `null` the rest of this
    repo's records use for "not measured" -- and this record is full of them,
    because a coverage-short cell has no WAPE and a one-day band has no
    interval.
    """
    record = {"a": float("nan"), "b": [1.0, float("inf")],
              "c": {"d": np.float64("nan"), "e": np.int64(3), "f": np.True_},
              "g": pd.NaT, "h": "text"}

    text = json.dumps(json_safe(record))

    assert "NaN" not in text and "Infinity" not in text
    parsed = json.loads(text)
    assert parsed["a"] is None
    assert parsed["b"] == [1.0, None]
    assert parsed["c"] == {"d": None, "e": 3, "f": True}
    assert parsed["g"] is None
    assert parsed["h"] == "text"


# --------------------------------------------------------------------------
# The alignment defect that invalidated the first live run (2026-08-28)
# --------------------------------------------------------------------------


def test_normalize_ts_preserves_a_non_contiguous_index():
    """Parsing must not silently re-key the column it is assigned back into.

    Every read in this harness filters rows before parsing -- `.isin(
    SUPPORTED_COUNTRIES)`, `.dropna()` -- so the frame reaching `normalize_ts`
    has a gappy index as a matter of course, never a `RangeIndex`. An earlier
    version rebuilt the result as `pd.Series(list(values))`, which carries a
    fresh positional `RangeIndex`; `frame["target"] = normalize_ts(...)` then
    aligned label-to-label and pulled each row's timestamp from whichever row
    happened to sit at that *position*.

    The failure is silent in the worst way: no exception, no nulls, and a column
    full of real timestamps that belong to other rows.
    """
    frame = pd.DataFrame(
        {"country_code": ["AT", "XX", "DE", "XX", "FR"],
         "raw": ["2026-08-13T00:00:00Z", "2026-08-14 00:00:00",
                 "2026-08-15T00:00:00Z", "2026-08-16 00:00:00",
                 "2026-08-17T00:00:00Z"]})
    kept = frame[frame["country_code"] != "XX"]        # index 0, 2, 4
    assert not kept.index.equals(pd.RangeIndex(len(kept)))

    kept = kept.copy()
    kept["parsed"] = normalize_ts(kept["raw"])

    assert kept["parsed"].notna().all(), "index alignment dropped rows to NaT"
    assert list(kept["parsed"]) == [pd.Timestamp("2026-08-13"),
                                    pd.Timestamp("2026-08-15"),
                                    pd.Timestamp("2026-08-17")]


def test_normalize_ts_still_accepts_a_bare_sequence():
    """`replica_state` passes a one-element list, not a Series."""
    assert normalize_ts(["2026-08-28T14:11:02.196Z"]).iloc[0] == \
        pd.Timestamp("2026-08-28 14:11:02.196")


def test_a_vintage_cannot_be_known_further_ahead_than_it_was_published():
    """The invariant the alignment defect violated, stated on the panel.

    A TSO day-ahead value selected at a cutoff must have been first seen at or
    before that cutoff -- so `generated_at - tso_first_seen` is non-negative,
    and the feature's lead over its target cannot exceed the longest lead the
    archive actually contains. The corrupted run reported a median lead of
    54.07h in a band whose rows sit 48-64h out, against a measured maximum
    forward lead of 47.07h; that arithmetic impossibility is what this pins.
    """
    tso = _vintages([
        ("AT", "2026-08-20 12:00", 100.0, "2026-08-19 06:00"),   # lead 30h
        ("AT", "2026-08-20 12:00", 110.0, "2026-08-20 06:00"),   # lead  6h
        ("AT", "2026-08-22 12:00", 120.0, "2026-08-21 06:00"),   # lead 30h
    ])
    cutoffs = pd.DatetimeIndex([pd.Timestamp("2026-08-20 07:00")])

    feature = feature_at_cutoffs(tso, cutoffs)

    age = (feature["generated_at"] - feature["tso_first_seen"])
    assert (age >= pd.Timedelta(0)).all(), "selected a vintage from the future"
    lead = (feature["target"] - feature["tso_first_seen"])
    assert lead.max() <= pd.Timedelta(hours=30)
    # The 08-22 target had not been published by the 08-20 07:00 cutoff.
    assert pd.Timestamp("2026-08-22 12:00") not in set(feature["target"])
