"""classify_unpaired distinguishes two `Pairs: 0` shapes (ABL-95).

Before this, every country with vintages but zero paired actuals read as
`no_paired_actuals` regardless of *why* nothing paired. That is right for a
zone that stopped publishing (GR), and wrong for a zone whose vintages simply
target days nobody has published yet — the ABL-95 shape, which cost a day of
investigation when xgboost-V014 joined the shadow rail late and its only two
vintages targeted days ahead of the actuals frontier.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.net_position import (
    EvalConfig, classify_unpaired, per_country_metrics,
)

FRONTIER = pd.Timestamp("2026-08-09 21:00")


def test_empty_targets_is_unclassified():
    diag = classify_unpaired(pd.Series([], dtype="datetime64[ns]"), FRONTIER)
    assert diag == {"rows": 0, "target_window": None, "after_actuals": 0,
                    "at_or_before_actuals": 0, "verdict": None}


def test_all_targets_after_frontier_is_awaiting_publication():
    targets = pd.Series(pd.date_range("2026-08-10", "2026-08-11", freq="h"))
    diag = classify_unpaired(targets, FRONTIER)
    assert diag["verdict"] == "awaiting_publication"
    assert diag["after_actuals"] == len(targets)
    assert diag["at_or_before_actuals"] == 0


def test_all_targets_at_or_before_frontier_is_no_zone_actuals():
    targets = pd.Series(pd.date_range("2026-08-01", "2026-08-02", freq="h"))
    diag = classify_unpaired(targets, FRONTIER)
    assert diag["verdict"] == "no_zone_actuals"
    assert diag["after_actuals"] == 0
    assert diag["at_or_before_actuals"] == len(targets)


def test_straddling_frontier_is_mixed():
    targets = pd.Series([FRONTIER - pd.Timedelta(hours=1),
                         FRONTIER + pd.Timedelta(hours=1)])
    diag = classify_unpaired(targets, FRONTIER)
    assert diag["verdict"] == "mixed"
    assert diag["after_actuals"] == 1
    assert diag["at_or_before_actuals"] == 1


def test_no_actuals_at_all_reads_as_no_zone_actuals_not_awaiting():
    # A frontier of None means we hold no actuals whatsoever, not that every
    # target is in the future — the two must not collapse to the same verdict.
    targets = pd.Series(pd.date_range("2026-08-01", periods=3, freq="h"))
    diag = classify_unpaired(targets, None)
    assert diag["after_actuals"] == 0
    assert diag["verdict"] == "no_zone_actuals"


# ---------------------------------------------------------------------------
# per_country_metrics integration: the two `Pairs: 0` shapes read differently
# ---------------------------------------------------------------------------

def _cfg():
    return EvalConfig(replica_db="unused")


def test_per_country_awaiting_publication_country_is_not_no_paired_actuals():
    scored = pd.DataFrame({"country_code": pd.Series([], dtype=object)})
    paired = pd.DataFrame({
        "country_code": ["V014"] * 2,
        "target_ts": [pd.Timestamp("2026-08-10"), pd.Timestamp("2026-08-11")],
    })
    out = per_country_metrics(scored, paired, _cfg(), actuals_max_ts=FRONTIER)
    m = out["V014"]
    assert m["n"] == 0
    assert m["coverage"] == "awaiting_publication"
    assert m["unpaired"]["verdict"] == "awaiting_publication"


def test_per_country_stopped_publishing_country_is_no_paired_actuals():
    scored = pd.DataFrame({"country_code": pd.Series([], dtype=object)})
    paired = pd.DataFrame({
        "country_code": ["GR"] * 2,
        "target_ts": [pd.Timestamp("2026-08-01"), pd.Timestamp("2026-08-02")],
    })
    out = per_country_metrics(scored, paired, _cfg(), actuals_max_ts=FRONTIER)
    m = out["GR"]
    assert m["n"] == 0
    assert m["coverage"] == "no_paired_actuals"
    assert m["unpaired"]["verdict"] == "no_zone_actuals"
