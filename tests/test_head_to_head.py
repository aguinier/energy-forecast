"""Head-to-head comparison must score two models on identical rows (ABL-68)."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.head_to_head import (MATERIAL_PCT, compare, pair,
                                         render_markdown)


def _fc(country, gen, hours, values):
    return pd.DataFrame({
        "country_code": country,
        "generated_at": gen,
        "target_ts": pd.date_range(f"{gen[:10]} 00:00", periods=hours, freq="h"),
        "forecast_value": values,
    })


def _actuals(country, gen, hours, values):
    return pd.DataFrame({
        "country_code": country,
        "target_ts": pd.date_range(f"{gen[:10]} 00:00", periods=hours, freq="h"),
        "actual": values,
    })


def test_pair_drops_rows_only_one_model_covers():
    """The champion's extra prod-pushed vintages must not enter the comparison.

    This is the defect the module exists for: scored on its own rows the
    champion covered 57 vintages to the challenger's 49, and the report-to-report
    read said the challenger won when the paired read said it lost.
    """
    a = _fc("FR", "2026-06-15", 4, [100.0, 100.0, 100.0, 100.0])
    b = _fc("FR", "2026-06-15", 2, [110.0, 110.0])
    act = _actuals("FR", "2026-06-15", 4, [100.0, 100.0, 500.0, 500.0])

    paired = pair(a, b, act)
    assert len(paired) == 2
    # The two hours where the champion is wildly wrong are excluded because the
    # challenger never forecast them.
    assert paired["actual"].tolist() == [100.0, 100.0]

    h = compare(paired, "champ", "chall", n_only_a=2, n_only_b=0)
    assert h.n_paired == 2 and h.n_only_a == 2
    assert h.pooled_mae_a_mw == 0.0 and h.pooled_mae_b_mw == 10.0
    assert h.pooled_delta_mw == 10.0  # challenger is worse, not better


def test_row_without_an_actual_scores_neither_model():
    a = _fc("BE", "2026-06-15", 3, [10.0, 10.0, 10.0])
    b = _fc("BE", "2026-06-15", 3, [20.0, 20.0, 20.0])
    act = _actuals("BE", "2026-06-15", 3, [10.0, np.nan, 10.0])

    paired = pair(a, b, act)
    assert len(paired) == 2

    h = compare(paired, "a", "b")
    assert h.countries[0].n == 2


def test_pass_through_country_reads_identical_not_tie():
    """V016 passes BG/LT/RO through uncorrected. That is a design fact, and
    reporting it as a 'tie' would hide that no correction ran at all."""
    a = _fc("BG", "2026-06-15", 4, [10.0, 20.0, 30.0, 40.0])
    b = _fc("BG", "2026-06-15", 4, [10.0, 20.0, 30.0, 40.0])
    act = _actuals("BG", "2026-06-15", 4, [11.0, 19.0, 33.0, 38.0])

    h = compare(pair(a, b, act), "V010", "V016")
    assert h.countries[0].verdict == "identical"
    assert h.n_identical == 1
    assert h.n_better == 0


def test_sub_material_gap_is_a_tie_not_a_win():
    """Three of V016's four 'wins' were under 0.5% (AT -0.1%, LV -0.2%,
    PL -0.4%). Counting those as wins manufactures a result out of noise."""
    a = _fc("AT", "2026-06-15", 4, [0.0, 0.0, 0.0, 0.0])
    b = _fc("AT", "2026-06-15", 4, [0.999, 0.999, 0.999, 0.999])
    act = _actuals("AT", "2026-06-15", 4, [1.0, 1.0, 1.0, 1.0])

    h = compare(pair(a, b, act), "V010", "V016")
    c = h.countries[0]
    assert c.delta_pct == pytest.approx(-99.9, abs=0.1)
    assert c.verdict == "better"          # a real 99.9% improvement
    assert h.n_materially_better == 1

    # Now a gap below the materiality floor.
    b2 = _fc("AT", "2026-06-15", 4, [0.0, 0.0, 0.0, 0.002])
    h2 = compare(pair(a, b2, act), "V010", "V016")
    assert abs(h2.countries[0].delta_pct) < MATERIAL_PCT
    assert h2.countries[0].verdict == "tie"
    assert h2.n_materially_better == 0


def test_empty_overlap_reports_zero_not_nan():
    """No overlap is 'we did not measure this', and must not render as a
    metric. NaN in a report table reads as a number to a hurried reader."""
    a = _fc("FR", "2026-06-15", 2, [1.0, 2.0])
    b = _fc("DE", "2026-06-15", 2, [1.0, 2.0])
    act = _actuals("FR", "2026-06-15", 2, [1.0, 2.0])

    h = compare(pair(a, b, act), "a", "b", n_only_a=2, n_only_b=2)
    assert h.n_paired == 0
    assert h.countries == []
    assert h.pooled_mae_a_mw == 0.0 and h.pooled_mae_b_mw == 0.0


def test_join_is_on_vintage_too_not_just_target_hour():
    """Two vintages forecasting the same hour are different predictions. Joining
    on the target hour alone would cross-multiply them."""
    def two_vintages(v1, v2):
        # Both vintages forecast the SAME target hours, which is what a daily
        # job actually produces: D+1 and D+2 views of the same day.
        f = pd.concat([_fc("FR", "2026-06-15", 2, v1),
                       _fc("FR", "2026-06-15", 2, v2)], ignore_index=True)
        f.loc[2:, "generated_at"] = "2026-06-14"
        return f

    a = two_vintages([1.0, 2.0], [3.0, 4.0])
    b = two_vintages([1.0, 2.0], [3.0, 4.0])
    act = _actuals("FR", "2026-06-15", 2, [1.0, 2.0])

    paired = pair(a, b, act)
    # 2 target hours x 2 vintages, not 2 x (2*2).
    assert len(paired) == 4
    assert paired["generated_at"].nunique() == 2


def test_markdown_states_scope_and_never_claims_comparability():
    a = _fc("FR", "2026-06-15", 4, [100.0] * 4)
    b = _fc("FR", "2026-06-15", 4, [110.0] * 4)
    act = _actuals("FR", "2026-06-15", 4, [100.0] * 4)

    h = compare(pair(a, b, act), "chronos-2-V010", "chronos-2-V016", n_only_a=8)
    md = render_markdown(h, "2026-06-17..2026-08-04", "2026-08-08 06:00 UTC")

    assert "not* comparable" in md
    assert "only in `chronos-2-V010`: 8" in md
    assert "worse" in md
    assert "| FR |" in md
