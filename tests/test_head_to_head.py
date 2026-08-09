"""Head-to-head comparison must score two models on identical rows (ABL-68),
and must recognise a run when the two models stamp it seconds apart (ABL-82)."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.head_to_head import (MATERIAL_PCT, MAX_RUN_SKEW, compare,
                                         pair, render_markdown)

# The rail's real shape: champion at 06:00:xx with microseconds, challengers a
# few seconds later truncated to the second. Measured on the live sidecar,
# 2026-08-09.
CHAMP_GEN = "2026-06-14 06:00:55.715745"
CHAL_GEN = "2026-06-14 06:01:08"


def _fc(country, gen, hours, values, target_day="2026-06-15"):
    return pd.DataFrame({
        "country_code": country,
        "generated_at": gen,
        "target_ts": pd.date_range(f"{target_day} 00:00", periods=hours, freq="h"),
        "forecast_value": values,
    })


def _actuals(country, hours, values, target_day="2026-06-15"):
    return pd.DataFrame({
        "country_code": country,
        "target_ts": pd.date_range(f"{target_day} 00:00", periods=hours, freq="h"),
        "actual": values,
    })


def test_pairs_across_the_second_level_skew_the_live_rail_produces():
    """ABL-82: the defect itself.

    The champion and the challengers are separate processes in
    run-net-position.ps1 and each stamps its own datetime.now(). Measured on the
    live sidecar 2026-08-09 they were 12.3 s apart, and an exact `generated_at`
    join paired 0 of 1,368 rows for every challenger — while printing a full
    report of 0.0 MW MAEs.
    """
    a = _fc("FR", CHAMP_GEN, 4, [100.0] * 4)
    b = _fc("FR", CHAL_GEN, 4, [110.0] * 4)
    act = _actuals("FR", 4, [100.0] * 4)

    paired, scope = pair(a, b, act)
    assert len(paired) == 4
    assert scope.n_only_a == 0 and scope.n_only_b == 0
    assert scope.max_skew_seconds == pytest.approx(12.28, abs=0.01)

    h = compare(paired, "chronos-2-V010", "chronos-2-V016", scope)
    assert h.measured
    assert h.n_vintages == 1          # one run, two stamps
    assert h.pooled_mae_a_mw == 0.0 and h.pooled_mae_b_mw == 10.0


def test_a_backfill_hours_later_is_a_different_run_and_is_not_paired():
    """The challenger backfill on 2026-08-07 ran 15h25m after that day's
    champion, so it saw a further day of actuals. Pairing it would credit the
    challenger for information the champion never had."""
    a = _fc("FR", "2026-06-14 06:00:33.466898", 4, [100.0] * 4)
    b = _fc("FR", "2026-06-14 21:25:35", 4, [110.0] * 4)
    act = _actuals("FR", 4, [100.0] * 4)

    paired, scope = pair(a, b, act)
    assert paired.empty
    assert scope.n_only_a == 4 and scope.n_only_b == 4

    h = compare(paired, "V010", "V016", scope)
    assert not h.measured


def test_same_cutoff_but_beyond_the_skew_bound_is_rejected_and_counted():
    """A cutoff bucket is 24h wide, so the skew bound is what stops two runs a
    day apart from pairing. Rejections are reported, never silent."""
    a = _fc("FR", "2026-06-13 11:30:00", 4, [100.0] * 4)
    b = _fc("FR", "2026-06-14 06:00:00", 4, [110.0] * 4)
    # Same as_of (both see actuals through 2026-06-14 22:00), 18.5h apart.
    paired, scope = pair(a, b, act := _actuals("FR", 4, [100.0] * 4))
    assert paired.empty
    assert scope.n_rejected_skew == 4      # four hours matched a cutoff, none a run

    # Widening the bound past the gap pairs them, which is the knob the gate
    # has if the pipeline ever gets that slow.
    paired2, scope2 = pair(a, b, act, max_run_skew=pd.Timedelta(hours=24))
    assert len(paired2) == 4 and scope2.n_rejected_skew == 0


def test_a_champion_rerun_pairs_the_closest_vintage_not_a_duplicate():
    """2026-08-06 has two champion vintages (06:00:44 and 10:52:22) sharing one
    cutoff. A day-level join would pair both to the single challenger vintage
    and count the challenger's hours twice."""
    a = pd.concat([_fc("FR", "2026-06-14 06:00:44.053283", 2, [100.0, 100.0]),
                   _fc("FR", "2026-06-14 10:52:22.362737", 2, [200.0, 200.0])],
                  ignore_index=True)
    b = _fc("FR", CHAL_GEN, 2, [110.0, 110.0])
    act = _actuals("FR", 2, [100.0, 100.0])

    paired, scope = pair(a, b, act)
    assert len(paired) == 2                      # not 4
    # The 06:00 vintage is the one 12s from the challenger, so it is the pair.
    assert paired["forecast_a"].tolist() == [100.0, 100.0]
    assert scope.n_only_a == 2                   # the 10:52 re-run, unmatched
    assert scope.n_only_b == 0


def test_pair_drops_rows_only_one_model_covers():
    """The champion's extra prod-pushed vintages must not enter the comparison.

    This is the defect the module exists for: scored on its own rows the
    champion covered 57 vintages to the challenger's 49, and the report-to-report
    read said the challenger won when the paired read said it lost.
    """
    a = _fc("FR", CHAMP_GEN, 4, [100.0, 100.0, 100.0, 100.0])
    b = _fc("FR", CHAL_GEN, 2, [110.0, 110.0])
    act = _actuals("FR", 4, [100.0, 100.0, 500.0, 500.0])

    paired, scope = pair(a, b, act)
    assert len(paired) == 2
    # The two hours where the champion is wildly wrong are excluded because the
    # challenger never forecast them.
    assert paired["actual"].tolist() == [100.0, 100.0]
    assert scope.n_only_a == 2 and scope.n_only_b == 0

    h = compare(paired, "champ", "chall", scope)
    assert h.n_paired == 2 and h.n_only_a == 2
    assert h.pooled_mae_a_mw == 0.0 and h.pooled_mae_b_mw == 10.0
    assert h.pooled_delta_mw == 10.0  # challenger is worse, not better


def test_row_without_an_actual_scores_neither_model():
    a = _fc("BE", CHAMP_GEN, 3, [10.0, 10.0, 10.0])
    b = _fc("BE", CHAL_GEN, 3, [20.0, 20.0, 20.0])
    act = _actuals("BE", 3, [10.0, np.nan, 10.0])

    paired, _ = pair(a, b, act)
    assert len(paired) == 2

    h = compare(paired, "a", "b")
    assert h.countries[0].n == 2


def test_pass_through_country_reads_identical_not_tie():
    """V016 passes BG/LT/RO through uncorrected. That is a design fact, and
    reporting it as a 'tie' would hide that no correction ran at all."""
    a = _fc("BG", CHAMP_GEN, 4, [10.0, 20.0, 30.0, 40.0])
    b = _fc("BG", CHAL_GEN, 4, [10.0, 20.0, 30.0, 40.0])
    act = _actuals("BG", 4, [11.0, 19.0, 33.0, 38.0])

    h = compare(pair(a, b, act)[0], "V010", "V016")
    assert h.countries[0].verdict == "identical"
    assert h.n_identical == 1
    assert h.n_better == 0


def test_sub_material_gap_is_a_tie_not_a_win():
    """Three of V016's four 'wins' were under 0.5% (AT -0.1%, LV -0.2%,
    PL -0.4%). Counting those as wins manufactures a result out of noise."""
    a = _fc("AT", CHAMP_GEN, 4, [0.0, 0.0, 0.0, 0.0])
    b = _fc("AT", CHAL_GEN, 4, [0.999, 0.999, 0.999, 0.999])
    act = _actuals("AT", 4, [1.0, 1.0, 1.0, 1.0])

    h = compare(pair(a, b, act)[0], "V010", "V016")
    c = h.countries[0]
    assert c.delta_pct == pytest.approx(-99.9, abs=0.1)
    assert c.verdict == "better"          # a real 99.9% improvement
    assert h.n_materially_better == 1

    # Now a gap below the materiality floor.
    b2 = _fc("AT", CHAL_GEN, 4, [0.0, 0.0, 0.0, 0.002])
    h2 = compare(pair(a, b2, act)[0], "V010", "V016")
    assert abs(h2.countries[0].delta_pct) < MATERIAL_PCT
    assert h2.countries[0].verdict == "tie"
    assert h2.n_materially_better == 0


def test_empty_overlap_reports_no_number_not_zero():
    """No overlap is 'we did not measure this'. It must not render as 0.0 MW
    MAE — a zeroed metric reads as a flawless forecast, and the ABL-82 join
    defect survived precisely because it printed one for three challengers."""
    a = _fc("FR", CHAMP_GEN, 2, [1.0, 2.0])
    b = _fc("DE", CHAL_GEN, 2, [1.0, 2.0])
    act = _actuals("FR", 2, [1.0, 2.0])

    paired, scope = pair(a, b, act)
    h = compare(paired, "a", "b", scope)
    assert h.n_paired == 0
    assert h.countries == []
    assert not h.measured
    assert h.pooled_mae_a_mw is None and h.pooled_mae_b_mw is None
    assert h.pooled_delta_mw is None and h.pooled_delta_pct is None
    assert h.to_dict()["pooled_mae_a_mw"] is None   # json null, not 0

    md = render_markdown(h, "all", "2026-08-09 06:00 UTC")
    assert "Not measured" in md
    assert "0.0 MW" not in md
    assert "% worse" not in md and "% better" not in md


def test_join_is_on_the_run_too_not_just_target_hour():
    """Two vintages forecasting the same hour are different predictions. Joining
    on the target hour alone would cross-multiply them."""
    def two_vintages(gen_late, gen_early, v1, v2):
        # Both vintages forecast the SAME target hours, which is what a daily
        # job actually produces: D+1 and D+2 views of the same day.
        return pd.concat([_fc("FR", gen_late, 2, v1),
                          _fc("FR", gen_early, 2, v2)], ignore_index=True)

    a = two_vintages(CHAMP_GEN, "2026-06-13 06:00:12.100000", [1.0, 2.0], [3.0, 4.0])
    b = two_vintages(CHAL_GEN, "2026-06-13 06:00:20", [1.0, 2.0], [3.0, 4.0])
    act = _actuals("FR", 2, [1.0, 2.0])

    paired, scope = pair(a, b, act)
    # 2 target hours x 2 runs, not 2 x (2*2).
    assert len(paired) == 4
    assert paired["run_as_of"].nunique() == 2
    assert scope.n_only_a == 0 and scope.n_only_b == 0


def test_markdown_states_scope_and_never_claims_comparability():
    a = _fc("FR", CHAMP_GEN, 4, [100.0] * 4)
    b = _fc("FR", CHAL_GEN, 4, [110.0] * 4)
    act = _actuals("FR", 4, [100.0] * 4)

    paired, scope = pair(a, b, act)
    scope.n_only_a = 8
    h = compare(paired, "chronos-2-V010", "chronos-2-V016", scope)
    md = render_markdown(h, "2026-06-17..2026-08-04", "2026-08-08 06:00 UTC")

    assert "not* comparable" in md
    assert "only in `chronos-2-V010`: 8" in md
    assert "worse" in md
    assert "| FR |" in md
    # The run rule is stated in the report, not only in the code.
    assert "actuals cutoff" in md


def test_default_skew_bound_is_wide_enough_for_the_rail_and_narrow_enough_to_split_runs():
    """Sized from measurement, not taste: the live champion->challenger gap is
    3.8-12.3 s and the two backfills that must NOT pair are 5h36m and 15h25m."""
    assert MAX_RUN_SKEW > pd.Timedelta(minutes=30)
    assert MAX_RUN_SKEW < pd.Timedelta(hours=5, minutes=36)
