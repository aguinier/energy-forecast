"""ABL-385: the arithmetic behind the registered decision margin.

The deliverable of ABL-385 is a number that future registrations cite instead of
a remembered noise floor. That makes the arithmetic load-bearing in a way a
one-off analysis script normally is not, so it is tested rather than trusted.

Three things are worth guarding, and they are the three that would fail
silently:

* **The margin's scaling.** delta_min has to fall as 1/sqrt(k). A margin that
  scaled as 1/k would tell a future gate read that 3 seeds buy what 9 actually
  buy, and nothing downstream would notice.
* **The range-to-CV conversion.** ABL-375, ABL-338 and ABL-253 all published
  max-minus-min. Converting those onto a sigma scale is what lets this issue say
  their margins were set against an understated floor, so d2 has to be right and
  has to be applied at the published seed count, not this issue's.
* **The reader finding the driver's files at all.** The driver names the sweep
  outputs and the reader parses those names back. They are two files with one
  convention between them, so the test generates names through the driver and
  parses them with the reader rather than hardcoding either side.
"""

import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.abl385_read_margin import (  # noqa: E402
    D2,
    Z,
    _percentile,
    _sd,
    _chi2_quantile,
    _variance_split,
    cv_interval,
    delta_min,
    parse_filename,
    range_to_cv,
    seeds_needed,
)
from scripts.abl385_run_sweep import _invocation  # noqa: E402


class TestMarginScaling:
    def test_margin_falls_as_one_over_root_k(self):
        """Quadrupling the seeds has to halve the margin, not quarter it.

        This is the whole reason a seed budget is worth anything: the return on
        more seeds is real but sublinear, and a gate read that assumed otherwise
        would under-buy seeds and call a gap it cannot see.
        """
        c = 0.05
        assert delta_min(c, c, 4) == pytest.approx(delta_min(c, c, 1) / 2)
        assert delta_min(c, c, 16) == pytest.approx(delta_min(c, c, 1) / 4)

    def test_equal_arms_reduce_to_the_closed_form(self):
        c, k = 0.032, 3
        assert delta_min(c, c, k) == pytest.approx(Z * math.sqrt(2) * c / math.sqrt(k))

    def test_seeds_needed_inverts_delta_min_exactly(self):
        """The two are used in the same sentence of the evidence pack.

        "a 4.5% gap would have needed N seeds" and "at k seeds the margin is X"
        have to be the same statement read in two directions, or the pack
        contradicts itself.
        """
        c_a, c_b = 0.032, 0.021
        for k in (1, 3, 5, 10, 12, 20):
            margin = delta_min(c_a, c_b, k)
            assert seeds_needed(c_a, c_b, margin) == pytest.approx(k)

    def test_positive_correlation_shrinks_the_margin(self):
        """The registration calls the independent margin conservative.

        That claim is only true if a positive correlation between two arms at
        matched seeds reduces the variance of their gap. If this ever inverts,
        the registration's "conservative" wording is wrong and the fleet number
        is understated rather than safe.
        """
        c_a, c_b = 0.04, 0.03
        independent = delta_min(c_a, c_b, 10, rho=0.0)
        correlated = delta_min(c_a, c_b, 10, rho=0.6)
        assert correlated < independent
        assert delta_min(c_a, c_b, 10, rho=-0.5) > independent

    def test_perfectly_correlated_equal_arms_have_no_gap_variance(self):
        """Two identical arms sharing a draw cannot disagree."""
        assert delta_min(0.05, 0.05, 3, rho=1.0) == pytest.approx(0.0)

    def test_variance_never_goes_negative(self):
        """A pathological rho must not produce a sqrt of a negative number."""
        assert delta_min(0.05, 0.05, 3, rho=1.5) == 0.0


class TestRangeConversion:
    def test_a_range_over_three_is_not_a_range_over_twelve(self):
        """The correction that changes how every published solar spread reads.

        Same underlying sigma, different seed count, different expected range.
        If these two ever came out equal the conversion would be a no-op and the
        issue's central methodological claim would be empty.
        """
        assert range_to_cv(0.1379, 3) != pytest.approx(range_to_cv(0.1379, 12))
        assert range_to_cv(0.1379, 12) < range_to_cv(0.1379, 3)

    def test_abl375_headline_converts_to_the_published_cv(self):
        """ABL-375's 13.79% range over 3 seeds is a CV of ~8.15%."""
        assert range_to_cv(0.1379, 3) == pytest.approx(0.0815, abs=5e-4)

    def test_d2_is_increasing_in_n(self):
        """More draws, wider expected range for a fixed sigma.

        The monotonicity is what makes "a 3-seed range understates the spread a
        12-seed range would show" true, which is the sentence the issue rests on.
        """
        counts = sorted(D2)
        assert all(D2[a] < D2[b] for a, b in zip(counts, counts[1:]))

    def test_round_trip_against_a_measured_cell(self):
        """The W1 DE CatBoost geometry cell, as a consistency check.

        A 10.11% range over 12 seeds and a directly measured CV of 3.21% are two
        estimators of the same quantity on the same data. They agree to within
        the sampling error of a 12-draw range, and that agreement is the
        evidence that the conversion applied to ABL-375's numbers is sound.
        """
        assert range_to_cv(0.1011, 12) == pytest.approx(0.0321, abs=0.004)


class TestVarianceSplit:
    def test_seed_and_window_variance_are_recovered_separately(self):
        """Constructed data with a known split.

        Three windows whose levels differ by a factor of e (so sd_window in logs
        is large) and an identical within-window pattern (so sd_seed is
        identical across them). The point of scope item 3 is that these two do
        not contaminate each other.
        """
        pattern = [-0.01, 0.0, 0.01]
        logs = {
            "W1": [math.log(100) + p for p in pattern],
            "W2": [math.log(100 * math.e) + p for p in pattern],
            "W3": [math.log(100 * math.e ** 2) + p for p in pattern],
        }
        split = _variance_split(logs)
        assert split["sd_seed_log"] == pytest.approx(_sd(pattern), rel=1e-9)
        assert split["sd_window_log"] == pytest.approx(1.0, rel=1e-9)
        # Seasonal level dwarfs the seed noise here, which is exactly why the
        # registration refuses to pool the two.
        assert split["seed_share_of_variance"] < 0.01

    def test_identical_windows_leave_only_seed_variance(self):
        logs = {"W1": [0.0, 0.1, 0.2], "W2": [0.0, 0.1, 0.2]}
        split = _variance_split(logs)
        assert split["sd_window_log"] == pytest.approx(0.0, abs=1e-12)
        assert split["seed_share_of_variance"] == pytest.approx(1.0)

    def test_sd_is_the_sample_sd(self):
        """ddof=1, as registered. The population sd would understate every CV."""
        assert _sd([1.0, 2.0, 3.0]) == pytest.approx(1.0)
        assert math.isnan(_sd([1.0]))


class TestPercentile:
    def test_matches_linear_interpolation(self):
        values = [1.0, 2.0, 3.0, 4.0]
        assert _percentile(values, 0) == 1.0
        assert _percentile(values, 100) == 4.0
        assert _percentile(values, 50) == pytest.approx(2.5)
        # (n-1)*0.9 = 2.7 -> between 3.0 and 4.0, seven tenths of the way.
        assert _percentile(values, 90) == pytest.approx(3.7)

    def test_single_unit_is_its_own_percentile(self):
        assert _percentile([0.042], 90) == 0.042


class TestDriverReaderNaming:
    """The reader has to find what the driver wrote.

    Generated through the driver rather than hardcoded, so a rename on either
    side fails here instead of producing an empty sweep that reads as "no cells
    found" - or worse, a partial one that reads as a real result.
    """

    @pytest.mark.parametrize(
        "forecast_type,arms,countries",
        [
            ("solar", ["control", "geometry"], ["AT", "BE", "DE", "FR"]),
            ("wind_onshore", ["control"], ["AT", "BE", "DE", "FR"]),
            ("wind_offshore", ["control"], ["BE", "FR"]),
            ("biomass", ["control"], ["BE", "FR"]),
            ("hydro_total", ["control"], ["BE", "FR"]),
        ],
    )
    @pytest.mark.parametrize("algorithm", ["catboost", "xgboost"])
    @pytest.mark.parametrize("tag", ["W1", "W6", "abl-2023-01-01", "abl-2026-03-01"])
    def test_every_registered_output_name_parses_back(
        self, forecast_type, arms, countries, algorithm, tag
    ):
        _, _, expected = _invocation(
            Path("reports/abl_385_sweep"), forecast_type, countries, algorithm,
            "42,1337", arms, "2026-07-13", "2026-08-11", tag, "2023-01-01",
        )
        parsed = parse_filename(expected)
        assert parsed == (tag, forecast_type, algorithm)

    def test_underscored_type_names_do_not_eat_the_tag(self):
        """`wind_onshore` and `hydro_total` contain the separator.

        A naive split on "_" would read `holdout_W1_wind_onshore_catboost` as
        tag `W1_wind` and lose the type, which would silently collapse two
        streams into one bucket in the fleet percentile.
        """
        assert parse_filename(Path("holdout_W1_wind_onshore_catboost.json")) == (
            "W1", "wind_onshore", "catboost",
        )
        assert parse_filename(Path("holdout_W3_hydro_total_xgboost.json")) == (
            "W3", "hydro_total", "xgboost",
        )

    def test_solar_names_carry_no_type_and_a_cleaned_suffix(self):
        assert parse_filename(Path("holdout_W2_catboost_cleaned.json")) == (
            "W2", "solar", "catboost",
        )

    def test_unrelated_files_are_ignored_rather_than_guessed_at(self):
        """The sweep directory is not guaranteed to hold only sweep output."""
        assert parse_filename(Path("abl_385_decision_margin.json")) is None
        assert parse_filename(Path("holdout_W1_lightgbm.json")) is None


class TestRegistrationProvenance:
    """Scope item 1 is "registration committed before the first fit, git
    timestamp as the evidence". The pack states that in its header, so the
    statement has to be computed from the two timestamps rather than typed.

    The failure that matters is the silent one: a pack that prints "frozen
    before the first registered fit" over a sweep that actually started first
    would launder a protocol violation into evidence. So the ordering is
    asserted in both directions.
    """

    COMMITTED_AT = "2026-08-13T13:14:40+02:00"

    def _sweep(self, tmp_path, offset_minutes, monkeypatch, dirty=""):
        """A one-file sweep whose fit sits `offset_minutes` from the commit."""
        import scripts.abl385_read_margin as reader

        monkeypatch.setattr(
            reader, "_git",
            lambda *a: (f"abc123def456\t{self.COMMITTED_AT}" if "log" in a else dirty),
        )
        fit = tmp_path / "holdout_W1_catboost_cleaned.json"
        fit.write_text("{}", encoding="utf-8")
        when = (datetime.fromisoformat(self.COMMITTED_AT)
                + timedelta(minutes=offset_minutes)).timestamp()
        os.utime(fit, (when, when))
        return reader.registration_provenance(tmp_path)

    def test_a_fit_after_the_commit_reads_as_ordered(self, tmp_path, monkeypatch):
        prov = self._sweep(tmp_path, +6, monkeypatch)
        assert prov["ordering"].startswith("ORDERED")
        assert "6.0 min" in prov["ordering"]
        assert prov["commit"] == "abc123def456"

    def test_a_fit_before_the_commit_reads_as_violated(self, tmp_path, monkeypatch):
        """The case the header must never launder into evidence."""
        prov = self._sweep(tmp_path, -6, monkeypatch)
        assert prov["ordering"].startswith("VIOLATED")
        assert "ORDERED" not in prov["ordering"]

    def test_a_modified_registration_is_reported_as_modified(self, tmp_path, monkeypatch):
        """A dirty config.json means the numbers came from an unfrozen copy."""
        prov = self._sweep(tmp_path, +6, monkeypatch,
                           dirty=" M experiments/ABL385/config.json")
        assert prov["registration_matches_commit"] is False
        assert "MODIFIED" in prov["working_tree_note"]

    def test_an_explicit_commit_overrides_but_the_ordering_still_stands(
            self, tmp_path, monkeypatch):
        import scripts.abl385_read_margin as reader

        monkeypatch.setattr(
            reader, "_git",
            lambda *a: (f"abc123def456\t{self.COMMITTED_AT}" if "log" in a else ""),
        )
        fit = tmp_path / "holdout_W1_catboost_cleaned.json"
        fit.write_text("{}", encoding="utf-8")
        when = (datetime.fromisoformat(self.COMMITTED_AT)
                + timedelta(minutes=3)).timestamp()
        os.utime(fit, (when, when))
        prov = reader.registration_provenance(tmp_path, override="deadbeef")
        assert prov["commit"] == "deadbeef"
        assert prov["ordering"].startswith("ORDERED")

    def test_an_empty_sweep_is_unknown_rather_than_ordered(self, tmp_path, monkeypatch):
        """No fits is not evidence of good ordering."""
        import scripts.abl385_read_margin as reader

        monkeypatch.setattr(
            reader, "_git",
            lambda *a: (f"abc123def456\t{self.COMMITTED_AT}" if "log" in a else ""),
        )
        prov = reader.registration_provenance(tmp_path)
        assert prov["ordering"].startswith("UNKNOWN")


class TestMarginPrecision:
    """The margin is an estimate. Section 1 of the pack quotes an interval for
    it, and that interval rests on a chi-square quantile computed by hand.

    The reader is deliberately dependency-free - the arithmetic behind a number
    a gate cites should be visible in one file - so the quantile is a
    Wilson-Hilferty approximation. That is a defensible choice only while it is
    actually close, so it is pinned against scipy's exact values here. scipy is
    in the rail venv but not imported by the reader.
    """

    #: scipy.stats.chi2.ppf, rail venv, scipy 1.18.0. The two dof that matter:
    #: 11 for a single 12-seed cell, 66 for a pair pooled over six windows.
    EXACT = {
        (0.975, 11): 21.9200492610212,
        (0.025, 11): 3.8157482522360993,
        (0.975, 66): 90.34890415884094,
        (0.025, 66): 45.43136314545968,
    }

    @pytest.mark.parametrize("p,dof", list(EXACT))
    def test_quantile_tracks_scipy_within_one_percent(self, p, dof):
        approx = _chi2_quantile(p, dof)
        exact = self.EXACT[(p, dof)]
        assert abs(approx - exact) / exact < 0.01

    def test_pooling_windows_tightens_the_interval(self):
        """The stated reason the registration reads six windows, not one.

        If this ever stopped holding, the pack's claim that pooling buys
        precision would be decoration rather than a result.
        """
        cell_low, cell_high = cv_interval(0.03, 11)
        pooled_low, pooled_high = cv_interval(0.03, 66)
        assert (cell_high - cell_low) > (pooled_high - pooled_low)
        # The figures quoted in section 1 of the pack.
        assert cell_low / 0.03 == pytest.approx(0.71, abs=0.02)
        assert cell_high / 0.03 == pytest.approx(1.70, abs=0.03)
        assert pooled_low / 0.03 == pytest.approx(0.86, abs=0.02)
        assert pooled_high / 0.03 == pytest.approx(1.19, abs=0.02)

    def test_the_interval_brackets_the_point_estimate(self):
        low, high = cv_interval(0.042, 66)
        assert low < 0.042 < high

    def test_a_degenerate_dof_is_not_an_interval(self):
        low, high = cv_interval(0.03, 0)
        assert math.isnan(low) and math.isnan(high)
