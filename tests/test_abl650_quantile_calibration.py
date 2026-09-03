"""ABL-650: the p10-p90 band must cover what its label claims.

The constraint the Board attached to this fix is that the served point forecast
is bit-identical before and after. That is checked here numerically on every
shape the calibration is applied in -- a frame, a raw array, and the dict the
serving path actually passes -- rather than argued from the algebra.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from src.quantile_calibration import (  # noqa: E402
    QCOLS,
    CalibrationRegistrationError,
    InsufficientCalibrationDataError,
    REGISTRY_PATH,
    ZoneCalibration,
    apply_zone_calibration,
    calibrate_quantile_array,
    calibrate_quantile_dict,
    fit_zone_calibration,
    load_registry,
    registered_calibration,
    verify_median_invariant,
)

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _frame(n=400, seed=0, spread=1.0, zones=("FR", "PT")):
    """Rows whose actuals come from a wider distribution than the emitted band
    -- the defect this module exists to correct."""
    rng = np.random.default_rng(seed)
    rows = []
    for zi, zone in enumerate(zones):
        centre = rng.normal(0, 500, n)
        for i in range(n):
            row = {"country_code": zone,
                   "generated_at": pd.Timestamp("2026-08-05") + pd.Timedelta(days=i // 24),
                   "target_ts": pd.Timestamp("2026-08-07") + pd.Timedelta(hours=i),
                   "actual": centre[i] + rng.normal(0, 300 * (1 + zi))}
            for q in LEVELS:
                # A narrow symmetric band around `centre`: sd 100 where the
                # actuals carry 300+.
                row[f"q{int(q * 100)}"] = centre[i] + spread * 100 * np.sqrt(2) * \
                    float(np.round(_probit(q), 6))
            row["forecast_value"] = row["q50"]
            rows.append(row)
    return pd.DataFrame(rows)


def _probit(p):
    from math import erf, sqrt
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# the constraint: the median cannot move
# ---------------------------------------------------------------------------

def test_apply_leaves_the_median_and_the_point_row_bit_identical():
    df = _frame()
    cal = {"FR": ZoneCalibration(1.7, 2.3, 400, 0.1, 0.1),
           "PT": ZoneCalibration(0.6, 3.1, 400, 0.1, 0.1)}
    out = apply_zone_calibration(df, cal)
    assert verify_median_invariant(df, out) == 0.0
    # Bit-identical, not merely close: the served point row is a separate column
    # and the calibration must not reach it at all.
    assert (out["q50"].to_numpy() == df["q50"].to_numpy()).all()
    assert (out["forecast_value"].to_numpy() == df["forecast_value"].to_numpy()).all()


def test_array_and_dict_forms_also_fix_the_median():
    rng = np.random.default_rng(3)
    block = np.sort(rng.normal(0, 100, (len(LEVELS), 24)), axis=0)
    cal = ZoneCalibration(1.4, 0.8, 100, 0.1, 0.1)
    out = calibrate_quantile_array(block, LEVELS, cal)
    med = LEVELS.index(0.5)
    assert (out[med] == block[med]).all()
    # every other level moved, or the multiplier did nothing
    assert not np.allclose(out[0], block[0])


def test_array_form_refuses_a_block_with_no_median_anchor():
    block = np.zeros((2, 4))
    with pytest.raises(ValueError, match="0.5 level as the anchor"):
        calibrate_quantile_array(block, [0.1, 0.9],
                                 ZoneCalibration(1.2, 1.2, 10, 0.1, 0.1))


# ---------------------------------------------------------------------------
# the map itself
# ---------------------------------------------------------------------------

def test_calibration_cannot_create_a_crossing():
    df = _frame(seed=5)
    cal = {z: ZoneCalibration(2.5, 0.3, 400, 0.1, 0.1) for z in ("FR", "PT")}
    out = apply_zone_calibration(df, cal)
    assert not (np.diff(out[list(QCOLS)].to_numpy(), axis=1) < 0).any()


def test_an_unregistered_zone_passes_through_untouched():
    df = _frame(zones=("FR", "XX"))
    out = apply_zone_calibration(df, {"FR": ZoneCalibration(2.0, 2.0, 400, 0.1, 0.1)})
    xx_before = df[df["country_code"] == "XX"][list(QCOLS)].to_numpy()
    xx_after = out[out["country_code"] == "XX"][list(QCOLS)].to_numpy()
    assert (xx_before == xx_after).all()


def test_the_fit_reaches_its_target_coverage_in_sample():
    """A sanity floor, explicitly in-sample: a calibration that cannot hit its
    own target on the rows it was fitted to is broken before out-of-sample
    behaviour is even a question."""
    df = _frame(n=600, seed=11)
    cal = fit_zone_calibration(df, ["FR"], alpha_lo=0.10, alpha_hi=0.10)
    out = apply_zone_calibration(df[df["country_code"] == "FR"], cal)
    cov = ((out["actual"] >= out["q10"]) & (out["actual"] <= out["q90"])).mean()
    assert 0.78 <= cov <= 0.86, cov


def test_the_fit_refuses_when_there_is_not_enough_data():
    df = _frame(n=3, seed=2, zones=("FR",))
    with pytest.raises(InsufficientCalibrationDataError):
        fit_zone_calibration(df, ["FR"], alpha_lo=0.001, alpha_hi=0.001)


def test_a_degenerate_zero_width_band_is_left_out_of_the_fit():
    """GR's stored band collapses to ~0 MW. Those rows carry no spread
    information and would send the normalised score to infinity."""
    df = _frame(n=200, seed=7, zones=("FR",))
    flat = df.head(50).index
    for col in QCOLS:
        df.loc[flat, col] = df.loc[flat, "q50"]
    cal = fit_zone_calibration(df, ["FR"])["FR"]
    assert cal.n_fit == len(df) - 50
    assert np.isfinite(cal.s_lo) and np.isfinite(cal.s_hi)


# ---------------------------------------------------------------------------
# the registration
# ---------------------------------------------------------------------------

def test_the_shipped_registration_loads_and_composes():
    models = load_registry()
    assert "chronos-2-V010" in models
    v016 = models["chronos-2-V016"]
    # V016's band is an affine image of the champion's, so it inherits the
    # champion's widening. The applied increment times the champion's applied
    # multiplier has to reproduce the total V016's band actually needs -- this
    # is the check that stops a change to one registration silently
    # double-widening the other.
    assert v016["upstream"] == "chronos-2-V010"
    for side in ("lo", "hi"):
        composed = v016[f"s_{side}_applied"] * models["chronos-2-V010"][f"s_{side}_applied"]
        assert abs(composed - v016[f"s_{side}_total"]) < 1e-3


def test_only_the_two_models_that_emit_quantiles_are_registered():
    """`baseline-V012` and `xgboost-V014` write zero rows to
    `forecast_quantiles` (ABL-595 s7a). Registering a band multiplier for a
    model with no band would be a number with nothing behind it."""
    models = load_registry()
    assert set(models) == {"chronos-2-V010", "chronos-2-V016"}


def test_a_missing_registry_means_no_calibration_not_an_error(tmp_path):
    assert load_registry(tmp_path / "absent.json") == {}
    assert registered_calibration("chronos-2-V010", ["FR"],
                                  tmp_path / "absent.json") == {}


def test_a_per_zone_registration_is_refused(tmp_path):
    """ABL-650 measured per-zone multipliers losing to no calibration at all on
    held-out vintages. The registry refuses the mode rather than leaving the
    losing option one edit away."""
    p = tmp_path / "reg.json"
    p.write_text(json.dumps({"models": {"m": {
        "mode": "per_zone", "alpha": 0.1, "s_lo_applied": 1.1,
        "s_hi_applied": 1.1, "fit_window": ["a", "b"], "fit_vintages": 20}}}),
        encoding="utf-8")
    with pytest.raises(CalibrationRegistrationError, match="per_zone"):
        load_registry(p)


def test_a_broken_composition_is_refused(tmp_path):
    p = tmp_path / "reg.json"
    p.write_text(json.dumps({"models": {
        "up": {"mode": "pooled", "alpha": 0.1, "s_lo_applied": 1.1,
               "s_hi_applied": 1.1, "s_lo_total": 1.1, "s_hi_total": 1.1,
               "fit_window": ["a", "b"], "fit_vintages": 20},
        "down": {"mode": "pooled", "alpha": 0.1, "s_lo_applied": 1.3,
                 "s_hi_applied": 1.3, "s_lo_total": 1.3, "s_hi_total": 1.3,
                 "fit_window": ["a", "b"], "fit_vintages": 20,
                 "upstream": "up"}}}), encoding="utf-8")
    with pytest.raises(CalibrationRegistrationError, match="s_lo_applied"):
        load_registry(p)


def test_calibrate_quantile_dict_applies_the_registered_multipliers():
    values = {q: np.full(24, 100.0 * (q - 0.5)) for q in LEVELS}
    out, cal = calibrate_quantile_dict(dict(values), "chronos-2-V010", "FR")
    assert cal is not None
    assert (out[0.5] == values[0.5]).all()
    spec = load_registry()["chronos-2-V010"]
    assert np.allclose(out[0.9], values[0.5] + spec["s_hi_applied"]
                       * (values[0.9] - values[0.5]))
    assert np.allclose(out[0.1], values[0.5] - spec["s_lo_applied"]
                       * (values[0.5] - values[0.1]))


def test_an_unregistered_model_serves_the_band_it_emitted():
    values = {q: np.full(4, float(q)) for q in LEVELS}
    out, cal = calibrate_quantile_dict(dict(values), "xgboost-V014", "FR")
    assert cal is None
    assert all((out[q] == values[q]).all() for q in LEVELS)


# ---------------------------------------------------------------------------
# nothing writes an uncalibrated band by accident
# ---------------------------------------------------------------------------

WRITERS_EXEMPT = {
    # Reconstructions replay a stored vintage as it was served on the day. A
    # calibration applied on replay would make the reconstruction differ from
    # the row it is supposed to reproduce, which is the opposite of the job.
    "scripts/reconstruct_v010_vintages.py":
        "serve-faithful replay of an already-served vintage",
    "scripts/apply_v016_to_vintages.py":
        "backtest replay over historical champion vintages",
    "src/db.py": "the storage primitive; calibration is the caller's decision "
                 "so a replay can opt out",
}


# A file reaches `forecast_quantiles` either by issuing the SQL itself or by
# calling the storage primitive that issues it. The champion's own serving path
# does the latter -- `scripts/forecast_chronos2.py` contains no INSERT at all --
# so a sweep keyed on the raw SQL alone is blind to the one path whose band is
# actually drawn on the dashboard. Both markers, or the guard cannot see the
# case it was written for.
QUANTILE_WRITE_MARKERS = (
    "INSERT OR REPLACE INTO forecast_quantiles",
    "save_quantile_forecasts",
)


def _swept_sources():
    """`{repo-relative path: source}` for every module the sweep looks at."""
    out = {}
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(("tests/", ".claude/")) or "__pycache__" in rel:
            continue
        out[rel] = path.read_text(encoding="utf-8", errors="ignore")
    return out


def _writes_quantiles(src):
    return any(marker in src for marker in QUANTILE_WRITE_MARKERS)


def _is_offender(rel, src):
    return (_writes_quantiles(src) and rel not in WRITERS_EXEMPT
            and "quantile_calibration" not in src)


def test_every_forecast_quantiles_writer_calibrates_or_says_why_not():
    """A new serving path that forgets the calibration would ship a band
    labelled p10-p90 that is not one -- the exact defect this issue is about,
    reintroduced silently."""
    offenders = [rel for rel, src in _swept_sources().items()
                 if _is_offender(rel, src)]
    assert not offenders, (
        f"{offenders} write to forecast_quantiles (directly, or through "
        f"save_quantile_forecasts) without importing src.quantile_calibration. "
        f"Either calibrate the band or add the file to WRITERS_EXEMPT with the "
        f"reason.")


def test_the_sweep_sees_the_champions_own_serving_path():
    """`scripts/forecast_chronos2.py` writes the band that is actually drawn on
    the Net position tab, and it writes it through `save_quantile_forecasts`.
    Keying the sweep on the raw SQL alone left that file outside the guard
    entirely -- it passed by not matching, not by calibrating."""
    rel = "scripts/forecast_chronos2.py"
    src = (REPO / rel).read_text(encoding="utf-8")
    assert "INSERT OR REPLACE INTO forecast_quantiles" not in src, (
        "the raw-SQL marker now matches this file, so it no longer witnesses "
        "the gap the primitive marker was added to close")
    assert _writes_quantiles(src)
    assert not _is_offender(rel, src)
    # Negative control: the file passes because it calibrates, not because the
    # sweep cannot see it. Drop the import and the sweep has to catch it.
    assert _is_offender(rel, src.replace("quantile_calibration", "_removed_"))


def test_both_write_markers_match_a_real_file():
    """A marker that matches nothing is a guard that can never fire."""
    sources = _swept_sources()
    for marker in QUANTILE_WRITE_MARKERS:
        assert any(marker in src for src in sources.values()), (
            f"no swept file contains {marker!r}; the sweep is vacuous on that "
            f"marker and would pass whatever the writers do")


def test_the_serving_entry_points_call_the_calibration():
    for rel in ("scripts/forecast_chronos2.py", "scripts/forecast_challengers.py"):
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "calibrate_quantile_dict" in called, rel


def test_the_registry_path_resolves_inside_the_repo():
    assert REGISTRY_PATH == REPO / "experiments" / "net_position_quantile_calibration.json"
    assert REGISTRY_PATH.exists()


# ---------------------------------------------------------------------------
# the pre-registered revert
# ---------------------------------------------------------------------------

# ABL-674 registers the revert as: delete
# `experiments/net_position_quantile_calibration.json`. That restores the
# serving behaviour exactly -- `load_registry` reads a missing file as "no
# calibration", so every model serves the band its head emitted -- but it does
# NOT leave the suite green, because the tests below pin the *shipped*
# registration on purpose. Executed on the merged tree, the deletion gives
# 5 failed / 26 passed across the two affected files.
#
# Both halves matter. An accidental deletion has to stay loud, so these are not
# skipped. A deliberate one happens under time pressure inside the 10-vintage
# watch window, so the expected red is written down instead of discovered:
# `reports/abl_650_band_calibration.md` section 8.
DEREGISTRATION_REDLIST = {
    "tests/test_abl650_quantile_calibration.py::"
    "test_the_shipped_registration_loads_and_composes",
    "tests/test_abl650_quantile_calibration.py::"
    "test_only_the_two_models_that_emit_quantiles_are_registered",
    "tests/test_abl650_quantile_calibration.py::"
    "test_calibrate_quantile_dict_applies_the_registered_multipliers",
    "tests/test_abl650_quantile_calibration.py::"
    "test_the_registry_path_resolves_inside_the_repo",
    "tests/test_challenger_rail.py::"
    "test_v016_applies_the_fit_and_keeps_quantiles_ordered",
}

# Reads the shipped registry and passes without it: an unregistered model is
# served exactly as emitted whether the file is there or not.
DEREGISTRATION_STILL_GREEN = {
    "tests/test_abl650_quantile_calibration.py::"
    "test_an_unregistered_model_serves_the_band_it_emitted",
}

# `{function: index of the `path` parameter}`. A call that supplies a registry
# path is pointing at a tmp file, not at the shipped registration.
_DEFAULT_REGISTRY_READERS = {"load_registry": 0, "registered_calibration": 2,
                             "calibrate_quantile_dict": 3}


def _reads_the_shipped_registry(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id == "REGISTRY_PATH":
            return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            pos = _DEFAULT_REGISTRY_READERS.get(n.func.id)
            if pos is None:
                continue
            if len(n.args) > pos or any(k.arg == "path" for k in n.keywords):
                continue
            return True
    return False


def _tests_that_read_the_shipped_registry():
    found = set()
    for path in sorted((REPO / "tests").glob("test_*.py")):
        src = path.read_text(encoding="utf-8")
        if "quantile_calibration" not in src:
            continue
        rel = path.relative_to(REPO).as_posix()
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.FunctionDef)
                    and node.name.startswith("test_")
                    and _reads_the_shipped_registry(node)):
                found.add(f"{rel}::{node.name}")
    return found


def test_the_deregistration_redlist_covers_every_test_that_reads_the_registry():
    """The revert's expected failures are derived, not remembered.

    A test added later that reads the shipped registration would go red during
    the revert with nobody having written that down, and whoever is executing
    the revert mid-incident would have to decide on the spot whether it
    misfired. This fails at the time that test is written instead."""
    documented = DEREGISTRATION_REDLIST | DEREGISTRATION_STILL_GREEN
    found = _tests_that_read_the_shipped_registry()
    assert found == documented, (
        f"unclassified: {sorted(found - documented)}; "
        f"stale: {sorted(documented - found)}. Every test that reads the "
        f"shipped registration must be listed either in DEREGISTRATION_REDLIST "
        f"(goes red when the registration is deleted) or in "
        f"DEREGISTRATION_STILL_GREEN (passes either way), and the runbook in "
        f"reports/abl_650_band_calibration.md section 8 updated to match.")
