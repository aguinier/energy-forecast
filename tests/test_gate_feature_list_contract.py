"""ABL-395: the gate harnesses' feature lists, held to ABL-394's standard.

ABL-394 made `get_feature_columns()` a reviewed decision and asserted that every
name it declares is actually *produced*. That covers `scripts/train.py`. It does
not cover the two pre-registered gate harnesses, which are what the 37 remaining
ABL-316 pairs are fitted by: they declare their own list
(`solar_retrain.FEATURE_COLUMNS`, `wind_retrain.FEATURE_COLUMNS`) and hand it to
`RenewableFeatureBuilder` through `to_vector`. Nothing reviewed that list, and it
sat two names short of an ABL-338-current solar fit from ABL-253 through ABL-381
— 25 features where an ABL-338-current fit is 27 — while declaring nothing was
missing. Measured on the ABL-381 read: CH predicted negative in **80.5%** of
night hours, the exact defect ABL-335/ABL-338 were opened for.

The two harness lists behave differently from the `train.py` path in one way that
matters, and it is why this file is separate rather than a parametrisation of
`test_feature_list_contract.py`:

    `select_feature_columns` **drops** a declared name the frame does not carry
    and warns. `to_vector` **raises** on one the builder cannot produce.

So a gate fit cannot silently run short; it either builds all 27 or dies at the
first row. What it *could* do — and did — is declare 25 and be exactly right
about it. The assertions here are therefore about the list itself: that it is
frozen where a reviewer sees it change, and that the builder really produces
every name on it for every country an ABL-316 tranche can name.

Everything is measured against a synthetic replica under a **real** country code
(`CH`, one of the two ABL-381 pairs). The geometry features are pure functions of
`(country, hour)`, so the country code has to be one `solar_geometry` knows;
`test_solar_features.py`'s `XX` would silently contribute no geometry at all,
which is the failure this file exists to catch.
"""

import importlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from src import db  # noqa: E402
from src.evaluation.gate_artifacts import save_gate_artifact  # noqa: E402
from src.evaluation.solar_retrain import (  # noqa: E402
    FEATURE_COLUMNS as SOLAR_GATE_COLUMNS,
    build_vintage_frame,
    finite_training_rows,
)
from src.evaluation.wind_retrain import FEATURE_COLUMNS as WIND_GATE_COLUMNS  # noqa: E402
from src.solar_features import SOLAR_GEOMETRY_FEATURES  # noqa: E402
from src.solar_geometry import (  # noqa: E402
    NIGHT_ELEVATION_THRESHOLD_DEG, SOLAR_REPRESENTATIVE_POINTS,
)
from src.wind_features import RenewableFeatureBuilder, to_vector  # noqa: E402

MANIFEST = json.loads(
    (Path(__file__).parent / "feature_list_manifest.json").read_text(encoding="utf-8")
)
GATE = MANIFEST["gate_harness"]

ROOT = Path(__file__).parent.parent


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Imported by path, as `test_solar_gate_unreadable.py` does: these are scripts,
#: not package modules, and the registration tables live in them.
harness = _load("scripts_evaluate_solar_retrain_abl395",
                "scripts/evaluate_solar_retrain.py")
seed_spread = _load("scripts_abl376_night_seed_spread_abl395",
                    "scripts/abl376_night_seed_spread.py")
geometry_probe = _load("scripts_abl395_geometry_feature_probe",
                       "scripts/abl395_geometry_feature_probe.py")

#: A country with a solar representative point, so the geometry features are
#: actually computable — and one of the two pairs ABL-381 read at 25 features.
COUNTRY = "CH"
OBS = pd.Timestamp("2026-08-11 08:00:00")


def _epoch_hours(ts) -> float:
    return (ts - pd.Timestamp("2000-01-01")).total_seconds() / 3600.0


@pytest.fixture
def replica(tmp_path, monkeypatch):
    path = tmp_path / "replica.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE energy_generation (country_code TEXT, timestamp_utc TIMESTAMP,
            wind_offshore_mw REAL, wind_onshore_mw REAL, solar_mw REAL,
            data_quality TEXT DEFAULT 'actual');
        CREATE TABLE energy_renewable (country_code TEXT, timestamp_utc TIMESTAMP,
            wind_offshore_mw REAL DEFAULT 0, wind_onshore_mw REAL DEFAULT 0,
            solar_mw REAL DEFAULT 0, data_quality TEXT DEFAULT 'actual');
        CREATE TABLE weather_data (country_code TEXT, timestamp_utc TIMESTAMP,
            forecast_run_time TIMESTAMP, data_quality TEXT,
            temperature_2m_k REAL, wind_speed_10m_ms REAL, wind_speed_100m_ms REAL,
            shortwave_radiation_wm2 REAL, direct_radiation_wm2 REAL,
            diffuse_radiation_wm2 REAL);
        """
    )
    # A plain diurnal series, not a constant: `exclude_suspect_constant_runs`
    # nulls 24+ hours of a bit-identical value (ABL-188), which would empty the
    # frame these tests then measure.
    for ts in pd.date_range(OBS - pd.Timedelta(days=60), OBS + pd.Timedelta(days=4), freq="h"):
        value = 500.0 + 400.0 * np.sin(np.pi * ts.hour / 12.0) + _epoch_hours(ts) % 7
        con.execute(
            "INSERT INTO energy_generation VALUES (?, ?, ?, ?, ?, 'actual')",
            (COUNTRY, str(ts), value, value, value),
        )
        con.execute(
            "INSERT INTO energy_renewable VALUES (?, ?, ?, ?, ?, 'actual')",
            (COUNTRY, str(ts), value, value, value),
        )
    run = OBS - pd.Timedelta(days=30)
    for ts in pd.date_range(OBS - pd.Timedelta(days=60), OBS + pd.Timedelta(days=4), freq="h"):
        con.execute(
            "INSERT INTO weather_data VALUES (?, ?, ?, 'forecast', ?, ?, ?, ?, ?, ?)",
            (
                COUNTRY, str(ts), str(run),
                280.0 + ts.hour, 5.0 + ts.hour * 0.1, 8.0 + ts.hour * 0.1,
                100.0 + ts.hour * 2.0, 50.0 + ts.hour, 20.0 + ts.hour * 0.5,
            ),
        )
    con.commit()
    con.close()

    monkeypatch.setenv("ENERGY_DB_PATH", str(path))
    importlib.reload(config)
    importlib.reload(db)
    yield path
    monkeypatch.undo()
    importlib.reload(config)
    importlib.reload(db)


def _builder(forecast_type="solar"):
    return RenewableFeatureBuilder(
        COUNTRY, forecast_type, OBS - pd.Timedelta(days=45),
        OBS + pd.Timedelta(days=4), actuals_source="energy_generation",
    )


# ---------------------------------------------------------------------------
# 1. The harness list is reviewed
# ---------------------------------------------------------------------------


def test_solar_gate_list_still_matches_the_reviewed_manifest():
    """Changing `solar_retrain.FEATURE_COLUMNS` must change the manifest in the
    same commit. Do not regenerate this to get green: a gate fit's feature list
    is what the artifacts of every subsequent tranche are built on, and the
    scopes already read at the old list do not move with it."""
    assert list(SOLAR_GATE_COLUMNS) == GATE["solar"]["columns"]
    assert len(SOLAR_GATE_COLUMNS) == GATE["solar"]["n"] == 27


def test_wind_gate_list_still_matches_the_reviewed_manifest():
    assert list(WIND_GATE_COLUMNS) == GATE["wind"]["columns"]
    assert len(WIND_GATE_COLUMNS) == GATE["wind"]["n"] == 24


def test_the_geometry_pair_is_solar_only_in_both_harness_lists():
    """`_solar_geometry_features` returns `{}` for any non-solar type, so a wind
    list naming one of these would raise in `to_vector` on its first row."""
    assert set(SOLAR_GEOMETRY_FEATURES) <= set(SOLAR_GATE_COLUMNS)
    assert not set(SOLAR_GEOMETRY_FEATURES) & set(WIND_GATE_COLUMNS)


# ---------------------------------------------------------------------------
# 2. Every declared name is produced — for every country a tranche can name
# ---------------------------------------------------------------------------


def test_the_builder_produces_every_declared_solar_gate_feature(replica):
    """`to_vector` in its own right: 27 names in, 27 finite floats out, in the
    declared order (which is the order the artifact records and serving rebuilds
    a row from)."""
    row = _builder().row(OBS + pd.Timedelta(days=2), OBS, OBS)
    vector = to_vector(row, SOLAR_GATE_COLUMNS)

    assert list(vector) == list(SOLAR_GATE_COLUMNS)
    assert len(vector) == 27
    assert all(np.isfinite(value) for value in vector.values())


def test_every_supported_country_has_a_solar_representative_point():
    """The precondition the remaining ABL-316 solar tranches inherit.

    `_solar_geometry_features` contributes nothing for a country absent from
    `SOLAR_REPRESENTATIVE_POINTS`, and `to_vector` then raises — so a tranche
    naming such a country would die at its first fit row rather than quietly
    fitting 25. All 24 supported countries have a point today; this fails in the
    suite instead, on the commit that adds a country without one.
    """
    missing = [c for c in config.SUPPORTED_COUNTRIES if c not in SOLAR_REPRESENTATIVE_POINTS]
    assert not missing, (
        f"no solar representative point for {missing}; a solar gate fit for "
        f"these countries raises in to_vector on the geometry features"
    )


def test_to_vector_refuses_a_declared_name_the_builder_cannot_produce(replica):
    """The difference from `select_feature_columns`, pinned.

    The `train.py` path drops an unproducible declared name and warns. The gate
    path raises. Both are defensible; what is not defensible is not knowing which
    one a harness is on, because they fail in opposite directions.
    """
    row = _builder().row(OBS + pd.Timedelta(days=2), OBS, OBS)
    with pytest.raises(KeyError, match="does not produce feature"):
        to_vector(row, [*SOLAR_GATE_COLUMNS, "a_feature_nobody_builds"])


# ---------------------------------------------------------------------------
# 3. The fitted artifact declares — and produces — 27
# ---------------------------------------------------------------------------


def test_a_gate_fit_frame_carries_all_27_and_the_geometry_columns_vary(replica):
    """A present-but-constant column is a no-op that a length check would pass.

    `is_night` must actually separate the day and `sun_elevation_deg` must move
    with the hour, or the fix is cosmetic: the mechanism ABL-338 identified is
    that nothing in the 25-name vector distinguished "0 W/m2 because the sun is
    down" from "0 W/m2 at a dark winter dawn".
    """
    start = OBS + pd.Timedelta(days=1)
    frame = build_vintage_frame(_builder(), start, start + pd.Timedelta(days=2),
                                SOLAR_GATE_COLUMNS)
    fit, audit = finite_training_rows(frame, SOLAR_GATE_COLUMNS)

    assert audit["retained_rows"] > 0
    assert set(SOLAR_GATE_COLUMNS) <= set(fit.columns)
    assert np.isfinite(fit[list(SOLAR_GATE_COLUMNS)].to_numpy(dtype=float)).all()

    assert set(fit["is_night"].unique()) == {0.0, 1.0}
    assert fit["sun_elevation_deg"].nunique() > 12
    # The two carry different information (ABL-338), which is why both earn a
    # place: `is_night` thresholds the hour's *maximum* elevation while
    # `sun_elevation_deg` is its midpoint, so at a shoulder hour the midpoint sits
    # below the night threshold on an hour the mask calls day. A tree could
    # otherwise have found `is_night` as a split on the elevation column alone.
    day_min = fit.loc[fit["is_night"] == 0.0, "sun_elevation_deg"].min()
    assert day_min < NIGHT_ELEVATION_THRESHOLD_DEG


def test_the_written_artifact_declares_the_27_it_was_fitted_on(replica, tmp_path):
    """End to end, through the harness's own writer: what the fit trained on is
    what the artifact records, and `Forecaster.load` reads that list back.

    This is the evidence ABL-395 asks for in place of a hand count — an artifact
    with 25 names and an artifact fitted at 25 are indistinguishable after the
    fact, which is exactly how ABL-253 through ABL-381 went unnoticed.
    """
    from catboost import CatBoostRegressor

    from src.forecaster import Forecaster

    builder = _builder()
    start = OBS + pd.Timedelta(days=1)
    frame = build_vintage_frame(builder, start, start + pd.Timedelta(days=2),
                                SOLAR_GATE_COLUMNS)
    fit, _ = finite_training_rows(frame, SOLAR_GATE_COLUMNS)
    params = {"iterations": 5, "depth": 2, "verbose": 0, "random_seed": 42}
    model = CatBoostRegressor(**params)
    model.fit(fit[list(SOLAR_GATE_COLUMNS)], fit["actual"])

    path = save_gate_artifact(
        tmp_path / COUNTRY / "solar" / "model.joblib", model=model, builder=builder,
        algorithm="catboost", params=params, feature_columns=SOLAR_GATE_COLUMNS,
        fit_window=(start, start + pd.Timedelta(days=2)),
    )

    assert joblib.load(path)["feature_columns"] == list(SOLAR_GATE_COLUMNS)
    loaded = Forecaster.load(COUNTRY, "solar", path=str(path))
    assert loaded.feature_columns == list(SOLAR_GATE_COLUMNS)
    assert len(loaded.feature_columns) == 27
    assert loaded.training_source == "energy_generation"


# ---------------------------------------------------------------------------
# 4. A dispositioned scope does not follow the constant (ABL-395, ABL-404)
# ---------------------------------------------------------------------------
#
# The list moving is a real change to the challenger, measured: on the two
# ABL-381 pairs at seed 42, CH's 24-36h cell moves 8.16% -> 7.78% WAPE and BG's
# 18.89% -> 19.95%. So `SCOPE_FEATURES` is the same kind of registration
# `FIT_RULES` is, and for the same stated reason -- two gate reads are not
# comparable unless both say what they trained on.
#
# ABL-404: `SCOPE_FEATURES` is *not* one of `check_registration_tables`' three,
# deliberately -- an absence here resolves through `features_for` instead of
# aborting at import. That makes this file the only thing standing between a
# published read and a silent re-base, and until ABL-404 it enumerated the scopes
# it covered by hand. It covered two of the three that needed it.
#
# So nothing below names a scope in order to decide whether it is covered.
# Coverage is derived from what the repository has actually published, and the
# expectation from what the published run recorded.


def test_the_legacy_25_is_the_current_list_minus_the_geometry_pair():
    """Derived by subtraction, never a second copy. A hand-written 25 would drift
    from the live list the moment anything else is added to it."""
    assert harness.LEGACY_FEATURE_COLUMNS == tuple(
        c for c in SOLAR_GATE_COLUMNS if c not in SOLAR_GEOMETRY_FEATURES)
    assert len(harness.LEGACY_FEATURE_COLUMNS) == 25
    assert not set(SOLAR_GEOMETRY_FEATURES) & set(harness.LEGACY_FEATURE_COLUMNS)


def _tracked(path: Path) -> bool:
    """Is `path` committed evidence, as opposed to a local run's leftovers?

    Tracked-in-git, not `exists()`. The rule below turns on whether a read has
    been *published*, and an untracked file in a working tree is by definition
    not — keying on existence would have a developer's local gate run silently
    promote an open scope to dispositioned and fail the suite for it.

    A missing `git` raises `FileNotFoundError` out of here and a broken
    invocation raises below. Neither degrades to `False`: that would empty
    `DISPOSITIONED_SCOPES`, and an empty `parametrize` is a test that reports
    green by not running -- the exact way this guard failed before.
    """
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", path.relative_to(ROOT).as_posix()],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git ls-files failed for {path}: {result.stderr.strip()}")
    return result.returncode == 0


def _read_on(record: Path) -> tuple | None:
    """The feature list a committed machine record says its run was fitted on.

    `None` when the record does not say. That is not a gap: ABL-395 added
    `meta.feature_columns` (`0a363eb`) in the same change that made the harness
    list 27 (`d045b5f`), so a record that omits the field was written by a
    harness whose list was the 25 -- every solar gate read from ABL-253 through
    ABL-381. Absence dates the read rather than losing it.
    """
    meta = json.loads(record.read_text(encoding="utf-8"))["meta"]
    columns = meta.get("feature_columns")
    return tuple(columns) if columns else None


def _dispositioned_scopes():
    """Every scope whose read is published, and the list it was read on.

    Derived from `SCOPE_OUTPUTS` — which `check_registration_tables` already
    holds equal to `SCOPES` — so a tranche registered after this file was
    written is covered on the commit that publishes it, with nothing to
    hand-maintain here. That is ABL-404: the parametrize list used to be the
    literal `["abl253", "abl376"]`, and `abl316-t1b` was simply not in it.
    """
    found = []
    for scope, paths in sorted(harness.SCOPE_OUTPUTS.items()):
        record = ROOT / paths["json_out"]
        record_published = _tracked(record)
        if not record_published and not _tracked(ROOT / paths["report_out"]):
            continue  # never read, or read and not published: still open
        found.append((scope, _read_on(record) if record_published else None))
    return found


#: `abl253` reaches this list through its report alone: `SCOPE_OUTPUTS` sends its
#: machine record to `experiments/ABL253/results.json`, which `.gitignore:53`
#: takes by exact filename (ABL-196). So it lands on the `None` branch and is held
#: to the 25 by date. Every scope registered since takes the tracked `json_out`
#: form the `SCOPE_OUTPUTS` comment recommends, and is held to its own literal
#: names instead.
DISPOSITIONED_SCOPES = _dispositioned_scopes()


@pytest.mark.parametrize("scope,read_on", DISPOSITIONED_SCOPES,
                         ids=[scope for scope, _ in DISPOSITIONED_SCOPES])
def test_a_dispositioned_scope_still_resolves_to_the_list_it_was_read_on(scope, read_on):
    """A published read must reproduce, so its challenger must not move under it.

    Re-basing one onto a new list moves published numbers with nothing in
    `git status` to show it -- the ABL-387 failure mode with a feature list in
    place of a path. Measured on the `abl316-t1b` pairs at seed 42, the move from
    25 to 27 takes BG's 24-36h cell 18.89% -> 19.95% WAPE, across its 19.15%
    hour-of-day climatology, while the run still writes ABL-381's registered
    heading and still exits 0.

    The expectation is read out of the evidence, never asserted against a
    constant: where the record states `meta.feature_columns` this compares
    against those literal names, which is a stronger pin than any table entry --
    the record is what the fit itself wrote down.
    """
    expected = harness.LEGACY_FEATURE_COLUMNS if read_on is None else read_on
    assert harness.features_for(scope) == expected, (
        f"scope {scope!r} is published evidence read on {len(expected)} features "
        f"but resolves to {len(harness.features_for(scope))} today. Pin it in "
        f"`SCOPE_FEATURES` to the list it was read on -- a scoped run writes over "
        f"`{harness.SCOPE_OUTPUTS[scope]['report_out']}` in place and exits 0."
    )


def test_the_dispositioned_set_is_derived_and_still_covers_the_published_reads():
    """The floor under the derivation, because its failure mode is silence.

    `_dispositioned_scopes` returning `[]` would take the parametrised test above
    with it and report green. These four reads are cited in committed evidence
    packs, so they can only leave this set by being retired, which is a review
    event and should look like one.
    """
    covered = {scope for scope, _ in DISPOSITIONED_SCOPES}
    assert {"abl253", "abl376", "abl316-t1b", "abl316-t2a"} <= covered
    assert covered <= set(harness.SCOPES)


def test_a_published_read_that_recorded_its_own_list_needs_no_scope_features_row():
    """The distinction the rule turns on: *dispositioned*, not *pinned*.

    `abl316-t2a` is deliberately absent from `SCOPE_FEATURES` and inherits the
    27 -- fitting that tranche at 27 was the point of it (ABL-405). A guard that
    required every registered scope to be pinned would be wrong and would fail
    here, so the guard requires only that the resolved list still match the
    published one. This scope's record states its 27 literal names, so moving
    `FEATURE_COLUMNS` to 28 fails the test above without anyone maintaining a
    row for it.
    """
    assert "abl316-t2a" not in harness.SCOPE_FEATURES
    read_on = dict(DISPOSITIONED_SCOPES)["abl316-t2a"]
    assert read_on is not None and len(read_on) == 27
    assert harness.features_for("abl316-t2a") == read_on == tuple(SOLAR_GATE_COLUMNS)


def test_a_scope_that_registers_no_feature_set_gets_the_current_27():
    """The point of the change: a new ABL-316 tranche is fitted at 27 without
    touching this table, which is what unblocks the remaining 33."""
    assert harness.features_for("a-scope-nobody-registered") == tuple(SOLAR_GATE_COLUMNS)
    assert len(harness.features_for("a-scope-nobody-registered")) == 27


def test_every_registered_scope_has_a_feature_set_that_is_one_of_the_two():
    """A `SCOPE_FEATURES` entry naming something else is a silent third protocol.
    `features_for` would serve it happily; this is what makes it a review event."""
    for scope in harness.SCOPES:
        resolved = harness.features_for(scope)
        assert resolved in (harness.LEGACY_FEATURE_COLUMNS, tuple(SOLAR_GATE_COLUMNS)), scope


# ---------------------------------------------------------------------------
# 5. The ABL-376 probe's two arms stay 25 and 27, and never 29
# ---------------------------------------------------------------------------


def test_the_seed_spread_arms_are_the_two_lists_with_no_duplicates():
    """`abl376_night_seed_spread.py --with-geometry` used to *append* the geometry
    pair to `FEATURE_COLUMNS`. Once ABL-395 put them in that constant, the old
    form would hand CatBoost both columns twice and relabel the registered arm
    `legacy25` while fitting 27. Neither raises; both are wrong."""
    assert seed_spread.LEGACY_FEATURE_COLUMNS == harness.LEGACY_FEATURE_COLUMNS
    for arm in (seed_spread.LEGACY_FEATURE_COLUMNS, tuple(SOLAR_GATE_COLUMNS)):
        assert len(set(arm)) == len(arm)
    assert len(seed_spread.LEGACY_FEATURE_COLUMNS) == 25
    assert len(SOLAR_GATE_COLUMNS) == 27


def test_the_two_arms_of_the_abl395_probe_are_the_same_two_lists():
    assert geometry_probe.ARMS["f25"] == harness.LEGACY_FEATURE_COLUMNS
    assert geometry_probe.ARMS["f27"] == tuple(SOLAR_GATE_COLUMNS)


def test_the_abl395_sweep_reuses_abl376s_registered_seeds_verbatim():
    """Restated in the probe rather than imported (a script importing a script),
    so the two are pinned equal here instead of being allowed to drift. Reusing a
    seed set frozen before another issue's first fit is what makes it a set this
    issue's effect was not selected on -- and it keeps the two solar seed reads
    commensurable."""
    assert geometry_probe.SWEEP_SEEDS == seed_spread.SEEDS
    assert geometry_probe.GATE_SEED not in geometry_probe.SWEEP_SEEDS
