"""ABL-345: the solar gate harness must read the table the run named.

`scripts/evaluate_solar_retrain.py` is the harness behind **19 of the 37**
remaining ABL-316 pairs — the largest tranche. Until this issue it had no source
argument at all: it built its `RenewableFeatureBuilder` with no `actuals_source`
and screened contamination with a hardcoded `FROM energy_renewable`. ABL-342
made the *artifact* truthful about what was read; it did not give the harness a
way to read anything else.

Two read sites, and they are the ones this file pins:

1. **The builder.** Everything scored flows from it — the fitted target series,
   every lag and rolling feature, the D-7 and persistence baselines
   (`attach_baselines(..., builder._actuals)`) and the gate actuals. So the
   source is not one of several inputs; it selects the whole experiment.
2. **`_constant_runs`.** Its output drives `verdict` directly: a non-empty list
   turns a PASS into "PERFORMANCE PASS — HOLD FOR CONTAMINATION ADJUDICATION".
   Screening a table the model was never fitted on therefore moves the
   harness's disposition, not just its prose.

Why the assertion in `test_the_builder_is_handed_...` is on the *argument*
rather than on a scored number: the harness's own read sites are the defect
boundary, and `save_gate_artifact` (ABL-342, `tests/test_gate_artifact_writer.py`)
already proves that `builder.actuals_source` is what reaches the artifact and
then serving. Re-proving that here would need a full replica, weather archive,
sidecar and CatBoost fit to re-derive a property that is already pinned. What
was *not* pinned anywhere is that `main()` passes the flag through at all.

`_constant_runs` is the opposite case and is tested for real, on a replica whose
two tables deliberately disagree: `energy_renewable` carries a zero-fill run
that `energy_generation` contradicts — ABL-188's exact shape — so a screen that
reads the wrong table returns a demonstrably wrong list rather than a matching
one.
"""
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import db  # noqa: E402

COUNTRY = "XX"
START = pd.Timestamp("2026-01-01")
END = pd.Timestamp("2026-01-08")


def _load_harness():
    """Import `scripts/evaluate_solar_retrain.py` as a module object.

    Loaded under a name of its own so its `if __name__ == '__main__'` guard
    stays shut. Unlike `scripts/train.py` (ABL-340) this script imports `src` as
    a package, so no flat-name aliasing is needed.
    """
    spec = importlib.util.spec_from_file_location(
        "scripts_evaluate_solar_retrain", ROOT / "scripts" / "evaluate_solar_retrain.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


@pytest.fixture
def replica(tmp_path):
    """A replica whose two solar series disagree in the ABL-188 shape.

    `energy_renewable` holds a 48-hour run of exact 0.0 — long enough to trip
    `find_suspect_constant_runs`' 24-hour floor. `energy_generation` holds real
    varying generation over the identical instants, which is what makes the run
    contamination rather than a measured night. Empty `forecasts` and
    `energy_generation_forecast` tables exist so the scorecard loaders `main()`
    calls before the builder can run against this file.
    """
    path = tmp_path / "replica.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE energy_generation (country_code TEXT, timestamp_utc TIMESTAMP,
            solar_mw REAL, wind_onshore_mw REAL, wind_offshore_mw REAL);
        CREATE TABLE energy_renewable (country_code TEXT, timestamp_utc TIMESTAMP,
            solar_mw REAL DEFAULT 0, wind_onshore_mw REAL DEFAULT 0,
            wind_offshore_mw REAL DEFAULT 0);
        CREATE TABLE forecasts (country_code TEXT, forecast_type TEXT,
            target_timestamp_utc TIMESTAMP, generated_at TIMESTAMP,
            horizon_hours REAL, forecast_value REAL, model_name TEXT);
        CREATE TABLE energy_generation_forecast (country_code TEXT,
            target_timestamp_utc TIMESTAMP, forecast_type TEXT, solar_mw REAL);
        """
    )
    hours = pd.date_range(START, periods=48, freq="h")
    con.executemany(
        "INSERT INTO energy_renewable (country_code, timestamp_utc, solar_mw) VALUES (?,?,?)",
        [(COUNTRY, str(ts), 0.0) for ts in hours],
    )
    con.executemany(
        "INSERT INTO energy_generation (country_code, timestamp_utc, solar_mw) VALUES (?,?,?)",
        [(COUNTRY, str(ts), 100.0 + index) for index, ts in enumerate(hours)],
    )
    con.commit()
    con.close()
    return path


def test_the_contamination_screen_reads_the_table_that_was_fitted(replica):
    """The planted run exists in one table only, so the two screens disagree."""
    renewable = harness._constant_runs(str(replica), COUNTRY, START, END, "energy_renewable")
    generation = harness._constant_runs(str(replica), COUNTRY, START, END, "energy_generation")

    assert len(renewable) == 1
    assert renewable[0]["value"] == 0.0
    assert renewable[0]["duration_hours"] >= 24.0
    # Not "also finds a run": the fitted series has none, and reporting
    # `energy_renewable`'s would hold an `energy_generation` gate read for
    # adjudication over contamination in a table it never opened.
    assert generation == []


def test_the_screen_refuses_a_table_it_does_not_know(replica):
    """The table name is interpolated into SQL, so it is checked, not trusted.

    A typo must raise here rather than reach SQLite — an unknown name would
    otherwise be an OperationalError at best and, for any other real table with
    a `solar_mw` column, a silently wrong screen at worst.
    """
    with pytest.raises(ValueError, match="unknown renewable source table"):
        harness._constant_runs(str(replica), COUNTRY, START, END, "energy_renewables")


class _BuilderSpy(Exception):
    """Records the builder's construction arguments and stops the run there.

    Raising is deliberate: everything after the first `RenewableFeatureBuilder`
    needs a weather archive and a CatBoost fit, and none of it can change the
    argument already captured.
    """

    def __init__(self, args, kwargs):
        super().__init__("builder constructed")
        self.args = args
        self.kwargs = kwargs


def _source_main_asks_for(replica, monkeypatch, tmp_path, extra_argv):
    def spy(*args, **kwargs):
        raise _BuilderSpy(args, kwargs)

    monkeypatch.setattr(harness, "RenewableFeatureBuilder", spy)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_solar_retrain.py",
        "--replica-db", str(replica),
        # A path that does not exist, so `_load_forecasts` reads the replica's
        # empty `forecasts` table only and never touches a real sidecar.
        "--sidecar-db", str(tmp_path / "no-sidecar.db"),
        *extra_argv,
    ])
    with pytest.raises(_BuilderSpy) as caught:
        harness.main()
    return caught.value


def test_the_builder_is_handed_the_source_the_run_named(replica, monkeypatch, tmp_path):
    captured = _source_main_asks_for(replica, monkeypatch, tmp_path,
                                     ["--renewable-source", "energy_generation"])
    assert captured.kwargs["actuals_source"] == "energy_generation"


def test_an_unflagged_run_still_reads_the_default_table(replica, monkeypatch, tmp_path):
    """ABL-253 must reproduce. The default is `energy_renewable` because ABL-321
    withheld the global switch — 3 of the 10 serving pairs are materially worse
    on `energy_generation` — so an unflagged run has to read what it always did.

    Asserted as the resolved literal, not as `None`: the harness resolves once
    and records that string in `meta.training_source`, so a `None` reaching the
    builder here would mean the report could name a table the run did not read.
    """
    captured = _source_main_asks_for(replica, monkeypatch, tmp_path, [])
    assert captured.kwargs["actuals_source"] == db.RENEWABLE_TYPE_SOURCE_TABLE
    assert captured.kwargs["actuals_source"] == "energy_renewable"


def test_an_unknown_source_is_rejected_before_anything_is_fitted(replica, monkeypatch,
                                                                 capsys):
    """A typo must not reach a fit. Asserted on the message as well as the exit
    code, because a harness that does not know the flag at all also exits 2 —
    the same failure this file exists to prevent, passing as a success."""
    monkeypatch.setattr(harness, "RenewableFeatureBuilder", None)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_solar_retrain.py", "--replica-db", str(replica),
        "--renewable-source", "energy_generations",
    ])
    with pytest.raises(SystemExit) as caught:
        harness.main()
    assert caught.value.code == 2
    stderr = capsys.readouterr().err
    assert "argument --renewable-source: invalid choice" in stderr
    for known in db._RENEWABLE_TYPE_SOURCES:
        assert known in stderr


def _result(source, constant_runs=()):
    return {
        "meta": {"generated_at": "2026-08-12 00:00 UTC", "replica_db": "r.db",
                 "replica_bytes": 1, "training_source": source,
                 "fit_window": {"start": "2026-01-14", "end_exclusive": "2026-07-11"},
                 "gate_window": {"start": "2026-07-11", "end_exclusive": "2026-08-10"}},
        "verdict": "PASS", "recommendation": "-",
        "gate_cells": [], "country_d2": [],
        "training": [{"country": COUNTRY, "algorithm": "catboost",
                      "audit": {"retained_rows": 1, "intended_rows": 1, "unique_targets": 1,
                                "excluded_missing_actual_or_feature": 0, "degraded_lag_1d_rows": 0},
                      "constant_runs": list(constant_runs), "artifact_sha256": "abc"}],
    }


def test_the_report_names_the_table_it_read():
    """Two runs of this report are not comparable unless both say which table
    they read — 9 months of history against 5.6 years, and one of the two
    zero-fills what the other leaves NULL."""
    for source in db._RENEWABLE_TYPE_SOURCES:
        assert f"contamination screen: `{source}`" in harness.render_markdown(_result(source))


def test_the_report_does_not_report_a_table_the_run_never_opened():
    """The DE zero-fill run and the FR New Year's read are findings about
    `energy_renewable`. Under an `energy_generation` run they would be a clean
    bill of health for a table that had nothing to do with the numbers above
    them."""
    renewable = harness.render_markdown(_result("energy_renewable"))
    generation = harness.render_markdown(_result("energy_generation"))

    assert "known DE zero-fill run" in renewable
    assert "known DE zero-fill run" not in generation
    assert "New Year's Day" in renewable
    assert "New Year's Day" not in generation
    # The screen itself still reports, and says where it looked.
    assert "no ≥24-hour bit-identical solar run in `energy_generation`" in generation


def test_a_reported_run_says_which_table_it_was_found_in():
    run = {"start": "2026-01-01 00:00:00", "end": "2026-01-03 00:00:00",
           "value": 0.0, "n_rows": 49, "duration_hours": 48.0}
    markdown = harness.render_markdown(_result("energy_generation", [run]))
    assert f"suspect solar runs for {COUNTRY} in `energy_generation`" in markdown
    assert "known DE zero-fill run" not in markdown
