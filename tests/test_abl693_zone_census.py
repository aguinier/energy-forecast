"""ABL-693: the census must see a mid-window population change.

The defect it exists for: `chronos-2-V010` served 19 zones through the
2026-09-05 vintage and 18 from 2026-09-06 on, and every artifact downstream --
`build_gate_scope`'s `countries_measured`, a pooled coverage figure, a
per-zone band-width clause -- reported a number without reporting that the
panel behind it had changed.
"""
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "abl693_zone_census", REPO / "scripts" / "abl693_zone_census.py")
census = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(census)


ZONES_19 = ["AT", "BE", "BG", "CZ", "DE", "EE", "ES", "FI", "FR", "HR",
            "HU", "LT", "LV", "NL", "PL", "PT", "RO", "SI", "SK"]


def _sidecar(path, rows):
    """rows: (model_name, generated_at, country_code, n_hours)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE forecasts (
            id INTEGER PRIMARY KEY, country_code TEXT, forecast_type TEXT,
            target_timestamp_utc TIMESTAMP, generated_at TIMESTAMP,
            horizon_hours INTEGER, forecast_value REAL,
            model_name TEXT, model_version TEXT)""")
    conn.execute("""
        CREATE TABLE forecast_quantiles (
            id INTEGER PRIMARY KEY, country_code TEXT, forecast_type TEXT,
            target_timestamp_utc TIMESTAMP, generated_at TIMESTAMP,
            quantile REAL, forecast_value REAL, model_name TEXT)""")
    for model, generated_at, country, hours in rows:
        for hour in range(hours):
            conn.execute(
                "INSERT INTO forecasts (country_code, forecast_type, "
                "target_timestamp_utc, generated_at, horizon_hours, "
                "forecast_value, model_name, model_version) "
                "VALUES (?, 'net_position', ?, ?, ?, 0.0, ?, 'v')",
                (country, f"2026-09-20 {hour:02d}:00:00", generated_at,
                 hour, model))
    conn.commit()
    conn.close()
    return path


def _rows(model, days, zones, hours=24):
    return [(model, f"{day} 06:00:00", zone, hours)
            for day in days for zone in zones]


def _census(tmp_path, rows, **kwargs):
    db_path = _sidecar(tmp_path / "sidecar.db", rows)
    conn = census._connect_readonly(db_path)
    try:
        population = census.read_population(conn, "forecasts", "net_position")
    finally:
        conn.close()
    model = next(iter(population))
    return census.census_for_model(population[model], **kwargs)


def test_a_constant_population_is_reported_constant(tmp_path):
    days = ["2026-09-03", "2026-09-04", "2026-09-05"]
    report = _census(tmp_path, _rows("chronos-2-V010", days, ZONES_19))
    assert report["population_constant"] is True
    assert report["zones_left"] == []
    assert report["zones_entered"] == []
    assert len(report["reference_zones"]) == 19


def test_a_zone_that_stops_being_served_is_reported_with_its_boundary(tmp_path):
    """The real shape: PT served through 09-05, absent from 09-06 on. The
    boundary vintage is the finding -- 'PT is missing' does not tell a gate
    read which of its vintages are 19-zone and which are 18."""
    kept = [z for z in ZONES_19 if z != "PT"]
    rows = (_rows("chronos-2-V010", ["2026-09-04", "2026-09-05"], ZONES_19)
            + _rows("chronos-2-V010",
                    ["2026-09-06", "2026-09-07", "2026-09-08"], kept))
    report = _census(tmp_path, rows)

    assert report["population_constant"] is False
    assert report["zones_left"] == [{"zone": "PT",
                                     "last_served": "2026-09-05",
                                     "first_absent": "2026-09-06"}]
    assert report["zones_entered"] == []
    missing = {v["vintage_day"]: v["missing_vs_reference"]
               for v in report["per_vintage"]}
    assert missing["2026-09-05"] == []
    assert missing["2026-09-06"] == ["PT"]
    assert missing["2026-09-08"] == ["PT"]


def test_the_panel_balance_shows_how_far_the_pooled_weight_moved(tmp_path):
    """A pooled figure is weighted by these shares whether or not it says so.
    PT at 2 of 5 vintages carries 2/(18*5+2) of the rows, not 1/19."""
    kept = [z for z in ZONES_19 if z != "PT"]
    rows = (_rows("chronos-2-V010", ["2026-09-04", "2026-09-05"], ZONES_19)
            + _rows("chronos-2-V010",
                    ["2026-09-06", "2026-09-07", "2026-09-08"], kept))
    report = _census(tmp_path, rows)

    pt = report["panel_balance"]["PT"]
    assert pt["vintages_served"] == 2
    assert pt["vintages_in_window"] == 5
    assert pt["completeness"] == 0.4
    assert pt["rows"] == 2 * 24
    # Shares are rounded to 6 dp for readability, so the tolerance is the
    # rounding, not the arithmetic.
    assert pt["row_share"] == pytest.approx(48 / ((18 * 5 + 2) * 24), abs=5e-7)
    assert pt["balanced_share"] == pytest.approx(1 / 19, abs=5e-7)

    at = report["panel_balance"]["AT"]
    assert at["completeness"] == 1.0
    assert at["row_share"] > at["balanced_share"]


def test_a_zone_that_appears_mid_window_is_reported_too(tmp_path):
    """Both directions move a pooled number. An addition is not benign just
    because nothing was lost."""
    without = [z for z in ZONES_19 if z != "PT"]
    rows = (_rows("chronos-2-V010", ["2026-09-04"], without)
            + _rows("chronos-2-V010", ["2026-09-05", "2026-09-06"], ZONES_19))
    report = _census(tmp_path, rows)

    assert report["population_constant"] is False
    assert report["zones_entered"] == [{"zone": "PT",
                                        "first_served": "2026-09-05"}]
    assert report["zones_left"] == []


def test_a_zone_absent_for_the_whole_window_needs_a_registered_reference(tmp_path):
    """The trap this census would otherwise walk into. Once PT has been gone
    for longer than the window, the union over the window no longer contains
    it, and 'the population did not change' becomes true and useless. A
    registered zone list is the only reference that can see a zone that was
    already missing when the window opened."""
    without = [z for z in ZONES_19 if z != "PT"]
    rows = _rows("chronos-2-V010", ["2026-09-07", "2026-09-08"], without)

    union_ref = _census(tmp_path / "a", rows)
    assert union_ref["population_constant"] is True
    assert union_ref["reference_source"] == "window_union"
    assert "PT" not in union_ref["reference_zones"]

    registered = _census(tmp_path / "b", rows, expect_zones=ZONES_19)
    assert registered["population_constant"] is False
    assert registered["reference_source"] == "expect_zones"
    assert all(v["missing_vs_reference"] == ["PT"]
               for v in registered["per_vintage"])
    assert registered["panel_balance"]["PT"]["vintages_served"] == 0


def test_a_vintage_that_produced_nothing_is_counted_as_a_calendar_gap(tmp_path):
    """The 2026-09-02 run refused all 19 zones, so it wrote no row and has no
    key in the population at all. Every per-zone check walks straight past it;
    only the calendar sees it. `population_constant` stays True here because
    the zone set really did not change -- the two facts are reported apart so
    neither can stand in for the other."""
    rows = _rows("chronos-2-V010",
                 ["2026-09-01", "2026-09-03", "2026-09-04"], ZONES_19)
    report = _census(tmp_path, rows)

    assert report["population_constant"] is True
    assert report["vintage_calendar_complete"] is False
    assert report["vintage_days_missing"] == ["2026-09-02"]
    assert report["vintage_days_in_span"] == 4
    assert len(report["vintage_days"]) == 3


def test_a_calendar_gap_alone_still_fails_the_exit_code(tmp_path, monkeypatch,
                                                       capsys):
    """A pooled read over 'ten vintages' that is really nine is the same
    hazard as a zone leaving, so it gets the same exit code."""
    rows = _rows("chronos-2-V010",
                 ["2026-09-01", "2026-09-03"], ZONES_19)
    db_path = _sidecar(tmp_path / "sidecar.db", rows)
    monkeypatch.setattr(sys, "argv",
                        ["abl693_zone_census.py", "--sidecar-db", str(db_path)])
    assert census.main() == 1
    out = capsys.readouterr().out
    assert "CONSTANT" in out
    assert "NO VINTAGE 2026-09-02" in out


def test_models_are_censused_independently(tmp_path):
    """The defect is per model: the two Chronos models dropped PT and the two
    non-Chronos ones did not. Pooling the models would report neither."""
    kept = [z for z in ZONES_19 if z != "PT"]
    rows = (_rows("chronos-2-V010", ["2026-09-05"], ZONES_19)
            + _rows("chronos-2-V010", ["2026-09-06"], kept)
            + _rows("baseline-V012", ["2026-09-05", "2026-09-06"], ZONES_19))
    db_path = _sidecar(tmp_path / "sidecar.db", rows)
    conn = census._connect_readonly(db_path)
    try:
        population = census.read_population(conn, "forecasts", "net_position")
    finally:
        conn.close()

    chronos = census.census_for_model(population["chronos-2-V010"])
    baseline = census.census_for_model(population["baseline-V012"])
    assert chronos["population_constant"] is False
    assert baseline["population_constant"] is True


def test_an_empty_selection_exits_2_because_unknown_is_not_constant(
        tmp_path, monkeypatch, capsys):
    """ABL-370's rule, applied to a census: exit 0 with nothing read would tell
    a gate harness the panel was fine."""
    db_path = _sidecar(tmp_path / "sidecar.db", [])
    monkeypatch.setattr(sys, "argv",
                        ["abl693_zone_census.py", "--sidecar-db", str(db_path)])
    assert census.main() == 2
    assert "not 'constant'" in capsys.readouterr().out


def test_the_exit_code_carries_the_finding(tmp_path, monkeypatch, capsys):
    """1 means the population moved. A caller that only checks 'did it crash'
    is the failure mode this replaces."""
    kept = [z for z in ZONES_19 if z != "PT"]
    rows = (_rows("chronos-2-V010", ["2026-09-05"], ZONES_19)
            + _rows("chronos-2-V010", ["2026-09-06"], kept))
    db_path = _sidecar(tmp_path / "sidecar.db", rows)

    monkeypatch.setattr(sys, "argv",
                        ["abl693_zone_census.py", "--sidecar-db", str(db_path)])
    assert census.main() == 1
    out = capsys.readouterr().out
    assert "CHANGED" in out
    assert "LEFT    PT" in out

    monkeypatch.setattr(sys, "argv",
                        ["abl693_zone_census.py", "--sidecar-db", str(db_path),
                         "--until", "2026-09-06"])
    assert census.main() == 0
    assert "CONSTANT" in capsys.readouterr().out


def test_the_census_cannot_write_to_the_sidecar(tmp_path):
    """It is opened through the read-only URI. A census that can write is a
    census that can be blamed for what it measured."""
    db_path = _sidecar(tmp_path / "sidecar.db",
                       _rows("chronos-2-V010", ["2026-09-05"], ["AT"]))
    conn = census._connect_readonly(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError,
                           match="readonly|read-only|attempt to write"):
            conn.execute("DELETE FROM forecasts")
    finally:
        conn.close()


def test_quantiles_carry_their_own_population(tmp_path):
    """ABL-677 pools 10-90 coverage out of `forecast_quantiles`, so that table
    is the one whose panel has to be censused for a coverage read -- assuming
    it matches `forecasts` is an assumption, not a check."""
    db_path = _sidecar(tmp_path / "sidecar.db",
                       _rows("chronos-2-V010", ["2026-09-05"], ZONES_19))
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO forecast_quantiles (country_code, forecast_type, "
        "target_timestamp_utc, generated_at, quantile, forecast_value, "
        "model_name) VALUES ('AT', 'net_position', '2026-09-20 00:00:00', "
        "'2026-09-05 06:00:00', 0.1, 0.0, 'chronos-2-V010')")
    conn.commit()
    conn.close()

    conn = census._connect_readonly(db_path)
    try:
        points = census.read_population(conn, "forecasts", "net_position")
        quantiles = census.read_population(conn, "forecast_quantiles",
                                           "net_position")
    finally:
        conn.close()

    assert len(points["chronos-2-V010"]["2026-09-05"]) == 19
    assert len(quantiles["chronos-2-V010"]["2026-09-05"]) == 1


def test_an_unknown_table_is_refused_rather_than_interpolated(tmp_path):
    """The table name reaches an f-string in the query. It is checked against a
    fixed tuple first."""
    db_path = _sidecar(tmp_path / "sidecar.db", [])
    conn = census._connect_readonly(db_path)
    try:
        with pytest.raises(ValueError, match="unknown table"):
            census.read_population(conn, "forecasts; DROP TABLE forecasts",
                                   "net_position")
    finally:
        conn.close()
