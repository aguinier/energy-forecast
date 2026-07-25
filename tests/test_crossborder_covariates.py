"""Cross-border flow covariates must be homogeneous aggregates
(import / export / net), NOT per-neighbour, so Chronos-2 global fine-tuning
receives identical covariate keys for every country.

See docs/superpowers/specs/2026-07-25-homogeneous-crossborder-covariates-design.md
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

FLOW_KEYS = {"flow__total_export_mw", "flow__total_import_mw", "flow__net_mw"}


def _seed_db(path, rows):
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE crossborder_flows "
        "(country_from TEXT, country_to TEXT, timestamp_utc TEXT, flow_mw REAL)"
    )
    con.executemany(
        "INSERT INTO crossborder_flows "
        "(country_from, country_to, timestamp_utc, flow_mw) VALUES (?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()


def _load_input_builder(monkeypatch, db_path):
    monkeypatch.setenv("ENERGY_DB_PATH", str(db_path))
    import config
    importlib.reload(config)
    from src.chronos2 import input_builder
    importlib.reload(input_builder)
    return input_builder


def test_returns_three_aggregate_keys_not_per_neighbour(monkeypatch, tmp_path):
    db = tmp_path / "t.db"
    _seed_db(db, [
        # hour 0: XX->AA +100 (export), XX->BB -40 (import)
        ("XX", "AA", "2024-01-01 00:00:00", 100.0),
        ("XX", "BB", "2024-01-01 00:00:00", -40.0),
        # hour 1: XX->AA -30 (import), XX->BB +50 (export), XX->CC +20 (export; CC only here)
        ("XX", "AA", "2024-01-01 01:00:00", -30.0),
        ("XX", "BB", "2024-01-01 01:00:00", 50.0),
        ("XX", "CC", "2024-01-01 01:00:00", 20.0),
    ])
    ib = _load_input_builder(monkeypatch, db)

    # Window must extend past t0 + serve lag: flows seeded at t0 are only presented
    # LAG hours later, so a window ending before that legitimately contains nothing.
    out = ib._load_crossborder_flow_covariates(
        "XX", "2024-01-01 00:00:00", "2024-01-30 00:00:00"
    )

    # Exactly the 3 fixed aggregate keys — no per-neighbour flow__AA/BB/CC.
    assert set(out.keys()) == FLOW_KEYS

    # Values are presented lagged by the serve lag (see the lag test below); the
    # aggregation arithmetic itself is what this test pins down.
    lag = pd.Timedelta(hours=ib.CROSSBORDER_SERVE_LAG_HOURS)
    h0 = pd.Timestamp("2024-01-01 00:00:00") + lag
    h1 = pd.Timestamp("2024-01-01 01:00:00") + lag
    exp, imp, net = out["flow__total_export_mw"], out["flow__total_import_mw"], out["flow__net_mw"]
    assert exp.loc[h0] == 100.0 and exp.loc[h1] == 70.0    # 100 ; 50+20
    assert imp.loc[h0] == 40.0 and imp.loc[h1] == 30.0     # 40  ; 30
    assert net.loc[h0] == 60.0 and net.loc[h1] == 40.0     # 100-40 ; 70-30


def test_flows_are_lagged_by_serve_lag_no_lookahead(monkeypatch, tmp_path):
    """Cross-border flows must be lagged by CROSSBORDER_SERVE_LAG_HOURS in BOTH
    training and inference, so the model never sees flow data that would not have
    been published at forecast time (the ~26h publication lag + D+2 horizon).

    See docs/superpowers/specs/2026-07-25-crossborder-lag-parity-design.md
    """
    db = tmp_path / "t.db"
    _seed_db(db, [
        ("XX", "AA", "2024-01-01 00:00:00", 100.0),
        ("XX", "AA", "2024-01-01 01:00:00", 200.0),
    ])
    ib = _load_input_builder(monkeypatch, db)
    lag = ib.CROSSBORDER_SERVE_LAG_HOURS

    out = ib._load_crossborder_flow_covariates(
        "XX", "2024-01-01 00:00:00", "2024-01-30 00:00:00"
    )
    net = out["flow__net_mw"]

    t0 = pd.Timestamp("2024-01-01 00:00:00")
    # The flow observed at t0 must be presented lag hours LATER...
    assert net.loc[t0 + pd.Timedelta(hours=lag)] == 100.0
    assert net.loc[t0 + pd.Timedelta(hours=lag + 1)] == 200.0
    # ...and must NOT be visible at its original (unlagged) timestamp.
    assert t0 not in net.index or pd.isna(net.loc[t0])


def test_empty_country_still_returns_three_keys(monkeypatch, tmp_path):
    """A country with no cross-border data must STILL return all 3 keys
    (as empty series) — never {} — or heterogeneity reappears at fine-tune time."""
    db = tmp_path / "t.db"
    _seed_db(db, [])
    ib = _load_input_builder(monkeypatch, db)

    out = ib._load_crossborder_flow_covariates(
        "ZZ", "2024-01-01 00:00:00", "2024-01-02 00:00:00"
    )

    assert set(out.keys()) == FLOW_KEYS
    for k in FLOW_KEYS:
        assert out[k].empty
