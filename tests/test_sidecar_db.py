"""Sidecar output DB: with FORECAST_OUTPUT_DB set, writes go to the sidecar,
reads keep hitting the main (replica) DB."""
import importlib
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _fresh_db_module(monkeypatch, tmp_path, sidecar: bool):
    monkeypatch.setenv("ENERGY_DB_PATH", str(tmp_path / "replica.db"))
    if sidecar:
        monkeypatch.setenv("FORECAST_OUTPUT_DB", str(tmp_path / "sidecar.db"))
    else:
        monkeypatch.delenv("FORECAST_OUTPUT_DB", raising=False)
    import config
    importlib.reload(config)
    from src import db
    importlib.reload(db)
    return db


def test_write_connection_targets_sidecar_when_set(monkeypatch, tmp_path):
    db = _fresh_db_module(monkeypatch, tmp_path, sidecar=True)
    with db.get_connection(readonly=False) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
    side = sqlite3.connect(tmp_path / "sidecar.db")
    assert side.execute("SELECT count(*) FROM t").fetchone()[0] == 1


def test_replica_untouched_by_writes_when_sidecar_set(monkeypatch, tmp_path):
    # Seed a replica so we can prove it stays pristine.
    rep = sqlite3.connect(tmp_path / "replica.db")
    rep.execute("CREATE TABLE existing (x INTEGER)")
    rep.commit(); rep.close()
    db = _fresh_db_module(monkeypatch, tmp_path, sidecar=True)
    with db.get_connection(readonly=False) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
    rep = sqlite3.connect(tmp_path / "replica.db")
    names = {r[0] for r in rep.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "t" not in names


def test_read_connection_targets_replica_even_with_sidecar(monkeypatch, tmp_path):
    rep = sqlite3.connect(tmp_path / "replica.db")
    rep.execute("CREATE TABLE marker (x INTEGER)")
    rep.commit(); rep.close()
    db = _fresh_db_module(monkeypatch, tmp_path, sidecar=True)
    with db.get_connection(readonly=True) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "marker" in names


def test_write_connection_targets_main_db_when_unset(monkeypatch, tmp_path):
    db = _fresh_db_module(monkeypatch, tmp_path, sidecar=False)
    with db.get_connection(readonly=False) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
    rep = sqlite3.connect(tmp_path / "replica.db")
    assert rep.execute("SELECT count(*) FROM sqlite_master WHERE name='t'").fetchone()[0] == 1
