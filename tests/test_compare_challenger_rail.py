"""`compare_challenger.py` end to end on live-rail-shaped vintages (ABL-82).

The unit tests cover the pairing rule. This covers the *seam* that actually
broke: the script used to write the join key down a second time to compute its
one-sided counts, so the script and the module could disagree about what a
vintage is — and did, silently, for every challenger on the live sidecar.

It also pins the exit-code contract the C2c promotion gate reads: a comparison
that paired nothing exits **1**, so a gate script cannot mistake "not measured"
for "no difference".
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

cc = importlib.import_module("compare_challenger")

CHAMPION = "chronos-2-V010"
CHALLENGER = "chronos-2-V016"

# The live rail's real stamps, measured on the sidecar 2026-08-09: the champion
# writes microseconds, the challenger process starts 12.3 s later and truncates
# to the second. Nothing in this shape is exotic - it is simply two processes.
CHAMP_GEN = "2026-08-08 06:00:55.715745"
CHAL_GEN = "2026-08-08 06:01:08"
# The 2026-08-07 shadow backfill: same calendar day as that day's champion run,
# 15h25m later, and therefore a day of actuals better informed.
BACKFILL_GEN = "2026-08-08 21:25:35"

TARGETS = [f"2026-08-10 {h:02d}:00:00" for h in range(6)]


def _sidecar(path: Path, chal_gen: str) -> str:
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, country_code TEXT, forecast_type TEXT,
        target_timestamp_utc TIMESTAMP, generated_at TIMESTAMP, horizon_hours INTEGER,
        forecast_value REAL, model_name TEXT, model_version TEXT)""")
    rows = []
    for i, t in enumerate(TARGETS):
        rows.append(("FR", "net_position", t, CHAMP_GEN, 48, 1000.0, CHAMPION, "v"))
        rows.append(("FR", "net_position", t, chal_gen, 48, 1100.0, CHALLENGER, "v"))
    con.executemany("""INSERT INTO forecasts (country_code, forecast_type,
        target_timestamp_utc, generated_at, horizon_hours, forecast_value,
        model_name, model_version) VALUES (?,?,?,?,?,?,?,?)""", rows)
    con.commit()
    con.close()
    return str(path)


def _replica(path: Path) -> str:
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE net_position (
        id INTEGER PRIMARY KEY AUTOINCREMENT, country_code TEXT,
        timestamp_utc TIMESTAMP, net_position_mw REAL)""")
    con.executemany(
        "INSERT INTO net_position (country_code, timestamp_utc, net_position_mw) "
        "VALUES (?,?,?)", [("FR", t, 1000.0) for t in TARGETS])
    con.commit()
    con.close()
    return str(path)


def _run(monkeypatch, capsys, sidecar, replica, out_dir):
    monkeypatch.setattr(sys, "argv", [
        "compare_challenger.py",
        "--champion-db", sidecar, "--challenger-db", sidecar,
        "--challenger", CHALLENGER, "--replica-db", replica,
        "--out-dir", str(out_dir)])
    code = cc.main()
    return code, (out_dir / "latest.md").read_text(encoding="utf-8")


def test_a_twelve_second_gap_is_one_run_and_scores(tmp_path, monkeypatch, capsys):
    sidecar = _sidecar(tmp_path / "sidecar.db", CHAL_GEN)
    replica = _replica(tmp_path / "replica.db")
    code, md = _run(monkeypatch, capsys, sidecar, replica, tmp_path / "out")

    assert code == 0
    assert f"**Paired rows:** {len(TARGETS)}" in md
    # The champion is exact and the challenger is 100 MW high on every hour.
    assert "**0.0 MW** MAE" in md and "**100.0 MW** MAE" in md
    assert "widest paired vintage gap **12.3 s**" in md


def test_a_backfill_is_not_paired_and_the_script_exits_nonzero(
        tmp_path, monkeypatch, capsys):
    """The challenger ran 15h25m later on more actuals. Refusing to score that
    is the point; exiting 0 with a report of zeros is what ABL-82 was."""
    sidecar = _sidecar(tmp_path / "sidecar.db", BACKFILL_GEN)
    replica = _replica(tmp_path / "replica.db")
    code, md = _run(monkeypatch, capsys, sidecar, replica, tmp_path / "out")

    assert code == 1
    assert "Not measured" in md
    assert "0.0 MW" not in md
    assert "NOT MEASURED" in capsys.readouterr().err

    # And `--max-run-skew-hours` does NOT buy it back, however wide. The two
    # vintages saw different actuals, so they are in different cutoff buckets
    # and no candidate pair is ever formed; the skew bound only ever separates
    # runs *within* one cutoff. An information mismatch is not a tolerance
    # problem and must not have a flag that makes it look like one.
    monkeypatch.setattr(sys, "argv", [
        "compare_challenger.py",
        "--champion-db", sidecar, "--challenger-db", sidecar,
        "--challenger", CHALLENGER, "--replica-db", replica,
        "--max-run-skew-hours", "999", "--stdout"])
    assert cc.main() == 1
    assert "Not measured" in capsys.readouterr().out


def test_missing_replica_is_reported_not_scored(tmp_path, monkeypatch):
    sidecar = _sidecar(tmp_path / "sidecar.db", CHAL_GEN)
    monkeypatch.setattr(sys, "argv", [
        "compare_challenger.py",
        "--champion-db", sidecar, "--challenger-db", sidecar,
        "--challenger", CHALLENGER,
        "--replica-db", str(tmp_path / "nope.db"), "--stdout"])
    assert cc.main() == 2
