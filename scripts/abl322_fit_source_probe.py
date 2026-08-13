#!/usr/bin/env python3
"""
ABL-322 / ABL-355: which database did the pilot gate read actually fit on?

ABL-355 found that `--replica-db` governed only the incumbent, TSO and
contamination screen, while the fitted series and the weather archive came from
`config.DATABASE_PATH` (that is, `ENERGY_DB_PATH`). The ABL-322 pilot gate read
predates that fix and its generated report records only the one path, so the
numbers are trustworthy only if the two resolved to the same data.

This probe reconstructs the answer from the workstation rather than from the
unrecorded environment variable, and it is kept because it is the evidence
behind "the pilot's numbers stand, no re-run needed". It asks three things:

  --candidates   Which local files could have been the fit source at all: does
                 the path exist, and does it carry an `energy_generation`
                 table? A file without that table cannot be the source of a fit
                 whose recorded `training_source` is `energy_generation`.

  --compare      For the candidates that survive, digest the DE and NL
                 `wind_offshore` target series and the `weather_data` archive
                 over the builder's full span. Identical digests mean the
                 choice between them could not have moved the gate read.

  --arithmetic   Reproduce the report's own counts from the replica's coverage:
                 15-minute rows / 4 (the ABL-332 hourly mean) = unique fit
                 targets, x 8 vintages = fit rows, and the gate-window count =
                 the reported n.

Nothing here writes. Every query opens SQLite read-only with `mode=ro`.

    .venv/Scripts/python.exe scripts/abl322_fit_source_probe.py --candidates
    .venv/Scripts/python.exe scripts/abl322_fit_source_probe.py --compare
    .venv/Scripts/python.exe scripts/abl322_fit_source_probe.py --arithmetic
"""

import argparse
import hashlib
import sqlite3
import sys
from pathlib import Path

# The ABL-322 pilot windows, as registered in experiments/ABL322/config.json.
FIT_START, FIT_END = "2026-01-14", "2026-07-11"
GATE_START, GATE_END = "2026-07-11", "2026-08-10"
# The builder is constructed at fit_start - 14d and runs through gate_end, so
# this is the whole span any fitted row could have been drawn from.
SPAN_LO, SPAN_HI = "2025-12-31", "2026-08-10"

VINTAGES = 8          # the measured eight-vintage schedule
SUB_HOURLY = 4        # 15-minute source resolution for DE and NL

# Every plausible fit source on this workstation, including the wrong ones --
# the point of the probe is that the wrong ones are eliminated, not omitted.
CANDIDATES = {
    "live replica": r"C:\Code\able\data\energy_dashboard.db",
    "ops backup": r"C:\Code\able\backups_ops\ops_backup_2026-08-12.db",
    "partial snapshot (decoy)": r"C:\Code\able\energy-data-gathering\energy_dashboard.db",
    "config fallback, ENERGY_DB_PATH unset": r"C:\data\energy_dashboard.db",
    "checked-in .env value": r"C:\Code\energy-data-gathering\energy_dashboard.db",
}

TARGET_SQL = (
    "SELECT timestamp_utc, wind_offshore_mw FROM energy_generation "
    "WHERE country_code=? AND timestamp_utc>=? AND timestamp_utc<? "
    "ORDER BY timestamp_utc"
)


def ro_connect(path: str) -> sqlite3.Connection:
    """Open `path` read-only. Raises if it does not exist, which is the point:
    a missing file is a dead configuration, not an empty result set."""
    uri = f"file:{Path(path).resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=60.0)


def has_table(con: sqlite3.Connection, name: str) -> bool:
    return bool(con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (name,)).fetchone()[0])


def digest(con: sqlite3.Connection, sql: str, params) -> tuple:
    h = hashlib.sha256()
    n = 0
    for row in con.execute(sql, params):
        h.update(repr(row).encode())
        n += 1
    return n, h.hexdigest()[:16]


def cmd_candidates() -> int:
    print("Candidate fit sources - a file must exist AND carry `energy_generation`.\n")
    for label, path in CANDIDATES.items():
        if not Path(path).exists():
            print(f"  [ELIMINATED] {label}\n"
                  f"               {path}\n"
                  f"               does not exist; `mode=ro` raises rather than creating it")
            continue
        size = Path(path).stat().st_size
        with ro_connect(path) as con:
            ok = has_table(con, "energy_generation")
        verdict = "VIABLE" if ok else "ELIMINATED"
        why = "" if ok else "; no `energy_generation` table"
        print(f"  [{verdict}] {label}\n"
              f"               {path}\n"
              f"               {size:,} bytes{why}")
    return 0


def cmd_compare() -> int:
    viable = [(label, path) for label, path in CANDIDATES.items()
              if Path(path).exists() and _carries_table(path)]
    if len(viable) < 2:
        print(f"{len(viable)} viable candidate(s); nothing to compare.")
        return 0
    print(f"Comparing {len(viable)} viable candidates over {SPAN_LO} .. {SPAN_HI}.")
    print("Identical digests mean the choice between them cannot move the gate read.\n")
    seen = {}
    for label, path in viable:
        print(f"=== {label}  ({Path(path).stat().st_size:,} bytes)")
        fingerprint = []
        with ro_connect(path) as con:
            for cc in ("DE", "NL"):
                n, d = digest(con, TARGET_SQL, (cc, SPAN_LO, SPAN_HI))
                print(f"    {cc} wind_offshore target series: n={n:,} sha256[:16]={d}")
                fingerprint.append((cc, n, d))
            if has_table(con, "weather_data"):
                row = con.execute(
                    "SELECT COUNT(*), MAX(forecast_run_time) FROM weather_data "
                    "WHERE timestamp_utc>=? AND timestamp_utc<?",
                    (SPAN_LO, SPAN_HI)).fetchone()
                print(f"    weather_data: n={row[0]:,} max forecast_run_time={row[1]}")
                fingerprint.append(("weather_data", row[0], row[1]))
            else:
                print("    weather_data: ABSENT")
                fingerprint.append(("weather_data", None, None))
        seen[label] = tuple(fingerprint)
    distinct = set(seen.values())
    print()
    if len(distinct) == 1:
        print("IDENTICAL across every viable candidate - the unrecorded "
              "ENERGY_DB_PATH could not have changed the fitted data.")
        return 0
    print("DIVERGENT - the candidates differ over this window, so which one the "
          "environment named DOES matter and the gate read needs a re-run.")
    return 1


def _carries_table(path: str) -> bool:
    with ro_connect(path) as con:
        return has_table(con, "energy_generation")


def cmd_arithmetic() -> int:
    path = CANDIDATES["live replica"]
    print(f"Reproducing the report's counts from {path}\n")
    ok = True
    with ro_connect(path) as con:
        for cc in ("DE", "NL"):
            fit = con.execute(
                "SELECT COUNT(*) FROM energy_generation WHERE country_code=? "
                "AND wind_offshore_mw IS NOT NULL AND timestamp_utc>=? AND timestamp_utc<?",
                (cc, FIT_START, FIT_END)).fetchone()[0]
            gate = con.execute(
                "SELECT COUNT(*) FROM energy_generation WHERE country_code=? "
                "AND wind_offshore_mw IS NOT NULL AND timestamp_utc>=? AND timestamp_utc<?",
                (cc, GATE_START, GATE_END)).fetchone()[0]
            targets, rem = divmod(fit, SUB_HOURLY)
            gate_n, gate_rem = divmod(gate, SUB_HOURLY)
            print(f"  {cc}: fit window {fit:,} sub-hourly rows / {SUB_HOURLY} "
                  f"= {targets:,} unique fit targets"
                  f"{'' if not rem else f' (+{rem} partial)'}")
            print(f"      x {VINTAGES} vintages = {targets * VINTAGES:,} fit rows "
                  f"(report: 34,176)")
            print(f"      gate window {gate:,} / {SUB_HOURLY} = {gate_n:,} "
                  f"(report n for 24-36h and 36-48h: 720)"
                  f"{'' if not gate_rem else f' (+{gate_rem} partial)'}")
            if targets * VINTAGES != 34_176 or gate_n != 720:
                ok = False
    print()
    print("Reproduces exactly." if ok else "DOES NOT reproduce - investigate.")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", action="store_true",
                        help="which local files could have been the fit source at all")
    parser.add_argument("--compare", action="store_true",
                        help="digest the viable candidates over the builder's span")
    parser.add_argument("--arithmetic", action="store_true",
                        help="reproduce the report's counts from the replica")
    args = parser.parse_args()
    if not (args.candidates or args.compare or args.arithmetic):
        parser.error("choose at least one of --candidates, --compare, --arithmetic")
    rc = 0
    if args.candidates:
        rc |= cmd_candidates()
    if args.compare:
        if args.candidates:
            print()
        rc |= cmd_compare()
    if args.arithmetic:
        if args.candidates or args.compare:
            print()
        rc |= cmd_arithmetic()
    return rc


if __name__ == "__main__":
    sys.exit(main())
