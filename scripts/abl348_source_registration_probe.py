"""ABL-348 -- does a change of training source void the ABL-195/ABL-253 registration?

Read-only. Opens the replica through `db.load_renewable_type_data`, which is the
same loader the gate harnesses train and score from, so what this measures is
what a harness would actually see: `data_quality='actual'` only, NULL dropped,
duplicate instants collapsed (disagreeing spellings nulled), and the ABL-188
suspect-constant-run screen applied. Hand-rolled SQL would measure a different
series than the one under registration.

Writes nothing to either database. Emits JSON on stdout / to --out.

    ENERGY_DB_PATH=C:\\Code\\able\\data\\energy_dashboard.db \\
      .venv\\Scripts\\python.exe scripts/abl348_source_registration_probe.py --out reports/abl_348_probe.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db  # noqa: E402

# The loader is deliberately chatty about dedup and ABL-188 exclusions; we count
# them ourselves against raw SQL, so silence the per-pair warnings.
logging.getLogger('src.db').setLevel(logging.ERROR)
logging.getLogger('src.data_quality').setLevel(logging.ERROR)

# --------------------------------------------------------------------------
# The registered windows. Verbatim from experiments/ABL195/config.json and
# experiments/ABL253/config.json -- identical in both.
# --------------------------------------------------------------------------
FIT_START = pd.Timestamp('2026-01-14 00:00:00')
FIT_END = pd.Timestamp('2026-07-11 00:00:00')      # exclusive
GATE_START = pd.Timestamp('2026-07-11 00:00:00')
GATE_END = pd.Timestamp('2026-08-10 00:00:00')     # exclusive
LOOKBACK_START = FIT_START - pd.Timedelta(days=14)  # 2025-12-31, the builder's reach

FIT_HOURS = int((FIT_END - FIT_START).total_seconds() // 3600)     # 4344
GATE_HOURS = int((GATE_END - GATE_START).total_seconds() // 3600)  # 720

# Registered minimum n per primary band (95% of intended 720/720/480).
REGISTERED_MIN_N = {'24-36h': 684, '36-48h': 684, '48-64h': 456}

SOURCES = ('energy_renewable', 'energy_generation')

# The 37 tranche pairs, listed rather than discovered: every ABL-318
# `data_verdict == TRAIN` pair except the ten already-serving pairs and the two
# ABL-322 pilot pairs (DE/NL wind_offshore), which carry their own registration.
# `reports/abl_318_renewable_data_audit.csv` is gitignored, so deriving the list
# from it at run time would make this probe unreproducible from the repo alone.
# `--audit-csv` cross-checks the literal against the audit when it is available.
PILOT_PAIRS = {('DE', 'wind_offshore'), ('NL', 'wind_offshore')}

TRANCHE_PAIRS: tuple[tuple[str, str], ...] = tuple(sorted(
    [(cc, 'solar') for cc in (
        'BG', 'CH', 'CZ', 'EE', 'ES', 'FI', 'GR', 'HR', 'HU', 'IT',
        'LT', 'LV', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK')]
    + [(cc, 'wind_onshore') for cc in (
        'BG', 'CH', 'CZ', 'EE', 'ES', 'FI', 'GR', 'HR', 'HU', 'IT',
        'LT', 'LV', 'NL', 'NO', 'PL', 'PT', 'RO', 'SE')]
))

#: BG and CH are the pairs the CEO measured as level across both tables.
LEVEL_PAIRS = {('BG', 'solar'), ('BG', 'wind_onshore'),
               ('CH', 'solar'), ('CH', 'wind_onshore')}


def crosscheck_audit(audit_csv: Path) -> str:
    """Confirm the hardcoded list still matches the ABL-318 audit, if present."""
    if not audit_csv.exists():
        return f'not cross-checked: {audit_csv} absent (gitignored)'
    frame = pd.read_csv(audit_csv)
    derived = sorted(
        (row.country, row.stream)
        # `verdict`, not `data_verdict`: the ten already-serving pairs carry
        # verdict ALREADY-COVERED with data_verdict TRAIN. 39 - 2 pilot = 37.
        for row in frame[frame['verdict'] == 'TRAIN'].itertuples()
        if (row.country, row.stream) not in PILOT_PAIRS
    )
    if derived != sorted(TRANCHE_PAIRS):
        missing = set(derived) - set(TRANCHE_PAIRS)
        extra = set(TRANCHE_PAIRS) - set(derived)
        raise SystemExit(
            f"tranche list disagrees with {audit_csv}: missing {sorted(missing)}, "
            f"extra {sorted(extra)}"
        )
    return f'cross-checked against {audit_csv}: {len(derived)} pairs, exact match'


def hourly(frame: pd.DataFrame) -> pd.Series:
    """The :00 instant of each hour -- the served cadence and the ABL-195 read."""
    if frame.empty:
        return pd.Series(dtype=float)
    ts = frame['timestamp_utc']
    on_hour = frame[(ts.dt.minute == 0) & (ts.dt.second == 0)]
    return pd.Series(
        on_hour['target_value'].to_numpy(),
        index=pd.DatetimeIndex(on_hour['timestamp_utc']),
    ).dropna()


def d7_scores(series: pd.Series) -> dict:
    """Literal seasonal-naive D-7 over the gate window, on this series alone.

    A gate hour is scorable only where both the actual and its D-7 lag exist,
    which is what bounds the n any primary band can reach.
    """
    gate_index = pd.date_range(GATE_START, GATE_END, freq='h', inclusive='left')
    actual = series.reindex(gate_index)
    lagged = series.reindex(gate_index - pd.Timedelta(days=7))
    lagged.index = gate_index
    both = actual.notna() & lagged.notna()
    n = int(both.sum())
    if n == 0:
        return {'n_gate_hours': int(actual.notna().sum()), 'n_d7_scorable': 0,
                'd7_wape_pct': None, 'd7_mae_mw': None, 'mean_actual_mw': None}
    a = actual[both].to_numpy(dtype=float)
    b = lagged[both].to_numpy(dtype=float)
    denom = float(np.abs(a).sum())
    return {
        'n_gate_hours': int(actual.notna().sum()),
        'n_d7_scorable': n,
        'd7_wape_pct': round(100.0 * float(np.abs(a - b).sum()) / denom, 2) if denom else None,
        'd7_mae_mw': round(float(np.abs(a - b).mean()), 1),
        'mean_actual_mw': round(float(a.mean()), 1),
    }


def raw_span(conn: sqlite3.Connection, country: str, stream: str, source: str) -> dict:
    """Unbounded first/last reported instant -- the depth claim under test."""
    col = db.RENEWABLE_TYPE_COLUMNS[stream]
    row = conn.execute(
        f"SELECT MIN(timestamp_utc), MAX(timestamp_utc), COUNT(*) FROM {source} "
        f"WHERE country_code = ? AND data_quality = 'actual' AND ({col}) IS NOT NULL",
        (country,),
    ).fetchone()
    return {'first_ts': row[0], 'last_ts': row[1], 'n_rows': int(row[2] or 0)}


def raw_window_count(conn: sqlite3.Connection, country: str, stream: str,
                     source: str, start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Distinct on-the-hour instants before dedup/ABL-188 -- the screen's cost."""
    col = db.RENEWABLE_TYPE_COLUMNS[stream]
    row = conn.execute(
        f"SELECT COUNT(DISTINCT timestamp_utc) FROM {source} "
        f"WHERE country_code = ? AND data_quality = 'actual' AND ({col}) IS NOT NULL "
        f"AND timestamp_utc >= ? AND timestamp_utc < ? "
        f"AND (CAST(substr(timestamp_utc, 15, 2) AS INTEGER) = 0)",
        (country, start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S')),
    ).fetchone()
    return int(row[0] or 0)


def measure_pair(conn: sqlite3.Connection, country: str, stream: str) -> dict:
    out = {'country': country, 'stream': stream, 'sources': {}}
    series_by_source = {}

    for source in SOURCES:
        frame = db.load_renewable_type_data(
            country, stream,
            LOOKBACK_START.strftime('%Y-%m-%d %H:%M:%S'),
            GATE_END.strftime('%Y-%m-%d %H:%M:%S'),
            source=source,
        )
        series = hourly(frame)
        series_by_source[source] = series

        fit = series[(series.index >= FIT_START) & (series.index < FIT_END)]
        look = series[(series.index >= LOOKBACK_START) & (series.index < FIT_START)]

        record = raw_span(conn, country, stream, source)
        record.update(d7_scores(series))
        record['n_fit_hours'] = int(fit.notna().sum())
        record['fit_hours_intended'] = FIT_HOURS
        record['n_lookback_hours'] = int(look.notna().sum())
        record['raw_gate_hours_prescreen'] = raw_window_count(
            conn, country, stream, source, GATE_START, GATE_END)
        record['raw_fit_hours_prescreen'] = raw_window_count(
            conn, country, stream, source, FIT_START, FIT_END)
        record['screen_cost_gate_hours'] = record['raw_gate_hours_prescreen'] - record['n_gate_hours']
        record['screen_cost_fit_hours'] = record['raw_fit_hours_prescreen'] - record['n_fit_hours']
        out['sources'][source] = record

    # Do the two tables hold the same series where they overlap?
    ren, gen = series_by_source['energy_renewable'], series_by_source['energy_generation']
    gate_index = pd.date_range(GATE_START, GATE_END, freq='h', inclusive='left')
    a = ren.reindex(gate_index)
    b = gen.reindex(gate_index)
    both = a.notna() & b.notna()
    n_both = int(both.sum())
    if n_both:
        diff = (a[both] - b[both]).to_numpy(dtype=float)
        mean_level = float(np.abs(b[both].to_numpy(dtype=float)).mean())
        out['gate_agreement'] = {
            'n_both': n_both,
            'n_exact_equal': int(np.count_nonzero(diff == 0.0)),
            'mean_abs_diff_mw': round(float(np.abs(diff).mean()), 3),
            'mean_abs_diff_pct_of_level': (
                round(100.0 * float(np.abs(diff).mean()) / mean_level, 3) if mean_level else None),
            'max_abs_diff_mw': round(float(np.abs(diff).max()), 1),
        }
    else:
        out['gate_agreement'] = {'n_both': 0}

    ren_bar = out['sources']['energy_renewable'].get('d7_wape_pct')
    gen_bar = out['sources']['energy_generation'].get('d7_wape_pct')
    out['d7_bar_delta_pp'] = (
        round(ren_bar - gen_bar, 2) if (ren_bar is not None and gen_bar is not None) else None)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replica-db', default=os.environ.get('ENERGY_DB_PATH'))
    parser.add_argument('--audit-csv', default='reports/abl_318_renewable_data_audit.csv')
    parser.add_argument('--out')
    args = parser.parse_args()

    replica = args.replica_db or db.config.DATABASE_PATH
    if not Path(replica).exists():
        print(f"replica not found: {replica}", file=sys.stderr)
        return 2
    os.environ['ENERGY_DB_PATH'] = str(replica)
    db.config.DATABASE_PATH = str(replica)

    audit_note = crosscheck_audit(Path(args.audit_csv))
    print(audit_note, file=sys.stderr)
    conn = sqlite3.connect(f'file:{Path(replica).as_posix()}?mode=ro', uri=True)

    results = []
    for country, stream in TRANCHE_PAIRS:
        record = measure_pair(conn, country, stream)
        record['level_pair'] = (country, stream) in LEVEL_PAIRS
        results.append(record)
        print(f"  measured {country}/{stream}", file=sys.stderr)
    conn.close()

    payload = {
        'issue': 'ABL-348',
        'replica_db': str(replica),
        'replica_bytes': Path(replica).stat().st_size,
        'access': 'mode=ro, uri=True; nothing written to either database',
        'interpreter': sys.version,
        'registered_windows': {
            'lookback_start': str(LOOKBACK_START),
            'fit': [str(FIT_START), str(FIT_END)],
            'gate': [str(GATE_START), str(GATE_END)],
            'gate_hours': GATE_HOURS,
            'fit_hours': FIT_HOURS,
            'registered_min_n': REGISTERED_MIN_N,
        },
        'tranche_list_provenance': audit_note,
        'n_pairs': len(results),
        'pairs': results,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding='utf-8')
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
