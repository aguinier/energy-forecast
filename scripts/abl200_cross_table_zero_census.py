"""ABL-200 cross-table zero census: what the sibling rule excludes that the
24h duration rule does not.

Read-only. Opens the replica through the read-only SQLite URI and writes
nothing to it. Every number in reports/abl_200_cross_table_zero_disproof.md
comes from here; re-run it rather than quoting the report, because this
database self-repairs -- an approved remediation once acted on an enumeration
of which 48% of the rows had already healed four minutes later.

Usage:
  python scripts/abl200_cross_table_zero_census.py
  python scripts/abl200_cross_table_zero_census.py --quantiles 0.9 0.95 0.99 1.0
  python scripts/abl200_cross_table_zero_census.py --json-out reports/abl_200_census.json
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src import db
from src.data_quality import (
    SIBLING_DISPROOF_MIN_CALIBRATION_ROWS,
    SIBLING_DISPROOF_QUANTILE,
    adjudicate_zeros_against_sibling,
    find_suspect_constant_runs,
)

#: ABL-348's registered windows. Reported per pair because a row this rule
#: excludes inside the fit window changes a registered gate's training set,
#: which per ABL-401 is a new pre-registration and never a re-read of a
#: published path.
FIT_WINDOW = ("2026-01-14", "2026-07-11")
GATE_WINDOW = ("2026-07-11", "2026-08-10")


def read_series(conn, table, country_code, target_col):
    """The same frame `db._read_renewable_series` builds, against an explicitly
    read-only connection -- parsed instants, one row per instant."""
    query = f"""
        SELECT timestamp_utc, ({target_col}) as target_value
        FROM {table}
        WHERE country_code = ?
          AND data_quality = 'actual'
          AND ({target_col}) IS NOT NULL
        ORDER BY timestamp_utc
    """
    df = pd.read_sql_query(query, conn, params=(country_code,))
    if df.empty:
        return df
    df['timestamp_utc'] = pd.to_datetime(
        df['timestamp_utc'], format='mixed', utc=True
    ).dt.tz_localize(None)
    if df['timestamp_utc'].duplicated().any():
        spellings = df.groupby('timestamp_utc')['target_value']
        disagreeing = spellings.nunique(dropna=False) > 1
        collapsed = spellings.last()
        collapsed[disagreeing] = float('nan')
        df = collapsed.reset_index()[['timestamp_utc', 'target_value']]
    return df


def duration_rule_mask(series):
    """Rows the existing ABL-188 guard already nulls, at the published
    resolution -- the baseline this rule is marginal to."""
    mask = pd.Series(False, index=series.index)
    for run in find_suspect_constant_runs(series, value_col='target_value'):
        mask |= ((series['timestamp_utc'] >= run.start)
                 & (series['timestamp_utc'] <= run.end))
    return mask


def census_pair(conn, country_code, renewable_type, quantiles):
    target_col = db.RENEWABLE_TYPE_COLUMNS[renewable_type]
    series = read_series(conn, db.RENEWABLE_TYPE_SOURCE_TABLE, country_code, target_col)
    if series.empty:
        return None
    sibling = read_series(
        conn, db.RENEWABLE_ZERO_DISPROOF_SOURCE, country_code, target_col
    )
    already = duration_rule_mask(series)

    row = {
        'country': country_code,
        'renewable_type': renewable_type,
        'n_observed': int(series['target_value'].notna().sum()),
        'n_exact_zero': int((series['target_value'] == 0.0).sum()),
        'n_excluded_by_duration_rule': int(already.sum()),
        'sibling_rows': int(len(sibling)),
    }
    for q in quantiles:
        verdict = adjudicate_zeros_against_sibling(series, sibling, quantile=q)
        marginal = verdict.mask & ~already
        key = f'q{q:g}'
        row[f'evaluable_{key}'] = verdict.evaluable
        row[f'floor_{key}'] = verdict.floor
        row[f'calibration_n_{key}'] = verdict.calibration_n
        row[f'n_disproved_{key}'] = verdict.n_disproved
        row[f'n_marginal_{key}'] = int(marginal.sum())
        if q == SIBLING_DISPROOF_QUANTILE:
            row['reason'] = verdict.reason
            row['max_disproving_value'] = verdict.max_disproving_value
            ts = series.loc[marginal, 'timestamp_utc']
            row['marginal_first'] = str(ts.min()) if len(ts) else None
            row['marginal_last'] = str(ts.max()) if len(ts) else None
            row['n_marginal_in_fit_window'] = int(
                ((ts >= FIT_WINDOW[0]) & (ts < FIT_WINDOW[1])).sum()
            )
            row['n_marginal_in_gate_window'] = int(
                ((ts >= GATE_WINDOW[0]) & (ts < GATE_WINDOW[1])).sum()
            )
    return row


def main():
    parser = argparse.ArgumentParser(description=(
        "ABL-200 read-only census: rows the cross-table zero rule excludes "
        "that the 24h duration rule does not."
    ))
    parser.add_argument('--quantiles', type=float, nargs='+',
                        default=[0.90, 0.95, 0.99, 1.00],
                        help='calibration quantiles to report the rule at')
    parser.add_argument('--json-out', default=None,
                        help='write the per-pair census here')
    parser.add_argument('--db', default=None,
                        help='replica path; defaults to config.DATABASE_PATH')
    args = parser.parse_args()

    path = args.db or config.DATABASE_PATH
    print(f"replica (read-only): {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    rows = []
    for country_code in config.SUPPORTED_COUNTRIES:
        for renewable_type in db.RENEWABLE_TYPE_COLUMNS:
            row = census_pair(conn, country_code, renewable_type, args.quantiles)
            if row is not None:
                rows.append(row)
    conn.close()

    table = pd.DataFrame(rows)
    registered = f'q{SIBLING_DISPROOF_QUANTILE:g}'
    marginal_col = f'n_marginal_{registered}'

    print(f"\npairs censused: {len(table)}")
    print(f"pairs the rule refuses to adjudicate (no calibration population): "
          f"{int((~table[f'evaluable_{registered}']).sum())}")
    print(f"\nmarginal exclusions (this rule, beyond the 24h duration rule):")
    for q in args.quantiles:
        col = f'n_marginal_q{q:g}'
        firing = table[table[col] > 0]
        print(f"  q={q:<5} {int(table[col].sum()):6d} rows over {len(firing):3d} pairs"
              + ("   <- registered" if q == SIBLING_DISPROOF_QUANTILE else ""))

    firing = table[table[marginal_col] > 0].sort_values(marginal_col, ascending=False)
    print(f"\nper pair at the registered q={SIBLING_DISPROOF_QUANTILE}:")
    cols = ['country', 'renewable_type', 'n_observed', 'n_exact_zero',
            'n_excluded_by_duration_rule', f'floor_{registered}', marginal_col,
            'max_disproving_value', 'n_marginal_in_fit_window',
            'n_marginal_in_gate_window']
    print(firing[cols].to_string(index=False))

    in_fit = int(table['n_marginal_in_fit_window'].sum())
    in_gate = int(table['n_marginal_in_gate_window'].sum())
    print(f"\nABL-348 fit window  {FIT_WINDOW}: {in_fit} rows over "
          f"{int((table['n_marginal_in_fit_window'] > 0).sum())} pairs")
    print(f"ABL-348 gate window {GATE_WINDOW}: {in_gate} rows over "
          f"{int((table['n_marginal_in_gate_window'] > 0).sum())} pairs")
    if in_fit:
        print("  -> a registered scope covering any of those pairs has a "
              "changed training set. Per ABL-401 that is a NEW pre-registration, "
              "not a re-read of a published path.")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            'meta': {
                'replica_db': str(path),
                'registered_quantile': SIBLING_DISPROOF_QUANTILE,
                'min_calibration_rows': SIBLING_DISPROOF_MIN_CALIBRATION_ROWS,
                'training_source': db.RENEWABLE_TYPE_SOURCE_TABLE,
                'disproof_source': db.RENEWABLE_ZERO_DISPROOF_SOURCE,
                'fit_window': list(FIT_WINDOW),
                'gate_window': list(GATE_WINDOW),
            },
            'pairs': json.loads(table.to_json(orient='records')),
        }, indent=2))
        print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
