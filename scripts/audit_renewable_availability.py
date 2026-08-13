#!/usr/bin/env python
"""
ABL-318 -- per-country availability and contamination audit for the three
renewable streams ABL-316 wants to extend to every country: solar,
wind_onshore, wind_offshore.

Produces one verdict per (country, stream) for every country in
config.SUPPORTED_COUNTRIES -- 24 x 3 = 72 rows, no gaps. A pair missing from
the output is a bug in this script, not a country without data.

    TRAIN               clean usable history; train on the full window
    TRAIN-FROM-<date>   usable, but the early window is damaged
    EXCLUDE-NO-DATA     stream not reported for this country (column is NULL)
    EXCLUDE-NO-FLEET    reported, but never once non-zero
    EXCLUDE-CONTAMINATED  data exists but is not trustworthy enough to train on
    ALREADY-COVERED     a model artifact exists today

One verdict beyond the six ABL-318 asked for, flagged rather than folded into
a neighbouring one because the distinction is decision-relevant:

    EXCLUDE-INSUFFICIENT-HISTORY  reported, real, non-trivial fleet, but under
                                  MIN_TRAIN_DAYS of history -- a "come back
                                  later", not a "never". Calling this
                                  CONTAMINATED would be false (the data is
                                  clean) and NO-DATA would be false (it is
                                  reported).

Two tables carry these streams and they do not agree:

  energy_generation  NaN-preserving. Declares "not reported" as NULL.
                     Covers 2021-01-01 -> now for every supported country.
  energy_renewable   the table src/db.py:load_renewable_type_data actually
                     trains from. Its mapper initialises every renewable
                     column to 0.0 before checking the source frame (ABL-188),
                     so it cannot say "not reported" at all, and its per-country
                     history is much shorter.

Both are censused. The verdict is set on energy_generation, which is the only
one of the two that can distinguish a NULL from a 0.0; the energy_renewable
census is carried alongside so the cost of training from the status-quo source
is visible per row.

Read-only by construction: the replica is opened with the SQLite read-only URI
form and this script issues no DML of any kind.

Usage:
    .venv/Scripts/python.exe scripts/audit_renewable_availability.py
    .venv/Scripts/python.exe scripts/audit_renewable_availability.py --out reports/abl_318_renewable_data_audit.md
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.data_quality import StreamQuality, summarize_stream  # noqa: E402

STREAMS = {
    'solar': 'solar_mw',
    'wind_onshore': 'wind_onshore_mw',
    'wind_offshore': 'wind_offshore_mw',
}

PRIMARY_TABLE = 'energy_generation'
STATUS_QUO_TABLE = 'energy_renewable'

# --- verdict thresholds -----------------------------------------------------
# A renewable target is weather-driven and strongly seasonal; anything under a
# full year of usable history cannot see every season once and is not a
# defensible training window for it.
MIN_TRAIN_DAYS = 365
# Fraction of the expected observations (span / cadence) that must actually be
# present and unflagged from a candidate start date onwards.
MIN_DENSITY = 0.95
# Suspect-constant coverage below this fraction of the window is noise to note,
# not grounds to move the start date.
NEGLIGIBLE_SUSPECT_FRAC = 0.01
# All-time peak below this is a demonstrator or a metering artefact, not a
# fleet: it sits under the rounding error of every other series on the
# dashboard and cannot repay a model. Quoted explicitly in every note that
# uses it so the threshold is arguable rather than hidden.
NEGLIGIBLE_FLEET_MW = 50.0

# Countries/streams holding a model artifact today (models/<CC>/<stream>/).
# Discovered from disk at runtime; this is only the fallback for a checkout
# without models/.
FALLBACK_ARTIFACTS = {
    ('AT', 'solar'), ('AT', 'wind_onshore'),
    ('BE', 'solar'), ('BE', 'wind_onshore'), ('BE', 'wind_offshore'),
    ('DE', 'solar'), ('DE', 'wind_onshore'), ('DE', 'wind_offshore'),
    ('FR', 'solar'), ('FR', 'wind_onshore'), ('FR', 'wind_offshore'),
}


# ============================================================================
# DB access -- read-only, and verified to be the live replica
# ============================================================================

def resolve_db_path() -> str:
    """
    ENERGY_DB_PATH wins over config.DATABASE_PATH.

    A git worktree has no .env (it is gitignored), so config.DATABASE_PATH
    degrades to a bare \\data\\energy_dashboard.db there and the nearest real
    file on this box is a stale 3.0 GB partial snapshot whose numbers look
    fine. Fail loudly rather than audit the decoy.
    """
    path = os.environ.get('ENERGY_DB_PATH') or str(config.DATABASE_PATH)
    p = Path(path)
    if not p.is_file():
        raise SystemExit(
            f"replica not found at {p!s}. Pass ENERGY_DB_PATH explicitly "
            f"(a worktree has no .env)."
        )
    return p.resolve().as_posix()


def connect_readonly(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)


def assert_live_replica(conn: sqlite3.Connection, db_path: str) -> Dict[str, str]:
    """
    Refuse to audit a stale partial snapshot.

    The decoy has zero energy_generation rows (the table does not exist in it
    at all) and stops in 2024; the live replica is current to the hour.
    """
    meta = {'db_path': db_path, 'db_size_gb': f"{Path(db_path).stat().st_size / 1e9:.2f}"}
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = {PRIMARY_TABLE, STATUS_QUO_TABLE} - tables
    if missing:
        raise SystemExit(f"{db_path} is missing {sorted(missing)} -- this is not the live replica.")

    latest = conn.execute(f"SELECT MAX(timestamp_utc) FROM {PRIMARY_TABLE}").fetchone()[0]
    meta['energy_generation_latest'] = latest
    age_h = (pd.Timestamp.now("UTC").tz_localize(None)
             - pd.to_datetime(latest, format='mixed')).total_seconds() / 3600.0
    meta['energy_generation_age_hours'] = f"{age_h:.1f}"
    if age_h > getattr(config, 'DB_STALE_AFTER_HOURS', 48):
        raise SystemExit(
            f"{PRIMARY_TABLE} latest row is {age_h:.1f}h old ({latest}) -- stale database, "
            f"refusing to publish an availability audit from it."
        )
    return meta


def load_country_stream(conn, table: str, country: str, value_col: str) -> pd.DataFrame:
    """
    All rows for one country/column. No date predicate: energy_renewable stores
    T-separated timestamps and energy_generation space-separated ones, and a
    SQL string range over the two silently drops rows (the bug that made
    ABL-188's first census undercount at 5,096 instead of 6,408). Filtering
    happens in pandas after a real datetime parse.
    """
    df = pd.read_sql_query(
        f"SELECT timestamp_utc, {value_col} AS target_value FROM {table} "
        f"WHERE country_code = ? ORDER BY timestamp_utc",
        conn, params=(country,),
    )
    if not df.empty:
        # Same normalisation the production read path applies
        # (db.py:load_renewable_type_data): a minority of rows carry a UTC
        # offset, so parse to UTC first, then drop the tz to get the naive-UTC
        # convention the rest of the module uses.
        df['timestamp_utc'] = pd.to_datetime(
            df['timestamp_utc'], format='mixed', utc=True
        ).dt.tz_localize(None)
        df = df.sort_values('timestamp_utc').reset_index(drop=True)
    return df


# ============================================================================
# Verdict engine
# ============================================================================

def _density(times: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """
    Share of the calendar days between start and end that are fully observed.

    Counted in *distinct hours covered per day*, not in observations, because
    most of these series change resolution mid-history -- CZ, FR, IT, LT, PL
    and others are hourly early and quarter-hourly later. Measuring against a
    single median cadence scores the hourly half of such a series at ~25%
    coverage and condemns a clean country as contaminated. A day is complete
    when it holds >= 23 of its 24 hours (23 admits the DST spring-forward day).
    """
    window = times[(times >= start) & (times <= end)]
    if window.empty:
        return 0.0
    total_days = max(int((end.normalize() - start.normalize()).days) + 1, 1)
    hours_per_day = window.dt.floor('h').drop_duplicates().dt.normalize().value_counts()
    complete_days = int((hours_per_day >= 23).sum())
    return complete_days / total_days


def choose_clean_start(df: pd.DataFrame, q: StreamQuality) -> Optional[pd.Timestamp]:
    """
    Earliest start date from which the remaining history is both long enough
    and dense enough to train on.

    Candidates are the first observation and every damage boundary -- the end
    of each suspect constant run and the far side of each long gap. The first
    candidate that leaves >= MIN_TRAIN_DAYS at >= MIN_DENSITY wins. Returns
    None when no candidate qualifies.
    """
    if q.first_ts is None or q.last_ts is None:
        return None

    times = pd.to_datetime(df['timestamp_utc'], format='mixed', errors='coerce')
    times = times[df['target_value'].notna().to_numpy() & times.notna()].sort_values()
    # A suspect run's rows are excluded from training, so they cannot count
    # towards density either.
    for run in q.suspect_runs:
        times = times[(times < run.start) | (times > run.end)]
    if times.empty:
        return None

    candidates = [q.first_ts]
    candidates += [run.end for run in q.suspect_runs]
    if q.longest_gap is not None:
        candidates.append(q.longest_gap.end)
    candidates = sorted({pd.Timestamp(c) for c in candidates})

    for cand in candidates:
        after = times[times > cand] if cand != q.first_ts else times[times >= cand]
        if after.empty:
            continue
        start = after.iloc[0]
        if (q.last_ts - start).days < MIN_TRAIN_DAYS:
            continue
        if _density(times, start, q.last_ts) >= MIN_DENSITY:
            return start
    return None


def decide(country: str, stream: str, df: pd.DataFrame,
           q: StreamQuality, has_artifact: bool) -> Dict[str, str]:
    """
    One verdict per country/stream, from the primary (NaN-preserving) census.

    ALREADY-COVERED is reported for pairs holding an artifact today, with the
    data verdict this audit would otherwise have given carried in data_verdict
    so a contaminated existing training window is still visible.
    """
    reasons: List[str] = []

    if q.n_nonnull == 0:
        verdict = 'EXCLUDE-NO-DATA'
        reasons.append(f"{q.n_rows} rows present for this country but "
                       f"{PRIMARY_TABLE}.{STREAMS[stream]} is NULL in every one "
                       f"-- ENTSO-E does not report this production type here")
    elif q.all_zero:
        verdict = 'EXCLUDE-NO-FLEET'
        reasons.append(f"reported and never once non-zero across n={q.n_nonnull} "
                       f"observations ({q.first_ts:%Y-%m-%d} to {q.last_ts:%Y-%m-%d}) "
                       f"-- a declared-zero series, not a missing one")
    elif q.max_value is not None and q.max_value < NEGLIGIBLE_FLEET_MW:
        verdict = 'EXCLUDE-NO-FLEET'
        reasons.append(f"reported, but all-time peak is {q.max_value:.1f} MW over "
                       f"n={q.n_nonnull} observations ({q.first_ts:%Y-%m-%d} to "
                       f"{q.last_ts:%Y-%m-%d}) -- below the {NEGLIGIBLE_FLEET_MW:.0f} MW "
                       f"demonstrator threshold, not a fleet worth modelling")
    elif q.span_days < MIN_TRAIN_DAYS:
        verdict = 'EXCLUDE-INSUFFICIENT-HISTORY'
        reasons.append(
            f"real and non-trivial (peak {q.max_value:.1f} MW) but only "
            f"{q.span_days:.0f}d of history ({q.first_ts:%Y-%m-%d} to "
            f"{q.last_ts:%Y-%m-%d}, n={q.n_nonnull}) -- under the {MIN_TRAIN_DAYS}d "
            f"needed to see every season once. Revisit, do not write off")
    else:
        suspect_frac = q.suspect_rows / q.n_nonnull if q.n_nonnull else 0.0
        clean_start = choose_clean_start(df, q)
        clean_enough = suspect_frac < NEGLIGIBLE_SUSPECT_FRAC
        long_gap_inside = q.longest_gap_hours >= 24 * 7

        if clean_start is None:
            verdict = 'EXCLUDE-CONTAMINATED'
            reasons.append(
                f"no start date leaves >= {MIN_TRAIN_DAYS}d of history at "
                f">= {MIN_DENSITY:.0%} density: n={q.n_nonnull} observations over "
                f"{q.span_days:.0f}d, {q.suspect_rows} rows in {len(q.suspect_runs)} "
                f"suspect constant run(s), longest gap {q.longest_gap_hours:.0f}h")
        elif clean_start <= q.first_ts and clean_enough and not long_gap_inside:
            verdict = 'TRAIN'
            reasons.append(f"n={q.n_nonnull} observations, {q.span_days:.0f}d, "
                           f"{q.pct_exact_zero:.1f}% exact zeros, longest gap "
                           f"{q.longest_gap_hours:.0f}h, no suspect constant run "
                           f">= 24h")
        elif clean_start <= q.first_ts:
            verdict = 'TRAIN'
            reasons.append(
                f"full window usable: n={q.n_nonnull}, {q.span_days:.0f}d; damage is "
                f"interior and small ({q.suspect_rows} rows, {suspect_frac:.2%}; "
                f"longest gap {q.longest_gap_hours:.0f}h) and is nulled at load by "
                f"exclude_suspect_constant_runs")
        else:
            verdict = f"TRAIN-FROM-{clean_start:%Y-%m-%d}"
            reasons.append(
                f"early window damaged; from {clean_start:%Y-%m-%d} there are "
                f"{(q.last_ts - clean_start).days}d at >= {MIN_DENSITY:.0%} density. "
                f"Before it: {q.suspect_rows} rows in {len(q.suspect_runs)} suspect "
                f"constant run(s), longest gap {q.longest_gap_hours:.0f}h")

    data_verdict = verdict
    if has_artifact:
        verdict = 'ALREADY-COVERED'

    return {'verdict': verdict, 'data_verdict': data_verdict, 'note': '; '.join(reasons)}


def duplicate_instants(df: pd.DataFrame) -> Dict[str, int]:
    """
    Rows whose UTC instant is already claimed by another row of the same table.

    energy_renewable's UNIQUE index is on (country_code, timestamp_utc) as a
    *string*, so '2025-11-09 23:00:00', '2025-11-09T23:00:00' and
    '2025-11-09T00:00:00+01:00' are three distinct keys for one instant and all
    three can be stored. `disagreeing` counts the instants where the duplicate
    rows do not even agree on the value -- there the table holds two different
    answers for the same hour and nothing downstream picks between them.
    energy_generation is clean on this axis; only energy_renewable is affected.
    """
    if df.empty:
        return {'dup_rows': 0, 'dup_disagreeing': 0}
    dup = df[df['timestamp_utc'].duplicated(keep=False)]
    if dup.empty:
        return {'dup_rows': 0, 'dup_disagreeing': 0}
    nunique = dup.groupby('timestamp_utc')['target_value'].nunique(dropna=False)
    return {'dup_rows': int(len(dup)), 'dup_disagreeing': int((nunique > 1).sum())}


def corroborate_zerofill(ren: pd.DataFrame, gen: pd.DataFrame) -> Dict[str, float]:
    """
    ABL-188's adjudication test, applied per country/stream.

    A 0.0 in energy_renewable is contamination only when the NaN-preserving
    sibling has real generation at the identical timestamp -- the two tables
    come from one A75 fetch, so a disagreement is one mapper losing what the
    other kept. Without this test the run detector over-reports: BE
    wind_offshore has 9 zero runs >= 24h in energy_renewable, but 8 of them sit
    on windows where energy_generation is also <= 0 (small negative house load
    in calm wind) and are genuine near-zero generation, not zero-fill. Only the
    9th -- where energy_generation peaks at 2,175 MW -- is the defect.
    """
    if ren.empty or gen.empty:
        return {'zerofill_rows': 0, 'zerofill_peak_mw': 0.0, 'calm_zero_rows': 0}
    g = gen.dropna(subset=['target_value']).set_index('timestamp_utc')['target_value']
    g = g[~g.index.duplicated()]
    r = ren.dropna(subset=['target_value'])
    # energy_renewable stores one instant under several string spellings, so
    # count each contradicted instant once (see duplicate_instants).
    r = r[~r['timestamp_utc'].duplicated()]
    r = r[r['target_value'] == 0.0]
    if r.empty:
        return {'zerofill_rows': 0, 'zerofill_peak_mw': 0.0, 'calm_zero_rows': 0,
                'zerofill_threshold_mw': 0.0}

    # "Contradicted" must mean materially non-zero, not merely > 0. Several of
    # these series carry a few MW of metering noise through the night, so a
    # bare `> 0` test marks every genuine solar night as contamination (it
    # inflated DE solar from ~6.4k rows to 15.3k before this threshold went in).
    # Scale the floor to the series: 1% of its own 99th percentile.
    threshold = 0.01 * float(g.quantile(0.99))
    aligned = g.reindex(r['timestamp_utc'].to_numpy()).dropna()
    contradicted = aligned[aligned > threshold]
    return {
        'zerofill_rows': int(len(contradicted)),
        'zerofill_peak_mw': round(float(contradicted.max()), 1) if len(contradicted) else 0.0,
        'calm_zero_rows': int((aligned <= threshold).sum()),
        'zerofill_threshold_mw': round(threshold, 2),
    }


def discover_artifacts(models_dir: Path) -> Dict[tuple, dict]:
    """
    What actually exists on disk per country/stream.

    The presence of a models/<CC>/<stream>/ directory proves nothing: the
    variant scaffolding (candidate/, centroid/, multipoint/, production/) is
    created for pairs that the daily job cannot serve. Forecaster.load reads
    exactly models/<CC>/<stream>/model.joblib and raises FileNotFoundError
    otherwise, so `servable` is that one file and nothing else. DE/wind_offshore
    is the case this distinction exists for: it holds production/model.joblib
    and candidate/model.joblib but no top-level model.joblib, and serves zero
    rows.
    """
    if not models_dir.is_dir():
        return {k: {'servable': True, 'variants': []} for k in FALLBACK_ARTIFACTS}
    found: Dict[tuple, dict] = {}
    for cc_dir in sorted(models_dir.iterdir()):
        if not cc_dir.is_dir() or cc_dir.name not in config.SUPPORTED_COUNTRIES:
            continue
        for stream in STREAMS:
            sd = cc_dir / stream
            if not sd.is_dir():
                continue
            variants = sorted(
                p.parent.name for p in sd.glob('*/model.joblib')
            )
            servable = (sd / 'model.joblib').is_file()
            if servable or variants:
                found[(cc_dir.name, stream)] = {
                    'servable': servable, 'variants': variants,
                }
    return found


def served_pairs(conn) -> Dict[tuple, int]:
    """Rows actually written to `forecasts` per country/stream -- a trained
    model that writes nothing serves nothing."""
    rows = conn.execute(
        "SELECT country_code, renewable_type, COUNT(*) FROM forecasts "
        "WHERE renewable_type IN ('solar','wind_onshore','wind_offshore') "
        "GROUP BY 1, 2"
    ).fetchall()
    return {(cc, rt): n for cc, rt, n in rows}


# ============================================================================
# Precedent check -- if the detector cannot re-find these, it is wrong
# ============================================================================

def precedent_check(conn) -> List[str]:
    """
    ABL-188 (DE solar) and ABL-198/199/200 (BE wind_offshore) are known
    zero-fill runs in energy_renewable. Re-find them, or this audit's negatives
    mean nothing.
    """
    lines = []
    for country, stream, label in (('DE', 'solar', 'ABL-188'),
                                   ('BE', 'wind_offshore', 'ABL-198/199/200')):
        df = load_country_stream(conn, STATUS_QUO_TABLE, country, STREAMS[stream])
        q = summarize_stream(df, 'target_value', STATUS_QUO_TABLE)
        zero_runs = [r for r in q.suspect_runs if r.value == 0.0]
        if not zero_runs:
            lines.append(f"FAIL {label}: no suspect zero run found for "
                         f"{country}/{stream} in {STATUS_QUO_TABLE}")
            continue
        biggest = max(zero_runs, key=lambda r: r.n_rows)
        lines.append(
            f"OK   {label}: {country}/{stream} {STATUS_QUO_TABLE} -- longest zero run "
            f"{biggest.start} -> {biggest.end} ({biggest.n_rows} rows, "
            f"{biggest.duration_hours:.2f}h); {len(zero_runs)} zero run(s) >= 24h "
            f"totalling {sum(r.n_rows for r in zero_runs)} rows")
    return lines


# ============================================================================
# Main
# ============================================================================

def audit(conn) -> List[dict]:
    artifacts = discover_artifacts(Path(__file__).resolve().parent.parent / 'models')
    served_rows = served_pairs(conn)
    rows = []

    for country in config.SUPPORTED_COUNTRIES:
        for stream, value_col in STREAMS.items():
            gen = load_country_stream(conn, PRIMARY_TABLE, country, value_col)
            ren = load_country_stream(conn, STATUS_QUO_TABLE, country, value_col)
            qg = summarize_stream(gen, 'target_value', PRIMARY_TABLE)
            qr = summarize_stream(ren, 'target_value', STATUS_QUO_TABLE)

            zf = corroborate_zerofill(ren, gen)
            dup_g = duplicate_instants(gen)
            dup_r = duplicate_instants(ren)
            art = artifacts.get((country, stream), {'servable': False, 'variants': []})
            served = served_rows.get((country, stream), 0)
            d = decide(country, stream, gen, qg, art['servable'])

            # A trained model that writes nothing serves nothing -- record
            # both sides of that gap rather than only the artifact.
            if art['variants'] and not art['servable']:
                d['note'] += (
                    f". ARTIFACT ORPHANED: {'/'.join(art['variants'])}/model.joblib "
                    f"exist(s) but models/{country}/{stream}/model.joblib -- the only "
                    f"path Forecaster.load reads -- does not, so forecast_daily.py "
                    f"skips it and {served} rows are served")
            elif art['servable'] and served == 0:
                d['note'] += ". ARTIFACT SERVES NOTHING: servable model.joblib present, 0 forecasts rows"

            rows.append({
                'country': country,
                'stream': stream,
                'verdict': d['verdict'],
                'data_verdict': d['data_verdict'],
                'first_ts': qg.first_ts, 'last_ts': qg.last_ts,
                'n_rows': qg.n_rows, 'n_nonnull': qg.n_nonnull,
                'pct_zero': round(qg.pct_exact_zero, 2),
                'max_mw': round(qg.max_value, 1) if qg.max_value is not None else None,
                'longest_zero_run_h': round(qg.longest_zero_run_hours, 2),
                'longest_gap_h': round(qg.longest_gap_hours, 2),
                'suspect_runs': len(qg.suspect_runs),
                'suspect_rows': qg.suspect_rows,
                'cadence_min': qg.cadence_minutes,
                # status quo (what training reads today)
                'ren_first_ts': qr.first_ts, 'ren_last_ts': qr.last_ts,
                'ren_n_rows': qr.n_rows, 'ren_n_nonnull': qr.n_nonnull,
                'ren_pct_zero': round(qr.pct_exact_zero, 2),
                'ren_longest_zero_run_h': round(qr.longest_zero_run_hours, 2),
                'ren_longest_gap_h': round(qr.longest_gap_hours, 2),
                'ren_suspect_runs': len(qr.suspect_runs),
                'ren_suspect_rows': qr.suspect_rows,
                # ABL-188 corroboration: renewable==0.0 while generation>0
                'ren_zerofill_rows': zf['zerofill_rows'],
                'ren_zerofill_peak_mw': zf['zerofill_peak_mw'],
                'ren_calm_zero_rows': zf['calm_zero_rows'],
                'ren_zerofill_threshold_mw': zf['zerofill_threshold_mw'],
                'dup_instant_rows': dup_g['dup_rows'],
                'ren_dup_instant_rows': dup_r['dup_rows'],
                'ren_dup_disagreeing': dup_r['dup_disagreeing'],
                'artifact_servable': art['servable'],
                'artifact_variants': ','.join(art['variants']),
                'served_rows': served,
                'skip_configured': stream in config.SKIP_RENEWABLE_TYPES.get(country, []),
                'note': d['note'],
            })
    return rows


def write_markdown(df: pd.DataFrame, out: Path, meta: Dict[str, str],
                   precedent: List[str], generated_at: str, csv_path: str) -> None:
    """Emit the 72-row verdict table. Regenerated, never hand-edited."""
    def fmt(v, nd=0):
        if pd.isna(v):
            return '-'
        return f"{v:,.{nd}f}" if isinstance(v, (int, float)) else str(v)

    def ts(v):
        return '-' if pd.isna(v) else str(v)[:16]

    lines = [
        "# ABL-318 — per-country data audit: solar, wind_onshore, wind_offshore",
        "",
        f"Generated `{generated_at}` by `scripts/audit_renewable_availability.py`.",
        f"Replica `{meta['db_path']}` ({meta['db_size_gb']} GB), `energy_generation` "
        f"current to `{meta['energy_generation_latest']}`. Read-only "
        f"(`file:...?mode=ro`); no write of any kind touched the replica.",
        "",
        "Verdicts are set on **`energy_generation`** (NaN-preserving, covers "
        "2021-01-01 → now for all 24 countries), not on `energy_renewable` "
        "(the table training reads today). Columns prefixed `ren_` census "
        "`energy_renewable` so the cost of the status-quo source is visible.",
        "",
        "## Precedent check",
        "",
        "The run detector must re-find the known contamination or its negatives "
        "mean nothing:",
        "",
        "```",
        *precedent,
        "```",
        "",
        "## Verdict table (24 countries × 3 streams = 72 rows)",
        "",
        "| country | stream | verdict | first actual | last actual | rows | non-null | % zero | peak MW | longest zero-run (h) | longest gap (h) | note |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['country']} | {r['stream']} | `{r['verdict']}` | {ts(r['first_ts'])} | "
            f"{ts(r['last_ts'])} | {fmt(r['n_rows'])} | {fmt(r['n_nonnull'])} | "
            f"{fmt(r['pct_zero'], 2)} | {fmt(r['max_mw'], 1)} | "
            f"{fmt(r['longest_zero_run_h'], 1)} | {fmt(r['longest_gap_h'], 1)} | "
            f"{r['note']} |"
        )

    lines += ["", "## Status-quo source census (`energy_renewable`)", "",
              "| country | stream | ren first | ren last | ren non-null | ren % zero | "
              "zero-filled rows | peak contradicted MW | dup instants | dup disagreeing |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['country']} | {r['stream']} | {ts(r['ren_first_ts'])} | "
            f"{ts(r['ren_last_ts'])} | {fmt(r['ren_n_nonnull'])} | "
            f"{fmt(r['ren_pct_zero'], 2)} | {fmt(r['ren_zerofill_rows'])} | "
            f"{fmt(r['ren_zerofill_peak_mw'], 1)} | {fmt(r['ren_dup_instant_rows'])} | "
            f"{fmt(r['ren_dup_disagreeing'])} |"
        )

    lines += ["", "## Verdict counts", ""]
    for stream in STREAMS:
        sub = df[df['stream'] == stream]
        counts = sub['data_verdict'].str.replace(
            r'TRAIN-FROM-.*', 'TRAIN-FROM-<date>', regex=True).value_counts()
        lines.append(f"**{stream}** — " + ", ".join(f"`{k}` {v}" for k, v in counts.items()))
        lines.append("")

    lines += [f"Full machine-readable output with every column: `{csv_path}`.", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='reports/abl_318_renewable_data_audit.md')
    ap.add_argument('--csv', default='reports/abl_318_renewable_data_audit.csv')
    args = ap.parse_args()

    db_path = resolve_db_path()
    conn = connect_readonly(db_path)
    try:
        meta = assert_live_replica(conn, db_path)
        print(f"replica: {meta['db_path']} ({meta['db_size_gb']} GB), "
              f"{PRIMARY_TABLE} current to {meta['energy_generation_latest']} "
              f"({meta['energy_generation_age_hours']}h old)")

        print("\n-- precedent check (ABL-188, ABL-198/199/200) --")
        precedent = precedent_check(conn)
        for line in precedent:
            print(line)

        print(f"\n-- auditing {len(config.SUPPORTED_COUNTRIES)} countries x "
              f"{len(STREAMS)} streams --")
        rows = audit(conn)
    finally:
        conn.close()

    expected = len(config.SUPPORTED_COUNTRIES) * len(STREAMS)
    assert len(rows) == expected, f"produced {len(rows)} rows, expected {expected}"

    df = pd.DataFrame(rows)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    print(f"\nwrote {args.csv} ({len(df)} rows)")

    for stream in STREAMS:
        sub = df[df['stream'] == stream]
        print(f"\n{stream}:")
        for verdict, n in sub['data_verdict'].str.replace(
                r'TRAIN-FROM-.*', 'TRAIN-FROM-<date>', regex=True).value_counts().items():
            print(f"   {verdict:24} {n}")

    generated_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    write_markdown(df, Path(args.out), meta, precedent, generated_at, args.csv)
    print(f"\nwrote {args.out}")
    print(f"generated_at (UTC): {generated_at}")


if __name__ == '__main__':
    main()
