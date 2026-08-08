"""V014 feature builder — serve-faithful features for the D+2 net-position challenger.

V014 is a *tabular* model, which changes the serve-time problem in a way the
Chronos-2 champion does not have. Chronos-2 consumes a context series that ends
where the observations end, so "what did the run actually hold?" is answered
once, by the context cutoff. A tabular model evaluates every feature **at the
target timestamp**, so each feature has to justify its own availability. A
single column read one hour past what the run held turns the whole backtest into
a claim about information nobody had.

Serve-faithfulness here holds **by construction, not by verification.** That is
forced on us: `fetched_at` and `publication_timestamp_utc` are last-write over a
rolling re-fetch window, not first-seen — every FR `net_position` row for targets
2026-08-01..07 carries the identical `fetched_at = 2026-08-07 00:34:48` — so an
"as-of" query over ingest metadata silently passes whatever you hand it (ABL-69
audit). The only sound construction is one documented per-source cutoff, derived
from the run instant, applied identically in training, backtest and serving.

**What is actually checked, and why the obvious check is worthless.** Filtering a
series to `index <= cutoff` and then asserting that its maximum index is inside
the cutoff is a tautology — it cannot fail, and a guard that cannot fail reads as
a guarantee while providing none. The construction that *can* go wrong is the
**lag arithmetic**: a same-hour lag of L hours reaches `max(target) - L`, and if
that is past the cutoff the feature is unavailable at serve time for the tail of
the target day. `assert_lag_is_serve_safe` checks exactly that, on every lag,
every build. It fires on a 48h lag and passes on 72h — see `SAME_HOUR_LAGS`.

## The cutoffs, and where they were measured

A run fires at **06:00 UTC on day D** for the whole of day **D+2**.

- **Day-ahead-published sources** (`net_position`, `energy_price`,
  `energy_load_forecast`, `energy_generation_forecast`): day X is published
  ~12:45 CET on X−1, so a 06:00Z run on D holds day D and *nothing* of D+1.
  Measured 2026-08-07: newest `net_position` target at a 06:00Z run is
  **D 21:00** for all 20 live countries without exception (ABL-28, re-verified
  under ABL-69). The tightest target hour is D+2 23:00, exactly 50h later —
  which is why the same-hour lags below start at 72h rather than 48h: a 48h
  same-hour lag reaches D 22:00 and D 23:00 for the target day's last two
  hours, and those two hours do not exist yet at run time.
- **`crossborder_flows`**: physical flows lag ~a day. ABL-74 measured FR's
  apparent 18h deficit and refuted it — a recovered 4-pass ingest write failure,
  not a publication lag — and decided **keep 72h and carry an explicit
  missingness indicator** rather than inflating every country's lag forever to
  pay for a transient bug. `xb_missing` is that indicator; see `_crossborder`.
- **Weather** is the one source legitimately evaluated *at the target hour*: a
  run issued at D−1 12:00Z covers 14 days ahead, so D+2 is comfortably inside
  it. It is admitted only as `data_quality = 'forecast'` with
  `forecast_run_time <= run_ts`.

## The weather archive does not reach the early backtest weeks

`weather_data` carries two populations and they are not interchangeable:
`data_quality = 'actual'` is reanalysis stamped with `forecast_run_time ==
timestamp_utc` (a nowcast — lead 0.0h for 100% of rows, 2023-01 through
2025-12), and `data_quality = 'forecast'` is the real issued-run archive, which
**begins 2026-01-11** (measured 2026-08-08: FR's earliest `forecast_run_time` is
`2026-01-11 18:00`, 193 distinct runs since). Measured DE forecast-quality rows
per backtest week: W01-W10 have **zero**, W11 has 696, W12 has 1,560.

So weather features are structurally absent for W01-W10 and this builder emits
them as NaN there — it does **not** fall back to the reanalysis, which would be
observed weather presented to the model as a forecast and would flatter every
weather-driven score in the backtest. XGBoost consumes the NaN natively.
`weather_available` records which regime each row is in, so a score can never be
read as weather-informed when it was not. The champion's loader filters to
`data_quality = 'forecast'` too (`src/chronos2/input_builder.py:237`), so
W01-W10 are weather-blind for *both* models and the comparison stays fair — but
neither model is in its serving configuration there.

`WEATHER_COLUMNS` names only columns that are actually populated. `pv_poa_wm2`
and `ghi_est_wm2` exist in the schema and are **NULL in 100% of forecast-quality
rows** (0 of 64,848 for FR, measured 2026-08-08); asking for them yields an
all-NaN feature that reads on an importance plot as "the model ignored solar
irradiance" rather than as "we never stored it".

## One database read per source, not one per run day

`SourceCache` loads each source once over the whole training span and every
window slices it in memory. Training spans ~1,300 run days per country; a
per-window query made that ~0.2 s x 1,300 x 19 countries. The cutoff filter is
identical either way — it is `_bounded`, applied on every slice — so this is a
speed change, not a semantics change.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("energy_forecast.v014")


# ---------------------------------------------------------------------------
# Serve-time constants. These are the model's contract with the schedule; a
# change here is a change to what V014 is allowed to know.
# ---------------------------------------------------------------------------

RUN_HOUR_UTC = 6
TARGET_DAY_OFFSET = 2

#: Newest target hour a day-ahead-published source holds at a 06:00Z run on D.
DAY_AHEAD_CUTOFF_HOUR = 21

#: Lag (hours, relative to the target hour) at which cross-border physical
#: flows are considered available. ABL-74.
CROSSBORDER_SERVE_LAG_HOURS = 72

#: Same-hour lags for the day-ahead sources. 72h is the freshest same-hour lag
#: that is safe at *every* hour of the target day (see module docstring), and
#: `assert_lag_is_serve_safe` is what holds that claim to account.
SAME_HOUR_LAGS = (72, 96, 120, 168)

#: Trailing windows (in days, ending at the run-day cutoff) for the run-anchored
#: aggregates. These are identical for all 24 target hours of a run.
TRAILING_WINDOWS_DAYS = (1, 3, 7, 28)

#: Weather columns that carry values on forecast-quality rows. See docstring.
WEATHER_COLUMNS = (
    "temperature_2m_k",
    "wind_speed_10m_ms",
    "wind_speed_100m_ms",
    "shortwave_radiation_wm2",
)

GENERATION_FORECAST_COLUMNS = {
    "solar_mw", "wind_onshore_mw", "wind_offshore_mw", "total_forecast_mw",
}


class ServeFaithfulnessError(AssertionError):
    """Raised when a feature would reach past what the run could have held."""


@dataclass(frozen=True)
class ServeWindow:
    """What one D+2 run is allowed to see.

    `run_ts` is the run instant (D 06:00Z). `day_ahead_cutoff` is the newest
    target timestamp any day-ahead-published source holds at that instant.
    `target_index` is the 24 hours of D+2 the run forecasts.
    """

    run_ts: pd.Timestamp
    day_ahead_cutoff: pd.Timestamp
    target_index: pd.DatetimeIndex

    @classmethod
    def for_run_day(cls, run_day) -> "ServeWindow":
        day = pd.Timestamp(run_day).normalize()
        run_ts = day + pd.Timedelta(hours=RUN_HOUR_UTC)
        cutoff = day + pd.Timedelta(hours=DAY_AHEAD_CUTOFF_HOUR)
        target_start = day + pd.Timedelta(days=TARGET_DAY_OFFSET)
        target_index = pd.date_range(target_start, periods=24, freq="h")
        return cls(run_ts=run_ts, day_ahead_cutoff=cutoff, target_index=target_index)

    @classmethod
    def for_target_day(cls, target_day) -> "ServeWindow":
        run_day = pd.Timestamp(target_day).normalize() - pd.Timedelta(days=TARGET_DAY_OFFSET)
        return cls.for_run_day(run_day)


def assert_lag_is_serve_safe(window: ServeWindow, lag_hours: int, what: str) -> None:
    """Fail if a same-hour lag reaches past what the run held.

    This is the check that can actually fire. A lag of L applied to target hour
    t reads `t - L`; the binding case is the *last* target hour, so the feature
    is available for the whole day only when `max(target) - L <= cutoff`. At the
    measured cutoff (D 21:00) and target day D+2, that admits L >= 50h and
    rejects the tempting 48h — which would reach D 22:00 and D 23:00, two hours
    that do not exist at a 06:00Z run.
    """
    reach = window.target_index.max() - pd.Timedelta(hours=lag_hours)
    if reach > window.day_ahead_cutoff:
        raise ServeFaithfulnessError(
            f"{what}: a {lag_hours}h lag reaches {reach} for target hour "
            f"{window.target_index.max()}, past the serve cutoff "
            f"{window.day_ahead_cutoff}. The run did not hold that value."
        )


def _bounded(series: pd.Series, cutoff: pd.Timestamp) -> pd.Series:
    """The cutoff, applied. Every feature derived from a source goes through it."""
    if series.empty:
        return series
    return series[series.index <= cutoff]


# ---------------------------------------------------------------------------
# Loading. One read per source over the whole span; windows slice in memory.
# ---------------------------------------------------------------------------

def _widened(start, end) -> tuple[str, str]:
    """Widen a window by a day on each side, for the string bound in SQL.

    Both timestamp separators occur in this database (`2026-07-20T00:00:00` and
    `2026-07-20 00:00:00`) and `'T'` (84) sorts above `' '` (32), so no single
    form is a correct bound while both exist — but wrapping the column in
    `REPLACE` forfeits the index seek on a multi-million-row table. Widening
    keeps the seek; the exact filter is re-applied in pandas, which has parsed
    both forms and is therefore separator-agnostic.
    """
    lo = (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    hi = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    return lo, hi


def _empty_series() -> pd.Series:
    """An absent source, still indexed by time.

    The index type matters, not just the emptiness. A bare `pd.Series(dtype=
    float)` carries a `RangeIndex`, and the trailing-aggregate and climatology
    helpers compare `series.index` against a `Timestamp` — which raises
    `TypeError: '>=' not supported between instances of 'numpy.ndarray' and
    'Timestamp'` rather than yielding the NaN feature the caller expects. That
    is a crash on exactly the country this model is supposed to handle
    gracefully: one that reports no price, or no flows, at all.
    """
    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def _to_series(df: pd.DataFrame, hourly_mean: bool = False) -> pd.Series:
    if df.empty:
        return _empty_series()
    ts = pd.to_datetime(df.iloc[:, 0], format="mixed", utc=True).dt.tz_localize(None)
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    series = pd.Series(np.asarray(values), index=pd.DatetimeIndex(ts)).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    if hourly_mean and not series.empty:
        # Prices and TSO forecasts are stored at 15-minute resolution in the
        # recent era and hourly before it; the target is hourly, so average
        # within the hour rather than picking one quarter arbitrarily.
        series = series.resample("h").mean().dropna()
    return series


def load_net_position(conn, country: str, start, end) -> pd.Series:
    lo, hi = _widened(start, end)
    return _to_series(pd.read_sql_query(
        "SELECT timestamp_utc, net_position_mw FROM net_position "
        "WHERE country_code = ? AND timestamp_utc >= ? AND timestamp_utc <= ? "
        "ORDER BY timestamp_utc", conn, params=(country, lo, hi)))


def load_price(conn, country: str, start, end) -> pd.Series:
    lo, hi = _widened(start, end)
    return _to_series(pd.read_sql_query(
        "SELECT timestamp_utc, price_eur_mwh FROM energy_price "
        "WHERE country_code = ? AND timestamp_utc >= ? AND timestamp_utc <= ? "
        "ORDER BY timestamp_utc", conn, params=(country, lo, hi)), hourly_mean=True)


def load_load_forecast(conn, country: str, start, end) -> pd.Series:
    lo, hi = _widened(start, end)
    return _to_series(pd.read_sql_query(
        "SELECT target_timestamp_utc, forecast_value_mw FROM energy_load_forecast "
        "WHERE country_code = ? AND forecast_type = 'day_ahead' "
        "AND target_timestamp_utc >= ? AND target_timestamp_utc <= ? "
        "ORDER BY target_timestamp_utc", conn, params=(country, lo, hi)), hourly_mean=True)


def load_generation_forecast(conn, country: str, column: str, start, end) -> pd.Series:
    if column not in GENERATION_FORECAST_COLUMNS:
        raise ValueError(f"unexpected generation-forecast column {column!r}")
    lo, hi = _widened(start, end)
    return _to_series(pd.read_sql_query(
        f"SELECT target_timestamp_utc, {column} FROM energy_generation_forecast "
        "WHERE country_code = ? AND forecast_type = 'day_ahead' "
        "AND target_timestamp_utc >= ? AND target_timestamp_utc <= ? "
        "ORDER BY target_timestamp_utc", conn, params=(country, lo, hi)), hourly_mean=True)


def load_net_crossborder_flow(conn, country: str, start, end) -> pd.Series:
    """Net physical exchange for `country`: exports minus imports, per hour.

    Nothing here fills a gap — missingness is the caller's business, because it
    needs to distinguish "no observation" from "zero flow" (ABL-74).
    """
    lo, hi = _widened(start, end)
    out = _to_series(pd.read_sql_query(
        "SELECT timestamp_utc, SUM(flow_mw) FROM crossborder_flows "
        "WHERE country_from = ? AND timestamp_utc >= ? AND timestamp_utc <= ? "
        "GROUP BY timestamp_utc ORDER BY timestamp_utc", conn, params=(country, lo, hi)))
    inn = _to_series(pd.read_sql_query(
        "SELECT timestamp_utc, SUM(flow_mw) FROM crossborder_flows "
        "WHERE country_to = ? AND timestamp_utc >= ? AND timestamp_utc <= ? "
        "GROUP BY timestamp_utc ORDER BY timestamp_utc", conn, params=(country, lo, hi)))
    if out.empty and inn.empty:
        return _empty_series()
    # An hour observed on only one direction is still an observation of that
    # direction; keep the union index and treat the unobserved side as absent.
    idx = out.index.union(inn.index)
    return out.reindex(idx).fillna(0.0) - inn.reindex(idx).fillna(0.0)


def load_weather_forecast_archive(conn, country: str, start, end,
                                  columns: Iterable[str] = WEATHER_COLUMNS) -> pd.DataFrame:
    """Every issued weather run covering `[start, end]`, as (target, run) rows.

    `data_quality = 'forecast'` is load-bearing: the `'actual'` rows in this
    table are reanalysis carrying `forecast_run_time == timestamp_utc`, so
    admitting them would hand the model observed weather for the target day.
    The freshest-run-per-target selection happens per window in
    `SourceCache.weather`, because which runs existed depends on the run instant.
    """
    cols = list(columns)
    lo, hi = _widened(start, end)
    df = pd.read_sql_query(
        f"SELECT timestamp_utc, forecast_run_time, {', '.join(cols)} FROM weather_data "
        "WHERE country_code = ? AND data_quality = 'forecast' "
        "AND timestamp_utc >= ? AND timestamp_utc <= ?",
        conn, params=(country, lo, hi))
    if df.empty:
        return pd.DataFrame(columns=["timestamp_utc", "forecast_run_time", *cols])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="mixed", utc=True).dt.tz_localize(None)
    df["forecast_run_time"] = pd.to_datetime(df["forecast_run_time"], format="mixed", utc=True).dt.tz_localize(None)
    return df.sort_values("forecast_run_time")


@dataclass
class SourceCache:
    """Every source V014 reads, loaded once for one country over one span.

    Neighbours are loaded lazily and shared, because a 19-country training run
    asks for DE's net position from six different countries' feature builds.
    """

    conn: sqlite3.Connection
    start: pd.Timestamp
    end: pd.Timestamp
    country: str
    _np: dict[str, pd.Series] = field(default_factory=dict)
    _price: dict[str, pd.Series] = field(default_factory=dict)
    _load_fc: Optional[pd.Series] = None
    _gen_fc: dict[str, pd.Series] = field(default_factory=dict)
    _xb: Optional[pd.Series] = None
    _weather: Optional[pd.DataFrame] = None

    def net_position(self, country: Optional[str] = None) -> pd.Series:
        cc = country or self.country
        if cc not in self._np:
            self._np[cc] = load_net_position(self.conn, cc, self.start, self.end)
        return self._np[cc]

    def price(self, country: Optional[str] = None) -> pd.Series:
        cc = country or self.country
        if cc not in self._price:
            self._price[cc] = load_price(self.conn, cc, self.start, self.end)
        return self._price[cc]

    def load_forecast(self) -> pd.Series:
        if self._load_fc is None:
            self._load_fc = load_load_forecast(self.conn, self.country, self.start, self.end)
        return self._load_fc

    def generation_forecast(self, column: str) -> pd.Series:
        if column not in self._gen_fc:
            self._gen_fc[column] = load_generation_forecast(
                self.conn, self.country, column, self.start, self.end)
        return self._gen_fc[column]

    def crossborder(self) -> pd.Series:
        if self._xb is None:
            self._xb = load_net_crossborder_flow(self.conn, self.country, self.start, self.end)
        return self._xb

    def weather(self, target_index: pd.DatetimeIndex,
                issued_before: pd.Timestamp) -> pd.DataFrame:
        """Freshest run per target hour among runs issued at or before `issued_before`."""
        if self._weather is None:
            self._weather = load_weather_forecast_archive(
                self.conn, self.country, self.start, self.end + pd.Timedelta(days=3))
        archive = self._weather
        if archive.empty:
            # dtype=float, not the default object: an object-dtype NaN column
            # is not a NaN XGBoost can consume natively, and the weather-blind
            # regime (W01-W10, and any country with no issued archive) is
            # exactly where this frame is empty.
            return pd.DataFrame(columns=list(WEATHER_COLUMNS), dtype=float)
        issued = pd.Timestamp(issued_before)
        window = archive[(archive["forecast_run_time"] <= issued)
                         & archive["timestamp_utc"].isin(target_index)]
        if window.empty:
            # dtype=float, not the default object: an object-dtype NaN column
            # is not a NaN XGBoost can consume natively, and the weather-blind
            # regime (W01-W10, and any country with no issued archive) is
            # exactly where this frame is empty.
            return pd.DataFrame(columns=list(WEATHER_COLUMNS), dtype=float)
        # Already sorted by forecast_run_time, so keep="last" is the freshest run.
        window = window.drop_duplicates("timestamp_utc", keep="last")
        return window.set_index("timestamp_utc")[list(WEATHER_COLUMNS)].sort_index()


def build_cache(conn: sqlite3.Connection, country: str, start, end) -> SourceCache:
    return SourceCache(conn=conn, country=country,
                       start=pd.Timestamp(start), end=pd.Timestamp(end))


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def _calendar_features(index: pd.DatetimeIndex, country: str) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["hour"] = index.hour
    out["day_of_week"] = index.dayofweek
    out["month"] = index.month
    out["day_of_year"] = index.dayofyear
    out["is_weekend"] = (index.dayofweek >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * index.hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * index.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * index.dayofyear / 365.25)
    out["is_holiday"] = _holiday_flags(index, country)
    return out


_HOLIDAY_CACHE: dict[tuple[str, int], object] = {}


def _holiday_flags(index: pd.DatetimeIndex, country: str) -> np.ndarray:
    """Public-holiday flag. Returns zeros when the calendar is unavailable —
    a missing calendar must not take the run down, and a wrong holiday flag is
    a far smaller error than a failed forecast."""
    years = sorted({int(y) for y in index.year})
    key = (country, years[0])
    if key not in _HOLIDAY_CACHE:
        try:
            import holidays

            _HOLIDAY_CACHE[key] = holidays.country_holidays(
                country, years=range(years[0] - 1, years[-1] + 2))
        except Exception:  # noqa: BLE001 - unsupported country codes raise plain exceptions
            logger.debug("no holiday calendar for %s; holiday flag left at 0", country)
            _HOLIDAY_CACHE[key] = None
    cal = _HOLIDAY_CACHE[key]
    if cal is None:
        return np.zeros(len(index), dtype=int)
    return np.array([1 if d.date() in cal else 0 for d in index], dtype=int)


def _same_hour_lags(series: pd.Series, window: ServeWindow, prefix: str,
                    lags: Iterable[int] = SAME_HOUR_LAGS) -> pd.DataFrame:
    """Same-hour lags, each checked against the serve cutoff before it is read."""
    index = window.target_index
    bounded = _bounded(series, window.day_ahead_cutoff)
    out = pd.DataFrame(index=index)
    for lag in lags:
        assert_lag_is_serve_safe(window, lag, f"{prefix}_lag{lag}h")
        out[f"{prefix}_lag{lag}h"] = bounded.reindex(index - pd.Timedelta(hours=lag)).to_numpy()
    return out


def _trailing_aggregates(series: pd.Series, window: ServeWindow, prefix: str) -> pd.DataFrame:
    """Run-anchored summaries of the window ending at the serve cutoff.

    These are identical for all 24 target hours of a run, which is the point:
    they carry the freshest information the run held (right up to D 21:00)
    without any per-hour lag arithmetic that could reach past it.
    """
    index = window.target_index
    cutoff = window.day_ahead_cutoff
    out = pd.DataFrame(index=index)
    bounded = _bounded(series, cutoff)
    for days in TRAILING_WINDOWS_DAYS:
        start = cutoff - pd.Timedelta(days=days) + pd.Timedelta(hours=1)
        chunk = bounded[bounded.index >= start]
        out[f"{prefix}_last{days}d_mean"] = chunk.mean() if len(chunk) else np.nan
        if days > 1:
            out[f"{prefix}_last{days}d_std"] = chunk.std() if len(chunk) > 1 else np.nan
            out[f"{prefix}_last{days}d_min"] = chunk.min() if len(chunk) else np.nan
            out[f"{prefix}_last{days}d_max"] = chunk.max() if len(chunk) else np.nan
    out[f"{prefix}_at_cutoff"] = bounded.iloc[-1] if len(bounded) else np.nan
    return out


def _hour_of_day_climatology(series: pd.Series, window: ServeWindow, prefix: str,
                             days: int = 28) -> pd.DataFrame:
    """Mean of the same hour-of-day over the trailing `days` ending at cutoff."""
    index = window.target_index
    cutoff = window.day_ahead_cutoff
    bounded = _bounded(series, cutoff)
    chunk = bounded[bounded.index > cutoff - pd.Timedelta(days=days)]
    out = pd.DataFrame(index=index)
    if chunk.empty:
        out[f"{prefix}_hod_clim{days}d"] = np.nan
        return out
    clim = chunk.groupby(chunk.index.hour).mean()
    out[f"{prefix}_hod_clim{days}d"] = [clim.get(h, np.nan) for h in index.hour]
    return out


def _crossborder(series: pd.Series, window: ServeWindow) -> pd.DataFrame:
    """Cross-border flow features at the ABL-74 lag, with explicit missingness.

    `xb_missing` is the whole reason this is a separate helper. Any country can
    lose several consecutive ingest passes at any moment — FI lost 24 in a row
    over 6 days in 2026-07/08 and nobody noticed — and the champion's aligner
    answers such a gap with `ffill(6).bfill(6).fillna(0.0)`, i.e. a fabricated
    0.0 MW flow presented as a measurement. A tabular model reading that sees
    "no flow across this border" where the truth is "we do not know". Here the
    gap stays NaN and the indicator names it, so the model can learn to
    distrust the column instead of reading a synthetic zero as signal.
    """
    index = window.target_index
    lag = CROSSBORDER_SERVE_LAG_HOURS
    # Flows are physical, not day-ahead: what the run holds is bounded by the
    # lag itself, not by the day-ahead publication cutoff.
    bounded = _bounded(series, index.max() - pd.Timedelta(hours=lag))
    out = pd.DataFrame(index=index)
    primary = bounded.reindex(index - pd.Timedelta(hours=lag)).to_numpy()
    out[f"xb_net_lag{lag}h"] = primary
    out[f"xb_net_lag{lag + 24}h"] = bounded.reindex(
        index - pd.Timedelta(hours=lag + 24)).to_numpy()
    out["xb_missing"] = np.isnan(primary).astype(int)
    return out


def build_features(cache: SourceCache, window: ServeWindow,
                   neighbours: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Build the V014 feature frame for one run: 24 rows, one per target hour.

    Every value in the returned frame was available at `window.run_ts`, by the
    construction documented at the top of this module: each source is bounded at
    its own cutoff before it is read, and each same-hour lag is checked against
    that cutoff by `assert_lag_is_serve_safe`.
    """
    index = window.target_index
    country = cache.country
    frames = [_calendar_features(index, country)]

    own = cache.net_position()
    frames.append(_same_hour_lags(own, window, "np"))
    frames.append(_trailing_aggregates(own, window, "np"))
    frames.append(_hour_of_day_climatology(own, window, "np"))

    price = cache.price()
    frames.append(_same_hour_lags(price, window, "price", lags=(72, 168)))
    frames.append(_trailing_aggregates(price, window, "price"))

    load_fc = cache.load_forecast()
    frames.append(_same_hour_lags(load_fc, window, "loadfc", lags=(72, 168)))
    frames.append(_trailing_aggregates(load_fc, window, "loadfc"))

    for column, prefix in (("solar_mw", "gensolar"), ("wind_onshore_mw", "genwind")):
        frames.append(_same_hour_lags(cache.generation_forecast(column), window,
                                      prefix, lags=(72, 168)))

    frames.append(_crossborder(cache.crossborder(), window))

    # Neighbours. Net position is a balance across borders, so the neighbours'
    # own balances and the price spread against them are the mechanism, not a
    # correlation: power flows toward the higher price.
    nb_price_cols = []
    for nb in list(neighbours or []):
        frames.append(_same_hour_lags(cache.net_position(nb), window,
                                      f"nb{nb}_np", lags=(72, 168)))
        col = _same_hour_lags(cache.price(nb), window, f"nb{nb}_price", lags=(72,))
        frames.append(col)
        nb_price_cols.append(col[f"nb{nb}_price_lag72h"])

    weather = cache.weather(index, issued_before=window.run_ts)
    wframe = pd.DataFrame(index=index)
    for col in WEATHER_COLUMNS:
        raw = (weather[col].reindex(index) if col in weather.columns
               else pd.Series(np.nan, index=index))
        wframe[f"wx_{col}"] = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
    wframe["weather_available"] = 0 if weather.empty else 1
    frames.append(wframe)

    features = pd.concat(frames, axis=1)
    if nb_price_cols:
        features["price_spread_nb_lag72h"] = (
            features["price_lag72h"] - pd.concat(nb_price_cols, axis=1).mean(axis=1))
    features.index.name = "target_timestamp_utc"
    return features


def build_training_frame(conn: sqlite3.Connection, country: str,
                         run_days: Iterable[pd.Timestamp],
                         neighbours: Optional[Iterable[str]] = None,
                         cache: Optional[SourceCache] = None) -> pd.DataFrame:
    """Stack `build_features` over many run days and attach the realised target.

    The target column is the *actual* net position for the target hour, joined
    after the fact. It is never a feature and is never bounded by the cutoff —
    it is the thing being predicted.
    """
    days = [pd.Timestamp(d).normalize() for d in run_days]
    if not days:
        return pd.DataFrame()
    if cache is None:
        span_start = min(days) - pd.Timedelta(days=35)
        span_end = max(days) + pd.Timedelta(days=TARGET_DAY_OFFSET + 1)
        cache = build_cache(conn, country, span_start, span_end)

    rows = []
    for run_day in days:
        window = ServeWindow.for_run_day(run_day)
        try:
            feats = build_features(cache, window, neighbours=neighbours)
        except ServeFaithfulnessError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("V014 %s: skipping run day %s (%s)", country, run_day, exc)
            continue
        feats["run_day"] = run_day
        rows.append(feats)

    if not rows:
        return pd.DataFrame()

    frame = pd.concat(rows)
    actual = load_net_position(conn, country, frame.index.min(), frame.index.max())
    frame["target_net_position_mw"] = actual.reindex(frame.index).to_numpy()
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Model input columns: everything except the target and the run bookkeeping."""
    excluded = {"target_net_position_mw", "run_day"}
    return [c for c in frame.columns if c not in excluded]
