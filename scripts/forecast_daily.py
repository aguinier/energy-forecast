#!/usr/bin/env python3
"""
Daily Forecast Generation Script

Generates forecasts for all countries and types with configurable horizons.
Default horizons: D+1 for load/renewable, D+2 for price.

Usage:
    python scripts/forecast_daily.py
    python scripts/forecast_daily.py --countries DE,FR
    python scripts/forecast_daily.py --horizon 1  # Force D+1 for all types
    python scripts/forecast_daily.py --dry-run
"""

import argparse
import logging
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List

# Add repo root to path (for the top-level `config` module and the `src` package)
REPO_ROOT = Path(__file__).parent.parent
# Runners whose `script` lives here are modules of a package, not loose files.
SRC_PACKAGE = 'src'
sys.path.insert(0, str(REPO_ROOT))

import config
from src.db import save_forecasts, create_forecasts_table, start_forecast_run, complete_forecast_run
from src.forecaster import Forecaster
from src.runner_report import (
    RECORD_COUNT_PREFIX,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_SUCCESS,
    STATUS_UNREPORTED,
    format_runner_summary,
    is_skip,
    parse_record_count,
    status_for_count,
)


def setup_logging():
    """Setup logging to file and console."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = config.LOGS_DIR / f"daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger('energy_forecast')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate D+2 daily forecasts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate forecasts for all countries
    python scripts/forecast_daily.py

    # Generate for specific countries
    python scripts/forecast_daily.py --countries DE,FR

    # Dry run (no database write)
    python scripts/forecast_daily.py --dry-run

    # Override target date
    python scripts/forecast_daily.py --target-date 2024-12-28
        """
    )

    parser.add_argument(
        '--countries',
        type=str,
        default='all',
        help='Comma-separated country codes or "all" (default: all)'
    )

    parser.add_argument(
        '--types',
        type=str,
        default='all',
        help='Comma-separated forecast types or "all" (default: all)'
    )

    parser.add_argument(
        '--target-date',
        type=str,
        default=None,
        help='Override target date YYYY-MM-DD (default: based on horizon)'
    )

    parser.add_argument(
        '--horizon',
        type=int,
        default=None,
        help='Override forecast horizon days (default: type-specific from config)'
    )

    parser.add_argument(
        '--horizons',
        type=str,
        default=None,
        help='Comma-separated list of horizons to generate (e.g., "1,2" for D+1 and D+2)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate forecasts but do not save to database'
    )

    parser.add_argument(
        '--include-non-production',
        action='store_true',
        help='Also run MODEL_RUNNERS entries marked "production": False '
             '(workstation-only runners; they need their own interpreter)'
    )

    return parser.parse_args()


def get_countries(countries_arg: str) -> List[str]:
    """Parse countries argument."""
    if countries_arg.lower() == 'all':
        return config.SUPPORTED_COUNTRIES
    return [c.strip().upper() for c in countries_arg.split(',')]


def get_forecast_types(types_arg: str) -> List[str]:
    """Parse forecast types argument."""
    if types_arg.lower() == 'all':
        return config.FORECAST_TYPES + config.RENEWABLE_TYPES
    return [t.strip().lower() for t in types_arg.split(',')]


def generate_forecast(
    country_code: str,
    forecast_type: str,
    reference_date: date,
    horizon_days: int,
    logger: logging.Logger
) -> dict:
    """
    Generate forecast for a single country/type.

    Args:
        country_code: ISO country code
        forecast_type: Type of forecast (load, price, renewable, etc.)
        reference_date: Date of forecast generation
        horizon_days: Days ahead to forecast (D+1, D+2, etc.)
        logger: Logger instance

    Returns:
        Dictionary with generation results
    """
    result = {
        'country_code': country_code,
        'forecast_type': forecast_type,
        'horizon_days': horizon_days,
        'runner': 'builtin',
        'status': STATUS_FAILED,
        'records': 0,
        'error': None
    }

    try:
        # Load trained model
        forecaster = Forecaster.load(country_code, forecast_type)

        # Generate forecast with specified horizon
        forecast_df = forecaster.predict_d2(reference_date=reference_date, horizon_days=horizon_days)

        # Add renewable_type column for individual renewable types
        if forecast_type in config.RENEWABLE_TYPES:
            forecast_df['renewable_type'] = forecast_type
        else:
            forecast_df['renewable_type'] = None

        # A model that returns no rows ran fine and produced nothing, which is
        # not the same thing as success (ABL-370).
        result['status'] = status_for_count(len(forecast_df))
        result['records'] = len(forecast_df)
        result['forecast_df'] = forecast_df

        if result['status'] == STATUS_EMPTY:
            logger.warning(f"[EMPTY] {country_code} {forecast_type}: ran, produced 0 forecasts")
        else:
            logger.info(f"[OK] {country_code} {forecast_type}: {len(forecast_df)} forecasts generated")

    except FileNotFoundError as e:
        # The one place that knows a result is "no model to run" rather than a
        # failure. Flagged, not inferred from the error text (ABL-370).
        result['error'] = f"Model not found: {e}"
        result['skipped'] = True
        logger.warning(f"[SKIP] {country_code} {forecast_type}: Model not trained yet")

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"[FAIL] {country_code} {forecast_type}: {e}")

    return result


def build_runner_command(
    runner: dict,
    args: List[str],
    repo_root: Path = REPO_ROOT,
) -> List[str]:
    """Build the argv for an external runner, launched the way its imports need.

    A runner hosted inside `src/` is a *module of the `src` package*: its
    siblings are imported relatively (`from .db import ...`), which only
    resolves when it has a parent package. Handing its path to the interpreter
    gives it none, and it dies at the import line with "attempted relative
    import with no known parent package" — which is exactly what happened to
    `src/tso_correction_forecaster.py` from ABL-340 until ABL-354: every BE
    solar / wind forecast from the `tso-correction` runner failed while the job
    still exited `[DONE]`.

    So a `src/`-hosted runner is launched as `python -m src.<module>`, with the
    subprocess `cwd` at the repo root so `src` and `config` resolve. Runners
    outside the package keep the by-path form; they have no parent package to
    need. See CLAUDE.md, "Importing this repo".
    """
    script = Path(runner['script'])
    python_exe = runner.get('python_executable') or 'python'

    if script.parts[0] == SRC_PACKAGE:
        launch = ['-m', '.'.join(script.with_suffix('').parts)]
    else:
        launch = [str(repo_root / script)]

    return [python_exe, *launch, *args]


def select_external_runners(
    runners: List[dict],
    include_non_production: bool = False,
) -> List[dict]:
    """The external runners a run should actually launch.

    Three keys, three separate questions, and conflating the last two is what
    ABL-606 was:

    - ``type``       -- is this a subprocess at all, or the in-process builtin?
    - ``enabled``    -- is this runner wired up, or parked (chronos-2, awaiting
                        fine-tuning, has no interpreter configured yet)?
    - ``production`` -- does this runner belong in the *scheduled* matrix?

    ``production`` was carried on every ``MODEL_RUNNERS`` entry, with a
    hand-written comment per entry explaining the choice, and **nothing read
    it**. Selection was ``type == 'external' and enabled``, so the two entries
    declared ``"production": False`` were launched on every scheduled run --
    including inside the production container, where their configured
    interpreters (`C:/Users/guill/...`) cannot exist. That is 8 of 440 matrix
    cells failing `Executable not found` on every run, forever:
    ``chronos-bolt-small`` at BE x price x {D+1, D+2} = 2, and
    ``tso-correction`` at BE x {solar, wind_onshore, wind_offshore} x
    {D+1, D+2} = 6.

    Neither has written a forecast row since 2026-03-03 (BE only, 504 and 3x506
    rows), and the dashboard deliberately does not register either model name --
    ``server/src/config/forecastModels.ts`` names them as stale and
    ``forecastModels.test.ts`` holds it. They are workstation experiments, and
    the ``tso-correction`` interpreter pin is deliberate (CLAUDE.md, "The
    interpreter is part of the configuration"). So the flag was right and the
    selection was wrong: make the flag load-bearing rather than move the paths.

    ``--include-non-production`` is how you get them back on a box that has
    those interpreters. It is an opt-in, so the scheduled job cannot acquire
    them by accident.
    """
    selected = [
        r for r in runners
        if r.get('type') == 'external' and r.get('enabled', False)
    ]
    if include_non_production:
        return selected
    return [r for r in selected if r.get('production', False)]


def run_external_model(
    runner: dict,
    country_code: str,
    forecast_type: str,
    horizon_days: int,
    reference_date: date,
    dry_run: bool,
    logger: logging.Logger
) -> dict:
    """
    Run an external model as a subprocess.

    The external script is responsible for saving its own forecasts to the DB
    (via its --save flag), so we don't append to all_forecasts. It reports how
    many rows it produced with `src.runner_report.emit_record_count`, which is
    the only channel we have into what it actually did — see that module for
    why the exit code alone is not enough (ABL-370).

    Args:
        runner: Runner config dict from MODEL_RUNNERS
        country_code: ISO country code
        forecast_type: Type of forecast
        horizon_days: Days ahead to forecast
        reference_date: Date of forecast generation
        dry_run: If True, don't pass --save to the subprocess
        logger: Logger instance

    Returns:
        Dictionary with generation results
    """
    import subprocess

    runner_name = runner['name']
    result = {
        'country_code': country_code,
        'forecast_type': forecast_type,
        'horizon_days': horizon_days,
        'runner': runner_name,
        'status': STATUS_FAILED,
        'records': 0,
        'error': None
    }

    run_args = [
        '--country', country_code,
        '--horizon', str(horizon_days),
        '--date', reference_date.strftime('%Y-%m-%d'),
    ]

    if not dry_run:
        run_args.append('--save')

    cmd = build_runner_command(runner, run_args)

    logger.info(f"[{runner_name}] Running: {country_code} {forecast_type} D+{horizon_days}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=str(REPO_ROOT),  # `-m src.x` resolves the package from here
        )

        if proc.returncode == 0:
            # Exit 0 says the runner did not crash. It does not say it produced
            # anything — the count line does (ABL-370). Absent, the count is
            # unknown, which is reported as unknown rather than as 0 or as OK.
            records = parse_record_count(proc.stdout)
            result['records'] = records
            result['status'] = status_for_count(records)
            where = f"{country_code} {forecast_type} D+{horizon_days}"

            if result['status'] == STATUS_SUCCESS:
                logger.info(f"[{runner_name}] OK: {where} ({records} rows)")
            elif result['status'] == STATUS_EMPTY:
                logger.warning(f"[{runner_name}] Ran, produced 0 forecasts: {where}")
            else:
                result['error'] = (
                    f"exited 0 without printing a {RECORD_COUNT_PREFIX}<n> line, "
                    "so what it produced is unknown"
                )
                logger.warning(f"[{runner_name}] No count reported: {where}")
        else:
            result['error'] = proc.stderr[-500:] if proc.stderr else f'Exit code {proc.returncode}'
            logger.warning(f"[{runner_name}] Failed ({proc.returncode}): {country_code} {forecast_type} D+{horizon_days}")
            if proc.stderr:
                logger.warning(f"[{runner_name}] stderr: {proc.stderr[-300:]}")

    except subprocess.TimeoutExpired:
        result['error'] = 'Subprocess timed out after 300s'
        logger.warning(f"[{runner_name}] Timeout: {country_code} {forecast_type} D+{horizon_days}")
    except FileNotFoundError as e:
        # `cmd[0]`, not `python_exe`: the interpreter name is local to
        # build_runner_command, and naming it here raised NameError *inside the
        # handler*, which no sibling `except` catches — a missing runner
        # interpreter killed the whole daily job instead of failing one result.
        result['error'] = f'Executable not found: {e}'
        logger.warning(f"[{runner_name}] Not found: {cmd[0]}")
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"[{runner_name}] Error: {e}")

    return result


def main():
    """Main forecast generation loop."""
    args = parse_args()
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("Daily Forecast Generation")
    logger.info("=" * 60)

    # Validate config
    try:
        config.validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Ensure forecasts table exists
    create_forecasts_table()

    # Parse arguments
    countries = get_countries(args.countries)
    forecast_types = get_forecast_types(args.types)

    # Determine run type and trigger source
    run_type = 'manual'
    trigger_source = 'script'

    # Check for environment variables that indicate scheduled execution
    import os
    if os.environ.get('FORECAST_RUN_TYPE'):
        run_type = os.environ.get('FORECAST_RUN_TYPE', 'scheduled')
    if os.environ.get('FORECAST_TRIGGER_SOURCE'):
        trigger_source = os.environ.get('FORECAST_TRIGGER_SOURCE', 'bat_file')

    # Start tracking the run (unless dry run)
    run_id = None
    start_time = datetime.now()
    log_file_path = str(config.LOGS_DIR / f"daily_{start_time.strftime('%Y%m%d_%H%M%S')}.log")

    if not args.dry_run:
        try:
            run_id = start_forecast_run(
                run_type=run_type,
                trigger_source=trigger_source,
                countries=countries,
                forecast_types=forecast_types,
            )
            logger.info(f"Forecast run ID: {run_id}")
        except Exception as e:
            logger.warning(f"Could not start run tracking: {e}")

    # Reference date is today (when generating the forecast)
    reference_date = date.today()

    # Determine target date if explicitly specified
    explicit_target_date = None
    if args.target_date:
        explicit_target_date = datetime.strptime(args.target_date, '%Y-%m-%d').date()

    # Horizon override options
    # --horizon N: single horizon for all types (backward compatible)
    # --horizons 1,2: list of horizons for all types
    horizon_override = args.horizon
    horizons_override = None
    if args.horizons:
        horizons_override = [int(h.strip()) for h in args.horizons.split(',')]

    logger.info(f"Reference date: {reference_date}")
    if horizon_override:
        logger.info(f"Horizon override: D+{horizon_override} for all types")
    elif horizons_override:
        logger.info(f"Horizons override: D+{',D+'.join(str(h) for h in horizons_override)} for all types")
    else:
        logger.info(f"Horizons: multi-horizon (D+1 and D+2 for all types)")
    logger.info(f"Countries: {len(countries)}")
    logger.info(f"Forecast types: {forecast_types}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("")

    # Track results
    results = []
    all_forecasts = []

    # Calculate total iterations (countries × types × horizons)
    def get_horizons(forecast_type):
        if horizon_override is not None:
            return [horizon_override]
        elif horizons_override is not None:
            return horizons_override
        else:
            return config.get_horizons_for_type(forecast_type)

    total = sum(len(get_horizons(ft)) for ft in forecast_types) * len(countries)
    completed = 0

    for country in countries:
        for forecast_type in forecast_types:
            # Get horizons to generate (can be multiple)
            horizons = get_horizons(forecast_type)

            for horizon_days in horizons:
                completed += 1

                # If explicit target date given, calculate reference date from it
                if explicit_target_date:
                    ref_date = explicit_target_date - timedelta(days=horizon_days)
                else:
                    ref_date = reference_date

                target_date = ref_date + timedelta(days=horizon_days)
                logger.info(f"[{completed}/{total}] Generating {country} {forecast_type} D+{horizon_days} -> {target_date}")

                result = generate_forecast(
                    country,
                    forecast_type,
                    ref_date,
                    horizon_days,
                    logger
                )
                results.append(result)

                if result['status'] == STATUS_SUCCESS and 'forecast_df' in result:
                    all_forecasts.append(result['forecast_df'])

    # ── External model runners (config-driven) ──
    external_runners = select_external_runners(
        config.MODEL_RUNNERS,
        include_non_production=args.include_non_production,
    )
    # Name the selection, so a run's `Total:` is attributable without re-reading
    # config: an external runner appearing or vanishing moves the count (ABL-606).
    held_back = [
        r['name'] for r in config.MODEL_RUNNERS
        if r.get('type') == 'external' and r.get('enabled', False)
        and r not in external_runners
    ]
    logger.info(
        f"External runners: {[r['name'] for r in external_runners] or 'none'}"
        + (f" (non-production, not run: {held_back})" if held_back else "")
    )

    for runner in external_runners:
        runner_name = runner['name']
        runner_countries = runner.get('countries', [])
        runner_types = runner.get('forecast_types', [])

        for country in countries:
            # Check if this runner handles this country
            if runner_countries != 'all' and country not in runner_countries:
                continue

            for forecast_type in forecast_types:
                # Check if this runner handles this forecast type
                if runner_types != 'all' and forecast_type not in runner_types:
                    continue

                for horizon_days in get_horizons(forecast_type):
                    if args.target_date:
                        ref_date = explicit_target_date - timedelta(days=horizon_days)
                    else:
                        ref_date = reference_date

                    result = run_external_model(
                        runner, country, forecast_type, horizon_days, ref_date, args.dry_run, logger
                    )
                    results.append(result)

    # Save all forecasts to database
    if all_forecasts and not args.dry_run:
        import pandas as pd
        combined_df = pd.concat(all_forecasts, ignore_index=True)
        records_saved = save_forecasts(combined_df)
        logger.info(f"\nSaved {records_saved} forecast records to database")
    elif args.dry_run:
        logger.info("\n[DRY RUN] Forecasts not saved to database")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Forecast Generation Summary")
    logger.info("=" * 60)

    success = sum(1 for r in results if r['status'] == STATUS_SUCCESS)
    empty = sum(1 for r in results if r['status'] == STATUS_EMPTY)
    unreported = sum(1 for r in results if r['status'] == STATUS_UNREPORTED)
    skipped = sum(1 for r in results if is_skip(r))
    failed = sum(1 for r in results if r['status'] == STATUS_FAILED) - skipped
    # `records` is None for a run that never reported one — unknown, not zero.
    total_forecasts = sum(r['records'] or 0 for r in results)

    logger.info(f"Reference date: {reference_date}")
    logger.info(
        f"Total: {len(results)}, Success: {success}, Empty: {empty}, "
        f"Unreported: {unreported}, Skipped: {skipped}, Failed: {failed}"
    )
    logger.info(f"Total forecasts: {total_forecasts}")

    # A zero inside that sum is invisible next to the in-process models'
    # thousands of rows, so break it out per runner (ABL-370).
    logger.info("")
    for line in format_runner_summary(results):
        logger.info(line)

    if failed > 0:
        logger.info("\nFailed models:")
        for r in results:
            if r['status'] == STATUS_FAILED and not is_skip(r):
                logger.info(f"  - {r['country_code']} {r['forecast_type']}: {r['error']}")

    # Complete the run tracking
    execution_time = (datetime.now() - start_time).total_seconds()
    successful_countries = list(set(r['country_code'] for r in results if r['status'] == STATUS_SUCCESS))
    successful_types = list(set(r['forecast_type'] for r in results if r['status'] == STATUS_SUCCESS))

    if run_id and not args.dry_run:
        try:
            error_message = None
            if failed > 0:
                failed_items = [f"{r['country_code']}/{r['forecast_type']}" for r in results
                               if r['status'] == STATUS_FAILED and not is_skip(r)]
                error_message = f"Failed: {', '.join(failed_items[:5])}" + ("..." if len(failed_items) > 5 else "")

            complete_forecast_run(
                run_id=run_id,
                status='completed' if failed == 0 else 'failed',
                countries_completed=successful_countries,
                types_completed=successful_types,
                forecasts_generated=total_forecasts,
                execution_time_seconds=execution_time,
                error_message=error_message,
                log_file_path=log_file_path,
            )
        except Exception as e:
            logger.warning(f"Could not complete run tracking: {e}")

    logger.info(f"\nExecution time: {execution_time:.1f}s")
    logger.info("\n[DONE] Forecast generation complete!")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
