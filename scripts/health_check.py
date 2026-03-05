#!/usr/bin/env python3
"""
Energy Forecast Health Check Script

Monitors system health and reports issues:
- Last forecast run time and status
- Production model coverage
- Data freshness
- Model file integrity

Usage:
    python scripts/health_check.py
    python scripts/health_check.py --json   # Output as JSON
    python scripts/health_check.py --alert  # Exit with error if issues found
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import config


def check_database_connection() -> Tuple[bool, str]:
    """Check if database is accessible."""
    try:
        from db import get_connection
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            return True, "Database connection OK"
    except Exception as e:
        return False, f"Database error: {e}"


def check_last_forecast_run() -> Tuple[str, Optional[dict]]:
    """
    Check when the last forecast run completed.

    Returns:
        Tuple of (status, run_info)
        status: 'healthy', 'warning', 'critical', 'unknown'
    """
    try:
        from db import get_latest_forecast_run
        run = get_latest_forecast_run()

        if not run:
            return 'unknown', None

        run_time = datetime.fromisoformat(run['run_timestamp'])
        hours_ago = (datetime.now() - run_time).total_seconds() / 3600

        status = 'healthy' if hours_ago < 24 else 'warning' if hours_ago < 48 else 'critical'

        return status, {
            'last_run': run['run_timestamp'],
            'status': run['status'],
            'forecasts_generated': run['forecasts_generated'],
            'hours_ago': round(hours_ago, 1),
        }
    except Exception as e:
        return 'unknown', {'error': str(e)}


def check_production_models() -> Tuple[str, dict]:
    """
    Check production model coverage.

    Returns:
        Tuple of (status, coverage_info)
    """
    try:
        from db import get_deployed_models
        deployed = get_deployed_models()

        if not deployed:
            return 'warning', {
                'total_deployed': 0,
                'message': 'No production models deployed',
            }

        # Count by type
        by_type = {}
        for model in deployed:
            ftype = model['forecast_type']
            by_type[ftype] = by_type.get(ftype, 0) + 1

        total = len(deployed)
        expected = len(config.SUPPORTED_COUNTRIES) * 5  # 5 main forecast types
        coverage = total / expected if expected > 0 else 0

        # Be more lenient - any deployed models is a good start
        status = 'healthy' if total >= 10 else 'warning' if total >= 1 else 'critical'

        return status, {
            'total_deployed': total,
            'expected': expected,
            'coverage_pct': round(coverage * 100, 1),
            'by_type': by_type,
        }
    except Exception as e:
        return 'unknown', {'error': str(e)}


def check_model_files() -> Tuple[str, dict]:
    """
    Check that model files exist on disk for deployed models.

    Returns:
        Tuple of (status, file_info)
    """
    models_dir = config.MODELS_DIR

    if not models_dir.exists():
        return 'critical', {'error': 'Models directory does not exist'}

    found = []
    missing = []

    # Check deployed models from database
    try:
        from db import get_deployed_models
        deployed = get_deployed_models()

        for model in deployed:
            country = model['country_code']
            ftype = model['forecast_type']

            # Check new production location first
            model_path = models_dir / country / ftype / 'production' / 'model.joblib'
            if model_path.exists():
                found.append(f"{country}/{ftype}")
            else:
                # Also check old location
                old_path = models_dir / country / ftype / 'model.joblib'
                if old_path.exists():
                    found.append(f"{country}/{ftype}")
                else:
                    missing.append(f"{country}/{ftype}")

        if not deployed:
            # No deployments yet, check a sample of expected models
            for country in config.SUPPORTED_COUNTRIES[:5]:
                for ftype in ['load', 'price']:
                    model_path = models_dir / country / ftype / 'production' / 'model.joblib'
                    old_path = models_dir / country / ftype / 'model.joblib'
                    if model_path.exists() or old_path.exists():
                        found.append(f"{country}/{ftype}")
                    else:
                        missing.append(f"{country}/{ftype}")

    except Exception as e:
        return 'unknown', {'error': str(e)}

    total_checked = len(found) + len(missing)
    coverage = len(found) / total_checked if total_checked > 0 else 0

    status = 'healthy' if coverage >= 0.9 else 'warning' if coverage >= 0.7 else 'critical'

    return status, {
        'models_found': len(found),
        'models_missing': len(missing),
        'sample_missing': missing[:5] if missing else [],
    }


def check_data_freshness() -> Tuple[str, dict]:
    """
    Check that source data is recent.

    Returns:
        Tuple of (status, freshness_info)
    """
    try:
        from db import get_connection

        with get_connection() as conn:
            cursor = conn.cursor()

            # Check energy_load
            cursor.execute("""
                SELECT MAX(timestamp_utc) as latest
                FROM energy_load
            """)
            load_latest = cursor.fetchone()[0]

            # Check energy_price
            cursor.execute("""
                SELECT MAX(timestamp_utc) as latest
                FROM energy_price
            """)
            price_latest = cursor.fetchone()[0]

        results = {}
        overall_status = 'healthy'

        for name, latest in [('load', load_latest), ('price', price_latest)]:
            if latest:
                latest_dt = datetime.fromisoformat(latest.replace('Z', '+00:00').replace('+00:00', ''))
                hours_ago = (datetime.now() - latest_dt).total_seconds() / 3600

                status = 'healthy' if hours_ago < 24 else 'warning' if hours_ago < 48 else 'critical'

                results[name] = {
                    'latest': latest,
                    'hours_ago': round(hours_ago, 1),
                    'status': status,
                }

                if status == 'critical':
                    overall_status = 'critical'
                elif status == 'warning' and overall_status != 'critical':
                    overall_status = 'warning'
            else:
                results[name] = {'error': 'No data found'}
                overall_status = 'critical'

        return overall_status, results
    except Exception as e:
        return 'unknown', {'error': str(e)}


def run_health_check() -> Dict:
    """
    Run all health checks and return aggregated results.
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'healthy',
        'checks': {},
    }

    # Run individual checks
    checks = [
        ('database', check_database_connection),
        ('last_forecast_run', check_last_forecast_run),
        ('production_models', check_production_models),
        ('model_files', check_model_files),
        ('data_freshness', check_data_freshness),
    ]

    for check_name, check_fn in checks:
        try:
            status, info = check_fn()
            results['checks'][check_name] = {
                'status': status if isinstance(status, str) else 'ok' if status else 'error',
                'info': info,
            }

            # Update overall status
            if status == 'critical':
                results['overall_status'] = 'critical'
            elif status == 'warning' and results['overall_status'] != 'critical':
                results['overall_status'] = 'warning'
        except Exception as e:
            results['checks'][check_name] = {
                'status': 'error',
                'info': {'error': str(e)},
            }
            results['overall_status'] = 'critical'

    return results


def format_text_report(results: Dict) -> str:
    """Format results as human-readable text."""
    lines = [
        "=" * 60,
        "Energy Forecast Health Check Report",
        f"Timestamp: {results['timestamp']}",
        f"Overall Status: {results['overall_status'].upper()}",
        "=" * 60,
        "",
    ]

    status_icons = {
        'healthy': '[OK]',
        'ok': '[OK]',
        'warning': '[WARN]',
        'critical': '[CRIT]',
        'unknown': '[????]',
        'error': '[ERR]',
    }

    for check_name, check_result in results['checks'].items():
        status = check_result['status']
        icon = status_icons.get(status, '[????]')

        lines.append(f"{icon} {check_name.replace('_', ' ').title()}")

        info = check_result.get('info', {})
        if isinstance(info, dict):
            for key, value in info.items():
                if key != 'error':
                    lines.append(f"      {key}: {value}")
                else:
                    lines.append(f"      ERROR: {value}")
        elif isinstance(info, str):
            lines.append(f"      {info}")

        lines.append("")

    return "\n".join(lines)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Check energy forecast system health',
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON instead of text',
    )

    parser.add_argument(
        '--alert',
        action='store_true',
        help='Exit with non-zero code if any check fails',
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only output on warning/critical status',
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    results = run_health_check()

    # Output
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        if args.quiet and results['overall_status'] == 'healthy':
            pass
        else:
            print(format_text_report(results))

    # Exit code
    if args.alert:
        if results['overall_status'] == 'critical':
            return 2
        elif results['overall_status'] == 'warning':
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
