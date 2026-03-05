#!/bin/bash
#
# Scheduler Setup for Daily Energy Forecasts
#
# This script helps set up a cron job to run forecasts daily at 18:00.
#
# Usage:
#   bash scripts/scheduler_setup.sh
#

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"

echo "=============================================="
echo "Energy Forecast Scheduler Setup"
echo "=============================================="
echo ""
echo "Project directory: $PROJECT_DIR"
echo ""

# Check if Python is available
PYTHON_PATH=$(which python3 2>/dev/null || which python 2>/dev/null)
if [ -z "$PYTHON_PATH" ]; then
    echo "[ERROR] Python not found in PATH"
    exit 1
fi
echo "Python: $PYTHON_PATH"

# Check if forecast_daily.py exists
if [ ! -f "$PROJECT_DIR/scripts/forecast_daily.py" ]; then
    echo "[ERROR] forecast_daily.py not found"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

echo ""
echo "=============================================="
echo "Cron Job Configuration"
echo "=============================================="
echo ""
echo "The following cron entry will run forecasts daily at 18:00:"
echo ""

CRON_ENTRY="0 18 * * * cd $PROJECT_DIR && $PYTHON_PATH scripts/forecast_daily.py >> logs/cron_daily.log 2>&1"

echo "  $CRON_ENTRY"
echo ""

read -p "Do you want to install this cron job? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if cron entry already exists
    if crontab -l 2>/dev/null | grep -q "forecast_daily.py"; then
        echo ""
        echo "[WARNING] A forecast cron job already exists:"
        crontab -l 2>/dev/null | grep "forecast_daily.py"
        echo ""
        read -p "Do you want to replace it? (y/N) " -n 1 -r
        echo ""

        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi

        # Remove existing entry
        crontab -l 2>/dev/null | grep -v "forecast_daily.py" | crontab -
    fi

    # Add new cron entry
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

    echo ""
    echo "[OK] Cron job installed successfully!"
    echo ""
    echo "Current crontab:"
    crontab -l | grep "forecast_daily.py"
else
    echo ""
    echo "To manually add the cron job, run:"
    echo "  crontab -e"
    echo ""
    echo "And add this line:"
    echo "  $CRON_ENTRY"
fi

echo ""
echo "=============================================="
echo "Additional Commands"
echo "=============================================="
echo ""
echo "View logs:"
echo "  tail -f $PROJECT_DIR/logs/cron_daily.log"
echo ""
echo "Test manually:"
echo "  cd $PROJECT_DIR && python scripts/forecast_daily.py --dry-run"
echo ""
echo "Remove cron job:"
echo "  crontab -l | grep -v 'forecast_daily.py' | crontab -"
echo ""
