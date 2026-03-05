#!/bin/bash
set -e

# Pass environment variables to cron (cron runs in a clean env)
printenv | grep -E '^(ENERGY_DB_PATH|PATH)=' > /etc/environment

echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting energy forecast scheduler" >> /app/logs/cron_forecast.log
echo "Schedule: 07:00, 14:00, 15:30, 19:00" >> /app/logs/cron_forecast.log

# Start cron in the foreground
exec cron -f
