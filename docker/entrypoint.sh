#!/bin/bash
set -e

# ABL-596: honour an explicit command override.
#
# This ENTRYPOINT used to run cron unconditionally, so *any* `docker run
# <image> <command>` -- a version probe, a debug shell, a CI smoke check --
# silently discarded the command and started a second cron scheduler off this
# image, writing production forecasts on the real 07/14/15:30/19 schedule.
# Run what the caller actually asked for; start the scheduler only when no
# command was given, which is the `docker compose up` path (neither the
# Dockerfile nor the compose service sets a CMD).
#
# Keep this branch above the cron setup below: a probe must not write
# "Starting energy forecast scheduler" into the mounted log volume either.
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

# Pass environment variables to cron (cron runs in a clean env)
printenv | grep -E '^(ENERGY_DB_PATH|PATH)=' > /etc/environment

echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting energy forecast scheduler" >> /app/logs/cron_forecast.log
echo "Schedule: 07:00, 14:00, 15:30, 19:00" >> /app/logs/cron_forecast.log

# Start cron in the foreground
exec cron -f
