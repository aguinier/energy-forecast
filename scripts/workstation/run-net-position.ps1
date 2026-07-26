# Daily net-position forecast on the workstation (acceptance).
# Reads the replica, writes ONLY to the sidecar DB (FORECAST_OUTPUT_DB).
# Scheduled at 08:00, after the 07:00 able-db-sync replica refresh.
$ErrorActionPreference = "Stop"
$Repo   = "C:\Code\able\energy-forecast"
$LogDir = "C:\Code\able\logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force $LogDir | Out-Null }
Start-Transcript -Path (Join-Path $LogDir "net-position-forecast.log") -Append | Out-Null
try {
    $env:ENERGY_DB_PATH     = "C:\Code\able\data\energy_dashboard.db"
    $env:FORECAST_OUTPUT_DB = "C:\Code\able\data\forecasts_local.db"
    & "$Repo\.venv\Scripts\python.exe" "$Repo\scripts\forecast_chronos2.py" `
        --experiment V010 --types net_position --countries all --save-to-db
    if ($LASTEXITCODE -ne 0) { throw "forecast_chronos2.py exited $LASTEXITCODE" }

    # Ship the run to the dashboard so it is visible outside this box. The
    # forecast itself is the job's real output, so a push failure is reported
    # but does not fail the run - the next run re-pushes, and the endpoint
    # replaces rather than duplicates a vintage.
    if ($env:DASHBOARD_WRITE_TOKEN) {
        if (-not $env:DASHBOARD_API_URL) { $env:DASHBOARD_API_URL = "http://192.168.86.36:3001" }
        & "$Repo\.venv\Scripts\python.exe" "$Repo\scripts\push_net_position_forecast.py"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Forecast saved locally but push to $($env:DASHBOARD_API_URL) failed (exit $LASTEXITCODE)."
        }
    } else {
        Write-Host "DASHBOARD_WRITE_TOKEN not set - forecast stays in the sidecar only."
    }
}
finally {
    Stop-Transcript | Out-Null
}
