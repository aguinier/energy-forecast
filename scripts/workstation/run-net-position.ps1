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
}
finally {
    Stop-Transcript | Out-Null
}
