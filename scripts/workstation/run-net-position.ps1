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

    # Challengers, in shadow, on the same serve-time inputs (ABL-68). They read
    # the champion vintage written above and store their own model_name rows in
    # the sidecar. They are never pushed: push_net_position_forecast.py names
    # the champion and filters on it. A challenger failure must not cost us the
    # champion's run or its push, so this is reported and continues.
    & "$Repo\.venv\Scripts\python.exe" "$Repo\scripts\forecast_challengers.py" `
        --experiments V012,V016 --countries all --save-to-db
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "forecast_challengers.py exited $LASTEXITCODE - one or more challengers produced nothing this run."
    }

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

    # Score every stored vintage against the actuals that have arrived since
    # (ABL-30 B4). Runs daily but writes ISO-week-keyed reports, so the weekly
    # artifact is always fresh; reports/net_position_eval/latest.md is the one
    # to read. Eval failure is reported but does not fail the forecast run.
    #
    # Once per model: the eval scores every stored *vintage* automatically, but
    # it is scoped to one model_name per invocation (--model), so a challenger
    # is only scored if it is named here.
    #
    # Each challenger gets its own --out-dir. The script always writes latest.md
    # alongside the week-tagged report, so sharing one directory would leave
    # latest.md holding whichever model ran last -- and ABL-30/ABL-34 both read
    # that path expecting the champion. The champion therefore keeps the
    # existing directory untouched.
    $evalRoot = "$Repo\reports\net_position_eval"
    $models = @(
        @{ Name = "chronos-2-V010"; OutDir = $evalRoot },
        @{ Name = "baseline-V012";  OutDir = "$evalRoot\V012" },
        @{ Name = "chronos-2-V016"; OutDir = "$evalRoot\V016" }
    )
    foreach ($m in $models) {
        & "$Repo\.venv\Scripts\python.exe" "$Repo\scripts\evaluate_net_position.py" `
            --model $m.Name --out-dir $m.OutDir
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "evaluate_net_position.py --model $($m.Name) exited $LASTEXITCODE - its vintages are unscored this run."
        }
    }
}
finally {
    Stop-Transcript | Out-Null
}
