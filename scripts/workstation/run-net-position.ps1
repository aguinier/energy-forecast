# Daily net-position forecast on the workstation (acceptance).
# Reads the replica, writes ONLY to the sidecar DB (FORECAST_OUTPUT_DB).
# Scheduled at 08:00, after the 07:00 able-db-sync replica refresh.
#
# This is the *job*. The scheduled task does not call it directly - it calls
# run-net-position-serving.ps1, which hard-resets the serving checkout to
# origin/main, logs the resolved SHA, and then invokes this file from that
# tree (ABL-692). Run this file by hand only for a one-off; the transcript and
# the commit witness line belong to the launcher.
#
# Three inputs deliberately do NOT come from $Repo. $Repo is a disposable,
# hard-reset checkout that holds only git-tracked code, and each of these lives
# outside git:
#
#   $Venv       .venv/ is gitignored, and ABL-69 pins the rail interpreter to
#               this exact path. An xgboost-3.3.0 artifact loaded under the
#               conda 2.1.4 silently resets the fitted intercept, so the
#               interpreter is part of the configuration, not a detail.
#   $ModelsDir  models/ is gitignored (4.2 GB). Only forecast_challengers.py
#               reads it - V014's artifacts are models/net_position/V014, 17 MB
#               of it. forecast_chronos2.py never resolves MODELS_DIR for the
#               champion because V010 is fine_tune=false and so serves the
#               pretrained Chronos-2; tests/test_abl692_serving_checkout.py
#               holds that assumption.
#   $EvalRoot   reports/net_position_eval/ is gitignored, and ABL-30/ABL-34
#               read it at the dev-checkout path. Writing it under $Repo
#               instead would leave every existing reader on a directory that
#               silently stopped updating - the same class of failure ABL-692
#               is about.
#
# The --candidate-backtest files, by contrast, ARE tracked, so they come from
# $Repo on purpose: the backtest a vintage is scored against then moves with
# the code that produced it.
[CmdletBinding()]
param(
    [string] $Repo      = "C:\Code\able\energy-forecast-serving",
    [string] $Venv      = "C:\Code\able\energy-forecast\.venv",
    [string] $ModelsDir = "C:\Code\able\energy-forecast\models",
    [string] $EvalRoot  = "C:\Code\able\energy-forecast\reports\net_position_eval"
)
$ErrorActionPreference = "Stop"
$Python = "$Venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Rail interpreter not found at $Python (ABL-69)." }

$env:ENERGY_DB_PATH     = "C:\Code\able\data\energy_dashboard.db"
$env:FORECAST_OUTPUT_DB = "C:\Code\able\data\forecasts_local.db"
& $Python "$Repo\scripts\forecast_chronos2.py" `
    --experiment V010 --types net_position --countries all --save-to-db
if ($LASTEXITCODE -ne 0) { throw "forecast_chronos2.py exited $LASTEXITCODE" }

# Challengers, in shadow, on the same serve-time inputs (ABL-68). They read
# the champion vintage written above and store their own model_name rows in
# the sidecar. A challenger failure must not cost us the champion's run or
# its push, so this is reported and continues.
#
# --models-dir is passed explicitly because config.MODELS_DIR resolves against
# the repo root, and this job no longer runs from the checkout that holds the
# artifacts. Without it V014 logs "no trained model" for all 24 countries,
# writes nothing, and still exits 0.
& $Python "$Repo\scripts\forecast_challengers.py" `
    --experiments V012,V014,V016 --countries all --models-dir $ModelsDir --save-to-db
if ($LASTEXITCODE -ne 0) {
    Write-Warning "forecast_challengers.py exited $LASTEXITCODE - one or more challengers produced nothing this run."
}

# Ship the run to the dashboard so it is visible outside this box, and so
# the ABL-70 promotion gate accrues a scored vintage for each challenger,
# not only the champion (ABL-175). push_net_position_forecast.py pushes
# chronos-2-V010, baseline-V012, xgboost-V014 and chronos-2-V016
# independently, each under its own name; one model having nothing to push
# (exit 2) or failing to push (exit 1) does not stop the others - the
# script always attempts every registered model and reports per-model
# status. The forecast itself is the job's real output, so a push problem
# is reported but does not fail the run - the next run re-pushes, and the
# endpoint replaces rather than duplicates a vintage.
if ($env:DASHBOARD_WRITE_TOKEN) {
    if (-not $env:DASHBOARD_API_URL) { $env:DASHBOARD_API_URL = "http://192.168.86.36:3001" }
    & $Python "$Repo\scripts\push_net_position_forecast.py"
    if ($LASTEXITCODE -eq 1) {
        Write-Warning "Forecast saved locally but at least one model failed to push to $($env:DASHBOARD_API_URL) (exit 1) - see the per-model summary above."
    } elseif ($LASTEXITCODE -eq 2) {
        Write-Warning "Forecast saved locally; at least one model had nothing new to push to $($env:DASHBOARD_API_URL) (exit 2) - see the per-model summary above."
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
$models = @(
    @{ Name = "chronos-2-V010"; OutDir = $EvalRoot;        Backtest = "$Repo\comparison_net_position_servefaithful.json" },
    @{ Name = "baseline-V012";  OutDir = "$EvalRoot\V012"; Backtest = "$Repo\experiments\V012\backtest_W01_W12.json" },
    @{ Name = "xgboost-V014";   OutDir = "$EvalRoot\V014"; Backtest = "$Repo\experiments\V014\backtest_W01_W12.json" },
    @{ Name = "chronos-2-V016"; OutDir = "$EvalRoot\V016"; Backtest = "$Repo\experiments\V016\backtest_W01_W12.json" }
)
foreach ($m in $models) {
    & $Python "$Repo\scripts\evaluate_net_position.py" `
        --model $m.Name --out-dir $m.OutDir --candidate-backtest $m.Backtest
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "evaluate_net_position.py --model $($m.Name) exited $LASTEXITCODE - its vintages are unscored this run."
    }
}
