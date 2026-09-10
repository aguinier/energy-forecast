# One-time install of the net-position serving checkout (ABL-692).
#
# Creates a clone that no agent works in, so the daily forecast stops executing
# whatever branch the shared dev tree happens to be parked on. Idempotent: on a
# checkout that already exists it verifies and reports, and changes nothing.
#
# This script is deliberately split from the re-pointing of the scheduled task.
# Creating the clone is inert - nothing runs it until the task points at it -
# whereas re-pointing the task changes what production forecasts tomorrow
# morning. Pass -UpdateScheduledTask only if you own that decision.
#
#   .\install-serving-checkout.ps1                       # clone + print next step
#   .\install-serving-checkout.ps1 -UpdateScheduledTask  # clone + re-point the task

[CmdletBinding()]
param(
    [string] $Serving   = "C:\Code\able\energy-forecast-serving",
    [string] $RemoteUrl = "https://github.com/aguinier/energy-forecast.git",
    [string] $Branch    = "main",
    [string] $TaskName  = "able-net-position-forecast",
    [switch] $UpdateScheduledTask
)

$ErrorActionPreference = "Stop"

$DevCheckout = "C:\Code\able\energy-forecast"
$servingFull = [IO.Path]::GetFullPath($Serving).TrimEnd('\')
$devFull     = [IO.Path]::GetFullPath($DevCheckout).TrimEnd('\')
if ($servingFull -ieq $devFull) {
    throw ("Refusing to install over $servingFull - that is the shared dev " +
           "checkout. The whole point of ABL-692 is that serving gets its own tree.")
}

if (Test-Path (Join-Path $Serving ".git")) {
    Write-Host "Serving checkout already present at $servingFull - verifying."
} else {
    if ((Test-Path $Serving) -and (Get-ChildItem -Force $Serving | Select-Object -First 1)) {
        throw "$servingFull exists and is not empty, but is not a git checkout. Inspect it by hand."
    }
    Write-Host "Cloning $RemoteUrl -> $servingFull (branch $Branch)"
    & git clone --branch $Branch --origin origin $RemoteUrl $Serving
    if ($LASTEXITCODE -ne 0) { throw "git clone exited $LASTEXITCODE" }
}

& git -C $Serving fetch --prune origin
if ($LASTEXITCODE -ne 0) { throw "git fetch exited $LASTEXITCODE" }
& git -C $Serving checkout $Branch
if ($LASTEXITCODE -ne 0) { throw "git checkout $Branch exited $LASTEXITCODE" }
& git -C $Serving reset --hard "origin/$Branch"
if ($LASTEXITCODE -ne 0) { throw "git reset --hard exited $LASTEXITCODE" }

$sha = (& git -C $Serving rev-parse HEAD).Trim()
Write-Host "Serving checkout ready: $servingFull @ $sha (origin/$Branch)"

# The gitignored inputs the job resolves outside $Serving. A clone cannot carry
# these, which is exactly why run-net-position.ps1 takes them as parameters -
# report them here so a broken install is visible now, not at 08:00.
$externals = @(
    @{ Label = "rail interpreter (ABL-69)"; Path = "$DevCheckout\.venv\Scripts\python.exe" },
    @{ Label = "V014 artifacts";            Path = "$DevCheckout\models\net_position\V014" },
    @{ Label = "eval report root";          Path = "$DevCheckout\reports\net_position_eval" }
)
foreach ($e in $externals) {
    $state = if (Test-Path $e.Path) { "OK     " } else { "MISSING" }
    Write-Host ("  {0}  {1,-26} {2}" -f $state, $e.Label, $e.Path)
}

$launcher = "$servingFull\scripts\workstation\run-net-position-serving.ps1"
if (-not (Test-Path $launcher)) {
    throw ("$launcher is absent - origin/$Branch does not carry the ABL-692 " +
           "launcher yet. Land that PR before re-pointing the task.")
}

$newArgs = ('//B //Nologo "C:\Users\guill\bin\run-hidden.vbs" "powershell.exe" ' +
            '-WindowStyle Hidden -NoProfile -ExecutionPolicy Bypass -File ' + $launcher)

if ($UpdateScheduledTask) {
    $task = Get-ScheduledTask -TaskName $TaskName
    $old  = $task.Actions[0]
    Write-Host "Old action arguments: $($old.Arguments)"
    $action = New-ScheduledTaskAction -Execute $old.Execute -Argument $newArgs
    Set-ScheduledTask -TaskName $TaskName -Action $action | Out-Null
    $now = (Get-ScheduledTask -TaskName $TaskName).Actions[0].Arguments
    Write-Host "New action arguments: $now"
    Write-Host "Scheduled task '$TaskName' now runs the serving checkout."
} else {
    Write-Host ""
    Write-Host "Next step - re-point the scheduled task (NOT done by this run):"
    Write-Host "  .\install-serving-checkout.ps1 -UpdateScheduledTask"
    Write-Host ""
    Write-Host "It will set '$TaskName' to run:"
    Write-Host "  $launcher"
    Write-Host ""
    Write-Host "Then confirm the fix at the next 08:00 run with:"
    Write-Host "  Select-String 'net-position serving commit:' C:\Code\able\logs\net-position-forecast.log | Select-Object -Last 1"
}
