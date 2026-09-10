# Entry point for the scheduled task `able-net-position-forecast` (ABL-692).
#
# WHY THIS FILE EXISTS
#
# Until 2026-09-10 the task ran the job straight out of
# C:\Code\able\energy-forecast - the checkout agents use as a scratch working
# tree. That tree had been parked on a feature branch since 2026-08-28, so
# every serving change merged to origin/main after 2026-08-27 16:10 was inert
# in production for twelve days, including the ABL-650 band recalibration
# (PR #107). Nothing broke and nothing alerted, because there is no deploy step
# here at all: the cron executes whatever happens to be on disk.
#
# This launcher supplies the two things that were missing.
#
#   1. It runs a checkout that no agent works in, hard-reset to origin/main, so
#      "merged" and "running" mean the same thing again.
#   2. It writes the resolved commit into the transcript, so every vintage is
#      attributable to a SHA.
#
# (2) is the one that matters more. A stale tree is recoverable the moment you
# can see it; before this line existed, nothing on disk recorded which code
# produced a given vintage, so staleness was undetectable from the log alone.
#
# A sync failure does NOT skip the forecast. A missing vintage is worse than a
# one-day-stale one, and the witness line is emitted either way - so a stale
# run becomes a loud, attributable stale run rather than a silent one.

[CmdletBinding()]
param(
    # The serving checkout. Created by install-serving-checkout.ps1. Nothing
    # else may work in it - this script hard-resets it on every run.
    [string] $Serving = "C:\Code\able\energy-forecast-serving",
    [string] $Branch  = "main",
    [string] $LogDir  = "C:\Code\able\logs"
)

$ErrorActionPreference = "Stop"

# The shared dev checkout. Hard-resetting this would discard whatever an agent
# has in progress and move HEAD under a live run - the hazard ABL-692 was filed
# to remove, not to automate. Refuse rather than "fix" someone else's tree.
$DevCheckout = "C:\Code\able\energy-forecast"

# Grep this prefix in net-position-forecast.log to attribute a vintage to code.
# Kept as a named constant because tests/test_abl692_serving_checkout.py and
# the ABL-692 runbook both assert on it.
$WitnessPrefix = "net-position serving commit:"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force $LogDir | Out-Null }
Start-Transcript -Path (Join-Path $LogDir "net-position-forecast.log") -Append | Out-Null
try {
    $servingFull = [IO.Path]::GetFullPath($Serving).TrimEnd('\')
    $devFull     = [IO.Path]::GetFullPath($DevCheckout).TrimEnd('\')
    if ($servingFull -ieq $devFull) {
        throw ("Refusing to serve from $servingFull - that is the shared dev " +
               "checkout (ABL-692). Point -Serving at a checkout no agent works in.")
    }
    if (-not (Test-Path (Join-Path $Serving ".git"))) {
        throw ("No git checkout at $Serving - run " +
               "scripts\workstation\install-serving-checkout.ps1 first (ABL-692).")
    }

    # Sync. Native git failures surface as a non-zero $LASTEXITCODE, not an
    # exception, so each step is checked explicitly. stderr is deliberately not
    # redirected: it belongs in the transcript, and redirecting a native
    # command's stderr in PS 5.1 wraps it as a NativeCommandError.
    $synced = $true
    & git -C $Serving fetch --prune origin
    if ($LASTEXITCODE -ne 0) { $synced = $false }
    if ($synced) {
        & git -C $Serving reset --hard "origin/$Branch"
        if ($LASTEXITCODE -ne 0) { $synced = $false }
    }

    # No `git clean` here, on purpose. The eval reports this job writes into
    # the tree are gitignored, and a future operator may well park a large
    # gitignored artifact directory alongside them; `clean -xdf` would delete
    # both. `reset --hard` already guarantees every *tracked* file matches
    # origin/$Branch, which is the property serving actually needs.

    $sha     = (& git -C $Serving rev-parse HEAD)
    $subject = (& git -C $Serving log -1 --format=%s)
    $when    = (& git -C $Serving log -1 --format=%cI)
    if ($sha) { $sha = $sha.Trim() }

    if ($synced) {
        Write-Host "$WitnessPrefix $sha (origin/$Branch, committed $when) $subject"
    } else {
        Write-Warning ("net-position serving sync FAILED - running the last " +
                       "successfully synced tree. This vintage may predate " +
                       "origin/$Branch.")
        Write-Host "$WitnessPrefix $sha (STALE - sync failed, committed $when) $subject"
    }

    $job = Join-Path $Serving "scripts\workstation\run-net-position.ps1"
    if (-not (Test-Path $job)) { throw "Job script not found at $job" }
    & $job -Repo $Serving
}
finally {
    Stop-Transcript | Out-Null
}
