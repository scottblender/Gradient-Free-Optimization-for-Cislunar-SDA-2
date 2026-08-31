# ---------------- run_baseline_soo.ps1 ----------------
$ErrorActionPreference = "Stop"

$MeasurementModels = @("ANGLES_ONLY", "ANGLES_RANGE")
$MissionTypes = @("LOW_THRUST_TRANSFER", "LUNAR_GATEWAY")
$ObserverCounts = @(3, 5, 7, 10)

# Only use 1, 3, 5 periods for Lunar Gateway
$GatewayPeriods = @(1, 3, 5)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$RunOpt = Join-Path $ProjectRoot "run_opt.m"
if (-not (Test-Path $RunOpt)) { throw "Cannot find run_opt.m at: $RunOpt" }
$ProjectRootMatlab = $ProjectRoot.Replace("'", "''")
$RunOptMatlab = $RunOpt.Replace("'", "''")

$MatlabExe = "C:\Program Files\MATLAB\R2025b\bin\matlab.exe"
if (-not (Test-Path $MatlabExe)) { throw "Cannot find matlab.exe at: $MatlabExe" }

$RunsRoot = Join-Path (Join-Path $ProjectRoot "results") "runs"
$BaselineRoot = Join-Path $RunsRoot "BASELINE"

New-Item -ItemType Directory -Force -Path $RunsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $BaselineRoot | Out-Null

function Get-MissionCode {
    param([string]$Mission)

    switch ($Mission) {
        "LOW_THRUST_TRANSFER" { return "lt" }
        "LUNAR_GATEWAY"       { return "lg" }
        "PERIODIC_ORBIT"      { return "po" }
        "BALLISTIC_TRANSFER"  { return "bt" }
        "TIME_OPT_TRANSFER"   { return "tt" }
        "FUEL_OPT_TRANSFER"   { return "ft" }
        default               { return $Mission.ToLower() }
    }
}

function Get-MeasCode {
    param([string]$MeasModel)

    switch ($MeasModel) {
        "ANGLES_ONLY"  { return "ao" }
        "ANGLES_RANGE" { return "ar" }
        default        { return $MeasModel.ToLower() }
    }
}

function Invoke-MatlabRun {
    param(
        [string]$RunDir,
        [string]$Alg,
        [int]$MaxEvals,
        [string]$MissionType,
        [string]$MeasModel,
        [int]$NumObservers,
        [int]$NPeriods,
        [bool]$UseScreening,
        [bool]$UseJ1,
        [bool]$UseJ2,
        [bool]$UseJ3,
        [int]$Seed = 0
    )

    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

    $env:OPTIMIZER_MODE = $Alg
    $env:MAX_EVALS = "$MaxEvals"
    $env:USE_PARALLEL_OPT = "1"
    $env:MISSION_TYPE = $MissionType
    $env:MEAS_MODEL = $MeasModel
    $env:NUM_OBSERVERS = "$NumObservers"
    $env:NPERIODS = "$NPeriods"
    $env:USE_SCREENING = $(if ($UseScreening) { "1" } else { "0" })
    $env:USE_J1 = $(if ($UseJ1) { "1" } else { "0" })
    $env:USE_J2 = $(if ($UseJ2) { "1" } else { "0" })
    $env:USE_J3 = $(if ($UseJ3) { "1" } else { "0" })
    $env:SEED = "$Seed"
    $env:RUN_DIR = $RunDir

    Push-Location $RunDir
    try {
        $cmd = @"
try
    cd(getenv('RUN_DIR'));
    addpath('$ProjectRootMatlab');
    setup_project;
    run('$RunOptMatlab');
catch ME
    disp(getReport(ME,'extended'));
    exit(1);
end
exit(0);
"@

        & "$MatlabExe" -batch $cmd *> "console.log"
    }
    finally {
        Pop-Location
    }

    Write-Host "Saved -> $RunDir"
}

# ---------------- Baseline GA ----------------
$gaBaselineEvals = 6000
$baselineSeed = 0

foreach ($meas in $MeasurementModels) {
    $measCode = Get-MeasCode $meas

    $AlgRoot = Join-Path $BaselineRoot "runs_GA"
    $MeasRoot = Join-Path $AlgRoot $measCode
    New-Item -ItemType Directory -Force -Path $AlgRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $MeasRoot | Out-Null

    foreach ($mission in $MissionTypes) {
        $MissionCode = Get-MissionCode $mission
        $MissionOutDir = Join-Path $MeasRoot $MissionCode
        New-Item -ItemType Directory -Force -Path $MissionOutDir | Out-Null

        foreach ($nObs in $ObserverCounts) {

            if ($mission -eq "LUNAR_GATEWAY") {
                $periodList = $GatewayPeriods
            }
            else {
                $periodList = @(1)
            }

            foreach ($nper in $periodList) {

                if ($mission -eq "LOW_THRUST_TRANSFER") {
                    $caseName = "b_ga${gaBaselineEvals}_${measCode}_o${nObs}"
                }
                else {
                    $caseName = "b_ga${gaBaselineEvals}_${measCode}_o${nObs}_p${nper}"
                }

                $RunDir = Join-Path $MissionOutDir $caseName

                Write-Host "`n============================="
                Write-Host "Baseline GA: [$mission] [$meas] $caseName"
                Write-Host "FE budget: $gaBaselineEvals"
                Write-Host "============================="

                Invoke-MatlabRun -RunDir $RunDir -Alg "GA" -MaxEvals $gaBaselineEvals `
                    -MissionType $mission -MeasModel $meas -NumObservers $nObs -NPeriods $nper `
                    -UseScreening $true -UseJ1 $true -UseJ2 $true -UseJ3 $true `
                    -Seed $baselineSeed
            }
        }
    }
}

Write-Host "`nBaseline runs complete."
Write-Host "Baseline -> $BaselineRoot"
# -------------------------------------------------
