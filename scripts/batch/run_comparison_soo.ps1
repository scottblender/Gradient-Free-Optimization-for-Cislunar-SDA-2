# ---------------- run_comparison_soo.ps1 ----------------
$ErrorActionPreference = "Stop"

$Algs = @("GA", "PSO", "BAYESIAN", "ABC", "ACO")
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
$ComparisonRoot = Join-Path $RunsRoot "COMPARISON"

New-Item -ItemType Directory -Force -Path $RunsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $ComparisonRoot | Out-Null

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
        [int]$MaxIters,
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
    $env:MAX_ITERS = "$MaxIters"
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

# ---------------- Comparison sweep ----------------
$itersSweep = 100
$sweepSeed = 0

$Cases = @(
    @{ screening=$true;  J1=$true;  J2=$true;  J3=$true  },
    @{ screening=$false; J1=$true;  J2=$true;  J3=$true  },
    @{ screening=$true;  J1=$true;  J2=$false; J3=$false },
    @{ screening=$true;  J1=$false; J2=$true;  J3=$false },
    @{ screening=$true;  J1=$false; J2=$false; J3=$true  }
)

foreach ($alg in $Algs) {
    $algCode = $alg.ToLower()
    $AlgRoot = Join-Path $ComparisonRoot "runs_$alg"
    New-Item -ItemType Directory -Force -Path $AlgRoot | Out-Null

    foreach ($meas in $MeasurementModels) {
        $measCode = Get-MeasCode $meas
        $MeasRoot = Join-Path $AlgRoot $measCode
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
                    foreach ($cc in $Cases) {

                        $screenCode = if ($cc.screening) { "1" } else { "0" }
                        $jCode = "$( [int]$cc.J1 )$( [int]$cc.J2 )$( [int]$cc.J3 )"

                        if ($mission -eq "LOW_THRUST_TRANSFER") {
                            $runName = "s_${algCode}${itersSweep}_${measCode}_o${nObs}_s${screenCode}_j${jCode}"
                        }
                        else {
                            $runName = "s_${algCode}${itersSweep}_${measCode}_o${nObs}_p${nper}_s${screenCode}_j${jCode}"
                        }

                        $RunDir = Join-Path $MissionOutDir $runName

                        Write-Host "`n============================="
                        Write-Host "Running: [$mission] [$meas] $runName"
                        Write-Host "============================="

                        Invoke-MatlabRun -RunDir $RunDir -Alg $alg -MaxIters $itersSweep `
                            -MissionType $mission -MeasModel $meas -NumObservers $nObs -NPeriods $nper `
                            -UseScreening $cc.screening -UseJ1 $cc.J1 -UseJ2 $cc.J2 -UseJ3 $cc.J3 `
                            -Seed $sweepSeed
                    }
                }
            }
        }
    }
}

Write-Host "`nComparison runs complete."
Write-Host "Comparison -> $ComparisonRoot"
# -------------------------------------------------
