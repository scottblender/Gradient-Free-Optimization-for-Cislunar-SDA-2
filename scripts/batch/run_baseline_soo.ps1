# ---------------- run_baseline_soo.ps1 ----------------
param(
    [string]$MatlabExe = "",
    [int]$EvalBudget = 6000,
    [int[]]$Seeds = (0..19),
    [switch]$Pilot
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$RunOpt = Join-Path $ProjectRoot "run_opt.m"
if (-not (Test-Path $RunOpt)) { throw "Cannot find run_opt.m at: $RunOpt" }

if ([string]::IsNullOrWhiteSpace($MatlabExe)) {
    $matlabCommand = Get-Command matlab.exe -ErrorAction SilentlyContinue
    if ($matlabCommand) {
        $MatlabExe = $matlabCommand.Source
    }
    else {
        $MatlabExe = "C:\Program Files\MATLAB\R2026a\bin\matlab.exe"
    }
}
if (-not (Test-Path $MatlabExe)) {
    throw "Cannot find matlab.exe. Pass -MatlabExe or add MATLAB to PATH."
}

if ($EvalBudget -lt 60 -or ($EvalBudget % 60) -ne 0) {
    throw "EvalBudget must be a positive multiple of 60 for GA and PSO."
}
if ($Seeds.Count -eq 0 -or ($Seeds | Where-Object { $_ -lt 0 }).Count -gt 0) {
    throw "Seeds must contain nonnegative integers."
}

$ProjectRootMatlab = $ProjectRoot.Replace("'", "''")
$RunOptMatlab = $RunOpt.Replace("'", "''")
$MeasurementNoiseSeed = 1001

function Get-MissionCode {
    param([string]$Mission)

    switch ($Mission) {
        "LOW_THRUST_TRANSFER" { return "lt" }
        "LUNAR_GATEWAY"       { return "lg" }
        "GATEWAY_IMPULSE"     { return "gi" }
        "PERIODIC_ORBIT"      { return "po" }
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
        [string]$StudyId,
        [string]$MissionType,
        [string]$MeasModel,
        [int]$NumObservers,
        [int]$NPeriods,
        [int]$Seed
    )

    $dataDir = Join-Path $RunDir "data"
    $stateFile = Join-Path $dataDir "optimization_run.mat"
    $trackingFile = Join-Path $dataDir "tracking_data.mat"

    if (Test-Path $trackingFile) {
        Write-Host "Skipping completed run -> $RunDir"
        return
    }
    if (Test-Path $stateFile) {
        throw "Incomplete run exists at $RunDir. Inspect or move it before resuming."
    }

    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

    $env:STUDY_ID = $StudyId
    $env:OPTIMIZER_MODE = $Alg
    $env:MAX_EVALS = "$EvalBudget"
    $env:USE_PARALLEL_OPT = "1"
    $env:MISSION_TYPE = $MissionType
    $env:MEAS_MODEL = $MeasModel
    $env:NUM_OBSERVERS = "$NumObservers"
    $env:NPERIODS = "$NPeriods"
    $env:USE_SCREENING = "1"
    $env:USE_J1 = "1"
    $env:USE_J2 = "1"
    $env:USE_J3 = "1"
    $env:SEED = "$Seed"
    $env:MEAS_NOISE_SEED = "$MeasurementNoiseSeed"
    $env:MAKE_PLOTS = "0"
    $env:IMPULSE_DV_MPS = "10"
    $env:IMPULSE_DIRECTION = "PROGRADE"
    $env:IMPULSE_DURATION_TU = "1.5"
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
        if ($LASTEXITCODE -ne 0) {
            throw "MATLAB failed with exit code $LASTEXITCODE. See $RunDir\console.log"
        }
    }
    finally {
        Pop-Location
    }

    Write-Host "Saved -> $RunDir"
}

# GA-only reproduction/sensitivity study from the original paper.
$StudyId = "reviewer2_baseline_v1"
$StudyFolder = "BASELINE"
$MeasurementModels = @("ANGLES_ONLY", "ANGLES_RANGE")
$MissionTypes = @("LOW_THRUST_TRANSFER", "LUNAR_GATEWAY", "GATEWAY_IMPULSE")
$ObserverCounts = @(3, 5, 7, 10)
$GatewayPeriods = @(1, 3, 5)

if ($Pilot) {
    if (-not $PSBoundParameters.ContainsKey("EvalBudget")) { $EvalBudget = 120 }
    if (-not $PSBoundParameters.ContainsKey("Seeds")) { $Seeds = @(0) }

    $StudyId = "reviewer2_baseline_pilot_v1"
    $StudyFolder = "BASELINE_PILOT"
    $MeasurementModels = @("ANGLES_ONLY")
    $ObserverCounts = @(3)
    $GatewayPeriods = @(1)
}

$BaselineRoot = Join-Path (Join-Path (Join-Path $ProjectRoot "results") "runs") $StudyFolder
New-Item -ItemType Directory -Force -Path $BaselineRoot | Out-Null

$TotalRuns = 0
foreach ($meas in $MeasurementModels) {
    foreach ($mission in $MissionTypes) {
        foreach ($nObs in $ObserverCounts) {
            $periodList = if ($mission -eq "LUNAR_GATEWAY") { $GatewayPeriods } else { @(1) }
            $TotalRuns += $periodList.Count * $Seeds.Count
        }
    }
}
$CompletedRuns = 0

foreach ($meas in $MeasurementModels) {
    $measCode = Get-MeasCode $meas
    foreach ($mission in $MissionTypes) {
        $missionCode = Get-MissionCode $mission
        $missionRoot = Join-Path (Join-Path (Join-Path $BaselineRoot "runs_GA") $measCode) $missionCode
        New-Item -ItemType Directory -Force -Path $missionRoot | Out-Null

        foreach ($nObs in $ObserverCounts) {
            $periodList = if ($mission -eq "LUNAR_GATEWAY") { $GatewayPeriods } else { @(1) }

            foreach ($nper in $periodList) {
                foreach ($seed in $Seeds) {
                    $seedCode = $seed.ToString("000")
                    if ($mission -eq "LUNAR_GATEWAY") {
                        $runName = "b_ga$($EvalBudget)_$($measCode)_o$($nObs)_p$($nper)_seed$($seedCode)"
                    }
                    else {
                        $runName = "b_ga$($EvalBudget)_$($measCode)_o$($nObs)_seed$($seedCode)"
                    }
                    $runDir = Join-Path $missionRoot $runName

                    $CompletedRuns++
                    $percent = [math]::Round(100 * $CompletedRuns / $TotalRuns, 1)
                    Write-Progress -Activity "Baseline GA study" -Status "$CompletedRuns of $TotalRuns | $mission | $meas | seed $seed" -PercentComplete $percent

                    Write-Host "`nBaseline GA: [$mission] [$meas] $runName"
                    Write-Host "FE budget: $EvalBudget | optimizer seed: $seed | measurement seed: $MeasurementNoiseSeed"

                    Invoke-MatlabRun -RunDir $runDir -Alg "GA" -StudyId $StudyId `
                        -MissionType $mission -MeasModel $meas -NumObservers $nObs `
                        -NPeriods $nper -Seed $seed
                }
            }
        }
    }
}

Write-Progress -Activity "Baseline GA study" -Completed
Write-Host "`nBaseline GA runs complete."
Write-Host "Baseline -> $BaselineRoot"
# -------------------------------------------------------
