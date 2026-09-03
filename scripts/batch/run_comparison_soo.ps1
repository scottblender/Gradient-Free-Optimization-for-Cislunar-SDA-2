# ---------------- run_comparison_soo.ps1 ----------------
param(
    [string]$MatlabExe = "",
    [int]$EvalBudget = 6000,
    [int[]]$Seeds = (0..19),
    [switch]$Pilot
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$RunOpt = Join-Path $ProjectRoot "run_opt.m"
$BatchEntry = Join-Path $ProjectRoot "scripts\batch\run_batch_entry.m"
if (-not (Test-Path $RunOpt)) { throw "Cannot find run_opt.m at: $RunOpt" }
if (-not (Test-Path $BatchEntry)) { throw "Cannot find MATLAB batch entry at: $BatchEntry" }

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

$BatchEntryMatlab = $BatchEntry.Replace("'", "''")
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
    $hasState = Test-Path $stateFile
    $hasTracking = Test-Path $trackingFile

    if ($hasState -and $hasTracking) {
        Write-Host "Skipping completed run -> $RunDir"
        return
    }

    if ($hasState -or $hasTracking) {
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $archiveRoot = Join-Path (Join-Path $ProjectRoot "results") "_INCOMPLETE_RUNS"
        $archiveStudy = Join-Path $archiveRoot $StudyId
        New-Item -ItemType Directory -Force -Path $archiveStudy | Out-Null

        $runLeaf = Split-Path -Leaf $RunDir
        $archiveDir = Join-Path $archiveStudy "$($runLeaf)_$stamp"
        $suffix = 1
        while (Test-Path $archiveDir) {
            $archiveDir = Join-Path $archiveStudy "$($runLeaf)_$stamp`_$suffix"
            $suffix++
        }

        Write-Warning "Incomplete run detected. Archiving before retry: $RunDir"
        Move-Item -Path $RunDir -Destination $archiveDir
        Write-Host "Archived incomplete run -> $archiveDir"
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
    $env:PROJECT_ROOT = $ProjectRoot

    Push-Location $RunDir
    try {
        $stdoutLog = Join-Path $RunDir "console.stdout.log"
        $stderrLog = Join-Path $RunDir "console.stderr.log"
        $consoleLog = Join-Path $RunDir "console.log"
        $batchCommand = "run('$BatchEntryMatlab')"

        try {
            $process = Start-Process `
                -FilePath $MatlabExe `
                -ArgumentList @("-batch", "`"$batchCommand`"") `
                -WorkingDirectory $RunDir `
                -RedirectStandardOutput $stdoutLog `
                -RedirectStandardError $stderrLog `
                -NoNewWindow `
                -Wait `
                -PassThru

            $matlabExitCode = $process.ExitCode

            if (Test-Path $consoleLog) { Remove-Item $consoleLog -Force }
            if (Test-Path $stdoutLog) { Get-Content $stdoutLog | Add-Content $consoleLog }
            if (Test-Path $stderrLog) { Get-Content $stderrLog | Add-Content $consoleLog }

            if ($matlabExitCode -ne 0) {
                throw "MATLAB failed with exit code $matlabExitCode. See $consoleLog"
            }
        }
        finally {
            Remove-Item $stdoutLog -Force -ErrorAction SilentlyContinue
            Remove-Item $stderrLog -Force -ErrorAction SilentlyContinue
        }
    }
    finally {
        Pop-Location
    }

    Write-Host "Saved -> $RunDir"
}

# Reviewer-facing optimizer comparison:
# 5 methods x 3 target cases x 20 seeds = 300 runs by default.
$StudyId = "reviewer2_comparison_v1"
$StudyFolder = "COMPARISON"
$Algs = @("GA", "PSO", "BAYESIAN", "ABC", "ACO")
$MeasurementModels = @("ANGLES_ONLY")
$MissionTypes = @("LUNAR_GATEWAY", "LOW_THRUST_TRANSFER", "GATEWAY_IMPULSE")
$ObserverCounts = @(3)
$GatewayPeriods = @(1)

if ($Pilot) {
    if (-not $PSBoundParameters.ContainsKey("EvalBudget")) { $EvalBudget = 1200 }
    if (-not $PSBoundParameters.ContainsKey("Seeds")) { $Seeds = @(0) }

    $StudyId = "reviewer2_comparison_pilot_1200_v1"
    $StudyFolder = "COMPARISON_PILOT_1200"
}

$ComparisonRoot = Join-Path (Join-Path $ProjectRoot "results") $StudyFolder
New-Item -ItemType Directory -Force -Path $ComparisonRoot | Out-Null

$TotalRuns = 0
foreach ($alg in $Algs) {
    foreach ($meas in $MeasurementModels) {
        foreach ($mission in $MissionTypes) {
            foreach ($nObs in $ObserverCounts) {
                $periodList = if ($mission -eq "LUNAR_GATEWAY") { $GatewayPeriods } else { @(1) }
                $TotalRuns += $periodList.Count * $Seeds.Count
            }
        }
    }
}
$CompletedRuns = 0

foreach ($alg in $Algs) {
    $algCode = $alg.ToLower()
    foreach ($meas in $MeasurementModels) {
        $measCode = Get-MeasCode $meas
        foreach ($mission in $MissionTypes) {
            $missionCode = Get-MissionCode $mission
            $missionRoot = Join-Path (Join-Path (Join-Path $ComparisonRoot "runs_$alg") $measCode) $missionCode
            New-Item -ItemType Directory -Force -Path $missionRoot | Out-Null

            foreach ($nObs in $ObserverCounts) {
                $periodList = if ($mission -eq "LUNAR_GATEWAY") { $GatewayPeriods } else { @(1) }

                foreach ($nper in $periodList) {
                    foreach ($seed in $Seeds) {
                        $seedCode = $seed.ToString("000")
                        if ($mission -eq "LUNAR_GATEWAY") {
                            $runName = "c_$($algCode)$($EvalBudget)_$($measCode)_o$($nObs)_p$($nper)_seed$($seedCode)"
                        }
                        else {
                            $runName = "c_$($algCode)$($EvalBudget)_$($measCode)_o$($nObs)_seed$($seedCode)"
                        }
                        $runDir = Join-Path $missionRoot $runName

                        $CompletedRuns++
                        $percent = [math]::Round(100 * $CompletedRuns / $TotalRuns, 1)
                        Write-Progress -Activity "Optimizer comparison study" -Status "$CompletedRuns of $TotalRuns | $alg | $mission | seed $seed" -PercentComplete $percent

                        Write-Host "`nComparison: [$alg] [$mission] [$meas] $runName"
                        Write-Host "FE budget: $EvalBudget | optimizer seed: $seed | measurement seed: $MeasurementNoiseSeed"

                        Invoke-MatlabRun -RunDir $runDir -Alg $alg -StudyId $StudyId `
                            -MissionType $mission -MeasModel $meas -NumObservers $nObs `
                            -NPeriods $nper -Seed $seed
                    }
                }
            }
        }
    }
}

Write-Progress -Activity "Optimizer comparison study" -Completed
Write-Host "`nComparison runs complete."
Write-Host "Comparison -> $ComparisonRoot"
# ---------------------------------------------------------
