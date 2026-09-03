# ---------------- run_runtime_comparison_1200_soo.ps1 ----------------
param(
    [string]$MatlabExe = "",
    [int[]]$Seeds = (0..19)
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
if ($Seeds.Count -eq 0 -or ($Seeds | Where-Object { $_ -lt 0 }).Count -gt 0) {
    throw "Seeds must contain nonnegative integers."
}

$BatchEntryMatlab = $BatchEntry.Replace("'", "''")

# Focused Reviewer 2 runtime/scaling comparison. All five methods use the
# same representative design so Bayesian surrogate/acquisition overhead can
# be compared directly against the population-based algorithms.
$StudyId = "reviewer2_runtime_comparison_1200_v1"
$StudyFolder = "RUNTIME_COMPARISON_1200"
$Algs = @("GA", "PSO", "BAYESIAN", "ABC", "ACO")
$EvalBudget = 1200
$MissionType = "LUNAR_GATEWAY"
$MeasModel = "ANGLES_ONLY"
$NumObservers = 3
$NPeriods = 1
$MeasurementNoiseSeed = 1001

$StudyRoot = Join-Path (Join-Path $ProjectRoot "results") $StudyFolder
New-Item -ItemType Directory -Force -Path $StudyRoot | Out-Null

function Invoke-MatlabRun {
    param(
        [string]$RunDir,
        [string]$Alg,
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

$TotalRuns = $Algs.Count * $Seeds.Count
$CompletedRuns = 0

foreach ($alg in $Algs) {
    $algCode = $alg.ToLower()
    $missionRoot = Join-Path (Join-Path (Join-Path $StudyRoot "runs_$alg") "ao") "lg"
    New-Item -ItemType Directory -Force -Path $missionRoot | Out-Null

    foreach ($seed in $Seeds) {
        $CompletedRuns++
        $seedCode = $seed.ToString("000")
        $runName = "rt_$($algCode)1200_ao_o3_p1_seed$($seedCode)"
        $runDir = Join-Path $missionRoot $runName

        $percent = [math]::Round(100 * $CompletedRuns / $TotalRuns, 1)
        Write-Progress -Activity "1200-FE runtime comparison" `
            -Status "$CompletedRuns of $TotalRuns | $alg | Lunar Gateway | seed $seed" `
            -PercentComplete $percent

        Write-Host "`nRuntime comparison: [$alg] [LUNAR_GATEWAY] [ANGLES_ONLY] $runName"
        Write-Host "Observers: 3 | periods: 1 | FE budget: 1200"
        Write-Host "Optimizer seed: $seed | measurement seed: $MeasurementNoiseSeed"

        Invoke-MatlabRun -RunDir $runDir -Alg $alg -Seed $seed
    }
}

Write-Progress -Activity "1200-FE runtime comparison" -Completed
Write-Host "`n1200-FE runtime comparison complete."
Write-Host "Runtime comparison -> $StudyRoot"
# ---------------------------------------------------------
