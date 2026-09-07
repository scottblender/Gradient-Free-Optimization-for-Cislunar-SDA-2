# ---------------- run_ga_objective_screening_soo.ps1 ----------------
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
    throw "EvalBudget must be a positive multiple of 60 for GA."
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
        default               { return $Mission.ToLower() }
    }
}

function Invoke-MatlabRun {
    param(
        [string]$RunDir,
        [string]$MissionType,
        [int]$Seed,
        [int]$UseScreening,
        [int]$UseJ1,
        [int]$UseJ2,
        [int]$UseJ3
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
    $env:OPTIMIZER_MODE = "GA"
    $env:MAX_EVALS = "$EvalBudget"
    $env:USE_PARALLEL_OPT = "1"
    $env:MISSION_TYPE = $MissionType
    $env:MEAS_MODEL = "ANGLES_ONLY"
    $env:NUM_OBSERVERS = "3"
    $env:NPERIODS = "1"
    $env:USE_SCREENING = "$UseScreening"
    $env:USE_J1 = "$UseJ1"
    $env:USE_J2 = "$UseJ2"
    $env:USE_J3 = "$UseJ3"
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

# GA-only objective/screening sensitivity study. This intentionally mirrors
# the five configurations in the original comparison-study table while
# separating that scientific sensitivity question from the optimizer
# benchmark and baseline studies.
$StudyId = "reviewer2_ga_objective_screening_v1"
$StudyFolder = "GA_OBJECTIVE_SCREENING"
$MissionTypes = @("LUNAR_GATEWAY", "LOW_THRUST_TRANSFER", "GATEWAY_IMPULSE")

$Cases = @(
    [pscustomobject]@{
        Code = "combined_on"
        Label = "Combined objective with screening"
        Screening = 1
        J1 = 1
        J2 = 1
        J3 = 1
    },
    [pscustomobject]@{
        Code = "combined_off"
        Label = "Combined objective without screening"
        Screening = 0
        J1 = 1
        J2 = 1
        J3 = 1
    },
    [pscustomobject]@{
        Code = "j1_only"
        Label = "Estimation-error only"
        Screening = 1
        J1 = 1
        J2 = 0
        J3 = 0
    },
    [pscustomobject]@{
        Code = "j2_only"
        Label = "Uncertainty only"
        Screening = 1
        J1 = 0
        J2 = 1
        J3 = 0
    },
    [pscustomobject]@{
        Code = "j3_only"
        Label = "Stability only"
        Screening = 1
        J1 = 0
        J2 = 0
        J3 = 1
    }
)

if ($Pilot) {
    if (-not $PSBoundParameters.ContainsKey("EvalBudget")) { $EvalBudget = 120 }
    if (-not $PSBoundParameters.ContainsKey("Seeds")) { $Seeds = @(0) }
    $StudyId = "reviewer2_ga_objective_screening_pilot_v1"
    $StudyFolder = "GA_OBJECTIVE_SCREENING_PILOT"
}

$StudyRoot = Join-Path (Join-Path $ProjectRoot "results") $StudyFolder
New-Item -ItemType Directory -Force -Path $StudyRoot | Out-Null

$TotalRuns = $Cases.Count * $MissionTypes.Count * $Seeds.Count
$CompletedRuns = 0

foreach ($case in $Cases) {
    foreach ($mission in $MissionTypes) {
        $missionCode = Get-MissionCode $mission
        $missionRoot = Join-Path (Join-Path (Join-Path $StudyRoot $case.Code) "ao") $missionCode
        New-Item -ItemType Directory -Force -Path $missionRoot | Out-Null

        foreach ($seed in $Seeds) {
            $CompletedRuns++
            $seedCode = $seed.ToString("000")
            $runName = "gaobj_$($case.Code)_$($missionCode)_ao_o3_p1_fe$($EvalBudget)_seed$($seedCode)"
            $runDir = Join-Path $missionRoot $runName

            $percent = [math]::Round(100 * $CompletedRuns / $TotalRuns, 1)
            Write-Progress -Activity "GA objective/screening study" `
                -Status "$CompletedRuns of $TotalRuns | $($case.Code) | $mission | seed $seed" `
                -PercentComplete $percent

            Write-Host "`nGA objective/screening: [$($case.Label)] [$mission] $runName"
            Write-Host "Screening: $($case.Screening) | J1/J2/J3: $($case.J1)/$($case.J2)/$($case.J3)"
            Write-Host "AO | observers: 3 | LG periods: 1 | FE budget: $EvalBudget"
            Write-Host "Optimizer seed: $seed | measurement seed: $MeasurementNoiseSeed"

            Invoke-MatlabRun -RunDir $runDir -MissionType $mission -Seed $seed `
                -UseScreening $case.Screening -UseJ1 $case.J1 `
                -UseJ2 $case.J2 -UseJ3 $case.J3
        }
    }
}

Write-Progress -Activity "GA objective/screening study" -Completed
Write-Host "`nGA objective/screening study complete."
Write-Host "Study -> $StudyRoot"
# ---------------------------------------------------------------
