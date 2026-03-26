# ---------------- run_all_soo.ps1 ----------------
$ErrorActionPreference = "Stop"

$Algs = @("GA", "PSO", "BAYESIAN", "ABC", "ACO")
$MissionTypes = @("LOW_THRUST_TRANSFER", "LUNAR_GATEWAY")
$ObserverCounts = @(3, 5, 7, 10)
$GatewayPeriods = @(1, 3, 5, 10)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunOpt = Join-Path $ProjectRoot "run_opt.m"
if (-not (Test-Path $RunOpt)) { throw "Cannot find run_opt.m at: $RunOpt" }

$MatlabExe = "C:\Program Files\MATLAB\R2025b\bin\matlab.exe"
if (-not (Test-Path $MatlabExe)) { throw "Cannot find matlab.exe at: $MatlabExe" }

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutDir = Join-Path $ProjectRoot ("runs_" + $Timestamp)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Invoke-MatlabRun {
    param(
        [string]$RunDir, [string]$Alg, [int]$MaxIters,
        [string]$MissionType, [int]$NumObservers, [int]$NPeriods,
        [bool]$UseScreening, [bool]$UseJ1, [bool]$UseJ2, [bool]$UseJ3
    )

    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

    $env:OPTIMIZER_MODE = $Alg
    $env:MAX_ITERS = "$MaxIters"
    $env:MISSION_TYPE = $MissionType
    $env:NUM_OBSERVERS = "$NumObservers"
    $env:NPERIODS = "$NPeriods"
    $env:USE_SCREENING = $(if ($UseScreening) { "1" } else { "0" })
    $env:USE_J1 = $(if ($UseJ1) { "1" } else { "0" })
    $env:USE_J2 = $(if ($UseJ2) { "1" } else { "0" })
    $env:USE_J3 = $(if ($UseJ3) { "1" } else { "0" })
    $env:RUN_DIR = $RunDir

    Push-Location $RunDir

    $cmd = @"
try
    cd(getenv('RUN_DIR'));
    addpath(genpath('$ProjectRoot'));
    run('$RunOpt');
catch ME
    disp(getReport(ME,'extended'));
    exit(1);
end
exit(0);
"@

    & "$MatlabExe" -batch $cmd *> "console.log"
    Pop-Location

    Write-Host "Saved -> $RunDir"
}

# ---------------- (A) Baseline GA ----------------
$gaBaselineIters = 600

foreach ($mission in $MissionTypes) {
    $MissionOutDir = Join-Path $OutDir $mission
    New-Item -ItemType Directory -Force -Path $MissionOutDir | Out-Null

    foreach ($nObs in $ObserverCounts) {

        if ($mission -eq "LUNAR_GATEWAY") {
            $periodList = $GatewayPeriods
        } else {
            $periodList = @(1)
        }

        foreach ($nper in $periodList) {

            if ($mission -eq "LOW_THRUST_TRANSFER") {
                $caseName = "BASELINE_GA_${gaBaselineIters}_obs${nObs}_screenON_J123"
            } else {
                $caseName = "BASELINE_GA_${gaBaselineIters}_obs${nObs}_nper${nper}_screenON_J123"
            }

            $RunDir = Join-Path $MissionOutDir $caseName

            Write-Host "`n============================="
            Write-Host "Baseline GA: [$mission] $caseName"
            Write-Host "============================="

            Invoke-MatlabRun -RunDir $RunDir -Alg "GA" -MaxIters $gaBaselineIters `
                -MissionType $mission -NumObservers $nObs -NPeriods $nper `
                -UseScreening $true -UseJ1 $true -UseJ2 $true -UseJ3 $true
        }
    }
}

# ---------------- (B) SOO Sweep ----------------
$itersSweep = 100

$Cases = @(
    @{ name="screenON_J123";  screening=$true;  J1=$true;  J2=$true;  J3=$true  },
    @{ name="screenOFF_J123"; screening=$false; J1=$true;  J2=$true;  J3=$true  },
    @{ name="screenON_J1";    screening=$true;  J1=$true;  J2=$false; J3=$false },
    @{ name="screenON_J2";    screening=$true;  J1=$false; J2=$true;  J3=$false },
    @{ name="screenON_J3";    screening=$true;  J1=$false; J2=$false; J3=$true  }
)

foreach ($mission in $MissionTypes) {
    $MissionOutDir = Join-Path $OutDir $mission

    foreach ($nObs in $ObserverCounts) {

        if ($mission -eq "LUNAR_GATEWAY") {
            $periodList = $GatewayPeriods
        } else {
            $periodList = @(1)
        }

        foreach ($nper in $periodList) {
            foreach ($alg in $Algs) {
                foreach ($cc in $Cases) {

                    if ($mission -eq "LOW_THRUST_TRANSFER") {
                        $runName = "SOO_${alg}_${itersSweep}_obs${nObs}_" + $cc.name
                    } else {
                        $runName = "SOO_${alg}_${itersSweep}_obs${nObs}_nper${nper}_" + $cc.name
                    }

                    $RunDir = Join-Path $MissionOutDir $runName

                    Write-Host "`n============================="
                    Write-Host "Running: [$mission] $runName"
                    Write-Host "============================="

                    Invoke-MatlabRun -RunDir $RunDir -Alg $alg -MaxIters $itersSweep `
                        -MissionType $mission -NumObservers $nObs -NPeriods $nper `
                        -UseScreening $cc.screening -UseJ1 $cc.J1 -UseJ2 $cc.J2 -UseJ3 $cc.J3
                }
            }
        }
    }
}

Write-Host "`nAll runs complete -> $OutDir"
# -------------------------------------------------