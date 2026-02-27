# ---------------- run_all_soo.ps1 ----------------
$ErrorActionPreference = "Stop"

# Optimizers to run (normal sweep)
$Algs = @("GA", "PSO", "BAYESIAN", "ABC", "ACO")

# Folder where this PS script lives (project root)
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

# Absolute path to your MATLAB script
$RunOpt = Join-Path $ProjectRoot "run_opt.m"
if (-not (Test-Path $RunOpt)) { throw "Cannot find run_opt.m at: $RunOpt" }

# MATLAB path (edit if needed)
$MatlabExe = "C:\Program Files\MATLAB\R2025b\bin\matlab.exe"
if (-not (Test-Path $MatlabExe)) { throw "Cannot find matlab.exe at: $MatlabExe" }

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutDir = Join-Path $ProjectRoot ("runs_" + $Timestamp)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Invoke-MatlabRun {
    param(
        [Parameter(Mandatory=$true)][string]$RunDir,
        [Parameter(Mandatory=$true)][string]$Alg,
        [Parameter(Mandatory=$true)][int]$MaxIters,
        [Parameter(Mandatory=$true)][bool]$UseScreening,
        [Parameter(Mandatory=$true)][bool]$UseJ1,
        [Parameter(Mandatory=$true)][bool]$UseJ2,
        [Parameter(Mandatory=$true)][bool]$UseJ3
    )

    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

    $env:OPTIMIZER_MODE = $Alg
    $env:MAX_ITERS = "$MaxIters"
    $env:USE_SCREENING = $(if ($UseScreening) { "1" } else { "0" })
    $env:USE_J1 = $(if ($UseJ1) { "1" } else { "0" })
    $env:USE_J2 = $(if ($UseJ2) { "1" } else { "0" })
    $env:USE_J3 = $(if ($UseJ3) { "1" } else { "0" })

    # NEW: tell MATLAB exactly where to write artifacts
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

# ---------------- Normal sweep (uses your default MAX_ITERS in run_opt unless you override it here) ----------------
foreach ($alg in $Algs) {
    $RunDir = Join-Path $OutDir ("SOO_" + $alg)
    Write-Host "`n============================="
    Write-Host "Running optimizer: $alg"
    Write-Host "============================="

    Invoke-MatlabRun -RunDir $RunDir -Alg $alg -MaxIters 100 -UseScreening $true -UseJ1 $true -UseJ2 $true -UseJ3 $true
}

# ---------------- Baseline GA grid (1000 iters) ----------------
# Cases:
# - screening ON/OFF
# - cost component toggles: {all on} + {J1 only} + {J2 only} + {J3 only}
$gaIters = 1000
$screeningCases = @($false, $true)

$costCases = @(
    @{ name="J123"; J1=$true;  J2=$true;  J3=$true  },
    @{ name="J1";   J1=$true;  J2=$false; J3=$false },
    @{ name="J2";   J1=$false; J2=$true;  J3=$false },
    @{ name="J3";   J1=$false; J2=$false; J3=$true  }
)

foreach ($sc in $screeningCases) {
    $scName = $(if ($sc) { "screenON" } else { "screenOFF" })

    foreach ($cc in $costCases) {
        $caseName = "GA_${gaIters}_" + $scName + "_" + $cc.name
        $RunDir = Join-Path $OutDir $caseName

        Write-Host "`n============================="
        Write-Host "Baseline GA: $caseName"
        Write-Host "============================="

        Invoke-MatlabRun -RunDir $RunDir -Alg "GA" -MaxIters $gaIters -UseScreening $sc `
            -UseJ1 $cc.J1 -UseJ2 $cc.J2 -UseJ3 $cc.J3
    }
}

Write-Host "`nAll runs complete -> $OutDir"
# -------------------------------------------------