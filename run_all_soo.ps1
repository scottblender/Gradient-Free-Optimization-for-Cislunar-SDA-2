# ---------------- run_all_soo.ps1 ----------------
$Algs = @("GA", "PSO", "BAYESIAN", "ABC", "ACO")

# Folder where this PS script lives (project root)
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

# Absolute path to your MATLAB script (your file name)
$RunOpt = Join-Path $ProjectRoot "run_opt.m"

if (-not (Test-Path $RunOpt)) {
    throw "Cannot find run_opt.m at: $RunOpt"
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutDir = Join-Path $ProjectRoot ("runs_" + $Timestamp)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# MATLAB R2025b path (edit if your install differs)
$MatlabExe = "C:\Program Files\MATLAB\R2025b\bin\matlab.exe"

if (-not (Test-Path $MatlabExe)) {
    throw "Cannot find matlab.exe at: $MatlabExe"
}

foreach ($alg in $Algs) {

    Write-Host "`n============================="
    Write-Host "Running optimizer: $alg"
    Write-Host "============================="

    $RunDir = Join-Path $OutDir $alg
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

    # Pass optimizer choice into MATLAB
    $env:OPTIMIZER_MODE = $alg

    Push-Location $RunDir

    # Add project root to path, run the script, log everything
    $cmd = "try, addpath(genpath('$ProjectRoot')); run('$RunOpt'); catch ME, disp(getReport(ME,'extended')); exit(1); end; exit(0);"

    & "$MatlabExe" -batch $cmd *> "console.log"

    Pop-Location

    Write-Host "Saved log -> $RunDir\console.log"
}

Write-Host "`nAll runs complete -> $OutDir"
# -------------------------------------------------
