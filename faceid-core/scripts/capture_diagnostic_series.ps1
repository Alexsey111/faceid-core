[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [string]$ProjectName = "faceid-core",
    [string]$HarnessPath = "",
    [string]$WorkerService = "worker",
    [string]$OutDir = "benchmarks\diagnostic_round_01",
    [string]$BaseUrl = "http://localhost:8000",
    [int]$Semaphore = 2,
    [int]$BatchSize = 8,
    [double]$BatchCollectTimeout = 0.008,
    [string]$Duration = "2m",
    [int]$WarmupSeconds = 10,
    [int]$LogTailLines = 200
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$DefaultProjectRoot = Resolve-Path -LiteralPath (Join-Path $ScriptRoot "..\..") | Select-Object -ExpandProperty Path

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = $DefaultProjectRoot
}
elseif (-not [System.IO.Path]::IsPathRooted($ProjectRoot)) {
    $candidateRoot = Join-Path (Get-Location).Path $ProjectRoot
    $ProjectRoot = (Resolve-Path -LiteralPath $candidateRoot -ErrorAction Stop).Path
}
else {
    $ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot -ErrorAction Stop).Path
}

if ([string]::IsNullOrWhiteSpace($HarnessPath)) {
    $HarnessPath = Join-Path $ScriptRoot "capture_multitest.ps1"
}
elseif (-not [System.IO.Path]::IsPathRooted($HarnessPath)) {
    $candidateHarness = Join-Path (Get-Location).Path $HarnessPath
    if (Test-Path -LiteralPath $candidateHarness) {
        $HarnessPath = (Resolve-Path -LiteralPath $candidateHarness -ErrorAction Stop).Path
    }
    else {
        $candidateHarness = Join-Path $ScriptRoot $HarnessPath
        $HarnessPath = (Resolve-Path -LiteralPath $candidateHarness -ErrorAction Stop).Path
    }
}
else {
    $HarnessPath = (Resolve-Path -LiteralPath $HarnessPath -ErrorAction Stop).Path
}

if (-not (Test-Path -LiteralPath $ProjectRoot)) {
    throw "Project root not found: $ProjectRoot"
}

if (-not (Test-Path -LiteralPath $HarnessPath)) {
    throw "Harness script not found: $HarnessPath"
}

Set-Location $ProjectRoot

function Write-Section {
    param([string]$Title)
    Write-Host "`n========== $Title ==========" -ForegroundColor Cyan
}

function Initialize-Directory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Get-Timestamp {
    return (Get-Date).ToString("yyyyMMdd_HHmmss")
}

function Convert-ArgumentListToString {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $escaped = foreach ($arg in $Arguments) {
        if ($null -eq $arg) {
            '""'
            continue
        }

        $needsQuotes = $arg -match '\s' -or $arg -match '"'
        if ($needsQuotes) {
            '"' + ($arg -replace '"', '\"') + '"'
        }
        else {
            $arg
        }
    }

    return ($escaped -join ' ')
}

function Start-Collector {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$LogPath
    )

    $stdoutPath = "$LogPath.stdout.tmp"
    $stderrPath = "$LogPath.stderr.tmp"

    foreach ($path in @($stdoutPath, $stderrPath, $LogPath)) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
        }
    }

    $argumentLine = Convert-ArgumentListToString -Arguments $Arguments
    $process = Start-Process `
        -FilePath (Get-Command "docker" -ErrorAction Stop).Source `
        -ArgumentList $argumentLine `
        -NoNewWindow `
        -PassThru `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath

    return [pscustomobject]@{
        Name = $Name
        Process = $process
        LogPath = $LogPath
        StdoutPath = $stdoutPath
        StderrPath = $stderrPath
    }
}

function Stop-Collector {
    param([object]$Collector)

    if ($null -eq $Collector) {
        return
    }

    try {
        if ($Collector.Process -and -not $Collector.Process.HasExited) {
            Stop-Process -Id $Collector.Process.Id -Force -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds 500
        }
    }
    catch {
    }

    foreach ($path in @($Collector.StdoutPath, $Collector.StderrPath)) {
        if (Test-Path -LiteralPath $path) {
            $content = Get-Content -Raw -LiteralPath $path
            if (-not [string]::IsNullOrWhiteSpace($content)) {
                $content | Out-File -FilePath $Collector.LogPath -Encoding utf8
            }
            Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
        }
    }
}

function Invoke-DockerComposeUpScale {
    param([int]$ReplicaCount)

    $composeArgs = @(
        "compose",
        "-p",
        $ProjectName,
        "up",
        "-d",
        "--build",
        "--scale",
        ("{0}={1}" -f $WorkerService, $ReplicaCount),
        "postgres",
        "redis",
        "minio",
        "api",
        $WorkerService
    )

    Write-Host ("docker {0}" -f ($composeArgs -join " ")) -ForegroundColor DarkGray
    & docker @composeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose up failed for worker=$ReplicaCount (exit_code=$LASTEXITCODE)"
    }
}

function Invoke-Multitest {
    param(
        [int]$WorkerCount,
        [int[]]$Rates,
        [string]$Label,
        [string]$LogPath
    )

    $ratesArg = ($Rates -join ",")

    $harnessArgs = @(
        "-SkipComposeUp",
        "-Scenario",
        "full",
        "-WorkerCount",
        $WorkerCount,
        "-Semaphore",
        $Semaphore,
        "-BatchSize",
        $BatchSize,
        "-BatchCollectTimeout",
        $BatchCollectTimeout,
        "-WarmupSeconds",
        0,
        "-BaseUrl",
        $BaseUrl,
        "-OutputDir",
        $OutDir,
        "-Label",
        $Label,
        "-Duration",
        $Duration,
        "-Rates"
        $ratesArg
    )

    $powershellExe = (Get-Command powershell.exe -ErrorAction Stop).Source
    $argumentLine = Convert-ArgumentListToString -Arguments (@(
        "-ExecutionPolicy", "Bypass",
        "-File", $HarnessPath
    ) + $harnessArgs)

    Write-Host ("{0} {1}" -f $powershellExe, $argumentLine) -ForegroundColor DarkGray

    $stdoutPath = "$LogPath.stdout.tmp"
    $stderrPath = "$LogPath.stderr.tmp"
    foreach ($path in @($stdoutPath, $stderrPath, $LogPath)) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
        }
    }

    $process = Start-Process `
        -FilePath $powershellExe `
        -ArgumentList $argumentLine `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath

    $stdout = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -Raw -LiteralPath $stdoutPath } else { "" }
    $stderr = if (Test-Path -LiteralPath $stderrPath) { Get-Content -Raw -LiteralPath $stderrPath } else { "" }
    $merged = @()
    if (-not [string]::IsNullOrWhiteSpace($stdout)) { $merged += $stdout.TrimEnd() }
    if (-not [string]::IsNullOrWhiteSpace($stderr)) { $merged += $stderr.TrimEnd() }
    if ($merged.Count -gt 0) {
        ($merged -join [Environment]::NewLine) | Out-File -FilePath $LogPath -Encoding utf8
    }
    else {
        "" | Out-File -FilePath $LogPath -Encoding utf8
    }

    Remove-Item -LiteralPath $stdoutPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue

    if ($process.ExitCode -ne 0) {
        Write-Warning "capture_multitest exited with code $($process.ExitCode) for $Label"
    }
}

if (-not (Test-Path -LiteralPath $ProjectRoot)) {
    throw "Project root not found: $ProjectRoot"
}

$runRoot = Join-Path $ProjectRoot $OutDir
Initialize-Directory -Path $runRoot
$seriesDir = Join-Path $runRoot ("{0}_{1}" -f (Get-Timestamp), "diagnostic_series")
Initialize-Directory -Path $seriesDir

Write-Section "Collectors"
$statsLog = Join-Path $seriesDir "docker_stats.log"
$logsLog = Join-Path $seriesDir "docker_worker_api.log"

$statsCollector = Start-Collector -Name "docker_stats" -Arguments @("stats", "--format", "table {{.Name}}`t{{.CPUPerc}}`t{{.MemUsage}}`t{{.NetIO}}`t{{.BlockIO}}") -LogPath $statsLog
$logsCollector = Start-Collector -Name "docker_logs" -Arguments @("compose", "-p", $ProjectName, "logs", "-f", "--tail=0", "api", $WorkerService) -LogPath $logsLog

try {
    Write-Section "Scale to 1 worker"
    Invoke-DockerComposeUpScale -ReplicaCount 1
    Start-Sleep -Seconds $WarmupSeconds
    Invoke-Multitest -WorkerCount 1 -Rates @(10, 12, 15) -Label "diag_w1" -LogPath (Join-Path $seriesDir "diag_w1.multitest.log")

    Write-Section "Scale to 4 workers"
    Invoke-DockerComposeUpScale -ReplicaCount 4
    Start-Sleep -Seconds $WarmupSeconds
    Invoke-Multitest -WorkerCount 4 -Rates @(15, 20) -Label "diag_w4" -LogPath (Join-Path $seriesDir "diag_w4.multitest.log")
}
finally {
    Stop-Collector -Collector $logsCollector
    Stop-Collector -Collector $statsCollector
}

Write-Section "Done"
Write-Host ("Artifacts saved to: {0}" -f (Resolve-Path -LiteralPath $seriesDir)) -ForegroundColor Green
