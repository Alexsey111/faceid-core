param(
    [int]$WorkerCount = 4,
    [int]$Semaphore = 2,
    [int]$BatchSize = 16,
    [double]$BatchCollectTimeout = 0.01,
    [int]$WarmupSeconds = 15,
    [int[]]$Rates = @(30, 50, 70),
    [int]$DurationSeconds = 180,
    [string]$BaseUrl = "http://localhost:8080",
    [string]$ImagePath = "tests/data_extended/person_011/1.jpg",
    [string]$OutputDir = "benchmarks",
    [string]$Label = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $root

$imageFullPath = Join-Path $root $ImagePath
if (-not (Test-Path $imageFullPath)) {
    throw "Image fixture not found: $imageFullPath"
}

if ([string]::IsNullOrWhiteSpace($Label)) {
    $Label = "w${WorkerCount}_s${Semaphore}_$(Get-Date -Format yyyyMMdd_HHmmss)"
}

$runDir = Join-Path $root (Join-Path $OutputDir $Label)
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$env:WORKER_SEMAPHORE = "$Semaphore"
$env:BATCH_SIZE = "$BatchSize"
$env:BATCH_COLLECT_TIMEOUT = "$BatchCollectTimeout"
$env:BASE_URL = $BaseUrl
$env:IMAGE_FILE = $imageFullPath

docker compose -p faceid-core -f (Join-Path $root "docker-compose.yml") `
    up -d --scale worker=$WorkerCount postgres redis minio api api_lb worker | Out-Host

if ($WarmupSeconds -gt 0) {
    Start-Sleep -Seconds $WarmupSeconds
}

foreach ($rate in $Rates) {
    $scenario = "worker${WorkerCount}_sem${Semaphore}_b${BatchSize}_r${rate}"
    $summaryPath = Join-Path $runDir "$scenario.summary.json"
    $logPath = Join-Path $runDir "$scenario.k6.log"

    $env:RATE = "$rate"
    $env:DURATION = "${DurationSeconds}s"
    $env:PREALLOCATED_VUS = 100
    $env:MAX_VUS = 500

    Write-Host "Running $scenario against $BaseUrl"

    $output = & k6 run `
        --summary-trend-stats "min,avg,med,p(90),p(95),p(99)" `
        --summary-export $summaryPath `
        (Join-Path $root "load_test_verify_async.js")

    $output | Out-File -FilePath $logPath -Encoding utf8
    Write-Host "saved summary -> $summaryPath"
    Write-Host "saved log -> $logPath"
}

Write-Host "benchmark run directory -> $runDir"
