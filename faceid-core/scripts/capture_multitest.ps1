param(
    [int]$WorkerCount = 4,
    [int]$Semaphore = 2,
    [int]$BatchSize = 8,
    [double]$BatchCollectTimeout = 0.05,
    [int]$WarmupSeconds = 20,
    [int[]]$Rates = @(20, 30, 40, 50, 60),
    [string]$Duration = "3m",
    [string]$BaseUrl = "http://localhost:8080",
    [string]$ImagePath = "tests/data/person1_small.b64.txt",
    [string]$OutputDir = "benchmarks",
    [string]$Label = "",
    [ValidateSet("full", "enqueue-only")]
    [string]$Scenario = "full",
    [switch]$SkipComposeUp
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $root

$imageFullPath = Join-Path $root $ImagePath
if (-not (Test-Path $imageFullPath)) {
    throw "Image fixture not found: $imageFullPath"
}

if ([string]::IsNullOrWhiteSpace($Label)) {
    $Label = "multitest_$(Get-Date -Format yyyyMMdd_HHmmss)"
}

$runDir = Join-Path $root (Join-Path $OutputDir $Label)
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$transcriptPath = Join-Path $runDir "run.transcript.log"
Start-Transcript -Path $transcriptPath -Force | Out-Null

function Get-VuPlan([int]$Rate) {
    switch ($Rate) {
        30 { return @{ preAllocated = 400; max = 800 } }
        40 { return @{ preAllocated = 1200; max = 2000 } }
        50 { return @{ preAllocated = 1600; max = 2400 } }
        60 { return @{ preAllocated = 2000; max = 3000 } }
        default {
            $preAllocated = [Math]::Max(50, [int][Math]::Ceiling($Rate * 2.0))
            $max = [Math]::Max($preAllocated + 50, [int][Math]::Ceiling($preAllocated * 1.5))
            return @{ preAllocated = $preAllocated; max = $max }
        }
    }
}

try {
    $env:WORKER_SEMAPHORE = "$Semaphore"
    $env:BATCH_SIZE = "$BatchSize"
    $env:BATCH_COLLECT_TIMEOUT = "$BatchCollectTimeout"
    $env:BASE_URL = $BaseUrl
    $env:IMAGE_FILE = $imageFullPath

    if (-not $SkipComposeUp) {
        $composeLogPath = Join-Path $runDir "docker-compose.up.log"
        docker compose -p faceid-core -f (Join-Path $root "docker-compose.yml") `
            up -d --build --scale worker=$WorkerCount postgres redis minio api api_lb worker 2>&1 | Tee-Object -FilePath $composeLogPath | Out-Host

        $composePsPath = Join-Path $runDir "docker-compose.ps.before.log"
        docker compose -p faceid-core -f (Join-Path $root "docker-compose.yml") `
            ps -a 2>&1 | Tee-Object -FilePath $composePsPath | Out-Host

        if ($WarmupSeconds -gt 0) {
            Start-Sleep -Seconds $WarmupSeconds
        }
    } else {
        Write-Host "Skipping compose up because SkipComposeUp was set."
    }

    foreach ($rate in $Rates) {
        $vuPlan = Get-VuPlan -Rate $rate
        $scenarioName = "r${rate}_w${WorkerCount}_s${Semaphore}_b${BatchSize}_$Scenario"
        $summaryPath = Join-Path $runDir "$scenarioName.summary.json"
        $logPath = Join-Path $runDir "$scenarioName.k6.log"

        $env:RATE = "$rate"
        $env:DURATION = $Duration
        $env:PRE_ALLOCATED_VUS = "$($vuPlan.preAllocated)"
        $env:PREALLOCATED_VUS = "$($vuPlan.preAllocated)"
        $env:MAX_VUS = "$($vuPlan.max)"

        $k6Script = if ($Scenario -eq "enqueue-only") {
            Join-Path $root "load_test_verify_enqueue_only.js"
        } else {
            Join-Path $root "load_test_async.js"
        }

        Write-Host "Running $scenarioName against $BaseUrl"
        Write-Host "VU plan: preAllocated=$($vuPlan.preAllocated) max=$($vuPlan.max)"

        & k6 run `
            --summary-trend-stats "min,avg,med,p(90),p(95),p(99),max" `
            --summary-export $summaryPath `
            $k6Script 2>&1 | Tee-Object -FilePath $logPath | Out-Host

        Write-Host "saved summary -> $summaryPath"
        Write-Host "saved log -> $logPath"
    }

    if (-not $SkipComposeUp) {
        $composePsAfterPath = Join-Path $runDir "docker-compose.ps.after.log"
        docker compose -p faceid-core -f (Join-Path $root "docker-compose.yml") `
            ps -a 2>&1 | Tee-Object -FilePath $composePsAfterPath | Out-Host
    }
}
finally {
    try {
        Stop-Transcript | Out-Null
    } catch {
    }
}

Write-Host "benchmark run directory -> $runDir"
