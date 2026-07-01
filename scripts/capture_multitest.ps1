param(
    [int]$WorkerCount = 4,
    [int]$Semaphore = 2,
    [int]$BatchSize = 8,
    [double]$BatchCollectTimeout = 0.05,
    [int]$WarmupSeconds = 20,
    [string]$Rates = "20,30,40,50,60",
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
Set-StrictMode -Version Latest

$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $root

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
        } else {
            $arg
        }
    }

    return ($escaped -join ' ')
}

function Invoke-NativeCommandSafely {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$LogPath,

        [switch]$EchoOutput
    )

    $resolvedBinary = (Get-Command $FilePath -ErrorAction Stop).Source

    $stdoutPath = "$LogPath.stdout.tmp"
    $stderrPath = "$LogPath.stderr.tmp"

    foreach ($path in @($stdoutPath, $stderrPath, $LogPath)) {
        if (Test-Path $path) {
            Remove-Item $path -Force -ErrorAction SilentlyContinue
        }
    }

    $argumentLine = Convert-ArgumentListToString -Arguments $Arguments

    $process = Start-Process `
        -FilePath $resolvedBinary `
        -ArgumentList $argumentLine `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath

    $stdout = if (Test-Path $stdoutPath) { Get-Content $stdoutPath -Raw } else { "" }
    $stderr = if (Test-Path $stderrPath) { Get-Content $stderrPath -Raw } else { "" }

    $merged = @()
    if (-not [string]::IsNullOrWhiteSpace($stdout)) {
        $merged += $stdout.TrimEnd()
    }
    if (-not [string]::IsNullOrWhiteSpace($stderr)) {
        $merged += $stderr.TrimEnd()
    }

    if ($merged.Count -gt 0) {
        ($merged -join [Environment]::NewLine) | Out-File -FilePath $LogPath -Encoding utf8
    } else {
        "" | Out-File -FilePath $LogPath -Encoding utf8
    }

    if ($EchoOutput) {
        if (-not [string]::IsNullOrWhiteSpace($stdout)) {
            Write-Host $stdout.TrimEnd()
        }
        if (-not [string]::IsNullOrWhiteSpace($stderr)) {
            Write-Warning $stderr.TrimEnd()
        }
    }

    Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue
    Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue

    return [pscustomobject]@{
        ExitCode = $process.ExitCode
        LogPath = $LogPath
    }
}

function Get-VuPlan {
    param([int]$Rate)

    switch ($Rate) {
        30 { return @{ preAllocated = 400;  max = 800  } }
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

function Get-NormalizedRates {
    param([object]$RatesValue)

    $flatRates = New-Object System.Collections.Generic.List[int]
    foreach ($item in @($RatesValue)) {
        $text = [string]$item
        foreach ($part in ($text -split "[,\s]+")) {
            $trimmed = $part.Trim()
            if ([string]::IsNullOrWhiteSpace($trimmed)) {
                continue
            }

            $null = $flatRates.Add([int]$trimmed)
        }
    }

    if ($flatRates.Count -eq 0) {
        throw "No rates provided"
    }

    return ,$flatRates.ToArray()
}

if (-not (Get-Command k6 -ErrorAction SilentlyContinue)) {
    throw "k6 is not installed or not found in PATH"
}

$imageFullPath = Join-Path $root $ImagePath
if (-not (Test-Path $imageFullPath)) {
    throw "Image fixture not found: $imageFullPath"
}

$Rates = Get-NormalizedRates -RatesValue $Rates

if ([string]::IsNullOrWhiteSpace($Label)) {
    $Label = "multitest_$(Get-Date -Format yyyyMMdd_HHmmss)"
}

$runDir = Join-Path $root (Join-Path $OutputDir $Label)
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$transcriptPath = Join-Path $runDir "run.transcript.log"
Start-Transcript -Path $transcriptPath -Force | Out-Null

try {
    $env:WORKER_SEMAPHORE = "$Semaphore"
    $env:BATCH_SIZE = "$BatchSize"
    $env:BATCH_COLLECT_TIMEOUT = "$BatchCollectTimeout"
    $env:BASE_URL = $BaseUrl
    $env:IMAGE_FILE = $imageFullPath

    if (-not $SkipComposeUp) {
        $composeFile = Join-Path $root "docker-compose.yml"

        $composeUpLogPath = Join-Path $runDir "docker-compose.up.log"
        $composeUpArgs = @(
            "compose"
            "-p"
            "faceid-core"
            "-f"
            $composeFile
            "up"
            "-d"
            "--build"
            "--scale"
            "worker=$WorkerCount"
            "postgres"
            "redis"
            "minio"
            "api"
            "api_lb"
            "worker"
        )

        $composeUpResult = Invoke-NativeCommandSafely `
            -FilePath "docker" `
            -Arguments $composeUpArgs `
            -LogPath $composeUpLogPath `
            -EchoOutput

        if ($composeUpResult.ExitCode -ne 0) {
            throw "docker compose up failed with exit code $($composeUpResult.ExitCode)"
        }

        $composePsPath = Join-Path $runDir "docker-compose.ps.before.log"
        $composePsArgs = @(
            "compose"
            "-p"
            "faceid-core"
            "-f"
            $composeFile
            "ps"
            "-a"
        )

        $composePsResult = Invoke-NativeCommandSafely `
            -FilePath "docker" `
            -Arguments $composePsArgs `
            -LogPath $composePsPath `
            -EchoOutput

        if ($composePsResult.ExitCode -ne 0) {
            Write-Warning "docker compose ps exited with code $($composePsResult.ExitCode)"
        }

        if ($WarmupSeconds -gt 0) {
            Start-Sleep -Seconds $WarmupSeconds
        }
    }
    else {
        Write-Host "Skipping compose up because SkipComposeUp was set."
    }

    $runStatuses = @()
    $rateValues = @(
        [string]$Rates -split "[,\s]+" |
        ForEach-Object { $_.Trim() } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
    if ($rateValues.Count -eq 0) {
        throw "No rates provided after normalization"
    }

    foreach ($rateValue in $rateValues) {
        $rate = [int]$rateValue
        $vuPlan = Get-VuPlan -Rate $rate
        $runName = "r${rate}_w${WorkerCount}_s${Semaphore}_b${BatchSize}_$Scenario"

        $summaryPath = Join-Path $runDir "$runName.summary.json"
        $k6LogPath = Join-Path $runDir "$runName.k6.log"
        $statusPath = Join-Path $runDir "$runName.status.json"

        $k6Script = if ($Scenario -eq "enqueue-only") {
            Join-Path $root "load_test_verify_enqueue_only.js"
        } else {
            Join-Path $root "load_test_async.js"
        }

        Write-Host "Running $runName against $BaseUrl"
        Write-Host "VU plan: preAllocated=$($vuPlan.preAllocated) max=$($vuPlan.max)"

        $k6Args = @(
            "run"
            "--summary-trend-stats"
            "min,avg,med,p(90),p(95),p(99),max"
            "--summary-export"
            $summaryPath
            "-e"
            "BASE_URL=$BaseUrl"
            "-e"
            "RATE=$rate"
            "-e"
            "DURATION=$Duration"
            "-e"
            "PRE_ALLOCATED_VUS=$($vuPlan.preAllocated)"
            "-e"
            "MAX_VUS=$($vuPlan.max)"
            $k6Script
        )

        $k6Result = Invoke-NativeCommandSafely `
            -FilePath "k6" `
            -Arguments $k6Args `
            -LogPath $k6LogPath `
            -EchoOutput

        Write-Host "saved summary -> $summaryPath"
        Write-Host "saved log -> $k6LogPath"
        Write-Host "benchmark run directory -> $runDir"

        if (-not (Test-Path $k6LogPath)) {
            throw "k6 log was not created: $k6LogPath"
        }

        if (-not (Test-Path $summaryPath)) {
            Write-Warning "summary.json was not created for $runName"
        }

        if ($k6Result.ExitCode -ne 0) {
            Write-Warning "k6 exited with code $($k6Result.ExitCode) for $runName"
        }

        $status = [pscustomobject]@{
            run_name = $runName
            scenario = $Scenario
            rate = $rate
            worker_count = $WorkerCount
            semaphore = $Semaphore
            batch_size = $BatchSize
            batch_collect_timeout = $BatchCollectTimeout
            duration = $Duration
            base_url = $BaseUrl
            summary_path = $summaryPath
            log_path = $k6LogPath
            summary_exists = (Test-Path $summaryPath)
            log_exists = (Test-Path $k6LogPath)
            exit_code = $k6Result.ExitCode
            finished_at = (Get-Date).ToString("s")
        }

        $status | ConvertTo-Json -Depth 5 | Out-File -FilePath $statusPath -Encoding utf8
        $runStatuses += $status
    }

    if (-not $SkipComposeUp) {
        $composeFile = Join-Path $root "docker-compose.yml"
        $composePsAfterPath = Join-Path $runDir "docker-compose.ps.after.log"
        $composePsAfterArgs = @(
            "compose"
            "-p"
            "faceid-core"
            "-f"
            $composeFile
            "ps"
            "-a"
        )

        $composePsAfterResult = Invoke-NativeCommandSafely `
            -FilePath "docker" `
            -Arguments $composePsAfterArgs `
            -LogPath $composePsAfterPath `
            -EchoOutput

        if ($composePsAfterResult.ExitCode -ne 0) {
            Write-Warning "docker compose ps after-run exited with code $($composePsAfterResult.ExitCode)"
        }
    }

    $seriesStatusPath = Join-Path $runDir "series.status.json"
    $runStatuses | ConvertTo-Json -Depth 5 | Out-File -FilePath $seriesStatusPath -Encoding utf8
}
finally {
    try {
        Stop-Transcript | Out-Null
    } catch {
    }
}

Write-Host "benchmark run directory -> $runDir"
