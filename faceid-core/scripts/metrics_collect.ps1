param(
    [string]$ProjectName = "faceid-core",
    [string]$MetricsUrl = "http://localhost:8080/metrics",
    [string]$BenchmarkDir = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-LatestBenchmarkDir {
    $dir = Get-ChildItem -Path ".\benchmarks" -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    return $dir
}

function Get-ContainerNamesByService([string]$serviceName) {
    $names = docker ps `
        --filter "label=com.docker.compose.project=$ProjectName" `
        --filter "label=com.docker.compose.service=$serviceName" `
        --format "{{.Names}}"

    return @($names | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function Get-WorkerJobCount([string]$containerName) {
    try {
        $logs = docker logs $containerName 2>&1
        if (-not $logs) {
            return 0
        }

        return @(
            $logs | Select-String -Pattern '\[METRICS\] job='
        ).Count
    }
    catch {
        return 0
    }
}

function Get-MetricLines([string]$metricsText, [string]$metricName) {
    return @(
        $metricsText -split "`n" |
        Where-Object {
            $_ -match ("^" + [regex]::Escape($metricName) + "(\{|\s)")
        }
    )
}

function Get-MetricNumericSum([string]$metricsText, [string]$metricName) {
    $lines = Get-MetricLines -metricsText $metricsText -metricName $metricName
    $sum = 0.0

    foreach ($line in $lines) {
        if ($line -match '([0-9]+(?:\.[0-9]+)?)\s*$') {
            $sum += [double]$matches[1]
        }
    }

    return $sum
}

function Get-MetricNumericMax([string]$metricsText, [string]$metricName) {
    $lines = Get-MetricLines -metricsText $metricsText -metricName $metricName
    $values = @()

    foreach ($line in $lines) {
        if ($line -match '([0-9]+(?:\.[0-9]+)?)\s*$') {
            $values += [double]$matches[1]
        }
    }

    if ($values.Count -eq 0) {
        return $null
    }

    return ($values | Measure-Object -Maximum).Maximum
}

function Add-SnapshotLine {
    param(
        [System.Collections.Generic.List[string]]$SnapshotLines,
        [string]$Line
    )

    $SnapshotLines.Add($Line)
    Write-Host $Line
}

function Add-NamedSnapshotValue {
    param(
        [System.Collections.Generic.List[string]]$SnapshotLines,
        [string]$Name,
        $Value
    )

    if ($null -ne $Value) {
        $line = "$Name=$Value"
        Add-SnapshotLine -SnapshotLines $SnapshotLines -Line $line
    }
}

function Get-JsonFile([string]$Path) {
    try {
        if (-not (Test-Path $Path)) {
            return $null
        }

        $raw = Get-Content $Path -Raw -ErrorAction Stop
        if ([string]::IsNullOrWhiteSpace($raw)) {
            return $null
        }

        return $raw | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        return $null
    }
}

if ([string]::IsNullOrWhiteSpace($BenchmarkDir)) {
    $latestDir = Get-LatestBenchmarkDir
    if ($latestDir) {
        $BenchmarkDir = $latestDir.FullName
    }
}

Write-Host "=== POST-RUN SNAPSHOT $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" -ForegroundColor Cyan
if (-not [string]::IsNullOrWhiteSpace($BenchmarkDir)) {
    Write-Host "benchmark_dir: $BenchmarkDir" -ForegroundColor DarkCyan
}
Write-Host ""

$snapshotLines = New-Object System.Collections.Generic.List[string]
$snapshotLines.Add("=== POST-RUN SNAPSHOT $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===")
if (-not [string]::IsNullOrWhiteSpace($BenchmarkDir)) {
    $snapshotLines.Add("benchmark_dir: $BenchmarkDir")
}

# Host CPU
Write-Host "--- Host CPU ---" -ForegroundColor Yellow
try {
    $cpu = Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1
    $cpuValue = [math]::Round($cpu.CounterSamples.CookedValue, 2)
    Add-NamedSnapshotValue -SnapshotLines $snapshotLines -Name "host_cpu_percent" -Value $cpuValue
}
catch {
    Add-SnapshotLine -SnapshotLines $snapshotLines -Line "host_cpu_percent=unavailable"
}

# Docker CPU / memory
Write-Host ""
Write-Host "--- Docker Stats ---" -ForegroundColor Yellow
try {
    $stats = docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
    $stats | Write-Host
    $snapshotLines.Add("docker_stats:")
    $snapshotLines.AddRange(@($stats))
}
catch {
    Write-Host "docker stats unavailable"
    $snapshotLines.Add("docker_stats=unavailable")
}

# Worker processed task counts from logs (hint only)
Write-Host ""
Write-Host "--- Worker Task Counts (log hint only) ---" -ForegroundColor Yellow
$workerServices = @("worker", "worker_fast", "worker_heavy", "worker_metrics")
foreach ($service in $workerServices) {
    $containers = Get-ContainerNamesByService $service
    if (-not $containers -or $containers.Count -eq 0) {
        $line = "${service}: no containers"
        Add-SnapshotLine -SnapshotLines $snapshotLines -Line $line
        continue
    }

    foreach ($container in $containers) {
        $count = Get-WorkerJobCount $container
        $line = "$container log_processed_jobs_hint=$count"
        Add-SnapshotLine -SnapshotLines $snapshotLines -Line $line
    }
}

# Metrics endpoint snapshot
Write-Host ""
Write-Host "--- Prometheus Metrics ---" -ForegroundColor Yellow
try {
    $metrics = (Invoke-WebRequest -Uri $MetricsUrl -UseBasicParsing -TimeoutSec 10).Content

    # Aggregated values
    $queuePending = Get-MetricNumericMax -metricsText $metrics -metricName "faceid_queue_jobs_pending"
    $inflight = Get-MetricNumericMax -metricsText $metrics -metricName "faceid_verify_inflight_current"
    $httpInflight = Get-MetricNumericMax -metricsText $metrics -metricName "faceid_verify_async_http_inflight"
    $workerUtil = Get-MetricNumericMax -metricsText $metrics -metricName "faceid_verify_worker_utilization"

    $completedTotal = Get-MetricNumericSum -metricsText $metrics -metricName "faceid_async_job_completed_total"
    $expiredTotal = Get-MetricNumericSum -metricsText $metrics -metricName "faceid_async_job_expired_total"
    $acceptedAsync = Get-MetricNumericSum -metricsText $metrics -metricName "faceid_verify_async_accepted_total"
    $acceptedJobs = Get-MetricNumericSum -metricsText $metrics -metricName "faceid_verify_accepted_jobs_total"

    Add-NamedSnapshotValue -SnapshotLines $snapshotLines -Name "queue_pending" -Value $queuePending
    Add-NamedSnapshotValue -SnapshotLines $snapshotLines -Name "inflight" -Value $inflight
    Add-NamedSnapshotValue -SnapshotLines $snapshotLines -Name "http_inflight" -Value $httpInflight
    Add-NamedSnapshotValue -SnapshotLines $snapshotLines -Name "worker_utilization" -Value $workerUtil
    Add-NamedSnapshotValue -SnapshotLines $snapshotLines -Name "async_completed_total" -Value $completedTotal
    Add-NamedSnapshotValue -SnapshotLines $snapshotLines -Name "async_expired_total" -Value $expiredTotal
    Add-NamedSnapshotValue -SnapshotLines $snapshotLines -Name "verify_async_accepted_total" -Value $acceptedAsync
    Add-NamedSnapshotValue -SnapshotLines $snapshotLines -Name "verify_accepted_jobs_total" -Value $acceptedJobs

    # Raw lines for debugging per-pid / per-label
    $rawMetricNames = @(
        "faceid_queue_jobs_pending",
        "faceid_verify_inflight_current",
        "faceid_verify_async_http_inflight",
        "faceid_verify_worker_utilization",
        "faceid_async_job_completed_total",
        "faceid_async_job_expired_total",
        "faceid_verify_rejected_jobs_total",
        "faceid_verify_async_accepted_total",
        "faceid_verify_accepted_jobs_total",
        "faceid_verify_async_status_total"
    )

    foreach ($metricName in $rawMetricNames) {
        $lines = Get-MetricLines -metricsText $metrics -metricName $metricName
        foreach ($line in $lines) {
            $trimmed = $line.Trim()
            if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
                Add-SnapshotLine -SnapshotLines $snapshotLines -Line $trimmed
            }
        }
    }
}
catch {
    Add-SnapshotLine -SnapshotLines $snapshotLines -Line "metrics_endpoint=unavailable"
}

# Timeout detection from latest benchmark logs
Write-Host ""
Write-Host "--- Timeout Check ---" -ForegroundColor Yellow
$timeoutHits = @()
if (-not [string]::IsNullOrWhiteSpace($BenchmarkDir) -and (Test-Path $BenchmarkDir)) {
    $k6Logs = Get-ChildItem -Path $BenchmarkDir -Filter "*.k6.log" -File -ErrorAction SilentlyContinue
    foreach ($log in $k6Logs) {
        $hits = Select-String -Path $log.FullName -Pattern "request timeout|wait_timeouts|thresholds have been crossed|Insufficient VUs|Request Failed" -ErrorAction SilentlyContinue
        if ($hits) {
            $timeoutHits += $hits
        }
    }
}

if ($timeoutHits.Count -gt 0) {
    Add-SnapshotLine -SnapshotLines $snapshotLines -Line "timeout_detected=YES"
    $timeoutHits | Select-Object -First 20 | ForEach-Object {
        Add-SnapshotLine -SnapshotLines $snapshotLines -Line $_.Line
    }
}
else {
    Add-SnapshotLine -SnapshotLines $snapshotLines -Line "timeout_detected=NO"
}

# K6 final progress snapshot
Write-Host ""
Write-Host "--- K6 Final Progress Snapshot ---" -ForegroundColor Yellow
if (-not [string]::IsNullOrWhiteSpace($BenchmarkDir) -and (Test-Path $BenchmarkDir)) {
    $k6Logs = Get-ChildItem -Path $BenchmarkDir -Filter "*.k6.log" -File -ErrorAction SilentlyContinue

    foreach ($log in $k6Logs) {
        Add-SnapshotLine -SnapshotLines $snapshotLines -Line "[$($log.Name)]"
        $progressLines = Get-Content $log.FullName | Where-Object {
            $_ -match 'running \(' -or
            $_ -match 'TOTAL RESULTS' -or
            $_ -match 'complete and .* interrupted iterations' -or
            $_ -match 'saved summary' -or
            $_ -match 'saved log'
        }

        if ($progressLines.Count -gt 0) {
            $lastLines = $progressLines | Select-Object -Last 5
            foreach ($line in $lastLines) {
                Add-SnapshotLine -SnapshotLines $snapshotLines -Line $line
            }
        }
        else {
            Add-SnapshotLine -SnapshotLines $snapshotLines -Line "no_progress_lines_found"
        }
    }
}

# Benchmark artifacts
Write-Host ""
Write-Host "--- Benchmark Artifacts ---" -ForegroundColor Yellow
if (-not [string]::IsNullOrWhiteSpace($BenchmarkDir) -and (Test-Path $BenchmarkDir)) {
    $summaryFiles = Get-ChildItem -Path $BenchmarkDir -Filter "*.summary.json" -File -ErrorAction SilentlyContinue
    $statusFiles = Get-ChildItem -Path $BenchmarkDir -Filter "*.status.json" -File -ErrorAction SilentlyContinue

    if (($summaryFiles | Measure-Object).Count -eq 0) {
        Add-SnapshotLine -SnapshotLines $snapshotLines -Line "summary_file=missing"
    }
    else {
        foreach ($file in $summaryFiles) {
            Add-SnapshotLine -SnapshotLines $snapshotLines -Line "summary_file=$($file.Name)"

            $json = Get-JsonFile -Path $file.FullName
            if ($null -ne $json) {
                if ($null -ne $json.metrics.http_req_failed) {
                    Add-SnapshotLine -SnapshotLines $snapshotLines -Line "$($file.Name):http_req_failed_rate=$($json.metrics.http_req_failed.rate)"
                }
                if ($null -ne $json.metrics.http_req_duration) {
                    Add-SnapshotLine -SnapshotLines $snapshotLines -Line "$($file.Name):http_req_duration_avg=$($json.metrics.http_req_duration.avg)"
                    Add-SnapshotLine -SnapshotLines $snapshotLines -Line "$($file.Name):http_req_duration_p95=$($json.metrics.http_req_duration.'p(95)')"
                }
                if ($null -ne $json.metrics.iterations) {
                    Add-SnapshotLine -SnapshotLines $snapshotLines -Line "$($file.Name):iterations_count=$($json.metrics.iterations.count)"
                    Add-SnapshotLine -SnapshotLines $snapshotLines -Line "$($file.Name):iterations_rate=$($json.metrics.iterations.rate)"
                }
                if ($null -ne $json.metrics.enqueue_accepted) {
                    Add-SnapshotLine -SnapshotLines $snapshotLines -Line "$($file.Name):enqueue_accepted_count=$($json.metrics.enqueue_accepted.count)"
                    Add-SnapshotLine -SnapshotLines $snapshotLines -Line "$($file.Name):enqueue_accepted_rate=$($json.metrics.enqueue_accepted.rate)"
                }
                if ($null -ne $json.metrics.completed) {
                    Add-SnapshotLine -SnapshotLines $snapshotLines -Line "$($file.Name):completed_count=$($json.metrics.completed.count)"
                    Add-SnapshotLine -SnapshotLines $snapshotLines -Line "$($file.Name):completed_rate=$($json.metrics.completed.rate)"
                }
            }
        }
    }

    foreach ($file in $statusFiles) {
        Add-SnapshotLine -SnapshotLines $snapshotLines -Line "status_file=$($file.Name)"

        $json = Get-JsonFile -Path $file.FullName
        if ($null -ne $json) {
            Add-SnapshotLine -SnapshotLines $snapshotLines -Line "$($file.Name):exit_code=$($json.exit_code)"
            Add-SnapshotLine -SnapshotLines $snapshotLines -Line "$($file.Name):summary_exists=$($json.summary_exists)"
            Add-SnapshotLine -SnapshotLines $snapshotLines -Line "$($file.Name):log_exists=$($json.log_exists)"
        }
    }
}

# API error tail
Write-Host ""
Write-Host "--- API Error Tail ---" -ForegroundColor Yellow
try {
    $apiContainers = Get-ContainerNamesByService "api"
    foreach ($container in $apiContainers) {
        $tail = docker logs $container --tail 30 2>&1 | Select-String -Pattern "error|timeout|Error|TIMEOUT|500|502|503"
        if ($tail) {
            Add-SnapshotLine -SnapshotLines $snapshotLines -Line "[$container]"
            $tail | ForEach-Object {
                Add-SnapshotLine -SnapshotLines $snapshotLines -Line $_.Line
            }
        }
    }
}
catch {
    Add-SnapshotLine -SnapshotLines $snapshotLines -Line "api_logs=unavailable"
}

if (-not [string]::IsNullOrWhiteSpace($BenchmarkDir) -and (Test-Path $BenchmarkDir)) {
    $snapshotPath = Join-Path $BenchmarkDir "postrun_snapshot.txt"
    $snapshotLines | Set-Content -Path $snapshotPath
    Write-Host ""
    Write-Host "snapshot_saved_to=$snapshotPath" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== END SNAPSHOT ===" -ForegroundColor Cyan
