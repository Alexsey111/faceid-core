[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [string]$ProjectName = "faceid-core",
    [string]$HarnessPath = "",
    [string]$WorkerService = "worker",
    [string]$OutDir = "benchmarks\scaling_round_01",

    [string]$BaseUrl = "http://localhost:8000",

    [int]$BaselineReplicaCount = 1,
    [int]$BaselineEnqueueRate = 40,
    [int]$BaselineCompletionRate15 = 15,
    [int]$BaselineCompletionRate20 = 20,
    [int]$ScaleCompletionRate = 20,
    [int]$PostSuccessCompletionRate22 = 22,
    [int]$PostSuccessCompletionRate25 = 25,
    [int]$PostSuccessEnqueueRate42 = 42,
    [int]$PostSuccessEnqueueRate45 = 45,
    [int[]]$ScaleReplicaSteps = @(4),

    [int]$Semaphore = 2,
    [int]$BatchSize = 8,
    [double]$BatchCollectTimeout = 0.008,
    [string]$Duration = "2m",
    [int]$WarmupSeconds = 20,
    [int]$LogTailLines = 200,

    [switch]$SkipBaseline,
    [switch]$SkipScale,
    [switch]$SkipPostSuccess,
    [switch]$NoWarmupPause,
    [switch]$CheckKeyEnvOnly,

    [string[]]$KeyEnvNames = @(
        "ENV",
        "APP_ROLE",
        "MODELS_DIR",
        "DATABASE_URL",
        "REDIS_URL",
        "REDIS_HOST",
        "PROMETHEUS_MULTIPROC_DIR",
        "CELERY_BROKER_URL",
        "CELERY_RESULT_BACKEND",
        "WORKER_COUNT",
        "WORKER_SEMAPHORE",
        "ASYNC_THROUGHPUT_PER_SEC",
        "BACKPRESSURE_MAX_QUEUE_DELAY_MS",
        "EMBED_BATCH_ENABLED",
        "EMBED_BATCH_SIZE",
        "EMBED_BATCH_TIMEOUT_MS",
        "EMBED_BATCH_MAX_WAIT_GUARD_MS",
        "LIVENESS_ENABLED",
        "FAISS_ENABLED",
        "ONNX_INTRA_OP_THREADS",
        "ONNX_INTER_OP_THREADS",
        "USE_FAST_PATH",
        "FAST_PATH_MAX_CONCURRENCY",
        "FAST_WORKER_URL",
        "FAST_WORKER_MAX_CONCURRENCY",
        "BATCH_SIZE",
        "BATCH_COLLECT_TIMEOUT",
        "OMP_NUM_THREADS",
        "ORT_INTRA_OP_NUM_THREADS",
        "ORT_INTER_OP_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "MAX_QUEUE_SIZE",
        "MAX_QUEUE_WAIT",
        "INFLIGHT_LIMIT"
    )
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

function New-SafeName {
    param([string]$Name)
    return ($Name -replace "[^a-zA-Z0-9_\-]", "_")
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

function Invoke-And-Capture {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$StdoutPath,
        [string]$StderrPath,
        [string]$WorkingDirectory
    )

    $resolvedBinary = (Get-Command $FilePath -ErrorAction Stop).Source

    foreach ($path in @($StdoutPath, $StderrPath)) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
        }
    }

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $resolvedBinary
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    if ($psi.PSObject.Properties.Name -contains "ArgumentList") {
        foreach ($arg in $Arguments) {
            [void]$psi.ArgumentList.Add([string]$arg)
        }
    }
    else {
        $psi.Arguments = Convert-ArgumentListToString -Arguments $Arguments
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $psi
    [void]$process.Start()

    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()

    $stdout | Out-File -FilePath $StdoutPath -Encoding utf8
    $stderr | Out-File -FilePath $StderrPath -Encoding utf8

    [pscustomobject]@{
        ExitCode = $process.ExitCode
        Stdout = $stdout
        Stderr = $stderr
    }
}

function Invoke-DockerCompose {
    param(
        [string[]]$Arguments,
        [string]$RunDir,
        [string]$Prefix
    )

    $stdoutPath = Join-Path $RunDir ("{0}.stdout.txt" -f $Prefix)
    $stderrPath = Join-Path $RunDir ("{0}.stderr.txt" -f $Prefix)

    Write-Host ("docker compose {0}" -f ($Arguments -join " ")) -ForegroundColor DarkGray

    $result = Invoke-And-Capture `
        -FilePath "docker" `
        -Arguments (@("compose", "-p", $ProjectName) + $Arguments) `
        -StdoutPath $stdoutPath `
        -StderrPath $stderrPath `
        -WorkingDirectory $ProjectRoot

    if ($result.ExitCode -ne 0) {
        throw "docker compose failed: $Prefix (exit_code=$($result.ExitCode))"
    }

    return $result
}

function Invoke-Docker {
    param(
        [string[]]$Arguments,
        [string]$RunDir,
        [string]$Prefix,
        [switch]$AllowFailure
    )

    $stdoutPath = Join-Path $RunDir ("{0}.stdout.txt" -f $Prefix)
    $stderrPath = Join-Path $RunDir ("{0}.stderr.txt" -f $Prefix)

    Write-Host ("docker {0}" -f ($Arguments -join " ")) -ForegroundColor DarkGray
    $result = Invoke-And-Capture -FilePath "docker" -Arguments $Arguments -StdoutPath $stdoutPath -StderrPath $stderrPath -WorkingDirectory $ProjectRoot

    if (-not $AllowFailure -and $result.ExitCode -ne 0) {
        throw "docker failed: $Prefix (exit_code=$($result.ExitCode))"
    }

    return $result
}

function Get-WorkerContainerIds {
    param([string]$RunDir)

    $workerIdsArgs = @(
        "ps", "-q",
        "--filter", "label=com.docker.compose.project=$ProjectName",
        "--filter", "label=com.docker.compose.service=$WorkerService"
    )

    $result = Invoke-Docker -Arguments $workerIdsArgs -RunDir $RunDir -Prefix "docker_ps_worker_ids"
    $ids = @($result.Stdout -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    return ,$ids
}

function Assert-WorkerReplicaCount {
    param(
        [string]$RunDir,
        [int]$ExpectedCount
    )

    $ids = Get-WorkerContainerIds -RunDir $RunDir
    if (@($ids).Count -ne $ExpectedCount) {
        throw "Expected $ExpectedCount worker containers, got $(@($ids).Count)"
    }
}

function Get-RunRoot {
    param([string]$Label)

    $safeLabel = New-SafeName -Name $Label
    $relative = Join-Path $OutDir ("{0}_{1}" -f (Get-Timestamp), $safeLabel)
    $absolute = Join-Path $ProjectRoot $relative
    Initialize-Directory -Path $absolute

    return [pscustomobject]@{
        Label = $Label
        SafeLabel = $safeLabel
        Relative = $relative
        Absolute = $absolute
    }
}

function Save-Text {
    param(
        [string]$Path,
        [string]$Content
    )
    $Content | Out-File -FilePath $Path -Encoding utf8
}

function Save-ComposeAndPs {
    param([string]$RunDir)

    Invoke-DockerCompose -Arguments @("config") -RunDir $RunDir -Prefix "compose_config" | Out-Null
    Invoke-DockerCompose -Arguments @("ps") -RunDir $RunDir -Prefix "compose_ps" | Out-Null

    Invoke-Docker -Arguments @("stats", "--no-stream", "--format", "table {{.Name}}`t{{.CPUPerc}}`t{{.MemUsage}}`t{{.NetIO}}`t{{.BlockIO}}") -RunDir $RunDir -Prefix "docker_stats" -AllowFailure | Out-Null

    $psArgs = @(
        "ps",
        "--filter", "label=com.docker.compose.project=$ProjectName",
        "--filter", "label=com.docker.compose.service=$WorkerService",
        "--format", "table {{.Names}}`t{{.Status}}`t{{.RunningFor}}"
    )
    Invoke-Docker -Arguments $psArgs -RunDir $RunDir -Prefix "docker_ps_workers" | Out-Null
}

function Save-WorkerEnv {
    param(
        [string]$RunDir,
        [string]$Prefix = "worker_env_snapshot"
    )

    $workerIds = Get-WorkerContainerIds -RunDir $RunDir
    $targetPath = Join-Path $RunDir ("{0}.txt" -f $Prefix)

    if (@($workerIds).Count -eq 0) {
        Save-Text -Path $targetPath -Content "No worker containers found."
        return
    }

    $lines = New-Object System.Collections.Generic.List[string]
    foreach ($id in $workerIds) {
        $nameResult = Invoke-Docker -Arguments @("inspect", "--format", "{{.Name}}", $id) -RunDir $RunDir -Prefix ("inspect_name_{0}" -f $id)
        $envResult = Invoke-Docker -Arguments @("inspect", "--format", "{{range .Config.Env}}{{println .}}{{end}}", $id) -RunDir $RunDir -Prefix ("inspect_env_{0}" -f $id)
        $name = $nameResult.Stdout.Trim()

        $lines.Add(("===== {0} =====" -f $name))
        $envLines = @($envResult.Stdout -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })

        if ($CheckKeyEnvOnly) {
            foreach ($key in $KeyEnvNames) {
                $matched = $envLines | Where-Object { $_ -like "$key=*" }
                if ($matched) {
                    foreach ($item in $matched) {
                        $lines.Add($item)
                    }
                }
                else {
                    $lines.Add(("{0}=<missing>" -f $key))
                }
            }
        }
        else {
            foreach ($item in ($envLines | Sort-Object)) {
                $lines.Add($item)
            }
        }

        $lines.Add("")
    }

    $lines | Out-File -FilePath $targetPath -Encoding utf8
}

function Save-WorkerLogs {
    param(
        [string]$RunDir,
        [string]$Prefix = "worker_logs"
    )

    Invoke-DockerCompose -Arguments @("logs", ("--tail={0}" -f $LogTailLines), $WorkerService) -RunDir $RunDir -Prefix $Prefix | Out-Null
}

function Wait-ForWarmup {
    param([string]$RunDir)

    if ($NoWarmupPause) {
        Write-Host "Warmup pause skipped." -ForegroundColor Yellow
        return
    }

    Write-Host ("Waiting {0}s for worker warmup..." -f $WarmupSeconds) -ForegroundColor Yellow
    Start-Sleep -Seconds $WarmupSeconds
    Save-WorkerLogs -RunDir $RunDir -Prefix "worker_logs_after_warmup"
}

function Start-ComposeStack {
    param(
        [string]$RunDir,
        [int]$ReplicaCount
    )

    Write-Section ("Compose up with {0} worker replica(s)" -f $ReplicaCount)
    Save-ComposeAndPs -RunDir $RunDir

    $env:WORKER_SEMAPHORE = "$Semaphore"
    $env:BATCH_SIZE = "$BatchSize"
    $env:BATCH_COLLECT_TIMEOUT = "$BatchCollectTimeout"

    $composeUpArgs = @(
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

    $result = Invoke-DockerCompose -Arguments $composeUpArgs -RunDir $RunDir -Prefix "compose_up"
    if ($result.ExitCode -ne 0) {
        throw "docker compose up failed (exit_code=$($result.ExitCode))"
    }

    Save-ComposeAndPs -RunDir $RunDir
    Save-WorkerEnv -RunDir $RunDir -Prefix "worker_env_after_up"
    Save-WorkerLogs -RunDir $RunDir -Prefix "worker_logs_after_up"
}

function Set-WorkerReplicaCount {
    param(
        [string]$RunDir,
        [int]$ReplicaCount
    )

    Write-Section ("Scale workers to {0}" -f $ReplicaCount)
    Save-ComposeAndPs -RunDir $RunDir

    $env:WORKER_SEMAPHORE = "$Semaphore"
    $env:BATCH_SIZE = "$BatchSize"
    $env:BATCH_COLLECT_TIMEOUT = "$BatchCollectTimeout"

    $composeScaleArgs = @(
        "up",
        "-d",
        "--scale",
        ("{0}={1}" -f $WorkerService, $ReplicaCount),
        $WorkerService
    )

    Invoke-DockerCompose -Arguments $composeScaleArgs -RunDir $RunDir -Prefix ("compose_scale_{0}" -f $ReplicaCount) | Out-Null

    Save-ComposeAndPs -RunDir $RunDir
    Save-WorkerEnv -RunDir $RunDir -Prefix ("worker_env_after_scale_{0}" -f $ReplicaCount)
    Wait-ForWarmup -RunDir $RunDir
    Save-ComposeAndPs -RunDir $RunDir
    Assert-WorkerReplicaCount -RunDir $RunDir -ExpectedCount $ReplicaCount
}

function Get-K6Metric {
    param(
        [object]$Summary,
        [string]$MetricName,
        [string]$FieldName
    )

    if ($null -eq $Summary -or $null -eq $Summary.metrics) {
        return $null
    }

    $metric = $Summary.metrics.PSObject.Properties[$MetricName]
    if ($null -eq $metric) {
        return $null
    }

    $metricValue = $metric.Value
    if ($null -eq $FieldName) {
        return $metricValue
    }

    $field = $metricValue.PSObject.Properties[$FieldName]
    if ($null -eq $field) {
        return $null
    }

    return $field.Value
}

function Read-SeriesStatus {
    param([string]$SeriesStatusPath)

    if (-not (Test-Path -LiteralPath $SeriesStatusPath)) {
        return @()
    }

    try {
        $json = Get-Content -Raw -Path $SeriesStatusPath | ConvertFrom-Json
        return @($json)
    }
    catch {
        return @()
    }
}

function Read-K6Summary {
    param([string]$SummaryPath)

    if ([string]::IsNullOrWhiteSpace($SummaryPath) -or -not (Test-Path -LiteralPath $SummaryPath)) {
        return $null
    }

    try {
        return Get-Content -Raw -Path $SummaryPath | ConvertFrom-Json
    }
    catch {
        return $null
    }
}

function Get-ExitCodeValue {
    param([object]$Value)

    if ($null -eq $Value) {
        return $null
    }

    if ($Value -is [System.Array] -and $Value.Count -gt 0) {
        $Value = $Value[0]
    }

    if ($Value.PSObject.Properties.Name -contains "ExitCode") {
        return [int]$Value.ExitCode
    }

    return $null
}

function Get-PropertyValue {
    param(
        [object]$Value,
        [string]$Name,
        $Default = $null
    )

    if ($null -eq $Value) {
        return $Default
    }

    if ($Value -is [System.Array] -and $Value.Count -gt 0) {
        $Value = $Value[0]
    }

    if ($Value.PSObject.Properties.Name -contains $Name) {
        return $Value.$Name
    }

    return $Default
}

function Get-StepExitCode {
    param([object[]]$Runs)

    if (-not $Runs -or @($Runs).Count -eq 0) {
        return 1
    }

    $failed = $Runs | Where-Object { (Get-ExitCodeValue -Value $_) -ne 0 }
    if (@($failed).Count -gt 0) {
        return 1
    }

    return 0
}

function Get-StepRunSucceeded {
    param([object[]]$Runs)

    if (-not $Runs -or @($Runs).Count -eq 0) {
        return $false
    }

    foreach ($run in $Runs) {
        $runExitCode = Get-ExitCodeValue -Value $run
        if ($null -eq $runExitCode -or $runExitCode -ne 0) {
            return $false
        }
    }

    return $true
}

function Get-StepShortNote {
    param(
        [object[]]$RunItems,
        [string]$Scenario
    )

    $lines = New-Object System.Collections.Generic.List[string]
    foreach ($item in $RunItems) {
        $summary = Get-PropertyValue -Value $item -Name "Summary"
        $rate = Get-PropertyValue -Value $item -Name "Rate" -Default "<unknown>"
        $exitCode = Get-ExitCodeValue -Value $item
        $queueP95 = $null
        $processingP95 = $null
        $clientP95 = $null
        $httpFailedRate = $null
        $completionFailed = $null
        $enqueueFailed = $null
        $completedCount = $null
        $acceptedCount = $null

        if ($null -ne $summary) {
            $httpFailedRate = Get-K6Metric -Summary $summary -MetricName "http_req_failed" -FieldName "value"
            $queueP95 = Get-K6Metric -Summary $summary -MetricName "queue_delay_ms" -FieldName "p(95)"
            $processingP95 = Get-K6Metric -Summary $summary -MetricName "processing_time_ms" -FieldName "p(95)"
            $clientP95 = Get-K6Metric -Summary $summary -MetricName "client_e2e_ms" -FieldName "p(95)"
            $completionFailed = Get-K6Metric -Summary $summary -MetricName "completion_failed" -FieldName "value"
            $enqueueFailed = Get-K6Metric -Summary $summary -MetricName "enqueue_failed" -FieldName "value"
            $completedCount = Get-K6Metric -Summary $summary -MetricName "completed" -FieldName "count"
            $acceptedCount = Get-K6Metric -Summary $summary -MetricName "enqueue_accepted" -FieldName "count"
        }

        $note = "rate={0}; exit={1}" -f $rate, $exitCode
        if ($Scenario -eq "full") {
            $note += "; completed={0}; completion_failed={1}; queue_p95={2}; processing_p95={3}; client_e2e_p95={4}" -f $completedCount, $completionFailed, $queueP95, $processingP95, $clientP95
        }
        else {
            $note += "; accepted={0}; enqueue_failed={1}; http_req_failed={2}" -f $acceptedCount, $enqueueFailed, $httpFailedRate
        }

        $lines.Add($note)
    }

    return $lines
}

function Save-RunResultNote {
    param(
        [string]$RunDir,
        [string]$StepLabel,
        [string]$Scenario,
        [int]$Rate,
        [int]$ReplicaCount,
        [int]$StepExitCode,
        [object[]]$RunItems
    )

    $lines = New-Object System.Collections.Generic.List[string]
    $null = $lines.Add("Step: $StepLabel")
    $null = $lines.Add(("Scenario: {0}" -f $Scenario))
    $null = $lines.Add(("Worker replicas: {0}" -f $ReplicaCount))
    $null = $lines.Add(("Step exit_code: {0}" -f $StepExitCode))
    $null = $lines.Add(("Rate: {0}" -f $Rate))
    $null = $lines.Add("")
    $null = $lines.Add("Runs:")
    foreach ($line in (Get-StepShortNote -RunItems $RunItems -Scenario $Scenario)) {
        $null = $lines.Add(("- {0}" -f $line))
    }

    $lines | Out-File -FilePath (Join-Path $RunDir "step_summary.txt") -Encoding utf8
}

function Invoke-HarnessRun {
    param(
        [string]$StepLabel,
        [string]$Scenario,
        [int]$Rate,
        [int]$ReplicaCount
    )

    $stepRoot = Get-RunRoot -Label $StepLabel
    Write-Section ("Harness run: {0}" -f $StepLabel)

    Save-ComposeAndPs -RunDir $stepRoot.Absolute
    Save-WorkerEnv -RunDir $stepRoot.Absolute -Prefix "worker_env_snapshot"
    Save-WorkerLogs -RunDir $stepRoot.Absolute -Prefix "worker_logs_before_run"

    $harnessOutputDir = Join-Path $stepRoot.Relative "harness"
    $runLabel = ("{0}_{1}" -f $stepRoot.SafeLabel, (Get-Timestamp))
    $harnessStdout = Join-Path $stepRoot.Absolute "harness.stdout.txt"
    $harnessStderr = Join-Path $stepRoot.Absolute "harness.stderr.txt"
    $harnessOut = Join-Path $stepRoot.Absolute "harness.output.txt"

    foreach ($path in @($harnessStdout, $harnessStderr, $harnessOut)) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
        }
    }

    $harnessExitCode = 0
    $global:LASTEXITCODE = 0
    try {
        $combined = & $HarnessPath `
            -SkipComposeUp `
            -Scenario $Scenario `
            -WorkerCount 0 `
            -Semaphore $Semaphore `
            -BatchSize $BatchSize `
            -BatchCollectTimeout $BatchCollectTimeout `
            -WarmupSeconds 0 `
            -BaseUrl $BaseUrl `
            -OutputDir $harnessOutputDir `
            -Label $runLabel `
            -Duration $Duration `
            -Rates $Rate 2>&1

        $combinedText = ($combined | Out-String)
        $combinedText | Out-File -FilePath $harnessStdout -Encoding utf8
        $combinedText | Out-File -FilePath $harnessOut -Encoding utf8

        if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
            $harnessExitCode = $LASTEXITCODE
        }
    }
    catch {
        $harnessExitCode = 1
        ($_ | Out-String) | Out-File -FilePath $harnessStderr -Encoding utf8
        ($_ | Out-String) | Out-File -FilePath $harnessOut -Encoding utf8
    }

    $seriesStatusPath = Join-Path $ProjectRoot $harnessOutputDir
    $seriesStatusPath = Join-Path $seriesStatusPath $runLabel
    $seriesStatusPath = Join-Path $seriesStatusPath "series.status.json"

    if (-not (Test-Path -LiteralPath $seriesStatusPath)) {
        Save-Text `
            -Path (Join-Path $stepRoot.Absolute "series_status_missing.txt") `
            -Content ("Missing series.status.json: {0}" -f $seriesStatusPath)
    }

    $runItems = New-Object System.Collections.Generic.List[object]
    $statusItems = Read-SeriesStatus -SeriesStatusPath $seriesStatusPath
    foreach ($status in $statusItems) {
        $summary = Read-K6Summary -SummaryPath $status.summary_path
        $null = $runItems.Add([pscustomobject]@{
            RunName = $status.run_name
            Rate = $status.rate
            Scenario = $status.scenario
            ExitCode = $status.exit_code
            SummaryPath = $status.summary_path
            Summary = $summary
        })
    }

    $stepExitCode = Get-StepExitCode -Runs $runItems

    $benchmarkCopyDir = $null
    if (Test-Path -LiteralPath $seriesStatusPath) {
        $benchmarkCopyDir = Split-Path -Parent $seriesStatusPath
    }

    $manifest = [pscustomobject]@{
        label = $runLabel
        scenario = $Scenario
        rate = $Rate
        harness_exit_code = $harnessExitCode
        benchmark_copy_dir = $benchmarkCopyDir
        parsed_summary = ($null -ne $statusItems -and @($statusItems).Count -gt 0)
        step_exit_code = $stepExitCode
    }
    $manifest | ConvertTo-Json -Depth 6 | Out-File -FilePath (Join-Path $stepRoot.Absolute "run_manifest.json") -Encoding utf8

    Save-ComposeAndPs -RunDir $stepRoot.Absolute
    Save-WorkerLogs -RunDir $stepRoot.Absolute -Prefix "worker_logs_after_run"
    Save-RunResultNote -RunDir $stepRoot.Absolute -StepLabel $StepLabel -Scenario $Scenario -Rate $Rate -ReplicaCount $ReplicaCount -StepExitCode $stepExitCode -RunItems $runItems

    return [pscustomobject]@{
        StepLabel = $StepLabel
        Scenario = $Scenario
        Rate = $Rate
        ReplicaCount = $ReplicaCount
        RunDir = $stepRoot.Absolute
        HarnessOutputDir = (Join-Path $ProjectRoot $harnessOutputDir)
        HarnessExitCode = $harnessExitCode
        ExitCode = $stepExitCode
        Runs = $runItems
    }
}

function Save-FinalSummary {
    param([System.Collections.Generic.List[object]]$Results)

    $target = Join-Path $ProjectRoot $OutDir
    Initialize-Directory -Path $target
    $markdown = Join-Path $target "series_summary.md"
    $jsonPath = Join-Path $target "series_summary.json"

    $lines = New-Object System.Collections.Generic.List[string]
    $null = $lines.Add("# Scaling series summary")
    $null = $lines.Add("")
    $null = $lines.Add(("Generated: {0}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")))
    $null = $lines.Add("")
    $null = $lines.Add("| Step | Scenario | Replica count | Rate | Exit code | Run count | Run dir | Harness output |")
    $null = $lines.Add("|---|---|---:|---:|---:|---:|---|---|")

    foreach ($item in $Results) {
        $rate = Get-PropertyValue -Value $item -Name "Rate" -Default "<unknown>"
        $stepLabel = Get-PropertyValue -Value $item -Name "StepLabel" -Default "<unknown>"
        $scenario = Get-PropertyValue -Value $item -Name "Scenario" -Default "<unknown>"
        $replicaCount = Get-PropertyValue -Value $item -Name "ReplicaCount" -Default "<unknown>"
        $runCount = @((Get-PropertyValue -Value $item -Name "Runs" -Default @())).Count
        $runDir = Get-PropertyValue -Value $item -Name "RunDir" -Default "<unknown>"
        $harnessOutputDir = Get-PropertyValue -Value $item -Name "HarnessOutputDir" -Default "<unknown>"
        $exitCode = Get-ExitCodeValue -Value $item
        $null = $lines.Add(("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |" -f $stepLabel, $scenario, $replicaCount, $rate, $exitCode, $runCount, $runDir, $harnessOutputDir))
    }

    $null = $lines.Add("")
    $null = $lines.Add("## Notes")
    $null = $lines.Add("")
    $null = $lines.Add("- The API base URL defaults to `http://localhost:8000`, which matches the published `api` port and avoids the unstable `api_lb` path.")
    $null = $lines.Add("- Compose is started explicitly before the first benchmark, because `deploy.replicas` is ignored by plain Docker Compose.")
    $null = $lines.Add("- Each step writes its own `step_summary.txt`, compose snapshots, worker env snapshot, and harness output under the step directory.")

    $lines | Out-File -FilePath $markdown -Encoding utf8
    $Results | ConvertTo-Json -Depth 8 | Out-File -FilePath $jsonPath -Encoding utf8
}

function Restore-BaselineReplicas {
    param([string]$RunDir)

    if ($BaselineReplicaCount -lt 0) {
        return
    }

    $current = Get-WorkerContainerIds -RunDir $RunDir
    $currentCount = @($current).Count
    Write-Host ("Current worker replicas before restore: {0}" -f $currentCount) -ForegroundColor DarkYellow
    if ($currentCount -eq $BaselineReplicaCount) {
        return
    }

    Write-Section ("Restore workers to baseline replica count {0}" -f $BaselineReplicaCount)
    Set-WorkerReplicaCount -RunDir $RunDir -ReplicaCount $BaselineReplicaCount
    Assert-WorkerReplicaCount -RunDir $RunDir -ExpectedCount $BaselineReplicaCount
}

# Main
Initialize-Directory -Path (Join-Path $ProjectRoot $OutDir)

$results = New-Object 'System.Collections.Generic.List[object]'

Write-Section "Initial compose snapshot"
$initialDir = Get-RunRoot -Label "initial_snapshot"
Start-ComposeStack -RunDir $initialDir.Absolute -ReplicaCount $BaselineReplicaCount
if (-not $NoWarmupPause) {
    Wait-ForWarmup -RunDir $initialDir.Absolute
}
Save-ComposeAndPs -RunDir $initialDir.Absolute
Save-WorkerEnv -RunDir $initialDir.Absolute -Prefix "worker_env_initial"
Save-WorkerLogs -RunDir $initialDir.Absolute -Prefix "worker_logs_initial"

if (-not $SkipBaseline) {
    $baselineEnqueue = Invoke-HarnessRun -StepLabel ("baseline_enqueue_r{0}" -f $BaselineEnqueueRate) -Scenario "enqueue-only" -Rate $BaselineEnqueueRate -ReplicaCount $BaselineReplicaCount
    $null = $results.Add($baselineEnqueue)

    $baselineCompletion15 = Invoke-HarnessRun -StepLabel ("baseline_completion_r{0}" -f $BaselineCompletionRate15) -Scenario "full" -Rate $BaselineCompletionRate15 -ReplicaCount $BaselineReplicaCount
    $null = $results.Add($baselineCompletion15)

    $baselineCompletion20 = Invoke-HarnessRun -StepLabel ("baseline_completion_r{0}" -f $BaselineCompletionRate20) -Scenario "full" -Rate $BaselineCompletionRate20 -ReplicaCount $BaselineReplicaCount
    $null = $results.Add($baselineCompletion20)
}
else {
    Write-Host "Baseline skipped." -ForegroundColor Yellow
}

if (-not $SkipScale) {
    foreach ($replicas in $ScaleReplicaSteps) {
        $scaleRun = Get-RunRoot -Label ("scale_r{0}" -f $replicas)
        Set-WorkerReplicaCount -RunDir $scaleRun.Absolute -ReplicaCount $replicas

        $scaleCompletion = Invoke-HarnessRun -StepLabel ("scale_r{0}_completion_r{1}" -f $replicas, $ScaleCompletionRate) -Scenario "full" -Rate $ScaleCompletionRate -ReplicaCount $replicas
        $null = $results.Add($scaleCompletion)

        $scaleCompletionRuns = Get-PropertyValue -Value $scaleCompletion -Name "Runs" -Default @()
        $scaleRunSucceeded = Get-StepRunSucceeded -Runs $scaleCompletionRuns
        $scaleCompletionExitCode = Get-ExitCodeValue -Value $scaleCompletion

        if (-not $SkipPostSuccess -and $scaleRunSucceeded) {
            $postCompletion22 = Invoke-HarnessRun -StepLabel ("scale_r{0}_post_completion_r{1}" -f $replicas, $PostSuccessCompletionRate22) -Scenario "full" -Rate $PostSuccessCompletionRate22 -ReplicaCount $replicas
            $null = $results.Add($postCompletion22)

            $postCompletion25 = Invoke-HarnessRun -StepLabel ("scale_r{0}_post_completion_r{1}" -f $replicas, $PostSuccessCompletionRate25) -Scenario "full" -Rate $PostSuccessCompletionRate25 -ReplicaCount $replicas
            $null = $results.Add($postCompletion25)

            $postEnqueue42 = Invoke-HarnessRun -StepLabel ("scale_r{0}_post_enqueue_r{1}" -f $replicas, $PostSuccessEnqueueRate42) -Scenario "enqueue-only" -Rate $PostSuccessEnqueueRate42 -ReplicaCount $replicas
            $null = $results.Add($postEnqueue42)

            $postEnqueue45 = Invoke-HarnessRun -StepLabel ("scale_r{0}_post_enqueue_r{1}" -f $replicas, $PostSuccessEnqueueRate45) -Scenario "enqueue-only" -Rate $PostSuccessEnqueueRate45 -ReplicaCount $replicas
            $null = $results.Add($postEnqueue45)
        }
        elseif ($SkipPostSuccess) {
            Write-Host "Post-success runs skipped by flag." -ForegroundColor Yellow
        }
        else {
            Write-Host ("Post-success runs skipped because completion run exit_code={0}" -f $scaleCompletionExitCode) -ForegroundColor Yellow
        }
    }
}
else {
    Write-Host "Scaling skipped." -ForegroundColor Yellow
}

Save-FinalSummary -Results $results

Write-Section "Cleanup"
Restore-BaselineReplicas -RunDir $initialDir.Absolute

Write-Section "Done"
Write-Host ("Artifacts saved to: {0}" -f (Resolve-Path -LiteralPath (Join-Path $ProjectRoot $OutDir))) -ForegroundColor Green
