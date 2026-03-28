param(
    [int]$Rate = 80,
    [int]$DurationSeconds = 180,
    [string]$WorkersLog = "workers_timeline.log",
    [string]$QueueLog = "queue_delay_timeline.log",
    [string]$K6Log = "k6_benchmark.log"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
Set-Location $root

if (Test-Path $WorkersLog) { Remove-Item $WorkersLog -Force }
if (Test-Path $QueueLog) { Remove-Item $QueueLog -Force }
if (Test-Path $K6Log) { Remove-Item $K6Log -Force }

$workersJob = Start-Job -ScriptBlock {
    param($logPath)
    while ($true) {
        $t = Get-Date -Format "HH:mm:ss"
        $n = (docker ps --format "{{.Names}}" | Select-String worker).Count
        Add-Content -Path $logPath -Value "t=$t workers=$n"
        Start-Sleep 1
    }
} -ArgumentList (Join-Path $root $WorkersLog)

$queueJob = Start-Job -ScriptBlock {
    param($logPath)
    while ($true) {
        $t = Get-Date -Format "HH:mm:ss"
        $v = docker compose exec -T redis redis-cli GET metrics:queue_delay_ms
        Add-Content -Path $logPath -Value "t=$t queue_delay_ms=$v"
        Start-Sleep 2
    }
} -ArgumentList (Join-Path $root $QueueLog)

try {
    $env:RATE = "$Rate"
    $k6Output = & k6 run --summary-trend-stats "min,avg,med,p(90),p(95),p(99)" (Join-Path $root "load_test_async.js")
    $k6Output | Out-File -FilePath $K6Log -Encoding utf8
}
finally {
    Stop-Job $workersJob | Out-Null
    Stop-Job $queueJob | Out-Null
    Remove-Job $workersJob -Force | Out-Null
    Remove-Job $queueJob -Force | Out-Null
}

Write-Host "saved workers log -> $WorkersLog"
Write-Host "saved queue log -> $QueueLog"
Write-Host "saved k6 log -> $K6Log"
