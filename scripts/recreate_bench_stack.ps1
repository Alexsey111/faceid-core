[CmdletBinding()]
param(
    [string]$ComposeFile = ".\docker-compose.yml",
    [string]$ProjectName = "faceid-core",
    [int]$WorkerReplicas = 4,
    [switch]$Build
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-WorkerContainers {
    docker ps `
        --filter "label=com.docker.compose.project=$ProjectName" `
        --filter "label=com.docker.compose.service=worker" `
        --format "{{.Names}}"
}

function Get-WorkerContainerCount {
    $names = @(Get-WorkerContainers)
    return @($names | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }).Count
}

function Get-WorkerRuntimeDump {
    $python = @'
import app.workers.verify_worker as w
print(getattr(w, "ENABLE_WORKER_EXPIRY", "<missing>"))
print(getattr(w, "MAX_QUEUE_WAIT_SEC", "<missing>"))
print(getattr(w, "MAX_JOB_AGE_MS", "<missing>"))
'@

    return $python | docker compose -p $ProjectName -f $ComposeFile exec -T worker python -
}

Write-Host "Stopping previous stack..."
docker compose -p $ProjectName -f $ComposeFile down --remove-orphans

$upArgs = @(
    "compose", "-p", $ProjectName, "-f", $ComposeFile,
    "up", "-d", "--force-recreate",
    "--scale", "worker=$WorkerReplicas"
)

if ($Build) {
    $upArgs += "--build"
}

Write-Host "Starting stack with worker replicas = $WorkerReplicas"
docker @upArgs

Write-Host "Waiting a moment for containers..."
Start-Sleep -Seconds 5

Write-Host "Validating worker fleet size..."
$actual = Get-WorkerContainerCount

if ($actual -ne $WorkerReplicas) {
    $workerNames = @(Get-WorkerContainers | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    Write-Host "Detected worker containers: $actual"
    $workerNames | ForEach-Object { Write-Host " - $_" }
    throw "Expected $WorkerReplicas worker containers, found $actual. Refusing to run benchmark."
}

Write-Host "Validating worker runtime..."
$runtimeDump = Get-WorkerRuntimeDump
Write-Host $runtimeDump

if ($runtimeDump -match "<missing>") {
    throw "Worker runtime does not contain expected expiry-toggle code. Abort benchmark."
}

if ($runtimeDump -match "15\.0|3000|5\.0") {
    throw "Worker runtime still looks like an old build. Abort benchmark."
}

Write-Host "Stack is ready."
