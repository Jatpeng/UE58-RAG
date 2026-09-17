[CmdletBinding()]
param(
    [int]$HttpPort = 6333,
    [string]$StoragePath = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$qdrantDir = Join-Path $repoRoot "tools\qdrant"
$qdrantExe = Join-Path $qdrantDir "qdrant.exe"
$configPath = Join-Path $qdrantDir "qdrant.yaml"
$stdoutLog = Join-Path $qdrantDir "qdrant_start.log"
$stderrLog = Join-Path $qdrantDir "qdrant_start.err.log"

if (-not (Test-Path -LiteralPath $qdrantExe)) {
    throw "Qdrant binary not found at $qdrantExe"
}

if (-not $StoragePath) {
    $StoragePath = Join-Path $repoRoot "data\qdrant_server"
}

$existing = Get-NetTCPConnection -LocalPort $HttpPort -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($existing) {
    Write-Host "Qdrant already listening on port $HttpPort (PID $($existing.OwningProcess))"
    exit 0
}

New-Item -ItemType Directory -Force -Path $StoragePath | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $StoragePath "snapshots") | Out-Null

$storageUnix = ($StoragePath -replace "\\", "/")
$configText = @"
storage:
  storage_path: $storageUnix
  snapshots_path: $storageUnix/snapshots
  on_disk_payload: true
service:
  http_port: $HttpPort
  grpc_port: $($HttpPort + 1)
  enable_cors: true
telemetry_disabled: true
"@
# Qdrant rejects UTF-8 BOM configs written by Windows PowerShell Set-Content -Encoding utf8.
[System.IO.File]::WriteAllText($configPath, $configText, [System.Text.UTF8Encoding]::new($false))

Write-Host "Starting Qdrant..."
Write-Host "  exe:     $qdrantExe"
Write-Host "  config:  $configPath"
Write-Host "  storage: $StoragePath"
Write-Host "  URL:     http://127.0.0.1:$HttpPort"

Remove-Item -LiteralPath $stdoutLog, $stderrLog -ErrorAction SilentlyContinue
$process = Start-Process -FilePath $qdrantExe `
    -ArgumentList @("--config-path", "qdrant.yaml") `
    -WorkingDirectory $qdrantDir `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

$ready = $false
foreach ($attempt in 1..30) {
    Start-Sleep -Seconds 1
    if ($process.HasExited) {
        $tail = @()
        if (Test-Path -LiteralPath $stderrLog) { $tail += Get-Content -LiteralPath $stderrLog -Tail 20 }
        if (Test-Path -LiteralPath $stdoutLog) { $tail += Get-Content -LiteralPath $stdoutLog -Tail 20 }
        throw ("Qdrant exited early (code $($process.ExitCode)). Log tail:`n" + ($tail -join "`n"))
    }
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$HttpPort/readyz" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {
        # still recovering large collections
    }
}

if ($ready) {
    Write-Host "Qdrant ready at http://127.0.0.1:$HttpPort (PID $($process.Id))"
} else {
    Write-Host "Qdrant process started (PID $($process.Id)), waiting for collection recovery..."
    Write-Host "Check http://127.0.0.1:$HttpPort/dashboard or $stdoutLog"
}
