[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$UnrealRoot,

    [switch]$SkipInstall,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$engineRoot = (Resolve-Path -LiteralPath $UnrealRoot).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

function Invoke-ProjectPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [int[]]$AllowedExitCodes = @(0)
    )

    & $venvPython @Arguments
    if ($AllowedExitCodes -notcontains $LASTEXITCODE) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

Push-Location $repoRoot
try {
    if ($SkipInstall -and -not (Test-Path -LiteralPath $venvPython)) {
        throw "Cannot use -SkipInstall before .venv has been created and dependencies installed."
    }
    if (-not (Test-Path -LiteralPath $venvPython)) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create .venv. Install Python 3.11 or newer and try again."
        }
    }

    if (-not $SkipInstall) {
        Invoke-ProjectPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-ProjectPython -Arguments @("-m", "pip", "install", "-e", ".[dev]")
    }

    $env:UE_ROOT = $engineRoot
    Invoke-ProjectPython -Arguments @("scripts/ingest_engine.py", "--dry-run")

    if ($ValidateOnly) {
        Write-Host "Validation succeeded. UE_ROOT=$engineRoot"
        return
    }

    Write-Host "Building the local UE 5.8 lexical index. This can take a long time."
    Invoke-ProjectPython -Arguments @("scripts/ingest_engine.py") -AllowedExitCodes @(0, 1)
    Invoke-ProjectPython -Arguments @("scripts/parse_engine.py") -AllowedExitCodes @(0, 1)
    Invoke-ProjectPython -Arguments @("scripts/chunk_engine.py")
    Invoke-ProjectPython -Arguments @(
        "scripts/build_lexical_index.py",
        "--rebuild",
        "--batch-size", "2048",
        "--progress-every", "100000"
    )

    Write-Host ""
    Write-Host "Setup complete. Start the local MCP server with:"
    Write-Host ".\.venv\Scripts\python.exe scripts\mcp_server.py"
} finally {
    Pop-Location
}
