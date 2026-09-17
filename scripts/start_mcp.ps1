[CmdletBinding()]
param(
    [ValidateSet("stdio", "sse", "streamable-http")]
    [string]$Transport = "stdio",
    [string]$HostAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,
    [switch]$EnableDense,
    [switch]$EnableRerank
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$indexPath = Join-Path $repoRoot "data\index\lexical.sqlite3"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Virtual environment not found. Run scripts\setup_local.ps1 first."
}
if (-not (Test-Path -LiteralPath $indexPath)) {
    throw "Lexical index not found. Run scripts\setup_local.ps1 first."
}

$arguments = @(
    "scripts/mcp_server.py",
    "--transport", $Transport,
    "--host", $HostAddress,
    "--port", $Port
)
if ($EnableDense) {
    $arguments += "--enable-dense"
}
if ($EnableRerank) {
    $arguments += "--enable-rerank"
}

Push-Location $repoRoot
try {
    & $venvPython @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
