@echo off
title UE 5.8 RAG Data Manager
cd /d "%~dp0"

rem Close a stale dashboard process before checking/starting the local service.
rem Only terminate the Python process that owns port 8765 and runs dashboard.py.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$dashboardProcessIds = @(Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique); foreach ($dashboardProcessId in $dashboardProcessIds) { $dashboardProcess = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $dashboardProcessId) -ErrorAction SilentlyContinue; if ($dashboardProcess -and $dashboardProcess.CommandLine -match 'dashboard\.py') { Stop-Process -Id $dashboardProcessId -Force -ErrorAction SilentlyContinue } }"
powershell -NoProfile -Command "for ($i=0; $i -lt 20; $i++) { if (-not (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue)) { break }; Start-Sleep -Milliseconds 250 }"

rem Ensure the vector database is running before opening the dashboard.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_qdrant.ps1"
if errorlevel 1 (
  echo.
  echo The vector database failed to start. Review the error above.
  pause
  exit /b 1
)

rem Open the page directly when the local dashboard is already running.
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8765/' -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; exit 1" >nul 2>&1
if not errorlevel 1 (
  start "" "http://127.0.0.1:8765/"
  exit /b 0
)

rem Prefer the project virtual environment, then fall back to system Python.
if exist ".venv\Scripts\python.exe" (
  set "RAG_PYTHON=.venv\Scripts\python.exe"
) else (
  where python >nul 2>&1
  if errorlevel 1 (
    echo Python was not found. Run the project setup first.
    pause
    exit /b 1
  )
  set "RAG_PYTHON=python"
)

"%RAG_PYTHON%" scripts\dashboard.py
if errorlevel 1 (
  echo.
  echo The RAG data manager failed to start. Review the error above.
  pause
)
