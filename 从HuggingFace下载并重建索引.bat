@echo off
setlocal EnableExtensions
title UE 5.8 RAG - Download Embeddings and Rebuild Index
cd /d "%~dp0"

echo.
echo =====================================================
echo   从 Hugging Face 下载向量数据并重建 Qdrant 索引
echo =====================================================
echo.

rem Prefer the project virtual environment, then fall back to system Python.
if exist ".venv\Scripts\python.exe" (
  set "RAG_PYTHON=.venv\Scripts\python.exe"
) else (
  where python >nul 2>&1
  if errorlevel 1 (
    echo 未找到 Python，请先完成项目环境配置。
    pause
    exit /b 1
  )
  set "RAG_PYTHON=python"
)

if not defined HF_REPO_ID set /p "HF_REPO_ID=请输入 Hugging Face Dataset 仓库名（例如 username/ue58-rag-embeddings）："
if not defined HF_REPO_ID (
  echo 未提供仓库名，已取消。
  pause
  exit /b 1
)

echo.
echo 警告：下载的 engine 数据将覆盖当前项目中的：
echo   data\chunks\engine\chunks.jsonl
echo   data\embeddings\engine\embeddings.npy*
set /p "CONFIRM=确认继续吗？输入 Y 继续，其他键取消："
if /I not "%CONFIRM%"=="Y" (
  echo 已取消。
  exit /b 0
)

set "DOWNLOAD_DIR=%~dp0hf_download"
set "DOWNLOAD_DIR=%DOWNLOAD_DIR:~0,-1%"

echo.
echo [1/3] 检查 Hugging Face 登录状态...
"%RAG_PYTHON%" -m huggingface_hub.cli.hf auth whoami >nul 2>&1
if errorlevel 1 (
  echo 尚未登录 Hugging Face，现在打开登录流程。
  "%RAG_PYTHON%" -m huggingface_hub.cli.hf auth login
  if errorlevel 1 (
    echo Hugging Face 登录失败。
    pause
    exit /b 1
  )
)

echo [2/3] 下载 engine 数据（支持断点续传）...
"%RAG_PYTHON%" -m huggingface_hub.cli.hf download "%HF_REPO_ID%" ^
  --repo-type=dataset ^
  --local-dir "%DOWNLOAD_DIR%" ^
  --include "engine/*"
if errorlevel 1 (
  echo 下载失败。请检查仓库名、网络或访问权限。
  pause
  exit /b 1
)

if not exist "%DOWNLOAD_DIR%\engine\chunks.jsonl" (
  echo 下载包中缺少 engine\chunks.jsonl。
  pause
  exit /b 1
)
if not exist "%DOWNLOAD_DIR%\engine\embeddings.npy" (
  echo 下载包中缺少 engine\embeddings.npy。
  pause
  exit /b 1
)
if not exist "%DOWNLOAD_DIR%\engine\embeddings.npy.ids.jsonl" (
  echo 下载包中缺少 engine\embeddings.npy.ids.jsonl。
  pause
  exit /b 1
)

echo 将下载的数据复制到项目目录...
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; New-Item -ItemType Directory -Force -Path 'data\chunks\engine','data\embeddings\engine' | Out-Null; Copy-Item '%DOWNLOAD_DIR%\engine\chunks.jsonl' 'data\chunks\engine\chunks.jsonl' -Force; Copy-Item '%DOWNLOAD_DIR%\engine\embeddings.npy*' 'data\embeddings\engine' -Force"
if errorlevel 1 (
  echo 复制下载文件失败。
  pause
  exit /b 1
)

echo [3/3] 重建 Qdrant 索引...
"%RAG_PYTHON%" scripts\build_index.py ^
  --input data\chunks\engine\chunks.jsonl ^
  --vectors data\embeddings\engine\embeddings.npy ^
  --ids data\embeddings\engine\embeddings.npy.ids.jsonl
if errorlevel 1 (
  echo 索引重建失败，请检查 Qdrant 服务是否已启动。
  pause
  exit /b 1
)

echo.
echo ================================================
echo   下载与索引重建完成
echo ================================================
pause
exit /b 0
