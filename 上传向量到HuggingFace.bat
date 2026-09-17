@echo off
setlocal EnableExtensions
title UE 5.8 RAG - Upload Embeddings to Hugging Face
cd /d "%~dp0"

echo.
echo ================================================
echo   UE 5.8 RAG 向量数据打包并上传 Hugging Face
echo ================================================
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

rem Use an environment variable when running unattended; otherwise ask once.
if not defined HF_REPO_ID set /p "HF_REPO_ID=请输入 Hugging Face Dataset 仓库名（例如 username/ue58-rag-embeddings）："
if not defined HF_REPO_ID (
  echo 未提供仓库名，已取消。
  pause
  exit /b 1
)

set "PACKAGE_DIR=%~dp0hf_package"
set "PACKAGE_DIR=%PACKAGE_DIR:~0,-1%"

echo.
echo [1/4] 检查向量与 chunks 文件...
for %%F in (
  "data\chunks\engine\chunks.jsonl"
  "data\embeddings\engine\embeddings.npy"
  "data\embeddings\engine\embeddings.npy.ids.jsonl"
  "data\embeddings\engine\embeddings.npy.manifest.json"
  "data\chunks\docs\chunks.jsonl"
  "data\embeddings\docs\embeddings.npy"
  "data\embeddings\docs\embeddings.npy.ids.jsonl"
  "data\embeddings\docs\embeddings.npy.manifest.json"
) do (
  if not exist "%%~F" (
    echo 缺少文件：%%~F
    pause
    exit /b 1
  )
)

echo [2/4] 整理上传包...
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $pkg='%PACKAGE_DIR%'; New-Item -ItemType Directory -Force -Path (Join-Path $pkg 'engine'),(Join-Path $pkg 'docs') | Out-Null; Copy-Item 'docs\huggingface_dataset_card.md' (Join-Path $pkg 'README.md') -Force; Copy-Item 'data\chunks\engine\chunks.jsonl' (Join-Path $pkg 'engine\chunks.jsonl') -Force; Copy-Item 'data\embeddings\engine\embeddings.npy*' (Join-Path $pkg 'engine') -Force; Copy-Item 'data\chunks\docs\chunks.jsonl' (Join-Path $pkg 'docs\chunks.jsonl') -Force; Copy-Item 'data\embeddings\docs\embeddings.npy*' (Join-Path $pkg 'docs') -Force"
if errorlevel 1 (
  echo 整理上传包失败。
  pause
  exit /b 1
)

echo [3/4] 生成 SHA256 校验文件...
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $pkg='%PACKAGE_DIR%'; $root=$pkg+'\'; Get-ChildItem $pkg -Recurse -File | Where-Object { $_.Name -ne 'sha256.csv' } | ForEach-Object { $h=Get-FileHash $_.FullName -Algorithm SHA256; [PSCustomObject]@{ Path=$_.FullName.Substring($root.Length).Replace('\','/'); SHA256=$h.Hash; Bytes=$_.Length } } | Export-Csv (Join-Path $pkg 'sha256.csv') -NoTypeInformation -Encoding UTF8"
if errorlevel 1 (
  echo 生成校验文件失败。
  pause
  exit /b 1
)

echo [4/4] 检查 Hugging Face 登录状态并上传...
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

rem Create the dataset repository if it does not exist. The repository is private by default.
set "HF_REPO_ID=%HF_REPO_ID%"
"%RAG_PYTHON%" -c "from huggingface_hub import HfApi; import os; HfApi().create_repo(os.environ['HF_REPO_ID'], repo_type='dataset', private=True, exist_ok=True)"
if errorlevel 1 (
  echo 创建或检查 Dataset 仓库失败。
  pause
  exit /b 1
)

set "HF_XET_HIGH_PERFORMANCE=1"
"%RAG_PYTHON%" -m huggingface_hub.cli.hf upload "%HF_REPO_ID%" "%PACKAGE_DIR%" --repo-type=dataset
if errorlevel 1 (
  echo 上传失败。可以重新运行本脚本，已上传的文件会自动跳过。
  pause
  exit /b 1
)

echo.
echo 上传完成：%HF_REPO_ID%
echo 本地上传包：%PACKAGE_DIR%
pause
exit /b 0
