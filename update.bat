@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
set "PY=.venv\Scripts\python.exe"
REM Ставим пакеты через uv (быстро), если uv.exe рядом; иначе откатываемся на pip.
set "PIPINSTALL=%PY% -m pip install"
if exist "uv.exe" set "PIPINSTALL=uv.exe pip install --python %PY%"

echo ========================================
echo   Higgs Audio Studio - Обновление
echo ========================================

where git >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Git не найден! https://git-scm.com/downloads
    pause
    exit /b 1
)

if exist ".git" (
    echo Обновление кода...
    git pull
    echo Обновление зависимостей...
    if exist "%PY%" %PIPINSTALL% -r requirements.txt
) else (
    echo Папка .git не найдена - скачивайте обновления вручную с GitHub.
)

REM Чиним CUDA-рантайм режиссёра на старых установках: llama (cu124) должна иметь СВОИ 12.4
REM cudart/cublas, а не 12.8 от torch (иначе "CUDA error: invalid argument" при включённом режиссёре).
if not exist "%PY%" goto :rt_done
%PIPINSTALL% nvidia-cuda-runtime-cu12==12.4.127 nvidia-cublas-cu12==12.4.5.8
for %%P in (cuda_runtime cublas) do for %%D in (cudart64_12.dll cublas64_12.dll cublasLt64_12.dll) do if exist ".venv\Lib\site-packages\nvidia\%%P\bin\%%D" copy /y ".venv\Lib\site-packages\nvidia\%%P\bin\%%D" ".venv\Lib\site-packages\llama_cpp\lib\%%D" >nul
:rt_done

echo Обновление завершено!
pause
