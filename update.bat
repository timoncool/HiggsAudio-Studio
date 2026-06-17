@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

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
    if exist "python\python.exe" python\python.exe -m pip install -r requirements.txt --no-warn-script-location
) else (
    echo Папка .git не найдена - скачивайте обновления вручную с GitHub.
)

REM Чиним CUDA-рантайм режиссёра на старых установках: llama (cu124) должна иметь СВОИ 12.4
REM cudart/cublas, а не 12.8 от torch (иначе "CUDA error: invalid argument" при включённом режиссёре).
if not exist "python\python.exe" goto :rt_done
python\python.exe -m pip install nvidia-cuda-runtime-cu12==12.4.127 nvidia-cublas-cu12==12.4.5.8 --no-warn-script-location
for %%P in (cuda_runtime cublas) do for %%D in (cudart64_12.dll cublas64_12.dll cublasLt64_12.dll) do if exist "python\Lib\site-packages\nvidia\%%P\bin\%%D" copy /y "python\Lib\site-packages\nvidia\%%P\bin\%%D" "python\Lib\site-packages\llama_cpp\lib\%%D" >nul
:rt_done

echo Обновление завершено!
pause
