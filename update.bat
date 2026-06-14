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

echo Обновление завершено!
pause
