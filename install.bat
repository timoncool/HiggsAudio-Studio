@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   Higgs Audio Studio - Установка
echo ========================================

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
set "TEMP=%SCRIPT_DIR%temp"
set "TMP=%SCRIPT_DIR%temp"

REM Интерпретатор uv-окружения (.venv) и быстрый установщик пакетов через uv.
set "PY=.venv\Scripts\python.exe"
set "UVPIP=uv.exe pip install --python %PY%"

if not exist "downloads" mkdir downloads
if not exist "temp" mkdir temp
if not exist "models" mkdir models
if not exist "cache" mkdir cache
if not exist "output" mkdir output
if not exist "voices" mkdir voices

REM ============================================================
REM  Шаг 1: Выбор GPU (torch 2.7.1 во всех ветках — под него есть prebuilt-ускорители)
REM ============================================================
echo.
echo Выберите GPU:
echo.
echo   1. NVIDIA GTX 10xx (Pascal)
echo   2. NVIDIA RTX 20xx (Turing)
echo   3. NVIDIA RTX 30xx (Ampere)
echo   4. NVIDIA RTX 40xx (Ada Lovelace)
echo   5. NVIDIA RTX 50xx (Blackwell)
echo   6. CPU only (без GPU)
echo.
set /p GPU_CHOICE="Введите номер (1-6): "

if "%GPU_CHOICE%"=="1" goto :gpu_10xx
if "%GPU_CHOICE%"=="2" goto :gpu_20xx
if "%GPU_CHOICE%"=="3" goto :gpu_30xx
if "%GPU_CHOICE%"=="4" goto :gpu_40xx
if "%GPU_CHOICE%"=="5" goto :gpu_50xx
if "%GPU_CHOICE%"=="6" goto :gpu_cpu
echo Неверный выбор!
pause
exit /b 1

:gpu_10xx
set "CUDA_VERSION=cu118"
set "CUDA_NAME=CUDA 11.8 (GTX 10xx)"
set "TORCH_VERSION=2.7.1"
set "TORCHAUDIO_VERSION=2.7.1"
goto :gpu_done
:gpu_20xx
set "CUDA_VERSION=cu126"
set "CUDA_NAME=CUDA 12.6 (RTX 20xx)"
set "TORCH_VERSION=2.7.1"
set "TORCHAUDIO_VERSION=2.7.1"
goto :gpu_done
:gpu_30xx
set "CUDA_VERSION=cu126"
set "CUDA_NAME=CUDA 12.6 (RTX 30xx)"
set "TORCH_VERSION=2.7.1"
set "TORCHAUDIO_VERSION=2.7.1"
goto :gpu_done
:gpu_40xx
set "CUDA_VERSION=cu128"
set "CUDA_NAME=CUDA 12.8 (RTX 40xx)"
set "TORCH_VERSION=2.7.1"
set "TORCHAUDIO_VERSION=2.7.1"
goto :gpu_done
:gpu_50xx
set "CUDA_VERSION=cu128"
set "CUDA_NAME=CUDA 12.8 (RTX 50xx)"
set "TORCH_VERSION=2.7.1"
set "TORCHAUDIO_VERSION=2.7.1"
goto :gpu_done
:gpu_cpu
set "CUDA_VERSION=cpu"
set "CUDA_NAME=CPU only"
set "TORCH_VERSION=2.7.1"
set "TORCHAUDIO_VERSION=2.7.1"
goto :gpu_done

:gpu_done
echo.
echo Выбрано: %CUDA_NAME%
echo.

REM ============================================================
REM  Шаг 2: uv + виртуальное окружение .venv (Python 3.12.9)
REM  uv качает python-build-standalone, создаёт venv и ставит пакеты
REM  (uv pip install — заметно быстрее обычного pip).
REM ============================================================
REM --- uv.exe: качаем, если ещё нет (нужен и для установки пакетов ниже) ---
if not exist "uv.exe" (
    echo [1/7] Скачиваю uv...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile 'downloads\uv.zip'}"
    powershell -Command "& {Expand-Archive -Path 'downloads\uv.zip' -DestinationPath 'downloads\uv_tmp' -Force}"
    copy /y "downloads\uv_tmp\uv.exe" "uv.exe" >nul
    rmdir /s /q "downloads\uv_tmp"
    del /f /q "downloads\uv.zip"
)
if not exist "uv.exe" (
    echo ОШИБКА: не удалось скачать uv!
    pause
    exit /b 1
)

if exist "%PY%" (
    echo [OK] Окружение .venv уже создано
) else (
    echo [1/7] Создаю .venv на Python 3.12.9 через uv...
    uv.exe venv --seed --python 3.12.9 ".venv"
    if not exist "%PY%" (
        echo ОШИБКА: не удалось создать .venv через uv!
        pause
        exit /b 1
    )
    echo [OK] Окружение .venv создано
)

REM ============================================================
REM  Шаг 3: базовый инструментарий (pip засеян в .venv через --seed)
REM ============================================================
echo [2/7] Обновляю pip/setuptools/wheel...
%UVPIP% --upgrade pip setuptools wheel

REM ============================================================
REM  Шаг 4: PyTorch 2.7.1
REM ============================================================
echo [3/7] Устанавливаю PyTorch %TORCH_VERSION% (%CUDA_NAME%)...
if "%CUDA_VERSION%"=="cpu" (
    %UVPIP% torch==%TORCH_VERSION% torchaudio==%TORCHAUDIO_VERSION%
) else (
    %UVPIP% torch==%TORCH_VERSION% torchaudio==%TORCHAUDIO_VERSION% --index-url https://download.pytorch.org/whl/%CUDA_VERSION%
)

REM ============================================================
REM  Шаг 5: Зависимости
REM ============================================================
echo [4/7] Устанавливаю зависимости...
%UVPIP% -r requirements.txt
REM Облачные голоса качаются через huggingface_hub (httpx). Выкидываем urllib3-future/niquests,
REM если приехали транзитивно — их битый HTTP/2 (hface) ломал скачивание голосов (ишью #2).
uv.exe pip uninstall --python %PY% urllib3-future niquests 2>nul

REM ============================================================
REM  Шаг 6: Triton для torch.compile (~2x). Скобки/for-блоки убраны (goto-поток).
REM  Higgs использует SDPA (flash-ядра встроены) — внешний Flash-Attention 2 НЕ нужен (модель его отвергает).
REM ============================================================
if "%CUDA_VERSION%"=="cpu" goto :after_accel
echo [5/7] Устанавливаю Triton для torch.compile...
%UVPIP% "triton-windows>=3.0.0,<3.4"
if exist ".venv\Include\Python.h" goto :after_accel
echo Копирую Python headers для Triton из базовой сборки uv...
set "BASEPY="
for /f "delims=" %%i in ('%PY% -c "import sys;print(sys.base_prefix)"') do set "BASEPY=%%i"
if not defined BASEPY goto :after_accel
if not exist "%BASEPY%\include\Python.h" goto :after_accel
if not exist ".venv\Include" mkdir ".venv\Include"
if not exist ".venv\libs" mkdir ".venv\libs"
xcopy /E /Y "%BASEPY%\include\*" ".venv\Include\" >nul 2>&1
xcopy /E /Y "%BASEPY%\libs\*" ".venv\libs\" >nul 2>&1
echo [OK] Python headers установлены
:after_accel

REM ============================================================
REM  Шаг 7: llama-cpp-python (GGUF-режиссёр, GPU) + ЕГО СОБСТВЕННЫЙ CUDA 12.4-рантайм.
REM  Колесо abetlen собрано под cu124 и не бандлит cudart/cublas. Брать их от torch (12.6/12.8)
REM  НЕЛЬЗЯ: ggml-cuda(12.4)+cuBLAS(12.8) рушит общий CUDA-контекст -> torch "invalid argument".
REM  Ставим llama её родной 12.4-рантайм (nvidia-вешалки), НЕ от torch.
REM ============================================================
echo [6/7] Устанавливаю llama-cpp-python (AI-режиссёр, GGUF)...
if "%CUDA_VERSION%"=="cpu" goto :llama_cpu
%UVPIP% llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
echo Ставлю CUDA 12.4-рантайм для llama (совпадает со сборкой ggml-cuda)...
%UVPIP% nvidia-cuda-runtime-cu12==12.4.127 nvidia-cublas-cu12==12.4.5.8
echo Кладу 12.4 cudart/cublas рядом с llama.dll...
for %%P in (cuda_runtime cublas) do for %%D in (cudart64_12.dll cublas64_12.dll cublasLt64_12.dll) do if exist ".venv\Lib\site-packages\nvidia\%%P\bin\%%D" copy /y ".venv\Lib\site-packages\nvidia\%%P\bin\%%D" ".venv\Lib\site-packages\llama_cpp\lib\%%D" >nul
if not exist ".venv\Lib\site-packages\llama_cpp\lib\cublasLt64_12.dll" echo [ВНИМАНИЕ] cublasLt64_12.dll не скопирован - режиссёр может не стартовать!
goto :after_llama
:llama_cpu
%UVPIP% llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
:after_llama

REM ============================================================
REM  Шаг 8: Стартовый voice-pack (пресеты голосов, тянется архивом с HF)
REM ============================================================
echo [+] Загружаю стартовый voice-pack...
if exist "voices\*.mp3" (
    echo [OK] Голоса уже на месте
) else (
    curl -L -o downloads\voice-pack.zip https://huggingface.co/datasets/nerualdreming/VibeVoice/resolve/main/voice-pack.zip
    if exist "downloads\voice-pack.zip" (
        powershell -Command "& {Expand-Archive -Path 'downloads\voice-pack.zip' -DestinationPath 'downloads\vp' -Force}"
        if exist "downloads\vp\voice-pack" (
            xcopy /E /Y /Q "downloads\vp\voice-pack\*" "voices\" >nul
        ) else (
            xcopy /E /Y /Q "downloads\vp\*" "voices\" >nul
        )
        echo [OK] Voice-pack установлен
    )
)
echo [7/7] Финализация...
echo %CUDA_VERSION%> cuda_version.txt

echo ========================================
echo   Установка завершена!
echo   Запуск: run.bat
echo ========================================
pause
