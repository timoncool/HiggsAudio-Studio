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
REM  Шаг 2: Python 3.12.9 embed
REM ============================================================
if exist "python\python.exe" (
    echo [OK] Python уже установлен
) else (
    echo [1/7] Скачиваю Python 3.12.9...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.9/python-3.12.9-embed-amd64.zip' -OutFile 'downloads\python.zip'}"
    powershell -Command "& {Expand-Archive -Path 'downloads\python.zip' -DestinationPath 'python' -Force}"
    cd python
    if exist "python312._pth" (
        echo python312.zip> python312._pth
        echo .>> python312._pth
        echo Lib\site-packages>> python312._pth
        echo ..\Lib\site-packages>> python312._pth
        echo import site>> python312._pth
    )
    cd ..
    echo [OK] Python 3.12.9 установлен
)

REM ============================================================
REM  Шаг 3: pip
REM ============================================================
if exist "python\Scripts\pip.exe" (
    echo [OK] pip уже установлен
) else (
    echo [2/7] Устанавливаю pip...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'downloads\get-pip.py'}"
    python\python.exe downloads\get-pip.py --no-warn-script-location
)
python\python.exe -m pip install --upgrade pip setuptools wheel --no-warn-script-location

REM ============================================================
REM  Шаг 4: PyTorch 2.7.1
REM ============================================================
echo [3/7] Устанавливаю PyTorch %TORCH_VERSION% (%CUDA_NAME%)...
if "%CUDA_VERSION%"=="cpu" (
    python\python.exe -m pip install torch==%TORCH_VERSION% torchaudio==%TORCHAUDIO_VERSION% --no-warn-script-location
) else (
    python\python.exe -m pip install torch==%TORCH_VERSION% torchaudio==%TORCHAUDIO_VERSION% --index-url https://download.pytorch.org/whl/%CUDA_VERSION% --no-warn-script-location
)

REM ============================================================
REM  Шаг 5: Зависимости
REM ============================================================
echo [4/7] Устанавливаю зависимости...
python\python.exe -m pip install -r requirements.txt --no-warn-script-location

REM ============================================================
REM  Шаг 6: Triton для torch.compile (~2x). Скобки/for-блоки убраны (goto-поток).
REM  Higgs использует SDPA (flash-ядра встроены) — внешний Flash-Attention 2 НЕ нужен (модель его отвергает).
REM ============================================================
if "%CUDA_VERSION%"=="cpu" goto :after_accel
echo [5/7] Устанавливаю Triton для torch.compile...
python\python.exe -m pip install "triton-windows>=3.0.0,<3.4" --no-warn-script-location
if exist "python\Include\Python.h" goto :after_accel
echo Скачиваю Python headers для Triton...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.9/amd64/dev.msi' -OutFile 'downloads\pydev.msi'}"
if not exist "downloads\pydev.msi" goto :after_accel
msiexec /a "downloads\pydev.msi" /qn TARGETDIR="%SCRIPT_DIR%downloads\pydev_extract"
if not exist "python\Include" mkdir "python\Include"
if not exist "python\libs" mkdir "python\libs"
xcopy /E /Y "downloads\pydev_extract\include\*" "python\Include\" >nul 2>&1
xcopy /E /Y "downloads\pydev_extract\libs\*" "python\libs\" >nul 2>&1
if exist "downloads\pydev_extract" rmdir /s /q "downloads\pydev_extract"
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
python\python.exe -m pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --no-warn-script-location
echo Ставлю CUDA 12.4-рантайм для llama (совпадает со сборкой ggml-cuda)...
python\python.exe -m pip install nvidia-cuda-runtime-cu12==12.4.127 nvidia-cublas-cu12==12.4.5.8 --no-warn-script-location
echo Кладу 12.4 cudart/cublas рядом с llama.dll...
for %%P in (cuda_runtime cublas) do for %%D in (cudart64_12.dll cublas64_12.dll cublasLt64_12.dll) do if exist "python\Lib\site-packages\nvidia\%%P\bin\%%D" copy /y "python\Lib\site-packages\nvidia\%%P\bin\%%D" "python\Lib\site-packages\llama_cpp\lib\%%D" >nul
if not exist "python\Lib\site-packages\llama_cpp\lib\cublasLt64_12.dll" echo [ВНИМАНИЕ] cublasLt64_12.dll не скопирован - режиссёр может не стартовать!
goto :after_llama
:llama_cpu
python\python.exe -m pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --no-warn-script-location
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
