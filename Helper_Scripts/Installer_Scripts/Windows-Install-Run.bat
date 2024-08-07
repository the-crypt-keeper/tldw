@echo off
setlocal enabledelayedexpansion

:: TLDW Windows Installer and Launcher Script

:: Check if TLDW is already installed
if exist "tldw\venv\Scripts\activate.bat" (
    goto :launch_tldw
)

:: Check for Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please download and install Python from https://www.python.org/downloads/windows/
    echo After installation, run this script again.
    pause
    exit /b 1
)

:: Check for curl installation
curl --version >nul 2>&1
if %errorlevel% neq 0 (
    echo curl is not installed or not in PATH.
    echo Please ensure you're using Windows 10 version 1803 or later, or install curl manually.
    echo After installation, run this script again.
    pause
    exit /b 1
)

:: Prompt for GPU installation
set /p gpu_support="Do you want to install with GPU support? (y/n): "
if /i "%gpu_support%"=="y" (
    set /p gpu_type="Choose GPU type (1 for CUDA, 2 for AMD): "
    if "!gpu_type!"=="1" (
        echo Please ensure your NVIDIA GPU drivers and CUDA are up to date.
        echo Visit https://developer.nvidia.com/cuda-downloads for instructions.
    ) else if "!gpu_type!"=="2" (
        echo Please ensure your AMD GPU drivers are up to date.
    ) else (
        echo Invalid choice. Defaulting to CPU installation.
        set gpu_support=n
    )
)

:: Download the latest release
echo Downloading the latest TLDW release...
curl -L -o tldw.zip https://github.com/rmusser01/tldw/archive/refs/heads/main.zip

:: Extract the downloaded zip file
echo Extracting files...
powershell -Command "Expand-Archive -Path 'tldw.zip' -DestinationPath '.' -Force"
move tldw-main tldw
cd tldw

:: Create and activate virtual environment
python -m venv .\venv
call .\venv\Scripts\activate.bat

:: Upgrade pip and install wheel
python -m pip install --upgrade pip wheel

:: Install PyTorch based on GPU support choice
if /i "%gpu_support%"=="y" (
    if "!gpu_type!"=="1" (
        pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
    ) else if "!gpu_type!"=="2" (
        pip install torch-directml
    )
) else (
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
    :: Update config.txt to use CPU
    powershell -Command "(gc config.txt) -replace 'cuda', 'cpu' | Out-File -encoding ASCII config.txt"
)

:: Install other requirements
pip install -r requirements.txt

:: Install ffmpeg
echo Installing ffmpeg...
curl -L -o ffmpeg.zip https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
powershell -Command "Expand-Archive -Path 'ffmpeg.zip' -DestinationPath 'ffmpeg' -Force"
move ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe .
rmdir /s /q ffmpeg
del ffmpeg.zip

:: Clean up
del ..\tldw.zip

echo Installation completed successfully!
echo Creating desktop shortcut...
powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%userprofile%\Desktop\TLDW.lnk'); $Shortcut.TargetPath = '%~f0'; $Shortcut.Save()"

timeout /T 5

:launch_tldw
echo Launching TLDW...
cd /d "%~dp0tldw"
call .\venv\Scripts\activate.bat
python summarize.py -gui -log INFO
exit /b 0