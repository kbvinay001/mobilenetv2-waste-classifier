@echo off
title GarbageSort AI -- Smart Waste Classification
color 0A

echo.
echo  +=======================================================+
echo  ^|        GarbageSort AI  ^|  v1.0                       ^|
echo  ^|   Smart Waste Classification ^| Edge-Ready AI         ^|
echo  +=======================================================+
echo.

:: Move to project root
cd /d "%~dp0"

:: Use the project venv if present, else fall back to py -3.12
if exist "venv\Scripts\python.exe" (
    echo  [INFO]  Using project venv (Python 3.12)...
    set PYTHON="%~dp0venv\Scripts\python.exe"
    set PIP="%~dp0venv\Scripts\pip.exe"
    set STREAMLIT="%~dp0venv\Scripts\streamlit.exe"
) else (
    echo  [INFO]  No venv found. Creating one with Python 3.12...
    py -3.12 -m venv venv
    set PYTHON="%~dp0venv\Scripts\python.exe"
    set PIP="%~dp0venv\Scripts\pip.exe"
    set STREAMLIT="%~dp0venv\Scripts\streamlit.exe"
)

:: Install dependencies if streamlit not found
%PYTHON% -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [INFO]  Installing dependencies (first-time setup, may take a few minutes)...
    %PIP% install tensorflow streamlit fpdf2 "qrcode[pil]" psutil plotly pandas opencv-python Pillow scikit-learn matplotlib seaborn
    if errorlevel 1 (
        echo  [ERROR] Install failed. Run: venv\Scripts\pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo  [OK]    Dependencies installed.
)

:: Check model exists
if not exist "models\transfer_learning_best.h5" (
    echo.
    echo  [WARN]  Model not found: models\transfer_learning_best.h5
    echo          Train the model first: venv\Scripts\python scripts\transfer_learning_model.py
    pause
    exit /b 1
)

:: Launch
echo.
echo  [OK]    Model ready. Starting Streamlit dashboard...
echo  [OK]    Open: http://localhost:8501
echo.
echo  Press Ctrl+C here to stop.
echo.

start "" http://localhost:8501
%STREAMLIT% run app\main.py --server.port 8501 --server.headless false

pause
