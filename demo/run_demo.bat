@echo off
REM FaceID Core desktop-demo launcher (Windows). Double-click to run.
REM Checks Python/Docker/deps, then starts demo/desktop_demo.py.
REM Unconditional pause at the end: window never closes by itself, traceback is visible.
setlocal enableextensions
cd /d "%~dp0\.."
title FaceID Core - desktop demo launcher
echo === FaceID Core - desktop demo ===
echo.

REM --- 1. Detect Python: python (PATH, where deps are installed) -> py -3 ---
REM Probe via --version (where is unreliable for App Paths / Store aliases here).
set "PY="
python --version >nul 2>nul
if not errorlevel 1 set "PY=python"
if not defined PY (
    py -3 --version >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)
if not defined PY (
    echo [ERROR] Python not found. Install Python 3.10+ from https://python.org
    echo         and tick "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo [1/4] Python: %PY%
%PY% --version
echo.

REM --- 2. Version >= 3.10 ---
%PY% -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.10+ required. Current:
    %PY% --version
    pause
    exit /b 1
)
echo [2/4] Python version OK
echo.

REM --- 3. Demo dependencies ---
%PY% -c "import cv2, requests, websocket, PIL" >nul 2>nul
if errorlevel 1 (
    echo [3/4] Installing demo deps (opencv/requests/websocket-client/Pillow)...
    %PY% -m pip install -r demo\requirements-demo.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies. Check network connection.
        pause
        exit /b 1
    )
) else (
    echo [3/4] Dependencies OK
)
echo.

REM --- 4. Docker ---
where docker >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Docker not found. Install Docker Desktop from https://docker.com and add to PATH.
    pause
    exit /b 1
)
docker info >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Docker daemon is not running. Start Docker Desktop and retry.
    pause
    exit /b 1
)
echo [4/4] Docker OK
echo.
echo [INFO] Starting desktop demo... Application window opens separately.
echo [INFO] Do NOT close this console - Python logs are written here.
echo       Close the demo via the app window (Stop service button + close).
echo.
%PY% demo\desktop_demo.py
set "RC=%errorlevel%"
echo.
echo === Application exited (exit code %RC%) ===
echo If a traceback (Traceback most recent call last) is shown above - send it in full.
echo If exit code is 0 - everything completed normally.
echo.
pause
endlocal