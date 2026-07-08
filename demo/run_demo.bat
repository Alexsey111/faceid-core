@echo off
REM FaceID Core desktop-demo launcher (Windows). Double-click to run.
REM ASCII-only (cmd on ru-Win misreads UTF-8 in .bat). Unconditional pause at :end.
setlocal enableextensions
cd /d "%~dp0\.."
title FaceID Core - desktop demo launcher
echo === FaceID Core - desktop demo ===
echo.

REM --- 1. Detect Python: prefer one that ALREADY has deps (filters Store stub) ---
REM A Microsoft Store stub without a real install has no cv2 -> import fails ->
REM we fall through. Prefer py launcher (system-wide, not a Store stub) over
REM python (PATH), targeting 3.11 where the user's demo deps already live.
set "PY="
py -3.11 -c "import cv2,requests,websocket,PIL" >nul 2>nul
if not errorlevel 1 set "PY=py -3.11"
if not defined PY (
    python -c "import cv2,requests,websocket,PIL" >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    py -3 -c "import cv2,requests,websocket,PIL" >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)
if not defined PY (
    REM none has deps yet; pick any available interpreter, we will pip-install below
    py -3.11 --version >nul 2>nul
    if not errorlevel 1 set "PY=py -3.11"
)
if not defined PY (
    python --version >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    py -3 --version >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)
if not defined PY (
    echo [ERROR] Python not found. Install Python 3.10+ from https://python.org
    echo         and tick "Add Python to PATH" during install, then rerun.
    goto :end
)
echo [1/4] Python: %PY%
%PY% --version
echo.

REM --- 2. Version >= 3.10 ---
%PY% -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.10+ required. Current:
    %PY% --version
    goto :end
)
echo [2/4] Python version OK
echo.

REM --- 3. Demo dependencies (install if missing) ---
%PY% -c "import cv2,requests,websocket,PIL" >nul 2>nul
if errorlevel 1 (
    echo [3/4] Installing demo deps (opencv/requests/websocket-client/Pillow)...
    %PY% -m pip install -r demo\requirements-demo.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies. Check network connection.
        goto :end
    )
) else (
    echo [3/4] Dependencies OK
)
echo.

REM --- 4. Docker ---
where docker >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Docker not found. Install Docker Desktop from https://docker.com and add to PATH.
    goto :end
)
docker info >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Docker daemon is not running. Start Docker Desktop and retry.
    goto :end
)
echo [4/4] Docker OK
echo.
echo [INFO] Starting desktop demo... Application window opens separately.
echo [INFO] Do NOT close this console - Python logs are written here.
echo       Close the demo via the app window (Stop service button + close).
echo.
%PY% demo\desktop_demo.py > demo\_demo_stdout.log 2>&1
set "RC=%errorlevel%"
echo.
echo === Application exited (exit code %RC%) ===
if %RC% neq 0 (
    echo --- traceback from demo\_demo_stdout.log ---
    type demo\_demo_stdout.log
    echo --- end of log ---
)
echo.

:end
echo.
echo === Launcher finished. Press any key to close this window. ===
pause
endlocal