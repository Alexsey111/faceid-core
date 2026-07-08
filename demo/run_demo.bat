@echo off
chcp 65001 >nul
REM Лаунчер desktop-демо FaceID Core (Windows). Двойной клик запускает.
REM Проверяет Python/Docker/зависимости, поднимает demo-стек через приложение.
cd /d "%~dp0\.."
echo === FaceID Core — desktop demo ===
echo.

REM --- 1. Python ---
where python >nul 2>nul
if errorlevel 1 (
    echo [ОШИБКА] Python не найден. Установите Python 3.10+ с https://python.org и добавьте в PATH.
    pause
    exit /b 1
)

REM --- 2. Версия Python >=3.10 ---
python -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ОШИБКА] Требуется Python 3.10+. Текущая версия:
    python --version
    pause
    exit /b 1
)

REM --- 3. Зависимости демо ---
python -c "import cv2, requests, websocket, PIL" >nul 2>nul
if errorlevel 1 (
    echo [ИНФО] Устанавливаю зависимости демо (opencv/requests/websocket-client/Pillow)...
    python -m pip install -r demo\requirements-demo.txt
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось установить зависимости. Проверьте подключение к сети.
        pause
        exit /b 1
    )
)

REM --- 4. Docker ---
where docker >nul 2>nul
if errorlevel 1 (
    echo [ОШИБКА] Docker не найден. Установите Docker Desktop с https://docker.com и добавьте в PATH.
    pause
    exit /b 1
)
docker info >nul 2>nul
if errorlevel 1 (
    echo [ОШИБКА] Docker-демон не запущен. Запустите Docker Desktop и повторите.
    pause
    exit /b 1
)

REM --- 5. Запуск приложения ---
echo [ИНФО] Запуск desktop-демо... Окно откроется отдельно.
echo [ИНФО] Не закрывайте это консольное окно — туда пишутся логи Python.
echo.
python demo\desktop_demo.py
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Приложение завершилось с ошибкой (см. вывод выше).
    pause
)