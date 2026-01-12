@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   Mistral Indian Law - Full Stack
echo ========================================
echo.

REM Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    set "VENV_PATH=.venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    set "VENV_PATH=venv\Scripts\activate.bat"
) else (
    echo ERROR: Virtual environment not found!
    echo Please create a virtual environment first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Get IP address for network access
set "IP="
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    if "!IP!"=="" (
        set "IP=%%a"
        set "IP=!IP: =!"
    )
)

echo Starting Backend and Frontend...
echo.
echo Backend Configuration:
echo   - Host: 0.0.0.0 (network accessible)
echo   - Port: 2347
echo   - Max GPU Memory: 5.0 GB (for 6GB VRAM)
echo   - Virtual Environment: %VENV_PATH%
echo.

REM Start backend in new window with venv activation
start "Mistral Indian Law - Backend" cmd /k "cd /d %~dp0 && call %VENV_PATH% && python -m uvicorn backend.main:app --host 0.0.0.0 --port 2347 --reload"

REM Wait a bit for backend to start
timeout /t 3 /nobreak >nul

REM Check if frontend node_modules exists
if not exist "frontend\node_modules" (
    echo WARNING: Frontend node_modules not found!
    echo Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
    echo.
)

REM Start frontend in new window
start "Mistral Indian Law - Frontend" cmd /k "cd /d %~dp0\frontend && npm run dev"

echo.
echo ========================================
echo   Servers Started
echo ========================================
echo.
echo Local Access (same computer):
echo   Backend: http://localhost:2347
echo   Frontend: http://localhost:5173
echo   Backend API Docs: http://localhost:2347/docs
echo.
if not "!IP!"=="" (
    echo Network Access (other devices on WiFi):
    echo   Backend: http://!IP!:2347
    echo   Frontend: http://!IP!:5173
    echo   Backend API Docs: http://!IP!:2347/docs
    echo.
    echo NOTE: Make sure frontend/.env has:
    echo   VITE_API_URL=http://!IP!:2347/chat
    echo   (Creating it now if it doesn't exist...)
    echo.
    if not exist "frontend\.env" (
        echo VITE_API_URL=http://!IP!:2347/chat > frontend\.env
        echo Created frontend/.env file
    )
) else (
    echo Network Access: Could not detect IP address
    echo.
)
echo.
echo ========================================
echo   Important Notes
echo ========================================
echo.
echo - Model loading may take 3-5 minutes on first startup
echo - Ensure you have at least 4GB free RAM before starting
echo - GPU memory is limited to 5GB (for 6GB VRAM systems)
echo - Check backend window for loading progress
echo.
echo Both servers are running in separate windows.
echo Close those windows to stop the servers.
echo.
echo To adjust memory settings, set environment variables:
echo   set MAX_MEMORY_GB=4.5  (reduce if OOM errors)
echo   set DEVICE_MAP=cpu     (use CPU only if GPU issues)
echo.
pause
endlocal

