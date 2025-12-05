@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   Mistral Indian Law - Full Stack
echo ========================================
echo.

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

REM Start backend in new window
start "Mistral Indian Law - Backend" cmd /k "cd /d %~dp0 && python -m uvicorn backend.main:app --host 0.0.0.0 --port 2347 --reload"

REM Wait a bit for backend to start
timeout /t 3 /nobreak >nul

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
echo Both servers are running in separate windows.
echo Close those windows to stop the servers.
echo.
pause
endlocal

