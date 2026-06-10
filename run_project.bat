@echo off
echo ========================================================
echo   KHOI CHAY HE THONG TOM TAT VAN BAN TIENG VIET
echo ========================================================
echo.

:: Path settings
set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

:: Start backend API in a new window
echo [*] Dang khoi chay Backend FastAPI (Port: 8000)...
start "Backend API - Vietnamese Summarizer" cmd /k "venv\Scripts\activate && python -m api.main"

:: Wait 2 seconds
timeout /t 2 /nobreak > nul

:: Start frontend dev server in a new window
echo [*] Dang khoi chay Frontend Vite (Port: 5173)...
cd frontend
start "Frontend UI - Vietnamese Summarizer" cmd /k "npm run dev"

echo.
echo ========================================================
echo [OK] Ca hai dich vu dang duoc khoi chay!
echo - API: http://localhost:8000
echo - Frontend: http://localhost:5173
echo ========================================================
pause
