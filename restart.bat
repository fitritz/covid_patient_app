@echo off
echo --- Stopping existing server processes ---

echo Stopping Flask backend (python.exe)...
taskkill /F /IM python.exe /T > nul

echo Stopping React frontend (node.exe)...
taskkill /F /IM node.exe /T > nul

echo.
echo --- Starting servers in new terminal windows ---
timeout /t 2 > nul

:: Get the directory of the batch file itself
set "BATCH_DIR=%~dp0"

:: Define paths relative to the batch file's location
set "VENV_PYTHON=%BATCH_DIR%.venv\Scripts\python.exe"
set "BACKEND_SCRIPT=%BATCH_DIR%webapp\backend\app.py"
set "FRONTEND_DIR=%BATCH_DIR%webapp\frontend"

echo Launching Flask Backend...
start "Flask Backend" cmd /k ""%VENV_PYTHON%" "%BACKEND_SCRIPT%""

echo Launching React Frontend...
start "React Frontend" cmd /k "cd /D ""%FRONTEND_DIR%"" && npm start"

echo.
echo --- Restart command issued. Check for new terminal windows. ---
