@echo off
REM Build Live Captions Tray Application
REM Creates a single executable Windows app with system tray support
cd /d "%~dp0.."

echo ============================================
echo    Building Live Captions Tray App
echo ============================================
echo.

REM Kill any running instance of the tray app
echo Stopping any running Live Captions Tray...
taskkill /F /IM "Live Captions.exe" 2>nul
if %errorlevel%==0 (
    echo Stopped running instance.
    timeout /t 1 /nobreak >nul
) else (
    echo No running instance found.
)
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Clean previous build
echo Cleaning previous build...
if exist "dist\Live Captions.exe" del "dist\Live Captions.exe"
if exist "build\Live Captions" rmdir /s /q "build\Live Captions"

REM Build with PyInstaller
echo.
echo Building executable...
pyinstaller "Live Captions Tray.spec" --clean

REM Check result
if exist "dist\Live Captions.exe" (
    echo.
    echo ============================================
    echo    Build Successful!
    echo ============================================
    echo.
    echo Output: dist\Live Captions.exe
    echo.
    echo Usage:
    echo   - Double-click to run
    echo   - Double-click tray icon to start/stop captions
    echo   - Right-click tray icon for options
    echo.
    dir "dist\Live Captions.exe"
) else (
    echo.
    echo Build failed! Check the output above for errors.
    exit /b 1
)

echo.
pause
