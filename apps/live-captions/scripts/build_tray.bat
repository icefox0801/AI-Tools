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

REM Generate build timestamp file (format: YYYYMMDDHHmm)
echo.
echo Generating build timestamp...
for /f %%I in ('powershell -Command "Get-Date -Format 'yyyyMMddHHmm'"') do set BUILD_TIME=%%I
echo %BUILD_TIME%> .build_time
echo Build time: %BUILD_TIME%

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
    
    REM Update Windows Startup registry if auto-start was previously enabled
    echo Updating Windows Startup registry...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$regPath = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'; $regName = 'LiveCaptions'; try { $existing = Get-ItemProperty -Path $regPath -Name $regName -ErrorAction SilentlyContinue; if ($existing) { $exePath = '%cd%\dist\Live Captions.exe'; Set-ItemProperty -Path $regPath -Name $regName -Value $exePath; Write-Host 'Auto-start registry updated with new executable path' } else { Write-Host 'Auto-start not previously enabled, skipping registry update' } } catch { Write-Warning \"Failed to update registry: $_\" }"
    echo.
    echo Usage:
    echo   - Double-click to run
    echo   - Double-click tray icon to start/stop captions
    echo   - Right-click tray icon for options
    echo   - Use tray menu to enable/disable auto-start at login
    echo.
    dir "dist\Live Captions.exe"
) else (
    echo.
    echo Build failed! Check the output above for errors.
    exit /b 1
)
