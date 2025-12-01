@echo off
REM Run Live Captions Tray Application (Development)
REM For testing without building the executable
cd /d "%~dp0.."

echo Starting Live Captions Tray...
echo.
echo - Double-click tray icon: Start/Stop with Whisper
echo - Right-click tray icon: Select backend or audio source
echo.

REM Use venv Python if available
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe live_captions_tray.py %*
) else if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" (
    "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" live_captions_tray.py %*
) else (
    python live_captions_tray.py %*
)
