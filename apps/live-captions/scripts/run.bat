@echo off
REM Run Live Captions (Caption Window Only)
cd /d "%~dp0.."

REM Use venv Python if available
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe live_captions.py %*
) else if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" (
    "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" live_captions.py %*
) else (
    python live_captions.py %*
)
