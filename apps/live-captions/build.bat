@echo off
cd /d "%~dp0"

echo Building Live Captions executable...
echo.

REM Activate venv and run pyinstaller
.venv\Scripts\pyinstaller.exe --noconfirm --onefile --windowed ^
    --name "Live Captions" ^
    --icon "icon.ico" ^
    --add-data "icon.ico;." ^
    live_captions.py

echo.
echo Build complete! Executable is in: dist\Live Captions.exe
echo You can pin this to your taskbar.
pause
