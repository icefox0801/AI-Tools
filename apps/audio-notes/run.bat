@echo off
REM Audio Notes - Run the Gradio web UI
cd /d "%~dp0"
..\live-captions\.venv\Scripts\python.exe audio_notes.py %*
