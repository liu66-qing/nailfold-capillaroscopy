@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
python scripts\windows_report_app.py
if errorlevel 1 pause
