@echo off
title ShieldAI - AI Review & Spam Detection Engine
color 0B

echo ============================================================
echo           STARTING SHIELDAI DETECTION ENGINE...
echo ============================================================
echo.

:: Move to the directory where this batch file is located
cd /d "%~dp0"

:: Start the Flask application (app.py opens the browser automatically)
python app.py

pause
