@echo off
cd /d "%~dp0"
call venv-web\Scripts\activate.bat
python app.py
pause