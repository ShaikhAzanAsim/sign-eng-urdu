@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
"C:\Users\QBS PC\PycharmProjects\ais-handler-template\.venv\Scripts\python.exe" %*
