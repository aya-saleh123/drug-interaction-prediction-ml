@echo off
title Drug Interaction API Server
cd /d "%~dp0backend"
echo ========================================================
echo  Starting Drug Interaction FastAPI Server on port 8000
echo ========================================================
py -3.10 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
