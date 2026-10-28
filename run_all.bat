@echo off
title Launch Drug Interaction App (Full Stack)
echo ========================================================
echo  Starting Backend (FastAPI :8000) and Frontend (:8501)
echo ========================================================

start "Backend Server" /min cmd /c "cd /d "%~dp0backend" && py -3.10 -m uvicorn main:app --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak >nul

start "Streamlit Frontend" cmd /c "cd /d "%~dp0frontend" && py -3.10 -m streamlit run app.py --server.port 8501"

timeout /t 2 /nobreak >nul
start http://localhost:8501
