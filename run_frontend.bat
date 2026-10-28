@echo off
title Drug Interaction Frontend (Streamlit)
cd /d "%~dp0frontend"
echo ========================================================
echo  Starting Streamlit Web Application on port 8501
echo ========================================================
py -3.10 -m streamlit run app.py --server.port 8501
pause
