@echo off
echo Starting Voice Controlled Local AI Agent...

:: Ensure output directory exists
mkdir output 2>nul
mkdir chroma_db 2>nul

echo Starting FastAPI Backend...
start cmd /k "uvicorn main:app --reload --port 8000"

echo Waiting for backend to initialize...
timeout /T 5 /NOBREAK >nul

echo Starting Streamlit Frontend...
start cmd /k "streamlit run frontend/streamlit_app.py"

echo Services have been started in new windows.
pause
