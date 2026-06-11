@echo off
echo Starting GhostTrace...
echo.
if not exist .env (
    echo WARNING: No .env file found. Copy .env.example to .env and add your ANTHROPIC_API_KEY.
    echo The demo (Harborview Capital) works without a key. Live traces require a key.
    echo.
)
uvicorn main:app --reload --host 127.0.0.1 --port 8006
pause
