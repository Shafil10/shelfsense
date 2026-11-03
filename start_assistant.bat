@echo off
REM Starts the free local AI model for the dashboard's "Ask about this audit" panel (Ollama, no internet needed).
REM Everything stays on E: (program in E:\tools\ollama, models in E:\ollama\models). Keep this window open.
REM First time only, in another window:  E:\tools\ollama\ollama.exe pull gemma3:4b
set OLLAMA_MODELS=E:\ollama\models
set OLLAMA_HOST=127.0.0.1:11434
REM Ollama keeps its key and history in the user profile (C:): point it at E: instead
set USERPROFILE=E:\ollama\home
set HOME=E:\ollama\home
if not exist "E:\tools\ollama\ollama.exe" (
  echo Ollama not found in E:\tools\ollama. See CLAUDE.md, section "AI assistant".
  pause
  exit /b 1
)
echo.
echo ShelfSense AI assistant is running on http://localhost:11434. Keep this window open. Close it to stop.
echo Then start the dashboard with start_dashboard.bat and open an audit.
echo.
"E:\tools\ollama\ollama.exe" serve
pause
