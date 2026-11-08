@echo off
REM ShelfSense phone demo: starts the server (window 1) and a free Cloudflare https link (this window).
REM Phones open  https://<the link shown below>/field/  and log in with a PIN from data\demo_pins.txt.
REM The link works from anywhere while this window is open. Close both windows to stop.
cd /d "%~dp0"
if not exist "E:\tools\cloudflared\cloudflared.exe" (
  echo cloudflared not found in E:\tools\cloudflared. See CLAUDE.md, section "Mobile app".
  pause
  exit /b 1
)
set SEED=
if not exist "data\shelfsense.db" set SEED=--seed-demo
start "ShelfSense server" cmd /k ".venv\Scripts\python.exe -m src.server.app %SEED%"
REM cloudflared looks for settings in the user profile (C:): point it at E: instead
set USERPROFILE=E:\tools\cloudflared\home
set HOME=E:\tools\cloudflared\home
if not exist "%USERPROFILE%" mkdir "%USERPROFILE%"
echo.
echo Starting the secure link. Look for the line with  https://....trycloudflare.com  below,
echo then open that address + /field/ on the phone. PINs: data\demo_pins.txt
echo The server needs a minute to load the detector the first time.
echo.
"E:\tools\cloudflared\cloudflared.exe" tunnel --no-autoupdate --url http://localhost:8000
pause
