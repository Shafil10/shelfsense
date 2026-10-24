@echo off
REM Starts the ShelfSense dashboard on this laptop (works without internet).
REM Laptop: http://localhost:8000   Other devices on the same Wi-Fi/hotspot: http://<this laptop's IPv4>:8000
cd /d "%~dp0"
echo.
echo ShelfSense dashboard is running. Keep this window open. Close it to stop.
echo Laptop:        http://localhost:8000
echo Other devices: use one of these addresses with :8000
ipconfig | findstr /c:"IPv4"
echo.
start "" http://localhost:8000
python -m http.server 8000 --bind 0.0.0.0 --directory app\mockup\site
pause
