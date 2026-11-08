@echo off
REM Opens only the free Cloudflare https link to a ShelfSense server that is ALREADY running on port 8000.
REM (start_demo.bat starts both the server and the link.) Close this window to close the link.
set USERPROFILE=E:\tools\cloudflared\home
set HOME=E:\tools\cloudflared\home
if not exist "%USERPROFILE%" mkdir "%USERPROFILE%"
echo.
echo Look for the line with  https://....trycloudflare.com  below. On the phone open that address + /field/
echo PINs: E:\ShelfSense\data\demo_pins.txt
echo.
"E:\tools\cloudflared\cloudflared.exe" tunnel --no-autoupdate --url http://localhost:8000
pause
