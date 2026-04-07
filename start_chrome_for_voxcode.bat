@echo off
echo Starting Chrome with remote debugging for VOXCODE...
echo.

REM Close existing Chrome instances
echo Closing existing Chrome instances...
taskkill /f /im chrome.exe 2>nul

REM Wait a moment
timeout /t 2 /nobreak > nul

echo Starting Chrome with remote debugging on port 9222...
echo.

REM Start Chrome with remote debugging
start "Chrome for VOXCODE" "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222

echo Chrome started!
echo.
echo You can now:
echo 1. Use your Chrome normally (bookmarks, logins, etc.)
echo 2. Run VOXCODE and give browser commands
echo 3. VOXCODE will control your Chrome browser
echo.
echo Press any key to exit...
pause >nul