@echo off
echo ========================================================
echo   Pushing Flood-AI All New Changes to GitHub...
echo   Repository: https://github.com/Ronakjain935/flood-ai.git
echo ========================================================
echo.
cd /d "%~dp0"
git push origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] All 20 datasets, live radar, and CI fixes are now live on GitHub!
) else (
    echo [NOTICE] If prompted, please sign in with your GitHub account in your browser.
)
echo ========================================================
pause
