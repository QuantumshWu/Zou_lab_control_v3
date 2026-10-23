@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\..\tools\ui.ps1"
set "ZLC_RESULT=%ERRORLEVEL%"
if not "%ZLC_RESULT%"=="0" if not "%ZLC_NO_PAUSE%"=="1" pause
exit /b %ZLC_RESULT%
