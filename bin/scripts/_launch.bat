@echo off
rem Shared launcher: the wrapper sets ZLC_MODE and this starts the v3 desktop
rem through tools\desktop.ps1: a build current for this checkout's sources, else
rem the release in bin\engine, else a build when a toolchain is here.
rem Once the app is started this console closes; it stays, paused, only when the
rem start itself fails (no executable, a failed build).
setlocal EnableExtensions DisableDelayedExpansion

if not defined ZLC_MODE (
  echo ZLC launcher error: wrapper did not select a mode.
  exit /b 2
)

for %%I in ("%~dp0..\..") do set "ZLC_HOME=%%~fI"
echo.
echo ============================================================
echo ZLC v3 - %ZLC_MODE%
echo Checkout: %ZLC_HOME%
echo ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%ZLC_HOME%\tools\desktop.ps1" -Mode %ZLC_MODE% %*
set "ZLC_STATUS=%ERRORLEVEL%"
if not "%ZLC_STATUS%"=="0" (
  echo.
  echo %ZLC_MODE% did not start ^(code %ZLC_STATUS%^).
  if "%ZLC_NO_PAUSE%"=="" pause
)
exit /b %ZLC_STATUS%