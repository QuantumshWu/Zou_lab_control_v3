@echo off
rem Build the UI, zlc-desktop, zlc-pulse-server and zlc-mcp from the committed
rem sources into bin\engine and bin\ai and their archive (target\engine-releases),
rem and commit bin\release.json, which names it, with the four programs. On a
rem development machine only (tools\dev_setup.ps1). --no-commit stops before the
rem commit; --publish (once the commit is pushed) makes the GitHub release that
rem lab PCs download, --publish --dry-run prints its command.
setlocal EnableExtensions DisableDelayedExpansion

set "ZLC_RELEASE_ARGS="
if /I "%~1"=="--no-commit" set "ZLC_RELEASE_ARGS=-NoCommit"
if /I "%~1"=="--publish" set "ZLC_RELEASE_ARGS=-Publish"
if /I "%~1"=="--publish" if /I "%~2"=="--dry-run" set "ZLC_RELEASE_ARGS=-Publish -DryRun"
if not "%~1"=="" if not defined ZLC_RELEASE_ARGS (
  echo Usage: build_release.bat [--no-commit ^| --publish [--dry-run]]
  if "%ZLC_NO_PAUSE%"=="" pause
  exit /b 2
)

for %%I in ("%~dp0..\..") do set "ZLC_HOME=%%~fI"
echo.
echo ============================================================
echo ZLC v3 - build the release
echo Checkout: %ZLC_HOME%
echo ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%ZLC_HOME%\tools\release.ps1" %ZLC_RELEASE_ARGS%
set "ZLC_STATUS=%ERRORLEVEL%"
echo.
if "%ZLC_STATUS%"=="0" (
  echo Done.
) else (
  echo build_release failed with code %ZLC_STATUS%. Read the messages above.
)
if "%ZLC_NO_PAUSE%"=="" pause
exit /b %ZLC_STATUS%
