@echo off
rem A development machine's toolchain into this checkout's .tools (tools\dev_setup.ps1):
rem portable Rust, lld-link, the MSVC SDK libraries, FreeType 2.6.1, Node.js and
rem TypeScript, and the crates of Cargo.lock. Lab PCs do not need it.
rem --dry-run prints what is present and what would be installed.
setlocal EnableExtensions DisableDelayedExpansion

set "ZLC_SETUP_ARGS="
if /I "%~1"=="--dry-run" set "ZLC_SETUP_ARGS=-DryRun"
if not "%~1"=="" if not defined ZLC_SETUP_ARGS (
  echo Usage: dev_setup.bat [--dry-run]
  if "%ZLC_NO_PAUSE%"=="" pause
  exit /b 2
)

for %%I in ("%~dp0..\..") do set "ZLC_HOME=%%~fI"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ZLC_HOME%\tools\dev_setup.ps1" %ZLC_SETUP_ARGS%
set "ZLC_STATUS=%ERRORLEVEL%"
if not "%ZLC_STATUS%"=="0" (
  echo.
  echo Development setup failed with code %ZLC_STATUS%. Read the messages above.
)
if "%ZLC_NO_PAUSE%"=="" pause
exit /b %ZLC_STATUS%
