@echo off
rem Double-click to check whether the FPGA part configured in
rem crates\zlc-pulse\fpga\board_config\streamer_config.json has the resources
rem (LUT / FF / DSP / BRAM) for the configured pulse-streamer geometry. Edit that
rem JSON to change the part, row count, delay depth, etc., then run this again.
rem The window stays open with the report. ZLC_PS_CONFIG names another config.
setlocal EnableExtensions DisableDelayedExpansion

for %%I in ("%~dp0..\..") do set "ZLC_HOME=%%~fI"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ZLC_HOME%\tools\fpga.ps1" -Action estimate
set "ZLC_STATUS=%ERRORLEVEL%"
echo.
if "%ZLC_STATUS%"=="0" (
  echo ZLC resource estimate: within the configured budget. Build and timing still require verification.
) else if "%ZLC_STATUS%"=="1" (
  echo ZLC resource estimate: INSUFFICIENT -- see the OVER BUDGET lines above.
) else (
  echo ZLC resource estimate failed with code %ZLC_STATUS% -- read the messages above.
)
echo You can close this window, or press any key to exit.
if "%ZLC_NO_PAUSE%"=="" pause
exit /b %ZLC_STATUS%
