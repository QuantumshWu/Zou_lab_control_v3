@echo off
rem Set up this checkout's science environment (.tools\science-env), the Python
rem the device and logic-node plugins run in, from science\pyproject.toml at
rem science\constraints.txt's versions, made from the Python v2 runs with (found
rem as v2's launchers find it: ZLC_PY_CMD, PATH, conda, python.org; 3.13 or newer)
rem and using its packages when it has them all. Changes no global Python, PATH or
rem other environment.
rem   install_requirements.bat            numpy, scipy, Pillow and the serial/VISA/USB links
rem   install_requirements.bat pylon      only Basler's pypylon, any version (~270 MB installed)
rem   install_requirements.bat slm-gpu    only CuPy/CUDA for the SLM solver (install_slm_gpu.bat)
rem   ... --dry-run                        print the plan, change nothing
setlocal EnableExtensions DisableDelayedExpansion

rem Before any shift: shift moves %0 too.
for %%I in ("%~dp0..\..") do set "ZLC_HOME=%%~fI"
set "ZLC_INSTALL_GROUP=runtime"
set "ZLC_INSTALL_DRY="
:zlc_arguments
if "%~1"=="" goto zlc_run
if /I "%~1"=="slm-gpu" (set "ZLC_INSTALL_GROUP=slm-gpu"& shift & goto zlc_arguments)
if /I "%~1"=="pylon" (set "ZLC_INSTALL_GROUP=pylon"& shift & goto zlc_arguments)
if /I "%~1"=="--dry-run" (set "ZLC_INSTALL_DRY=-DryRun"& shift & goto zlc_arguments)
echo Usage: install_requirements.bat [pylon ^| slm-gpu] [--dry-run]
if "%ZLC_NO_PAUSE%"=="" pause
exit /b 2

:zlc_run
echo.
echo ============================================================
echo ZLC v3 - install the science environment (%ZLC_INSTALL_GROUP%)
echo Checkout: %ZLC_HOME%
echo ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%ZLC_HOME%\tools\install_science.ps1" -Group %ZLC_INSTALL_GROUP% %ZLC_INSTALL_DRY%
set "ZLC_STATUS=%ERRORLEVEL%"
if not "%ZLC_STATUS%"=="0" (
  echo.
  echo Installation failed with code %ZLC_STATUS%. Read the messages above.
)
if "%ZLC_NO_PAUSE%"=="" pause
exit /b %ZLC_STATUS%
