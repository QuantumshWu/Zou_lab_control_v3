@echo off
rem Warm the SLM rearrangement GPU code on this machine, as v2's warm_slm_gpu,
rem without reinstalling: compile the CUDA module (CuPy keeps the binary) and
rem measure, for every band count, whether cuBLAS or a warp kernel computes the
rem analysis product faster on this GPU (tensor cores make cuBLAS faster; the
rem lab's T400 has none), where the SLM plugin reads it. install_slm_gpu.bat
rem does this after installing; run this after a GPU, driver or solver change
rem (the plugin also re-measures on first use, once slower). Close the SLM task
rem console first.
setlocal EnableExtensions DisableDelayedExpansion

for %%I in ("%~dp0..\..") do set "ZLC_HOME=%%~fI"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ZLC_HOME%\tools\warm_slm_gpu.ps1"
set "ZLC_RC=%ERRORLEVEL%"
if not "%ZLC_RC%"=="0" (
  echo.
  echo SLM GPU warmup failed -- see the message above.
)
if "%ZLC_NO_PAUSE%"=="" pause
exit /b %ZLC_RC%
