@echo off
rem One click for the SLM solver's GPU: install the slm-gpu extra (CuPy and the
rem CUDA runtime) into .tools\science-env, then warm it on this machine --
rem compile the CUDA module and measure, per band count, whether cuBLAS or the
rem warp kernel is faster on this GPU, where the SLM plugin reads it. Run it again
rem after a GPU or driver change (the plugin also re-measures on first use);
rem close the SLM task console first. --dry-run prints the plan.
setlocal EnableExtensions DisableDelayedExpansion
call "%~dp0install_requirements.bat" slm-gpu %*
exit /b %ERRORLEVEL%
