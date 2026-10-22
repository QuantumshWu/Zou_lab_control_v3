@echo off
rem PulseGUI@Zou lab alone, as v2's pulse_editor.bat; the connection stays the operator's choice.
setlocal DisableDelayedExpansion
set "ZLC_MODE=pulse_editor"
call "%~dp0_launch.bat" %*
exit /b %ERRORLEVEL%