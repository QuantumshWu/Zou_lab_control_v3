@echo off
rem Devices@Zou lab opens first; Init opens the Task Console, as v2's experiment.bat.
setlocal DisableDelayedExpansion
set "ZLC_MODE=experiment"
call "%~dp0_launch.bat" %*
exit /b %ERRORLEVEL%