@echo off
rem FigureViewer@Zou lab alone, as v2's figure_viewer.bat; a Figure archive dropped on it opens.
setlocal DisableDelayedExpansion
set "ZLC_MODE=figure_viewer"
call "%~dp0_launch.bat" %*
exit /b %ERRORLEVEL%