@echo off
rem Build and/or program the pulse board's FPGA from crates\zlc-pulse\fpga, as
rem v2's build_and_program.bat (tools\fpga.ps1 does the work).
setlocal EnableExtensions DisableDelayedExpansion

set "ZLC_ACTION="
set "ZLC_FPGA_ARGS="
if "%~1"=="" (set "ZLC_ACTION=build + program"& set "ZLC_FPGA_ARGS=-Action build_program")
if /I "%~1"=="--force-build" (set "ZLC_ACTION=build + program"& set "ZLC_FPGA_ARGS=-Action build_program -ForceBuild")
if /I "%~1"=="--build-only" (set "ZLC_ACTION=build"& set "ZLC_FPGA_ARGS=-Action build")
if /I "%~1"=="--program-only" (set "ZLC_ACTION=program"& set "ZLC_FPGA_ARGS=-Action program")
if /I "%~1"=="--flash" (set "ZLC_ACTION=program SPI flash (persists across power cycle)"& set "ZLC_FPGA_ARGS=-Action flash")
if /I "%~1"=="--diagnose" (set "ZLC_ACTION=hardware diagnose"& set "ZLC_FPGA_ARGS=-Action diagnose")
if "%~1"=="--help" goto zlc_help
if "%~1"=="/?" goto zlc_help
if not defined ZLC_ACTION (
  echo Unknown option: %~1
  echo.
  call :zlc_usage
  if "%ZLC_NO_PAUSE%"=="" pause
  exit /b 2
)

for %%I in ("%~dp0..\..") do set "ZLC_HOME=%%~fI"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ZLC_HOME%\tools\fpga.ps1" %ZLC_FPGA_ARGS%
set "ZLC_STATUS=%ERRORLEVEL%"
echo.
rem Outside a parenthesised block: the action's own words hold parentheses.
if not "%ZLC_STATUS%"=="0" goto zlc_failed
echo ZLC %ZLC_ACTION% completed successfully.
echo You can close this window, or press any key to exit.
goto zlc_end
:zlc_failed
echo ZLC %ZLC_ACTION% failed with code %ZLC_STATUS%.
echo Keep this window open and read the messages above.
:zlc_end
if "%ZLC_NO_PAUSE%"=="" pause
exit /b %ZLC_STATUS%

:zlc_help
call :zlc_usage
exit /b 0

:zlc_usage
echo Build/program the FINAL ZLC FPGA pulse streamer (one clean design, no variants).
echo Control path: JTAG-to-AXI master -^> AXI BRAM controller -^> row/scan BRAMs + loop/delay registers.
echo Engine: 1-tick (20 ns) FIFO prefetch + streamed autonomous ping-pong scan.
echo.
echo Usage:
echo   bin\scripts\build_and_program.bat               Build if needed, then PROGRAM volatile FPGA
echo   bin\scripts\build_and_program.bat --force-build Rebuild, then PROGRAM volatile FPGA
echo   bin\scripts\build_and_program.bat --build-only  Build only
echo   bin\scripts\build_and_program.bat --program-only Program existing bitstream (VOLATILE: lost on power-off)
echo   bin\scripts\build_and_program.bat --flash       Program the SPI flash so the program SURVIVES a power cycle
echo   bin\scripts\build_and_program.bat --diagnose    List Vivado hw targets/devices
echo.
echo --program-only loads the VOLATILE FPGA config (lost when the board powers off).  --flash writes
echo the bitstream into the board's SPI configuration flash so the FPGA AUTO-BOOTS from it on every
echo power-up -- run it ONCE and the program persists across reboots (no rebuild needed).  Requires the
echo board MODE jumper on SPI/QSPI boot; set ZLC_PS_CFGMEM_PART if your flash differs from the default
echo (run 'get_cfgmem_parts' in Vivado to list valid names).
echo.
echo The default mode SKIPS the (slow) synth+impl only when the Vivado build, target part,
echo engine/top HDL, create tcl, board XDC, streamer_config and generated geometry all match,
echo and the bitstream plus routed reports match their saved receipt; it then PROGRAMS that bit.
echo --force-build forces a rebuild before programming.  The qualified build receipt
echo is crates\zlc-pulse\fpga\build\ps\.zlc_src_hash.
echo.
echo Real build XDC:
echo   crates\zlc-pulse\fpga\board_config\board.xdc
echo   The physical pin map is defined by streamer_config.json ^(see board_config\README.md^).
echo   For a different board, replace board.xdc or set:
echo   set ZLC_PS_XDC=C:\path\to\board.xdc
echo.
echo Optional:
echo   set ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\2019.2\bin\vivado.bat
exit /b 0
