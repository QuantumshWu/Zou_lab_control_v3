# FPGA tree

The pulse board's hardware sources, as v2 built them: the frozen RTL and its
simulation benches (`pulse_streamer/`), the Vivado build/program Tcl, and the
board configuration (`board_config/`). The host side that talks to this RTL --
the BRAM image, the register mailbox, the transports and the pulse server -- is
the `zlc-pulse` crate around this folder.

## Build and program

`installation.exe`'s FPGA card runs these, as `bin\scripts\build_and_program.bat` and
`bin\scripts\estimate_resources.bat` do with v2's options:

```text
bin\scripts\build_and_program.bat                 build if the sources changed, then program (volatile)
bin\scripts\build_and_program.bat --force-build   rebuild, then program
bin\scripts\build_and_program.bat --build-only    build only; never touches the board
bin\scripts\build_and_program.bat --program-only  program the existing bitstream (lost on power-off)
bin\scripts\build_and_program.bat --flash         write the bitstream into the SPI flash (survives power-off)
bin\scripts\build_and_program.bat --diagnose      list the Vivado hardware targets and devices
bin\scripts\estimate_resources.bat                LUT/FF/DSP/BRAM estimate of the configured geometry
```

Building is not part of normal experiment work: the board runs the frozen,
qualified bitstream, and a rebuild is for an established RTL or deployment
defect. Every action first checks `board_config/streamer_config.json` with
nothing forgiven, so a build never uses a geometry the operator did not write.
A build regenerates `pulse_streamer/zlc_geometry.vh` and the build's `geom.tcl`
from that file and stops when the estimate is over the config's `target_pct`.
The default mode skips synthesis when the Vivado version, part, top, HDL,
create Tcl, XDC, config and geometry all match the receipt saved with the last
build (`build\ps\.zlc_src_hash`) and the bitstream and routed reports are
unchanged since; it then programs that bitstream. installation.exe's FPGA card
reads the same receipt to say whether the bitstream matches the sources, by the
Vivado identity the build kept beside it (`.zlc_vivado_id`; a build from before
it was kept is compared by `vivado -version`, as a build compares).

The build lives in `build\` (ignored by git). The short project name `ps`
keeps Vivado's debug-core temp path under the Windows MAX_PATH limit; check the
repository out at a short path if Vivado still reports it too long, or set
`ZLC_PS_BUILD_ROOT`. Inside a synced folder the build root is marked ignored
for Dropbox. Vivado is `ZLC_PS_VIVADO_BIN`, else the newest release under
`C:` or `D:` `\Xilinx\Vivado` or `\AMD\Vivado`, else `vivado.bat` on PATH --
the same rule the pulse server's JTAG transport uses. Other overrides:
`ZLC_PS_XDC` (another board's pin map), `ZLC_PS_FPGA_PART`,
`ZLC_PS_CFGMEM_PART` / `ZLC_PS_CFGMEM_SIZE_MB` (the SPI flash for `--flash`),
`ZLC_PS_HW_SERVER_URL`, and `ZLC_NO_PAUSE=1` for automation.

Every RTL, Tcl, XDC and config file here is v2's, byte for byte (line endings
aside). v2's `sim/diff` harness is not here: it compiled and packed pulses with
each compared checkout's own Python `zlc_pulse`, which v3 does not have; the
host compiler is covered by the pulse crate's goldens against v2.

## Hardware acceptance runbook

Software and RTL simulation do not program or flash a board. A human hardware
acceptance uses this order and keeps one receipt:

1. Record the checkout commit, `streamer_config.json` path, `board.id`, FPGA
   part, 50 MHz constraint and the printed build/project directories.
2. Run `--build-only`. Keep the timing status and the exact `.bit` and `.ltx`
   paths; building never programs or flashes.
3. Run `--diagnose` and require exactly one target, device and JTAG AXI, its
   part the configured one, before an explicit program.
4. After programming, read word 63 and require the expected layout fingerprint
   (`ZLC_LAYOUT_FINGERPRINT` in `pulse_streamer/zlc_geometry.vh`), then start
   `zlc-pulse-server --check-config` and record the geometry, clock and lane count.
5. With instruments disconnected or otherwise safe, verify reset/SAFE pin
   levels and clock gates, one finite pulse with a delayed DAC tail, DONE only
   after the physical drain, a loud sticky underflow/engine overflow, and the
   separate non-fatal UART LINK_ERROR for recovered CRC/framing faults.
6. Verify that truncated or invalid UART frames, a client disconnect, a failed
   SAFE and a second client's takeover all leave no owner until a stable
   physical SAFE.
7. Flash only as a separate, explicitly requested action after every volatile
   programming check passes.

The receipt records operator and time, checkout commit, config path and
`board.id`, part, Vivado version, build/project/bit/ltx paths, the selected
target/device/transport, the word-63 fingerprint, SAFE status and clock-word
readback, the timing result, each check above as pass/fail with its log, and
whether the image was programmed, flashed or not written. A manifest ID or a
layout fingerprint alone is not a board/build receipt.
