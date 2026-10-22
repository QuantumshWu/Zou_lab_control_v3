param(
    # status: installation.exe's check (one JSON line), nothing run or written.
    [ValidateSet('build_program', 'build', 'program', 'flash', 'diagnose', 'estimate', 'status')][string]$Action = 'build_program',
    [switch]$ForceBuild
)
# v2's build_and_program.bat / estimate_resources.bat, step for step, over this
# checkout's FPGA tree (crates\zlc-pulse\fpga). What v2 asked its Python for --
# the strict config check, the two geometry projections, the capacity estimate --
# is `zlc-pulse-server fpga` (the pulse crate owns that config), and so is the
# Vivado rule (`fpga --vivado`, the JTAG transport's). The Vivado runs, their environment and the build
# receipt are v2's. Exit codes are v2's: 0 success, 1 a refusal, 2 a missing
# source; a failed Vivado run passes its own code on.
$ErrorActionPreference = 'Stop'
trap { Write-Host "ERROR: $($_.Exception.Message)"; exit 1 }
. (Join-Path $PSScriptRoot 'tools_root.ps1')
$root = Get-ZlcCheckout
$fpga = Join-Path $root 'crates\zlc-pulse\fpga'
$streamer = Join-Path $fpga 'pulse_streamer'
$top = 'zlc_pulse_streamer_top'
$createTcl = 'create_project.tcl'
# Its `fpga` command; installation.exe names the one beside it (ZLC_PULSE_SERVER).
$tool = if ($env:ZLC_PULSE_SERVER) { $env:ZLC_PULSE_SERVER } else { Join-Path (Get-ZlcProducts) 'zlc-pulse-server.exe' }
function Stop-Build([string[]]$Lines, [int]$Code = 1) { $Lines | ForEach-Object { Write-Host $_ }; exit $Code }

if ($Action -eq 'estimate') {
    if (-not $env:ZLC_PS_CONFIG) { $env:ZLC_PS_CONFIG = Join-Path $fpga 'board_config\streamer_config.json' }
    Write-Host "Reading config: $($env:ZLC_PS_CONFIG)"
    Write-Host ''
    Invoke-Program { & $tool fpga --config $env:ZLC_PS_CONFIG }
    exit $LASTEXITCODE
}

# Vivado: ZLC_PS_VIVADO_BIN, else the newest release under the install roots, else vivado.bat / vivado on PATH.
function Find-Vivado {
    if ($env:ZLC_PS_VIVADO_BIN) { return $env:ZLC_PS_VIVADO_BIN }
    $found = "$(& $tool fpga --vivado | Select-Object -First 1)".Trim()
    if ($found -and (Test-Path -LiteralPath $found)) { return $found }
    if (Get-Command vivado.bat -ErrorAction SilentlyContinue) { return (Get-Command vivado.bat).Source }
    if (Get-Command vivado -ErrorAction SilentlyContinue) { return 'vivado' }
    $null
}

# :zlc_default_paths -- the build tree is ZLC_PS_BUILD_ROOT or fpga\build; the project its short 'ps' child (MAX_PATH).
foreach ($name in 'ZLC_PS_BUILD_ROOT', 'ZLC_PS_PROJECT_DIR', 'ZLC_PS_LOG_DIR') {
    $value = [Environment]::GetEnvironmentVariable($name, 'Process')
    if ($value -and -not $value.Trim()) { [Environment]::SetEnvironmentVariable($name, $null, 'Process') }
}
if (-not $env:ZLC_PS_BUILD_ROOT) { $env:ZLC_PS_BUILD_ROOT = Join-Path $fpga 'build' }
$buildRoot = $env:ZLC_PS_BUILD_ROOT
$env:ZLC_PS_PROJECT_ROOT = $buildRoot
if (-not $env:ZLC_PS_PROJECT_DIR) { $env:ZLC_PS_PROJECT_DIR = Join-Path $buildRoot 'ps' }
if (-not $env:ZLC_PS_LOG_DIR) { $env:ZLC_PS_LOG_DIR = Join-Path $buildRoot 'logs' }
$projectDir = $env:ZLC_PS_PROJECT_DIR
$impl = Join-Path $projectDir 'ps.runs\impl_1'
$bit = Join-Path $impl "$top.bit"
$reports = @('timing_summary', 'bus_skew', 'utilization') | ForEach-Object { Join-Path $impl "${top}_${_}_routed.rpt" }
$config = Join-Path $fpga 'board_config\streamer_config.json'
# The bitstream's inputs: the HDL, the geometry header a build derives from the config, the create Tcl, the XDC.
if (-not $env:ZLC_PS_XDC) { $env:ZLC_PS_XDC = Join-Path $fpga 'board_config\board.xdc' }
$xdc = $env:ZLC_PS_XDC
$engine = Join-Path $streamer 'zlc_period_streamer.v'
$topFile = Join-Path $streamer "$top.v"
$create = Join-Path $streamer $createTcl
$header = Join-Path $streamer 'zlc_geometry.vh'

# The build receipt, byte for byte v2's: sha256 of 'tool\0' + Vivado's -version output +
# 'part\0' + part + 'top\0' + top + each input's checkout-relative path, '\0' and bytes; then
# one over the bitstream and routed reports relative to the project; "<sources>:<artifacts>".
# program_fpga.tcl is not an input: it loads a bit and cannot change one.
function Get-RelativePath([string]$Path, [string]$Start) {
    # os.path.relpath on Windows, with '/' between the parts.
    $pathParts = @([IO.Path]::GetFullPath($Path) -split '\\' | Where-Object { $_ })
    $startParts = @([IO.Path]::GetFullPath($Start) -split '\\' | Where-Object { $_ })
    if ($pathParts[0] -ne $startParts[0]) { return $null }
    $common = 1
    while ($common -lt $pathParts.Count -and $common -lt $startParts.Count -and $pathParts[$common] -eq $startParts[$common]) { $common++ }
    $relative = @(@('..') * ($startParts.Count - $common)) + @($pathParts | Select-Object -Skip $common)
    if ($relative.Count) { $relative -join '/' } else { '.' }
}
function Get-Digest([byte[]]$Head, [string[]]$Files, [string]$Base) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $add = { param([byte[]]$bytes) [void]$sha.TransformBlock($bytes, 0, $bytes.Length, $null, 0) }
        & $add $Head
        foreach ($file in $Files) {
            if (-not $file -or -not (Test-Path -LiteralPath $file -PathType Leaf)) { return $null }
            $relative = Get-RelativePath $file $Base
            if ($null -eq $relative) { return $null }
            & $add ([Text.Encoding]::UTF8.GetBytes($relative) + [byte[]]@(0))
            & $add ([IO.File]::ReadAllBytes($file))
        }
        [void]$sha.TransformFinalBlock([byte[]]@(), 0, 0)
        -join ($sha.Hash | ForEach-Object { $_.ToString('x2') })
    }
    finally { $sha.Dispose() }
}
# Vivado's build identity, its -version output (run from the build root); -Saved: the one the
# last build's receipt was made with, kept beside it (else run, for a build from before it was kept).
$identityFile = Join-Path $projectDir '.zlc_vivado_id'
function Get-VivadoIdentity([switch]$Saved) {
    if ($Saved -and (Test-Path -LiteralPath $identityFile)) { return ,[IO.File]::ReadAllBytes($identityFile) }
    if (-not $vivado) { return $null }
    $identity = Join-Path $buildRoot '.zlc_vivado_id.tmp'
    Push-Location -LiteralPath $buildRoot
    try { Invoke-Program { & cmd.exe /d /c call $vivado -version '>' $identity '2>&1' } }
    finally { Pop-Location }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $identity)) {
        Write-Host 'ERROR: could not read the selected Vivado build identity.'
        Remove-Item -LiteralPath $identity -ErrorAction SilentlyContinue
        return $null
    }
    $bytes = [IO.File]::ReadAllBytes($identity)
    Remove-Item -LiteralPath $identity
    ,$bytes
}
function Get-SourceDigest([byte[]]$Identity) {
    $text = [Text.Encoding]::UTF8
    $head = $text.GetBytes("tool`0") + $Identity + $text.GetBytes("part`0$($env:ZLC_PS_FPGA_PART)") + $text.GetBytes("top`0$top")
    $geom = if ($env:ZLC_PS_GEOM_TCL -and (Test-Path -LiteralPath $env:ZLC_PS_GEOM_TCL)) { $env:ZLC_PS_GEOM_TCL } else { '' }
    Get-Digest $head @($engine, (Join-Path $streamer 'zlc_uart_bridge.v'), $topFile, $header, $create, $xdc, $config, $geom) $root
}
function Get-ArtifactDigest { Get-Digest ([byte[]]@()) (@($bit) + $reports) $projectDir }

# :zlc_check_prebuilt -- whether the last build's receipt still names these inputs and its
# outputs: $null when the inputs cannot be hashed (no Vivado identity, an input missing).
$receiptFile = Join-Path $projectDir '.zlc_src_hash'
function Test-Prebuilt([byte[]]$Identity) {
    $sources = if ($Identity) { Get-SourceDigest $Identity } else { $null }
    if (-not $sources) { return $null }
    $artifacts = Get-ArtifactDigest
    [bool]($artifacts -and "${sources}:$artifacts" -eq (Get-Content -LiteralPath $receiptFile -TotalCount 1))
}

if ($Action -eq 'status') {
    # One JSON line: the Vivado a build would run, the configured part (or why the config is
    # refused), the bitstream a build left, when, and whether the receipt of that build still
    # matches the sources (the build's own rule; $null: no receipt, or not comparable).
    $vivado = Find-Vivado
    $answer = @(Invoke-Program { & $tool fpga --require --config $config 2>&1 } | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
    $valid = $LASTEXITCODE -eq 0
    $built = (Test-Path -LiteralPath $bit) -and (Test-Path -LiteralPath $receiptFile) -and $valid -and $answer.Count
    if ($built) {
        if (-not $env:ZLC_PS_FPGA_PART) { $env:ZLC_PS_FPGA_PART = $answer[0] }
        if (-not $env:ZLC_PS_GEOM_TCL) { $env:ZLC_PS_GEOM_TCL = Join-Path $buildRoot 'geom.tcl' }
    }
    [ordered]@{ vivado = $vivado; version = $(if ($vivado -match '[\\/]Vivado[\\/]([^\\/]+)[\\/]') { $Matches[1] } else { $null }); config = $config
                part = $(if ($valid -and $answer.Count) { $answer[0] } else { $null }); config_error = $(if (-not $valid) { $answer -join ' ' } else { $null })
                bitstream = $(if (Test-Path -LiteralPath $bit) { $bit } else { $null })
                bitstream_built = $(if (Test-Path -LiteralPath $bit) { (Get-Item -LiteralPath $bit).LastWriteTime.ToString('yyyy-MM-dd HH:mm') } else { $null })
                bitstream_current = $(if ($built) { Test-Prebuilt (Get-VivadoIdentity -Saved) } else { $null }) } | ConvertTo-Json -Compress
    exit 0
}

$vivado = Find-Vivado
if (-not $vivado) { Stop-Build 'ERROR: Vivado was not found. Set ZLC_PS_VIVADO_BIN.' }
$env:ZLC_PS_VIVADO_BIN = $vivado
Write-Host "ZLC Vivado: $vivado"
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
# Dropbox must not sync -- and so lock -- Vivado's generated files: mark the build root ignored
# (an NTFS stream; harmless when the tree is not synced).
try { Set-Content -LiteralPath $buildRoot -Stream 'com.dropbox.ignored' -Value 1 } catch { }
Write-Host "ZLC build root: $buildRoot"

# :zlc_verify_sources
if (-not (Test-Path -LiteralPath $engine)) { Stop-Build "ERROR: missing FINAL engine HDL: $engine" 2 }
if (-not (Test-Path -LiteralPath $topFile)) { Stop-Build "ERROR: missing FINAL top HDL: $topFile" 2 }
if (-not (Test-Path -LiteralPath $create)) { Stop-Build "ERROR: missing FINAL build Tcl: $create" 2 }
$createText = [IO.File]::ReadAllText($create)
if (-not $createText.Contains('zlc_period_streamer.v')) { Stop-Build 'ERROR: create_project.tcl does not read the FINAL engine HDL.' 2 }
if (-not [IO.File]::ReadAllText($topFile).Contains("module $top")) { Stop-Build 'ERROR: FINAL top module name is wrong.' 2 }
if (-not [IO.File]::ReadAllText($engine).Contains('module zlc_period_streamer')) { Stop-Build 'ERROR: FINAL engine module name is wrong.' 2 }
foreach ($check in @(@('create_ip -name jtag_axi', 'create the JTAG-to-AXI master IP'), @('create_ip -name axi_bram_ctrl', 'create the AXI BRAM controller IP'),
                     @('blk_mem_gen_rows', 'create the period-row BRAM'), @('zlc_force_latency2', 'force the BRAM read latency to 2'))) {
    if (-not $createText.Contains($check[0])) { Stop-Build "ERROR: create_project.tcl does not $($check[1])." 2 }
}
if (-not (Test-Path -LiteralPath $xdc)) {
    Stop-Build @("ERROR: missing board XDC: $xdc", 'Put your board pin map at fpga\board_config\board.xdc -- see its README -- or set ZLC_PS_XDC.') 2
}
$xdcText = [IO.File]::ReadAllText($xdc)
if (-not $xdcText.Contains('[get_ports trig]')) { Stop-Build 'ERROR: selected XDC does not define the trig output.' 2 }
if ($xdcText.Contains('<PIN_CH')) { Stop-Build "ERROR: selected XDC still contains PIN_CH placeholders: $xdc" 2 }
Write-Host 'ZLC FINAL source contract: geometry from streamer_config.json, UART/JTAG-to-AXI, forced BRAM latency 2'
Write-Host "ZLC FINAL XDC: $xdc"

# :zlc_require_config + :zlc_resolve_part -- the config with nothing forgiven; its part unless ZLC_PS_FPGA_PART is set.
if (-not (Test-Path -LiteralPath $config)) { Stop-Build "ERROR: canonical FPGA config is missing: $config" }
$configuredPart = Invoke-Program { & $tool fpga --require --config $config } | Select-Object -First 1
if ($LASTEXITCODE -ne 0) { Stop-Build 'ERROR: streamer_config.json is invalid; build/program/flash/diagnose refuse fallback geometry.' }
if (-not $env:ZLC_PS_FPGA_PART) { $env:ZLC_PS_FPGA_PART = "$configuredPart".Trim() }
if (-not $env:ZLC_PS_FPGA_PART) { Stop-Build 'ERROR: canonical FPGA config did not provide fpga_part.' }
Write-Host "ZLC synthesis part: $($env:ZLC_PS_FPGA_PART) (from streamer_config.json / env)"

# :zlc_run_tcl -- Vivado in batch mode from the build root, one journal and log per script.
function Invoke-Tcl([string]$Name) {
    $script = Join-Path $streamer $Name
    if (-not (Test-Path -LiteralPath $script)) { Stop-Build "Missing Tcl script: $script" 2 }
    New-Item -ItemType Directory -Force -Path $env:ZLC_PS_LOG_DIR | Out-Null
    $stem = [IO.Path]::GetFileNameWithoutExtension($Name)
    Write-Host "ZLC direct Vivado path: $script"
    Write-Host "ZLC project dir: $projectDir"
    Push-Location -LiteralPath $buildRoot
    try { Invoke-Program { & $vivado -mode batch -journal (Join-Path $env:ZLC_PS_LOG_DIR "$stem.jou") -log (Join-Path $env:ZLC_PS_LOG_DIR "$stem.log") -source $script } }
    finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
function Invoke-FpgaProgram { Write-Host 'ZLC FPGA pulse streamer: program FINAL bitstream'; Invoke-Tcl 'program_fpga.tcl' }

if ($Action -eq 'diagnose') { Invoke-Tcl 'diagnose_hw_target.tcl'; exit 0 }
# --flash writes the EXISTING bitstream into the SPI configuration flash (survives a power cycle;
# needs the MODE jumper on SPI/QSPI boot); --program-only loads the volatile configuration.
if ($Action -eq 'flash') { Write-Host 'ZLC FPGA pulse streamer: program SPI flash (persists across power cycle)'; Invoke-Tcl 'program_flash.tcl'; exit 0 }
if ($Action -eq 'program') { Invoke-FpgaProgram; exit 0 }

# :zlc_emit_geom -- only a build derives files from the config: the RTL header in place, the geometry Tcl in the build root.
$geomFailure = @('', 'ERROR: failed to derive FPGA geometry from streamer_config.json.', "  config: $config", "  header: $header",
                 "  geom tcl: $(Join-Path $buildRoot 'geom.tcl')", '  The reason is printed above, from zlc-pulse-server fpga.')
Invoke-Program { & $tool fpga --config $config --emit-geometry-vh $header } | Out-Null
if ($LASTEXITCODE -ne 0) { Stop-Build $geomFailure }
if (-not $env:ZLC_PS_GEOM_TCL) {
    $geom = Join-Path $buildRoot 'geom.tcl'
    Invoke-Program { & $tool fpga --config $config --emit-geom-tcl $geom } | Out-Null
    if ($LASTEXITCODE -ne 0) { Stop-Build $geomFailure }
    $env:ZLC_PS_GEOM_TCL = $geom
}
Write-Host "ZLC RTL geometry header regenerated from config: $header"
Write-Host "ZLC geometry tcl: $($env:ZLC_PS_GEOM_TCL) (from streamer_config.json)"
# :zlc_print_capacity_estimate -- the config's own budget: an over-budget geometry is not worth an hour of Vivado.
Invoke-Program { & $tool fpga --config $config --part $env:ZLC_PS_FPGA_PART }
if ($LASTEXITCODE -ne 0) { exit 1 }

# Skip synthesis when the receipt still matches; the default mode then programs that bit.
# (Checked under --force-build too, as v2 did; only the skip is forced off.)
$identity = if (Test-Path -LiteralPath $receiptFile) { Get-VivadoIdentity } else { $null }
$prebuilt = (Test-Prebuilt $identity) -eq $true
if (-not $ForceBuild -and $prebuilt) {
    [IO.File]::WriteAllBytes($identityFile, $identity)
    Write-Host 'ZLC bitstream is up to date (sources unchanged since last build) -- skipping build.'
    Write-Host "ZLC   bit: $bit"
    Write-Host 'ZLC   (force a rebuild with: build_and_program.bat --force-build)'
    if ($Action -eq 'build_program') { Invoke-FpgaProgram }
    exit 0
}

Write-Host 'ZLC FPGA pulse streamer: build FINAL bitstream (period table + nested loops + autonomous scan, JTAG-to-AXI)'
Invoke-Tcl $createTcl
# :zlc_save_src_hash -- with the Vivado identity it was made with (installation.exe compares by it).
$identity = Get-VivadoIdentity
$sources = if ($identity) { Get-SourceDigest $identity } else { $null }
if (-not $sources) { Stop-Build 'ERROR: the bitstream inputs could not be hashed; refusing an unqualified build receipt.' }
$artifacts = Get-ArtifactDigest
if (-not $artifacts) { Stop-Build 'ERROR: bitstream or routed reports are missing; refusing an incomplete build receipt.' }
Set-Content -LiteralPath $receiptFile -Value "${sources}:$artifacts" -Encoding ascii
[IO.File]::WriteAllBytes($identityFile, $identity)
if ($Action -eq 'build_program') { Invoke-FpgaProgram }
exit 0
