param(
    # Default: the checkout's tools folder (tools\tools_root.ps1).
    [string]$ToolsRoot
)
if (-not $ToolsRoot) { . (Join-Path (Split-Path -Parent $PSScriptRoot) 'tools_root.ps1'); $ToolsRoot = Get-ZlcToolsRoot }

$ErrorActionPreference = 'Stop'
$toolchainRoot = Join-Path $ToolsRoot 'rust-1.99.1-x86_64-pc-windows-gnu'
$rustcPath = Join-Path $toolchainRoot 'bin\rustc.exe'
$cargoPath = Join-Path $toolchainRoot 'bin\cargo.exe'
if (-not (Test-Path -LiteralPath $rustcPath)) {
    throw 'Portable Rust is missing. Run install-portable-rust.ps1 first.'
}

$savedPath = $env:PATH
$savedRustc = $env:RUSTC
$savedCargoHome = $env:CARGO_HOME
$savedTargetDir = $env:CARGO_TARGET_DIR
try {
    $env:PATH = "$(Join-Path $toolchainRoot 'bin');$(Join-Path $toolchainRoot 'lib\rustlib\x86_64-pc-windows-gnu\bin');$savedPath"
    $env:RUSTC = $rustcPath
    $env:CARGO_HOME = Join-Path $ToolsRoot 'cargo-home'
    $env:CARGO_TARGET_DIR = Join-Path $PSScriptRoot 'target'
    & $cargoPath build --manifest-path (Join-Path $PSScriptRoot 'Cargo.toml') --release --offline
    if ($LASTEXITCODE -ne 0) { throw 'Release compilation failed.' }
    $executable = Join-Path $env:CARGO_TARGET_DIR 'release\zlc-v3-rust-smoke.exe'
    & $executable
    if ($LASTEXITCODE -ne 0) { throw 'Release executable failed.' }
    Write-Host "Executable: $executable"
}
finally {
    $env:PATH = $savedPath
    $env:RUSTC = $savedRustc
    $env:CARGO_HOME = $savedCargoHome
    $env:CARGO_TARGET_DIR = $savedTargetDir
}
