param(
    [ValidateSet('build', 'check', 'test', 'run')][string]$Action = 'build',
    [string[]]$Package = @('zlc-core'),
    [switch]$Offline
)
# cargo with the portable toolchain in the tools folder, into <checkout>\target;
# the caller's environment is restored afterwards.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'tools_root.ps1')
$paths = Get-ZlcToolchainPaths
if (-not (Test-Path -LiteralPath $paths.Cargo)) {
    throw "The portable Rust toolchain is not installed ($($paths.Cargo)). Run tools\dev_setup.ps1."
}
$names = @('PATH', 'RUSTC', 'CARGO_HOME', 'CARGO_TARGET_DIR', 'CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER', 'LIB', 'ZLC_FREETYPE_DIR', 'CARGO_ENCODED_RUSTFLAGS')
$saved = @{}
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
try {
    $env:PATH = "$($paths.Rust)\bin;$($paths.Rust)\lib\rustlib\x86_64-pc-windows-gnu\bin;$env:PATH"
    $env:RUSTC = Join-Path $paths.Rust 'bin\rustc.exe'
    $env:CARGO_HOME = $paths.CargoHome
    $env:CARGO_TARGET_DIR = Join-Path (Get-ZlcCheckout) 'target'
    $env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER = $paths.Linker
    $env:LIB = (@('crt\lib\x86_64', 'sdk\lib\um\x86_64', 'sdk\lib\ucrt\x86_64') | ForEach-Object { Join-Path $paths.Sdk $_ }) -join ';'
    # zlc-text links FreeType 2.6.1 built as Matplotlib's wheel builds it.
    if (-not $env:ZLC_FREETYPE_DIR) { $env:ZLC_FREETYPE_DIR = $paths.FreeType }
    # No machine path in what is built: source paths (panic locations, debug info) are named
    # relative to the checkout, the crates' cargo home and the toolchain -- the last matching
    # prefix wins, so the narrower ones follow -- and the PDB is named without its folder.
    $flags = @("--remap-path-prefix=$(Get-ZlcCheckout)=zlc", "--remap-path-prefix=$($paths.CargoHome)=cargo",
               "--remap-path-prefix=$($paths.Rust)=rust", '-Clink-arg=/PDBALTPATH:%_PDB%')
    $env:CARGO_ENCODED_RUSTFLAGS = $flags -join [char]0x1f
    $cargoArguments = @($Action, '--manifest-path', (Join-Path (Get-ZlcCheckout) 'Cargo.toml'), '--release', '--target', 'x86_64-pc-windows-msvc')
    foreach ($name in $Package) { $cargoArguments += @('-p', $name) }
    if ($Offline) { $cargoArguments += '--offline' }
    Invoke-Program { & $paths.Cargo @cargoArguments }
    if ($LASTEXITCODE -ne 0) { throw "cargo $Action failed with exit code $LASTEXITCODE" }
}
finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
}
