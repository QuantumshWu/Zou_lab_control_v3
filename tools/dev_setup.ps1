param(
    # Print what is present and what would be downloaded or built, and change nothing.
    [switch]$DryRun,
    # installation.exe's check: the dry run, then one JSON line of each step and whether it is present.
    [switch]$Status
)
# A development machine's toolchain, into <checkout>\.tools (or ZLC_V3_TOOLS):
# the portable Rust 1.99.1 (GNU host, MSVC target), lld-link, llvm-dlltool, the
# MSVC SDK/CRT libraries (xwin), FreeType 2.6.1 built as Matplotlib 3.10.8
# builds it (zig cc), Node.js and the UI's TypeScript, and the crates in
# Cargo.lock. Every archive is an official release checked against its
# publisher's SHA-256; what is already present is kept and skipped. Nothing
# global changes (no installer, registry, PATH or default toolchain). Lab PCs do
# not need this: they run the release (installation.exe) with the science environment alone.
$ErrorActionPreference = 'Stop'
trap { Write-Host "ERROR: $($_.Exception.Message)"; exit 1 }
. (Join-Path $PSScriptRoot 'tools_root.ps1')
$root = Get-ZlcCheckout
$tools = Get-ZlcToolsRoot
$paths = Get-ZlcToolchainPaths
$downloads = Join-Path $tools 'downloads'
$release = 'https://static.rust-lang.org/dist/2026-10-15'

if ($Status) { $DryRun = $true }
$steps = @()
function Step([string]$Name, [bool]$Present, [scriptblock]$Install, [string]$Detail) {
    $script:steps += [ordered]@{ name = $Name; present = $Present }
    if ($Present) { Write-Host "  present   $Name"; return }
    if ($DryRun) { Write-Host "  missing   $Name -- $Detail"; return }
    Write-Host "Installing $Name ..."
    & $Install
}

# One official archive in .tools\downloads, checked against its SHA-256 (fetched first when absent).
function Get-Archive([string]$Url, [string]$Sha256, [string]$Name = (Split-Path -Leaf $Url)) {
    New-Item -ItemType Directory -Force -Path $downloads | Out-Null
    $archive = Join-Path $downloads $Name
    if (-not (Test-Path -LiteralPath $archive) -or (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Sha256) {
        Write-Host "Downloading $Url"
        Invoke-Checked "Downloading $Url" { & curl.exe --fail --location --retry 3 --retry-delay 1 --silent --show-error --output $archive $Url }
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Sha256) { throw "SHA-256 mismatch: $archive" }
    $archive
}

function Invoke-Checked([string]$What, [scriptblock]$Body) {
    Invoke-Program $Body
    if ($LASTEXITCODE -ne 0) { throw "$What failed with code $LASTEXITCODE." }
}

Write-Host "Development toolchain in $tools"
$rustc = Join-Path $paths.Rust 'bin\rustc.exe'
$rustPresent = (Test-Path -LiteralPath $rustc) -and "$(Invoke-Quiet { & $rustc --version })".Contains($ZlcRustVersion)
Step "Rust $ZlcRustVersion (rustc, cargo, rust-std, rust-mingw)" $rustPresent {
    & (Join-Path $PSScriptRoot 'toolchain\install-portable-rust.ps1') -ToolsRoot $tools
} "tools\toolchain\install-portable-rust.ps1 -ToolsRoot $tools ($release)"

Step 'Rust MSVC-target standard library' (Test-Path -LiteralPath $paths.StdMsvc) {
    $archive = Get-Archive "$release/rust-std-$ZlcRustVersion-x86_64-pc-windows-msvc.tar.xz" '4fe09430ded40b8b3092b49f0a651178eb1f8154ef4d1f1a73f2a3595b2d956f'
    Invoke-Checked 'Extracting rust-std (MSVC)' { & tar.exe -xf $archive -C $paths.Rust --strip-components 2 "rust-std-$ZlcRustVersion-x86_64-pc-windows-msvc/rust-std-x86_64-pc-windows-msvc" }
} "$release/rust-std-$ZlcRustVersion-x86_64-pc-windows-msvc.tar.xz"

Step 'lld-link (the Rust distribution''s LLD)' (Test-Path -LiteralPath $paths.Linker) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $paths.Linker) | Out-Null
    Copy-Item -LiteralPath (Join-Path $paths.Rust 'lib\rustlib\x86_64-pc-windows-gnu\bin\rust-lld.exe') -Destination $paths.Linker
} 'a copy of the toolchain''s rust-lld.exe'

Step 'llvm-dlltool (LLVM tools'' llvm-ar)' (Test-Path -LiteralPath $paths.Dlltool) {
    $archive = Get-Archive "$release/llvm-tools-$ZlcRustVersion-x86_64-pc-windows-gnu.tar.xz" '7cc7089d8e6e8031adc8c9a2251d32c7e70c34e2ed472834b0753fa63ebd91b2'
    $folder = Split-Path -Parent $paths.Dlltool
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    Invoke-Checked 'Extracting llvm-ar' { & tar.exe -xf $archive -C $folder --strip-components 6 "llvm-tools-$ZlcRustVersion-x86_64-pc-windows-gnu/llvm-tools-preview/lib/rustlib/x86_64-pc-windows-gnu/bin/llvm-ar.exe" }
    # llvm-ar selects its dlltool driver by its executable name.
    Copy-Item -LiteralPath (Join-Path $folder 'llvm-ar.exe') -Destination $paths.Dlltool
} "$release/llvm-tools-$ZlcRustVersion-x86_64-pc-windows-gnu.tar.xz"

$sdkLibraries = @('crt\lib\x86_64\msvcrt.lib', 'crt\lib\x86_64\vcruntime.lib', 'crt\lib\x86_64\libcmt.lib', 'sdk\lib\um\x86_64\kernel32.lib', 'sdk\lib\ucrt\x86_64\ucrt.lib')
$sdkPresent = -not ($sdkLibraries | Where-Object { -not (Test-Path -LiteralPath (Join-Path $paths.Sdk $_)) })
Step 'MSVC SDK/CRT libraries and headers (xwin 0.10.0 splat; accepts the Microsoft SDK licence)' $sdkPresent {
    $archive = Get-Archive 'https://github.com/Jake-Shadle/xwin/releases/download/0.10.0/xwin-0.10.0-x86_64-pc-windows-msvc.tar.gz' '96e83665ef5c1406aabd0be603a8d08e67e604f36fda630b6807ee948a614e26'
    $xwin = Join-Path $tools 'xwin-0.10.0'
    New-Item -ItemType Directory -Force -Path $xwin | Out-Null
    Invoke-Checked 'Extracting xwin' { & tar.exe -xf $archive -C $xwin --strip-components 1 }
    & (Join-Path $xwin 'xwin.exe') --accept-license --arch x86_64 --cache-dir (Join-Path $tools 'xwin-cache') splat --disable-symlinks --output $paths.Sdk
    # xwin 0.10.0 still makes a versioned SDK-header alias on Windows despite
    # --disable-symlinks (privilege error 1314); the libraries are what the build links.
    $missing = @($sdkLibraries | Where-Object { -not (Test-Path -LiteralPath (Join-Path $paths.Sdk $_)) })
    if ($missing.Count) { throw "The SDK extraction did not supply: $($missing -join ', ')" }
} 'https://github.com/Jake-Shadle/xwin/releases/download/0.10.0 (then Microsoft''s SDK/CRT packages)'

$freetypeLib = Join-Path $paths.FreeType 'freetype.lib'
Step 'FreeType 2.6.1, built as Matplotlib 3.10.8 builds it (zig 0.17.0 cc, MSVC ABI)' (Test-Path -LiteralPath $freetypeLib) {
    $zigArchive = Get-Archive 'https://ziglang.org/download/0.17.0/zig-x86_64-windows-0.17.0.zip' 'b5663f69581dcf391293fbf16c06cb80d81d806545ce618b4d0bab7f0eb8c428'
    # Matplotlib's own pin of the FreeType release (subprojects/freetype-2.6.1.wrap).
    $freetypeArchive = Get-Archive 'https://download.savannah.gnu.org/releases/freetype/freetype-old/freetype-2.6.1.tar.gz' '0a3c7dfbda6da1e8fce29232e8e96d987ababbbf71ebc8c75659e4132c367014'
    $mplArchive = Get-Archive 'https://files.pythonhosted.org/packages/source/m/matplotlib/matplotlib-3.10.8.tar.gz' '2299372c19d56bcd35cf05a2738308758d32b9eaed2371898d8f5bd33f084aa3'
    $work = Join-Path $tools 'freetype-src'
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    if (-not (Test-Path -LiteralPath (Join-Path $tools 'zig-0.17.0\zig.exe'))) {
        Expand-Archive -LiteralPath $zigArchive -DestinationPath $work -Force
        Move-Item -LiteralPath (Join-Path $work 'zig-x86_64-windows-0.17.0') -Destination (Join-Path $tools 'zig-0.17.0')
    }
    $zig = Join-Path $tools 'zig-0.17.0\zig.exe'
    Invoke-Checked 'Extracting FreeType' { & tar.exe -xf $freetypeArchive -C $work }
    $meson = 'matplotlib-3.10.8/subprojects/packagefiles/freetype-2.6.1-meson'
    Invoke-Checked 'Extracting Matplotlib''s FreeType build files' { & tar.exe -xf $mplArchive -C $work "$meson/include/freetype/config/ftoption.h.in" "$meson/src/gzip/zconf.h" }
    $source = Join-Path $work 'freetype-2.6.1'
    $mesonDir = Join-Path $work ($meson -replace '/', '\')
    $config = Join-Path $paths.FreeType 'config'
    $objects = Join-Path $paths.FreeType 'obj'
    New-Item -ItemType Directory -Force -Path $config, $objects | Out-Null
    # Matplotlib's ftoption.h.in with every #mesondefine left undefined (no system zlib, bzip2, png, harfbuzz).
    $options = [IO.File]::ReadAllText((Join-Path $mesonDir 'include\freetype\config\ftoption.h.in')) -replace '(?m)^#mesondefine ([^\r\n]*)', '/* #undef $1 */'
    [IO.File]::WriteAllText((Join-Path $config 'ftoption.h'), $options)
    Copy-Item -LiteralPath (Join-Path $source 'include\freetype\config\ftconfig.h') -Destination (Join-Path $config 'ftconfig.h') -Force
    Copy-Item -LiteralPath (Join-Path $mesonDir 'src\gzip\zconf.h') -Destination (Join-Path $source 'src\gzip\zconf.h') -Force
    $libc = Join-Path $paths.FreeType 'msvc-libc.txt'
    $sdk = $paths.Sdk
    Set-Content -LiteralPath $libc -Encoding ascii -Value @("include_dir=$sdk\sdk\include\ucrt", "sys_include_dir=$sdk\crt\include", "crt_dir=$sdk\sdk\lib\ucrt\x86_64",
        "msvc_lib_dir=$sdk\crt\lib\x86_64", "kernel32_lib_dir=$sdk\sdk\lib\um\x86_64", 'gcc_dir=', 'darwin_sdk_dir=')
    $env:ZIG_LIBC = $libc
    $sources = 'src/autofit/autofit.c src/base/ftbase.c src/base/ftbbox.c src/base/ftbdf.c src/base/ftbitmap.c src/base/ftcid.c src/base/ftfntfmt.c ' +
               'src/base/ftfstype.c src/base/ftgasp.c src/base/ftglyph.c src/base/ftgxval.c src/base/ftinit.c src/base/ftlcdfil.c src/base/ftmm.c ' +
               'src/base/ftotval.c src/base/ftpatent.c src/base/ftpfr.c src/base/ftstroke.c src/base/ftsynth.c src/base/ftsystem.c src/base/fttype1.c ' +
               'src/base/ftwinfnt.c src/bdf/bdf.c src/bzip2/ftbzip2.c src/cache/ftcache.c src/cff/cff.c src/cid/type1cid.c src/gzip/ftgzip.c ' +
               'src/lzw/ftlzw.c src/pcf/pcf.c src/pfr/pfr.c src/psaux/psaux.c src/pshinter/pshinter.c src/psnames/psnames.c src/raster/raster.c ' +
               'src/sfnt/sfnt.c src/smooth/smooth.c src/truetype/truetype.c src/type1/type1.c src/type42/type42.c src/winfonts/winfnt.c builds/windows/ftdebug.c'
    $built = @()
    foreach ($file in $sources -split ' ') {
        $object = Join-Path $objects (($file -replace '/', '_') -replace '\.c$', '.obj')
        Invoke-Checked "zig cc $file" {
            & $zig cc -target x86_64-windows-msvc -O2 -c (Join-Path $source $file) -o $object -DFT2_BUILD_LIBRARY '-DFT_CONFIG_CONFIG_H=<ftconfig.h>' `
                '-DFT_CONFIG_OPTIONS_H=<ftoption.h>' "-I$config" "-I$(Join-Path $source 'include')" -isystem "$sdk\crt\include" -isystem "$sdk\sdk\include\ucrt" `
                -isystem "$sdk\sdk\include\um" -isystem "$sdk\sdk\include\shared" -Wno-everything
        }
        $built += $object
    }
    Invoke-Checked 'zig ar' { & $zig ar rcs $freetypeLib @built }
} 'zig 0.17.0 (ziglang.org), freetype-2.6.1.tar.gz (savannah, Matplotlib''s hash), matplotlib-3.10.8.tar.gz (PyPI, for its meson build files)'

$nodeVersion = '24.19.0'
Step "Node.js (npm for the UI build)" ([bool](Get-ZlcNpm)) {
    # nodejs.org publishes each release's SHA-256 list beside its archives.
    $name = "node-v$nodeVersion-win-x64.zip"
    $sums = Join-Path $downloads "node-v$nodeVersion-SHASUMS256.txt"
    New-Item -ItemType Directory -Force -Path $downloads | Out-Null
    Invoke-Checked 'Downloading the Node.js checksums' { & curl.exe --fail --location --silent --show-error --output $sums "https://nodejs.org/dist/v$nodeVersion/SHASUMS256.txt" }
    $line = Get-Content -LiteralPath $sums | Where-Object { $_ -match "^([0-9a-f]{64})\s+$([regex]::Escape($name))$" } | Select-Object -First 1
    if (-not $line) { throw "SHASUMS256.txt of Node.js $nodeVersion does not list $name." }
    $archive = Get-Archive "https://nodejs.org/dist/v$nodeVersion/$name" ($line -split '\s+')[0]
    Expand-Archive -LiteralPath $archive -DestinationPath $tools -Force
    Move-Item -LiteralPath (Join-Path $tools "node-v$nodeVersion-win-x64") -Destination $paths.Node
} "https://nodejs.org/dist/v$nodeVersion/node-v$nodeVersion-win-x64.zip (checked against that release's SHASUMS256.txt)"

$ui = Join-Path $root 'apps\desktop\ui'
Step 'TypeScript for the UI (npm ci from package-lock.json)' (Test-Path -LiteralPath (Join-Path $ui 'node_modules\typescript')) {
    $npm = Get-ZlcNpm
    $saved = $env:PATH
    try { $env:PATH = "$(Split-Path -Parent $npm);$env:PATH"; Invoke-Checked 'npm ci' { & $npm --prefix $ui ci --ignore-scripts } }
    finally { $env:PATH = $saved }
} 'npm ci --ignore-scripts in apps\desktop\ui'

# The crates of Cargo.lock in .tools\cargo-home, so every build runs --offline.
$cargoArguments = @('fetch', '--manifest-path', (Join-Path $root 'Cargo.toml'), '--locked', '--target', 'x86_64-pc-windows-msvc')
$fetched = $false
if (Test-Path -LiteralPath $paths.Cargo) {
    $savedHome = $env:CARGO_HOME
    $env:CARGO_HOME = $paths.CargoHome
    try { Invoke-Quiet { & $paths.Cargo @cargoArguments --offline } | Out-Null; $fetched = ($LASTEXITCODE -eq 0) } finally { $env:CARGO_HOME = $savedHome }
}
Step 'The crates of Cargo.lock (cargo fetch)' $fetched {
    $savedHome = $env:CARGO_HOME
    $env:CARGO_HOME = $paths.CargoHome
    try { Invoke-Checked 'cargo fetch' { & $paths.Cargo @cargoArguments } } finally { $env:CARGO_HOME = $savedHome }
} "cargo fetch --locked into $($paths.CargoHome) (crates.io, checked against Cargo.lock)"

# The science environment, with the plotting oracle the native engine is compared against
# (numba, llvmlite, matplotlib, cycler; no plugin imports them): present when install_science.ps1,
# by the rule its installs decide by, finds nothing of the two groups unmet.
$science = @(& (Join-Path $PSScriptRoot 'install_science.ps1') -Status | Where-Object { "$_".StartsWith('{') }) | Select-Object -Last 1
$science = if ($science) { $science | ConvertFrom-Json } else { $null }
$oraclePresent = $science -and ($science.exists -or $science.inherit) -and -not (@($science.unmet.core) + @($science.unmet.devices) + @($science.unmet.oracle)).Count
Step 'The science environment and the plotting oracle' $oraclePresent {
    foreach ($group in 'runtime', 'oracle') {
        & (Join-Path $PSScriptRoot 'install_science.ps1') -Group $group
        if ($LASTEXITCODE -ne 0) { throw "The science environment ($group) failed." }
    }
} 'tools\install_science.ps1 -Group runtime, then -Group oracle'

if ($Status) { ConvertTo-Json -Compress -InputObject @($steps); return }
if ($DryRun) { Write-Host ''; Write-Host 'Dry run: nothing was changed.'; return }
$gaps = @(Get-ZlcToolchainGaps)
if ($gaps.Count) { throw "Still missing: $($gaps -join ', ')" }
Write-Host ''
Write-Host 'The development machine is ready: bin\scripts\experiment.bat (tools\desktop.ps1) builds this checkout when its sources change.'
