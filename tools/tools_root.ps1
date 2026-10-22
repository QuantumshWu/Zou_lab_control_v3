# Where v3's tools and products live -- all inside this checkout:
#   .tools\            the science environment (every PC) and, on a development
#                      machine, the portable toolchain (tools\dev_setup.ps1);
#                      ZLC_V3_TOOLS names another tools folder explicitly
#   target\            development builds
#   experiment.exe, figure_viewer.exe, pulse_editor.exe, installation.exe
#                      the four programs (one launcher, in git)
#   bin\release.json   the release this checkout names (in git; tools\release.ps1 writes it)
#   bin\engine\        that release's engine: zlc-desktop, zlc-pulse-server and the page
#   bin\ai\            zlc-mcp, for AI clients (both not in git: tools\fetch_engine.ps1)
#   bin\scripts\       the .bat files
#   .local\            what running programs write: the WebView2 profile, MCP
#                      records, the desktop's logs, Vivado state and the plugins'
#                      caches (ZLC_V3_LOCAL)
# Dot-sourced by the scripts beside it.

function Get-ZlcCheckout { Split-Path -Parent $PSScriptRoot }

# The PYTHONPATH the script was started with: zlc-desktop's, which its plugin processes keep
# after science\src (Invoke-ZlcAsPlugin).
$ZlcCallerPythonPath = $env:PYTHONPATH

# Run a program, its stderr passed on: Windows PowerShell turns each stderr
# line into an error record when the output is redirected, which 'Stop' would
# make fatal. Its exit code stays in $LASTEXITCODE.
function Invoke-Program([scriptblock]$Body) {
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Body } finally { $ErrorActionPreference = $saved }
}

# A program's output with its stderr dropped.
function Invoke-Quiet([scriptblock]$Body) {
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    try { & $Body 2>$null } finally { $ErrorActionPreference = $saved }
}

function Get-ZlcToolsRoot {
    if ($env:ZLC_V3_TOOLS) { return $env:ZLC_V3_TOOLS }
    Join-Path (Get-ZlcCheckout) '.tools'
}

function Get-ZlcSciencePython {
    $python = Join-Path (Get-ZlcToolsRoot) 'science-env\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        throw "The science environment is not installed ($python). Run installation.exe."
    }
    $python
}

# The pinned portable toolchain tools\dev_setup.ps1 installs into the tools folder.
$ZlcRustVersion = '1.99.1'
function Get-ZlcToolchainPaths {
    $tools = Get-ZlcToolsRoot
    $rust = Join-Path $tools "rust-$ZlcRustVersion-x86_64-pc-windows-gnu"
    [ordered]@{
        Rust = $rust
        Cargo = Join-Path $rust 'bin\cargo.exe'
        StdMsvc = Join-Path $rust 'lib\rustlib\x86_64-pc-windows-msvc\lib'
        CargoHome = Join-Path $tools 'cargo-home'
        Linker = Join-Path $tools "lld-link-$ZlcRustVersion\lld-link.exe"
        Dlltool = Join-Path $tools "llvm-dlltool-$ZlcRustVersion\dlltool.exe"
        Sdk = Join-Path $tools 'msvc-sdk'
        FreeType = Join-Path $tools 'freetype-build'
        Node = Join-Path $tools 'node'
    }
}

# What a build still lacks (empty when this machine can build the desktop).
function Get-ZlcToolchainGaps {
    $paths = Get-ZlcToolchainPaths
    $gaps = @()
    foreach ($item in @(@('Rust', $paths.Cargo), @('Rust MSVC standard library', $paths.StdMsvc), @('lld-link', $paths.Linker),
                        @('MSVC SDK libraries', (Join-Path $paths.Sdk 'sdk\lib\um\x86_64\kernel32.lib')),
                        @('FreeType 2.6.1', (Join-Path $paths.FreeType 'freetype.lib')))) {
        if (-not (Test-Path -LiteralPath $item[1])) { $gaps += $item[0] }
    }
    if (-not (Get-ZlcNpm)) { $gaps += 'Node.js' }
    $gaps
}

# npm: the tools folder's portable Node, else one on PATH.
function Get-ZlcNpm {
    $portable = Join-Path (Get-ZlcToolchainPaths).Node 'npm.cmd'
    if (Test-Path -LiteralPath $portable) { return $portable }
    $found = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    $null
}

# The sources a desktop build is made of: crates\ (not their tests and
# examples, nor the FPGA tree, which no executable compiles in), the UI page,
# Cargo.toml/Cargo.lock and the build's own flags (dev.ps1, this file) -- the
# Python plugins are not among them.
$ZlcSourceSpec = @('crates', 'apps/desktop/ui', 'Cargo.toml', 'Cargo.lock', 'tools/dev.ps1', 'tools/tools_root.ps1',
                   ':(exclude,glob)crates/*/tests/**', ':(exclude,glob)crates/*/examples/**', ':(exclude)crates/zlc-pulse/fpga')

# Those sources as one hash of git's blob id of every file (line endings
# normalised, so a release built on one machine is current on another's
# clone); uncommitted edits and new files count. $null without git.
function Get-ZlcSourceStamp {
    $root = Get-ZlcCheckout
    $git = (Get-Command git.exe -ErrorAction SilentlyContinue).Source
    if (-not $git) { return $null }
    $ids = @{}
    $staged = @(Invoke-Quiet { & $git -C $root ls-files -s -- @ZlcSourceSpec })
    if ($LASTEXITCODE -ne 0 -or $staged.Count -eq 0) { return $null }
    foreach ($line in $staged) { $meta, $path = $line -split "`t", 2; $ids[$path] = ($meta -split ' ')[1] }
    foreach ($path in @(& $git -C $root ls-files -d -- @ZlcSourceSpec)) { $ids.Remove($path) }
    $changed = @(& $git -C $root ls-files -m -o --exclude-standard -- @ZlcSourceSpec |
        Where-Object { Test-Path -LiteralPath (Join-Path $root $_) -PathType Leaf } | Sort-Object -Unique)
    if ($changed.Count) {
        $hashes = @(& $git -C $root hash-object -- @changed)
        if ($LASTEXITCODE -ne 0 -or $hashes.Count -ne $changed.Count) { return $null }
        for ($i = 0; $i -lt $changed.Count; $i++) { $ids[$changed[$i]] = $hashes[$i] }
    }
    [string[]]$paths = $ids.Keys
    [Array]::Sort($paths, [StringComparer]::Ordinal)
    $text = ($paths | ForEach-Object { "$($ids[$_]) $_" }) -join "`n"
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { -join ($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($text)) | ForEach-Object { $_.ToString('x2') }) }
    finally { $sha.Dispose() }
}

# The engine the four programs start (bin\engine), and every program a build makes.
$ZlcEngineNames = @('zlc-desktop.exe', 'zlc-pulse-server.exe')
$ZlcProductNames = $ZlcEngineNames + @('zlc-mcp.exe', 'zlc-launch.exe')
# The four programs at the checkout's root: zlc-launch.exe under the name of the mode it starts.
$ZlcPrograms = @('experiment', 'figure_viewer', 'pulse_editor', 'installation')
# Where the releases are published (GitHub releases of this repository).
$ZlcRepository = 'QuantumshWu/Zou_lab_control_v3'
function Get-ZlcDevProducts { Join-Path (Get-ZlcCheckout) 'target\x86_64-pc-windows-msvc\release' }
function Get-ZlcReleaseProducts { Join-Path (Get-ZlcCheckout) 'bin\engine' }

# The stamp a products folder was built from (its SOURCE file's first line).
function Get-ZlcProductsStamp([string]$Folder) {
    $file = Join-Path $Folder 'SOURCE'
    if (-not (Test-Path -LiteralPath $file)) { return $null }
    $line = Get-Content -LiteralPath $file -TotalCount 1
    if ($line -match '^source ([0-9a-f]{64})$') { return $Matches[1] }
    $null
}

function Test-ZlcProducts([string]$Folder) {
    foreach ($name in $ZlcEngineNames) { if (-not (Test-Path -LiteralPath (Join-Path $Folder $name))) { return $false } }
    $true
}

# Build the UI and the executables into target\ and stamp them.
function Invoke-ZlcBuild([string]$Stamp) {
    $gaps = @(Get-ZlcToolchainGaps)
    if ($gaps.Count) { throw "This machine cannot build the desktop: missing $($gaps -join ', '). Run tools\dev_setup.ps1." }
    $root = Get-ZlcCheckout
    $ui = Join-Path $root 'apps\desktop\ui'
    $npm = Get-ZlcNpm
    Write-Host "Building the UI and zlc-desktop from $root (a few minutes, once per source change) ..."
    $savedPath = $env:PATH
    try {
        $env:PATH = "$(Split-Path -Parent $npm);$env:PATH"
        # Shown, not returned: this function's output is the products folder.
        if (-not (Test-Path -LiteralPath (Join-Path $ui 'node_modules\typescript'))) {
            Invoke-Program { & $npm --prefix $ui ci --ignore-scripts } | Out-Host
            if ($LASTEXITCODE -ne 0) { throw 'npm ci failed for apps\desktop\ui.' }
        }
        Invoke-Program { & $npm --prefix $ui run build } | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'The UI build (tsc) failed.' }
    }
    finally { $env:PATH = $savedPath }
    & (Join-Path $PSScriptRoot 'dev.ps1') build -Package zlc-desktop, zlc-pulse, zlc-bus, zlc-launch -Offline | Out-Host
    $folder = Get-ZlcDevProducts
    if ($Stamp) { Set-Content -LiteralPath (Join-Path $folder 'SOURCE') -Value "source $Stamp" -Encoding ascii }
    $folder
}

# The folder holding the engine, zlc-desktop.exe and zlc-pulse-server.exe:
# a development build in target\ that is current for these sources, else the
# release in bin\engine when it is, else a new build when this machine has the
# toolchain, else the release as it is (said so). -Build always builds.
function Get-ZlcProducts([switch]$Build) {
    $stamp = Get-ZlcSourceStamp
    $dev = Get-ZlcDevProducts
    $release = Get-ZlcReleaseProducts
    if ($Build) { return Invoke-ZlcBuild $stamp }
    if ($stamp) {
        if ((Test-ZlcProducts $dev) -and (Get-ZlcProductsStamp $dev) -eq $stamp) { return $dev }
        if ((Test-ZlcProducts $release) -and (Get-ZlcProductsStamp $release) -eq $stamp) { return $release }
    }
    if (@(Get-ZlcToolchainGaps).Count -eq 0) { return Invoke-ZlcBuild $stamp }
    if (Test-ZlcProducts $release) {
        Write-Host 'Using the release in bin\engine: these sources differ from the ones it was built from, and this machine has no toolchain to rebuild (tools\dev_setup.ps1).'
        return $release
    }
    throw 'There is no zlc-desktop.exe: neither bin\engine nor a build in target\. Start installation.exe (it downloads the release), or set up a development machine with tools\dev_setup.ps1.'
}

# A venv's base installation (home in the pyvenv.cfg beside Scripts\), else the interpreter's own folder.
function Get-ZlcBaseHome([string]$Python) {
    $folder = Split-Path -Parent $Python
    foreach ($config in (Join-Path (Split-Path -Parent $folder) 'pyvenv.cfg'), (Join-Path $folder 'pyvenv.cfg')) {
        $line = @(Get-Content -LiteralPath $config -ErrorAction SilentlyContinue | Where-Object { $_ -match '^\s*home\s*=\s*\S' } | Select-Object -First 1)
        if ($line.Count) { return ($line[0] -replace '^\s*home\s*=\s*', '').Trim() }
    }
    $folder
}

# The folders a base installation's DLLs live in, put first on PATH: for a conda base the ones
# conda activation adds (v2 _resolve_tools.bat), since numpy and ssl load DLLs from Library\bin.
function Get-ZlcBaseFolders([string]$BaseHome) {
    $folders = @($BaseHome)
    if (Test-Path -LiteralPath (Join-Path $BaseHome 'conda-meta') -PathType Container) {
        $folders += @('Library\mingw-w64\bin', 'Library\usr\bin', 'Library\bin', 'Scripts', 'bin', 'condabin') | ForEach-Object { Join-Path $BaseHome $_ }
    }
    $folders += Join-Path $BaseHome 'DLLs'
    @($folders | Where-Object { Test-Path -LiteralPath $_ -PathType Container })
}

# Run $Body ($Python its argument) as zlc-desktop runs a plugin process (zlc-plugin-host
# command() and plugin_environment()): the interpreter started without -I or -s, so site.py
# adds the user and system site-packages it would; no PYTHONHOME; the checkout's science\src
# before the caller's PYTHONPATH; the base installation's DLL folders first on PATH, a conda
# base told to search them. This process's environment is restored.
function Invoke-ZlcAsPlugin([string]$Python, [scriptblock]$Body) {
    $zlcNames = @('PATH', 'PYTHONHOME', 'PYTHONPATH', 'CONDA_DLL_SEARCH_MODIFICATION_ENABLE')
    $zlcSaved = @{}
    foreach ($name in $zlcNames) { $zlcSaved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
    try {
        $zlcHome = Get-ZlcBaseHome $Python
        Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
        $env:PYTHONPATH = (@(Join-Path (Get-ZlcCheckout) 'science\src') + @("$ZlcCallerPythonPath" -split ';' | Where-Object { $_ })) -join ';'
        $env:PATH = (@(Get-ZlcBaseFolders $zlcHome) + @($env:PATH)) -join ';'
        if ((Test-Path -LiteralPath (Join-Path $zlcHome 'conda-meta') -PathType Container) -and -not $env:CONDA_DLL_SEARCH_MODIFICATION_ENABLE) { $env:CONDA_DLL_SEARCH_MODIFICATION_ENABLE = '1' }
        & $Body $Python
    }
    finally {
        foreach ($name in $zlcNames) { [Environment]::SetEnvironmentVariable($name, $zlcSaved[$name], 'Process') }
    }
}

# Run $Body in the science environment as its plugin processes run, with their caches (the
# checkout's .local); its stderr is passed on.
function Invoke-ZlcPluginEnvironment([scriptblock]$Body) {
    $names = @('ZLC_PLUGIN_PYTHON', 'ZLC_V3_LOCAL', 'CUPY_CACHE_DIR')
    $saved = @{}
    foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
    try {
        $env:ZLC_PLUGIN_PYTHON = Get-ZlcSciencePython
        if (-not $env:ZLC_V3_LOCAL) { $env:ZLC_V3_LOCAL = Join-Path (Get-ZlcCheckout) '.local' }
        if (-not $env:CUPY_CACHE_DIR) { $env:CUPY_CACHE_DIR = Join-Path $env:ZLC_V3_LOCAL 'cupy' }
        $work = $Body   # Invoke-Program's own parameter is also named Body
        Invoke-ZlcAsPlugin $env:ZLC_PLUGIN_PYTHON { param($python) Invoke-Program { & $work $python } }
    }
    finally {
        foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
    }
}
