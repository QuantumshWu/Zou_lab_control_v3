param(
    # Default: the checkout's tools folder (tools\tools_root.ps1).
    [string]$ToolsRoot
)
if (-not $ToolsRoot) { . (Join-Path (Split-Path -Parent $PSScriptRoot) 'tools_root.ps1'); $ToolsRoot = Get-ZlcToolsRoot }

$ErrorActionPreference = 'Stop'
$rustVersion = '1.99.1'
$rustTarget = 'x86_64-pc-windows-gnu'
$releaseDate = '2026-10-15'
$toolchainRoot = Join-Path $ToolsRoot "rust-$rustVersion-$rustTarget"
$downloadRoot = Join-Path $ToolsRoot 'downloads'
$components = @(
    @{ Name = 'rustc'; Component = 'rustc'; Sha256 = 'a48f186defb3d57538b34e14a24a94f6e29906f191630b627dcb091198bae438' },
    @{ Name = 'cargo'; Component = 'cargo'; Sha256 = 'e4b9107e9f655d37200c884294e74930829f236a3e81982fc678e9fc56ae9a6b' },
    @{ Name = 'rust-std'; Component = "rust-std-$rustTarget"; Sha256 = 'b17e3bc962ff24b3bc47a750d838defed85b280746250e3a3ad029e6c1a29dd7' },
    @{ Name = 'rust-mingw'; Component = 'rust-mingw'; Sha256 = '1dca8e0d253565dbf21296eabc3e8d6e06e607e92c59dc00e3ecff902fe3d5ae' }
)

New-Item -ItemType Directory -Force -Path $downloadRoot, $toolchainRoot | Out-Null
foreach ($component in $components) {
    $archiveRoot = "$($component.Name)-$rustVersion-$rustTarget"
    $archiveName = "$archiveRoot.tar.xz"
    $archivePath = Join-Path $downloadRoot $archiveName
    $archiveUrl = "https://static.rust-lang.org/dist/$releaseDate/$archiveName"
    if (-not (Test-Path -LiteralPath $archivePath)) {
        Write-Host "Downloading $archiveUrl"
        & curl.exe --fail --location --retry 2 --silent --show-error --output $archivePath $archiveUrl
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $archiveName" }
    }
    $actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $component.Sha256) { throw "SHA256 mismatch: $archiveName" }
    Write-Host "Verified $archiveName ($((Get-Item -LiteralPath $archivePath).Length) bytes)"
    & tar.exe -xf $archivePath -C $toolchainRoot --strip-components 2 "$archiveRoot/$($component.Component)"
    if ($LASTEXITCODE -ne 0) { throw "Extraction failed: $archiveName" }
}

& (Join-Path $toolchainRoot 'bin\rustc.exe') --version --verbose
if ($LASTEXITCODE -ne 0) { throw 'Portable rustc could not run.' }
& (Join-Path $toolchainRoot 'bin\cargo.exe') --version
if ($LASTEXITCODE -ne 0) { throw 'Portable cargo could not run.' }
Write-Host "Portable toolchain: $toolchainRoot"
