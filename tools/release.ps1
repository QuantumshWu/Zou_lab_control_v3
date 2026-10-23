param(
    # Build and stage the release and its archive, print the commit it would make, and stop.
    [switch]$NoCommit,
    # Publish the release bin\release.json names, once its commit is pushed: a GitHub release
    # of that commit carrying the archive (gh). With -DryRun the command is only printed.
    [switch]$Publish,
    [switch]$DryRun
)
# The engine release a lab PC runs without a toolchain: build executables from the
# committed Rust sources; place the engine in bin\engine and zlc-mcp in bin\ai
# (neither is in git) and pack both into target\engine-releases\zlc-engine-<version>.zip; write
# bin\release.json, which names that release (version yyyy.M.d.N, its tag, the source stamp and
# commit it was built from, the archive's name, size and sha256); commit that file with the four
# programs at the checkout's root. installation.exe downloads the archive from the release's
# GitHub page (tools\fetch_engine.ps1). Only this script writes these.
$ErrorActionPreference = 'Stop'
trap { Write-Host "ERROR: $($_.Exception.Message)"; exit 1 }
. (Join-Path $PSScriptRoot 'tools_root.ps1')
$root = Get-ZlcCheckout
$record = Join-Path $root 'bin\release.json'
$archives = Join-Path $root 'target\engine-releases'

if ($Publish) {
    $release = Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
    $zip = Join-Path $archives $release.asset.name
    if (-not (Test-Path -LiteralPath $zip)) { throw "$zip is not here: publish from the machine that built the release." }
    if ((Get-Item -LiteralPath $zip).Length -ne $release.asset.size -or (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $release.asset.sha256) {
        throw "$zip is not the archive bin\release.json names."
    }
    # The release is the commit that wrote bin\release.json; GitHub tags it there, so it is pushed first.
    $at = "$(& git -C $root log -1 --format=%H -- bin/release.json)".Trim()
    if (-not $at) { throw 'bin\release.json is not committed.' }
    if (-not @(& git -C $root branch -r --contains $at).Count) { throw "Push $at (the release commit) first." }
    $arguments = @('release', 'create', $release.tag, $zip, '--repo', $ZlcRepository, '--target', $at, '--title', "v$($release.version)",
                   '--notes', "The engine built from commit $($release.commit); installation.exe downloads it.")
    Write-Host "gh $($arguments -join ' ')"
    if ($DryRun) { Write-Host 'Dry run: nothing was published.'; return }
    if (-not (Get-Command gh.exe -ErrorAction SilentlyContinue)) { throw 'gh (the GitHub CLI) is not on PATH.' }
    & gh.exe @arguments
    if ($LASTEXITCODE -ne 0) { throw 'gh release create failed.' }
    return
}

$gaps = @(Get-ZlcToolchainGaps)
if ($gaps.Count) { throw "A release is built on a development machine: missing $($gaps -join ', '). Run tools\dev_setup.ps1." }
$dirty = @(& git -C $root status --porcelain -- @ZlcSourceSpec)
if ($dirty.Count) { throw "Commit or discard these source changes first, so the release is built from a commit:`n$($dirty -join "`n")" }
$stamp = Get-ZlcSourceStamp
if (-not $stamp) { throw 'The source stamp needs git and a git checkout.' }
$commit = (& git -C $root rev-parse HEAD).Trim()
# Today's date, numbered after a release made earlier the same day.
$today = Get-Date -Format 'yyyy.M.d'
$previous = if (Test-Path -LiteralPath $record) { "$((Get-Content -LiteralPath $record -Raw | ConvertFrom-Json).version)" } else { '' }
$version = "$today.$(if ($previous -match "^$([regex]::Escape($today))\.(\d+)$") { [int]$Matches[1] + 1 } else { 1 })"

$built = Get-ZlcDevProducts
$complete = -not @($ZlcProductNames | Where-Object { -not (Test-Path -LiteralPath (Join-Path $built $_)) }).Count
if (-not ($complete -and (Get-ZlcProductsStamp $built) -eq $stamp)) { $built = Invoke-ZlcBuild $stamp }
if ((Get-ZlcSourceStamp) -ne $stamp) { throw 'Rust sources changed during the build. Commit the intended sources and build again before publishing.' }
# A release names no folder of the machine it was built on (dev.ps1 remaps every source path).
$machine = @((Get-ZlcCheckout), $env:USERPROFILE, (Get-ZlcToolsRoot)) | Where-Object { $_ } | ForEach-Object { $_.TrimEnd('\') }
foreach ($name in $ZlcProductNames) {
    $bytes = [IO.File]::ReadAllBytes((Join-Path $built $name))
    $text = [Text.Encoding]::GetEncoding(28591).GetString($bytes) + "`n" + [Text.Encoding]::Unicode.GetString($bytes) + "`n" +
            [Text.Encoding]::Unicode.GetString($bytes, 1, $bytes.Length - 1)
    foreach ($path in $machine) {
        if ($text.IndexOf($path, [StringComparison]::OrdinalIgnoreCase) -ge 0) { throw "$name names $path; a release carries no machine path." }
    }
}

# What the release holds after this run, as checkout-relative path -> source file: the four
# programs at the root (one launcher, named for its mode), the engine in
# bin\engine, zlc-mcp in bin\ai.
$files = [ordered]@{}
foreach ($program in $ZlcPrograms) { $files["$program.exe"] = Join-Path $built 'zlc-launch.exe' }
foreach ($name in $ZlcEngineNames) { $files["bin\engine\$name"] = Join-Path $built $name }
$files['bin\ai\zlc-mcp.exe'] = Join-Path $built 'zlc-mcp.exe'
if ((Get-ZlcSourceStamp) -ne $stamp) { throw 'Rust sources changed before staging the release.' }
# Files a previous release had that this one does not (a renamed module) go.
foreach ($folder in 'bin\engine', 'bin\ai') {
    $full = Join-Path $root $folder
    if (-not (Test-Path -LiteralPath $full)) { continue }
    foreach ($item in Get-ChildItem -LiteralPath $full -Recurse -File) {
        $relative = $item.FullName.Substring($root.Length + 1)
        if ($relative -ne 'bin\engine\SOURCE' -and -not $files.Contains($relative)) { Remove-Item -LiteralPath $item.FullName }
    }
}
foreach ($relative in $files.Keys) {
    $target = Join-Path $root $relative
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    Copy-Item -LiteralPath $files[$relative] -Destination $target -Force
}
Set-Content -LiteralPath (Join-Path (Get-ZlcReleaseProducts) 'SOURCE') -Value @("source $stamp", "commit $commit", "version $version") -Encoding ascii

# The archive: bin\engine and bin\ai as engine\ and ai\.
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
New-Item -ItemType Directory -Force -Path $archives | Out-Null
$asset = "zlc-engine-$version.zip"
$zip = Join-Path $archives $asset
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip }
$archive = [IO.Compression.ZipFile]::Open($zip, 'Create')
try {
    foreach ($part in 'engine', 'ai') {
        $folder = Join-Path $root "bin\$part"
        foreach ($item in Get-ChildItem -LiteralPath $folder -Recurse -File) {
            $entry = "$part/$($item.FullName.Substring($folder.Length + 1).Replace('\', '/'))"
            [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $item.FullName, $entry, [IO.Compression.CompressionLevel]::Optimal)
        }
    }
}
finally { $archive.Dispose() }
if ((Get-ZlcSourceStamp) -ne $stamp) { throw 'Rust sources changed before recording the release.' }
$release = [ordered]@{ version = $version; tag = "v$version"; source = $stamp; commit = $commit
                       asset = [ordered]@{ name = $asset; size = (Get-Item -LiteralPath $zip).Length; sha256 = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant() } }
Set-Content -LiteralPath $record -Value ($release | ConvertTo-Json -Depth 3) -Encoding ascii
Write-Host "Release v${version}: $($files.Count) files from $commit (sources $stamp), $zip ($([math]::Round($release.asset.size / 1MB, 1)) MB)."

$paths = @('bin/release.json') + @($ZlcPrograms | ForEach-Object { "$_.exe" })
$message = "Release v${version}: the engine and zlc-mcp from $($commit.Substring(0, 12)) (bin\release.json), the four programs"
if ($NoCommit) {
    Write-Host 'Not committed (-NoCommit). The commit would be:'
    Write-Host "  git add -- $($paths -join ' ')"
    Write-Host "  git commit -m `"$message`" -- $($paths -join ' ')"
    return
}
& git -C $root add -- @paths
if ($LASTEXITCODE -ne 0) { throw 'git add failed.' }
& git -C $root commit -m $message -- @paths
if ($LASTEXITCODE -ne 0) { throw 'git commit failed.' }
Write-Host "Committed. Push it, then publish the archive: bin\scripts\build_release.bat --publish"
