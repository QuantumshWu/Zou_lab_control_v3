param(
    # installation.exe's check: one JSON line -- the release this checkout names (bin\release.json),
    # the engine in bin\engine, and how they relate. Nothing changes.
    [switch]$Status,
    # installation.exe's online check: one JSON line -- the newest release published on GitHub and
    # whether it is newer than this checkout's (a short timeout). Nothing changes.
    [switch]$Latest,
    # Only put a downloaded engine (bin\engine.next) in place, waiting up to this many seconds for
    # the programs running from bin to end (the four programs' first step).
    [int]$Swap = -1,
    # Print what would be downloaded and where, and change nothing.
    [switch]$DryRun,
    # The program that asked (the four programs' bootstrap), named first.
    [string]$For = '',
    # A failure waits for Enter (the bootstrap's own console window).
    [switch]$Pause
)
# The engine a lab PC runs -- bin\engine (zlc-desktop, zlc-pulse-server, the page) and bin\ai
# (zlc-mcp) -- is not in git: bin\release.json names the release this checkout was released with
# (version, tag, the source stamp and commit it was built from, the asset's name, size and
# sha256), and this script puts that release in place. It downloads the asset from the GitHub
# release (ZLC_RELEASES names another server with the same paths), checks its size and sha256,
# extracts it beside the live engine (bin\engine.next, renamed in only once whole) and swaps it in
# by renaming: the live folder aside, the new one in. A folder a program runs from is never
# replaced; the download waits in bin\engine.next until the four programs next start.
$ErrorActionPreference = 'Stop'
trap { Write-Host "ERROR: $($_.Exception.Message)"; if ($Pause) { Read-Host 'Press Enter to close' | Out-Null }; exit 1 }
. (Join-Path $PSScriptRoot 'tools_root.ps1')
$root = Get-ZlcCheckout
$bin = Join-Path $root 'bin'
$record = Join-Path $bin 'release.json'
$engine = Get-ZlcReleaseProducts
$next = Join-Path $bin 'engine.next'
$parts = @('engine', 'ai')
$releases = if ($env:ZLC_RELEASES) { $env:ZLC_RELEASES.TrimEnd('/') } else { "https://github.com/$ZlcRepository/releases" }

# A release's identity as a SOURCE file states it (source, commit, version), or $null.
function Read-Source([string]$File) {
    if (-not (Test-Path -LiteralPath $File)) { return $null }
    $facts = [ordered]@{ source = $null; commit = $null; version = $null }
    foreach ($line in Get-Content -LiteralPath $File) { if ($line -match '^(source|commit|version) (\S+)$') { $facts[$Matches[1]] = $Matches[2] } }
    if ($facts.source) { [pscustomobject]$facts } else { $null }
}
function Get-Version([string]$Text) { $parsed = $null; if ([version]::TryParse("$Text", [ref]$parsed)) { $parsed } else { $null } }
# Whether a program runs from this folder: a running executable cannot be opened for writing.
function Test-InUse([string]$Folder) {
    foreach ($file in @(Get-ChildItem -LiteralPath $Folder -Filter '*.exe' -File -ErrorAction SilentlyContinue)) {
        try { [IO.File]::Open($file.FullName, 'Open', 'ReadWrite', 'None').Dispose() } catch { return $true }
    }
    $false
}
function Remove-Folder([string]$Folder) { if (Test-Path -LiteralPath $Folder) { [IO.Directory]::Delete($Folder, $true) } }

$expected = if (Test-Path -LiteralPath $record) { Get-Content -LiteralPath $record -Raw | ConvertFrom-Json } else { $null }
$installed = if (Test-ZlcProducts $engine) { Read-Source (Join-Path $engine 'SOURCE') } else { $null }
# What bin\engine.next holds: its SOURCE is the release's (copied there when it was staged).
$staged = Read-Source (Join-Path $next 'SOURCE')
$current = { param($facts) $facts -and $expected -and $facts.source -eq $expected.source -and $facts.version -eq $expected.version }

# Keep the independently runnable tool at the checkout root after an engine
# update. Older engine archives do not contain it and keep their old behavior.
function Sync-AtomWorkbench {
    $source = Join-Path $engine 'atom_workbench.exe'
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { return $true }
    $destination = Join-Path $root 'atom_workbench.exe'
    if ((Test-Path -LiteralPath $destination -PathType Leaf) -and
        (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -eq (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash) { return $true }
    $temporary = Join-Path $bin 'atom_workbench.next.exe'
    Copy-Item -LiteralPath $source -Destination $temporary -Force
    try {
        if (Test-Path -LiteralPath $destination) { [IO.File]::Replace($temporary, $destination, $null) }
        else { [IO.File]::Move($temporary, $destination) }
        return $true
    } catch {
        Write-Host 'Atom Workbench is still running: close it and start installation.exe again to finish its update.'
        return $false
    }
}

# Put bin\engine.next in place, folder by folder: the live one renamed aside, the new one renamed
# in, so each folder is always whole. A folder in use waits up to $Seconds, then stays as it is
# with its replacement left in bin\engine.next; one downloaded for another release is dropped.
# True when nothing is left to put in place.
function Invoke-Swap([int]$Seconds) {
    if (-not (Test-Path -LiteralPath $next)) { return (Sync-AtomWorkbench) }
    if (-not (& $current $staged)) { Remove-Folder $next; return $true }
    foreach ($part in $parts) {
        $new = Join-Path $next $part
        if (-not (Test-Path -LiteralPath $new)) { continue }
        $live = Join-Path $bin $part
        $deadline = (Get-Date).AddSeconds($Seconds)
        while ((Test-InUse $live) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 250 }
        if (Test-InUse $live) { continue }
        $old = "$live.old"
        Remove-Folder $old
        if (Test-Path -LiteralPath $live) { [IO.Directory]::Move($live, $old) }
        try { [IO.Directory]::Move($new, $live) }
        catch { if (Test-Path -LiteralPath $old) { [IO.Directory]::Move($old, $live) }; throw }
        Remove-Folder $old
    }
    if (@(Get-ChildItem -LiteralPath $next -Directory).Count) { return $false }
    Remove-Folder $next
    Sync-AtomWorkbench
}

if ($Swap -ge 0) {
    if (Invoke-Swap $Swap) { Write-Host 'Nothing waits to be put in place.' } else { Write-Host 'The downloaded program waits: a program still runs from bin.' }
    return
}

if ($Status) {
    # A development machine runs what it builds from these sources (tools\desktop.ps1, build_release).
    $built = $null
    if (-not @(Get-ZlcToolchainGaps).Count) {
        $stamp = Get-ZlcSourceStamp
        $built = @($engine, (Get-ZlcDevProducts)) | Where-Object { $stamp -and (Test-ZlcProducts $_) -and (Get-ZlcProductsStamp $_) -eq $stamp } | Select-Object -First 1
    }
    $relation = if (-not $expected) { 'unreleased' }
        elseif (& $current $installed) { 'match' }
        elseif (& $current $staged) { 'staged' }
        elseif ($built) { 'built' }
        elseif (-not $installed) { 'missing' }
        else {
            $have = Get-Version $installed.version; $want = Get-Version $expected.version
            if ($have -and $want -and $have -lt $want) { 'older' } elseif ($have -and $want -and $have -gt $want) { 'newer' } else { 'other' }
        }
    [ordered]@{ expected = $expected; installed = $installed; relation = $relation; built = $built } | ConvertTo-Json -Compress -Depth 4
    return
}

# GitHub from PowerShell 5.1: TLS 1.2, and the system proxy with this user's credentials.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
if ([Net.WebRequest]::DefaultWebProxy) { [Net.WebRequest]::DefaultWebProxy.Credentials = [Net.CredentialCache]::DefaultNetworkCredentials }

if ($Latest) {
    # releases/latest answers with a redirect to releases/tag/<the newest tag>.
    $request = [Net.WebRequest]::Create("$releases/latest")
    $request.AllowAutoRedirect = $false
    $request.Timeout = 5000
    $request.UserAgent = 'zlc-fetch-engine'
    $response = $request.GetResponse()
    try { $location = "$($response.Headers['Location'])" } finally { $response.Dispose() }
    $tag = if ($location -match '/tag/([^/?#]+)$') { [Uri]::UnescapeDataString($Matches[1]) } else { $null }
    $newer = $tag -and $expected -and (Get-Version ($tag -replace '^v', '')) -gt (Get-Version $expected.version)
    [ordered]@{ tag = $tag; newer = [bool]$newer } | ConvertTo-Json -Compress
    return
}

if ($For) {
    $Host.UI.RawUI.WindowTitle = "Zou lab control: the program for $For.exe"
    Write-Host "$For.exe needs the program of this checkout's release: this window puts it in place."
}
if (-not $expected) { throw "This checkout names no release ($record): on a development machine bin\scripts\build_release.bat makes one." }
$asset = $expected.asset
Write-Host "Release: v$($expected.version) (built from commit $($expected.commit.Substring(0, 12)))"
Write-Host "Program: $engine$(if ($installed) { " (now v$($installed.version), commit $("$($installed.commit)".Substring(0, [math]::Min(12, "$($installed.commit)".Length))))" } else { ' (none yet)' })"
$url = "$releases/download/$($expected.tag)/$($asset.name)"
$download = -not (& $current $installed) -and -not (& $current $staged)
if ($DryRun) {
    Write-Host ''
    Write-Host 'Dry run -- these steps would run:'
    if ($download) { Write-Host "  download $url ($([math]::Round($asset.size / 1MB, 1)) MB), check its size and sha256 $($asset.sha256), and extract it to $next" }
    if ((Test-Path -LiteralPath $next) -or $download) { Write-Host '  put bin\engine.next in place of bin\engine and bin\ai (now, or when the four programs next start if one runs from bin)' }
    else { Write-Host '  nothing: the release is in place' }
    return
}
if ($download) {
    $local = if ($env:ZLC_V3_LOCAL) { $env:ZLC_V3_LOCAL } else { Join-Path $root '.local' }
    $downloads = Join-Path $local 'downloads'
    New-Item -ItemType Directory -Force -Path $downloads | Out-Null
    $zip = Join-Path $downloads $asset.name
    Write-Host "Downloading $url ($([math]::Round($asset.size / 1MB, 1)) MB) ..."
    $request = [Net.WebRequest]::Create($url)
    $request.Timeout = 30000
    if ($request -is [Net.HttpWebRequest]) { $request.UserAgent = 'zlc-fetch-engine'; $request.ReadWriteTimeout = 60000 }
    $response = $request.GetResponse()
    $done = 0L
    try {
        $in = $response.GetResponseStream()
        $out = [IO.File]::Create("$zip.part")
        try {
            $buffer = New-Object byte[] (1MB)
            $shown = 0
            while (($count = $in.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $out.Write($buffer, 0, $count); $done += $count
                $tenth = [math]::Floor(10 * $done / [math]::Max(1, $asset.size))
                if ($tenth -gt $shown) { $shown = $tenth; Write-Host ('  {0,5:N1} of {1:N1} MB' -f ($done / 1MB), ($asset.size / 1MB)) }
            }
        } finally { $out.Dispose(); $in.Dispose() }
    } finally { $response.Dispose() }
    $sha = (Get-FileHash -LiteralPath "$zip.part" -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($done -ne $asset.size -or $sha -ne $asset.sha256) {
        Remove-Item -LiteralPath "$zip.part"
        throw "The download is not the release: $done bytes, sha256 $sha (wanted $($asset.size) bytes, $($asset.sha256))."
    }
    Move-Item -LiteralPath "$zip.part" -Destination $zip -Force
    Write-Host 'Checked: the size and sha256 are the release''s.'
    # Extracted beside, checked, then renamed to bin\engine.next whole.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $partial = "$next.partial"
    Remove-Folder $partial
    [IO.Compression.ZipFile]::ExtractToDirectory($zip, $partial)
    Remove-Item -LiteralPath $zip
    $unpacked = Join-Path $partial 'engine'
    if (-not ((Test-ZlcProducts $unpacked) -and (& $current (Read-Source (Join-Path $unpacked 'SOURCE'))))) {
        Remove-Folder $partial
        throw "The archive does not hold this release's engine (source $($expected.source))."
    }
    Copy-Item -LiteralPath (Join-Path $unpacked 'SOURCE') -Destination (Join-Path $partial 'SOURCE')
    Remove-Folder $next
    [IO.Directory]::Move($partial, $next)
    $staged = Read-Source (Join-Path $next 'SOURCE')
}
if (Invoke-Swap 0) { Write-Host "In place: v$($expected.version)." }
else { Write-Host "Downloaded v$($expected.version): a program still runs from bin, so it goes in place when the four programs next start (close them, then start installation.exe)." }
