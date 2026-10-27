param(
    # installation.exe's check: one JSON line -- the release this checkout names (bin\release.json),
    # the engine in bin\engine, and how they relate. Nothing changes.
    [switch]$Status,
    # installation.exe's online check: two JSON lines -- the newest release published on GitHub,
    # whether it is newer than this checkout's (a short timeout), and how this checkout stands
    # against the branch it follows on GitHub: GitHub's own count, then git's after a fetch (the
    # working tree is not touched).
    [switch]$Latest,
    # Bring this checkout up to the branch it follows on GitHub: a fetch and a fast-forward only,
    # refused (nothing changed) when this checkout has commits of its own or local edits to files
    # the update changes, saying why. installation.exe runs it before the program step unless GitHub
    # is known to have nothing newer.
    [switch]$Pull,
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
    # Replacing a running Atom Workbench leaves the old file as atom_workbench.exe~RF*.TMP: gone once it ended.
    Get-ChildItem -LiteralPath $root -Filter 'atom_workbench.exe~RF*.TMP' -File -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue }
    $source = Join-Path $engine 'atom_workbench.exe'
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { return $true }
    $destination = Join-Path $root 'atom_workbench.exe'
    if ((Test-Path -LiteralPath $destination -PathType Leaf) -and
        (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -eq (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash) { return $true }
    $temporary = Join-Path $bin 'atom_workbench.next.exe'
    Copy-Item -LiteralPath $source -Destination $temporary -Force
    try {
        # No backup: [NullString]::Value, since PowerShell passes $null to a string parameter as "".
        if (Test-Path -LiteralPath $destination) { [IO.File]::Replace($temporary, $destination, [NullString]::Value) }
        else { [IO.File]::Move($temporary, $destination) }
        return $true
    } catch {
        $reason = $_.Exception.GetBaseException()
        # A sharing violation (0x20) is a running Atom Workbench; anything else is said as it is.
        if ($reason -is [IO.IOException] -and ($reason.HResult -band 0xFFFF) -eq 32) {
            Write-Host 'Atom Workbench is still running: close it and start installation.exe again to finish its update.'
        } else { Write-Host "Atom Workbench was not updated: $($reason.Message)" }
        return $false
    }
}

function Test-AtomWorkbenchCurrent {
    $source = Join-Path $engine 'atom_workbench.exe'
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { return $true }
    $destination = Join-Path $root 'atom_workbench.exe'
    (Test-Path -LiteralPath $destination -PathType Leaf) -and
        (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -eq (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
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
    if (-not (Sync-AtomWorkbench)) { return $false }
    Remove-Folder $next
    $true
}

# git: on PATH, else GitHub Desktop's own (the newest app-* of this user), else Git for Windows.
function Find-Git {
    $found = (Get-Command git.exe -ErrorAction SilentlyContinue).Source
    if ($found) { return $found }
    $desktop = Join-Path $env:LOCALAPPDATA 'GitHubDesktop'
    $bundled = @(Get-ChildItem -LiteralPath $desktop -Directory -Filter 'app-*' -ErrorAction SilentlyContinue |
        Sort-Object { Get-Version ($_.Name -replace '^app-', '') } -Descending |
        ForEach-Object { Join-Path $_.FullName 'resources\app\git\cmd\git.exe' } | Where-Object { Test-Path -LiteralPath $_ })
    if ($bundled.Count) { return $bundled[0] }
    foreach ($base in @($env:ProgramFiles, ${env:ProgramFiles(x86)}, (Join-Path $env:LOCALAPPDATA 'Programs'))) {
        if ($base -and (Test-Path -LiteralPath (Join-Path $base 'Git\cmd\git.exe'))) { return (Join-Path $base 'Git\cmd\git.exe') }
    }
    $null
}

# git -C <checkout> <arguments>, never asking for credentials, ended after $Seconds: its exit code
# and its output lines (stdout, then stderr).
function Invoke-Git([string]$Git, [string[]]$Arguments, [int]$Seconds = 60) {
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = $Git
    $start.Arguments = (@('-C', $root) + $Arguments | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }) -join ' '
    $start.UseShellExecute = $false; $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true; $start.RedirectStandardError = $true
    $start.EnvironmentVariables['GIT_TERMINAL_PROMPT'] = '0'
    $start.EnvironmentVariables['GCM_INTERACTIVE'] = 'never'
    $process = [Diagnostics.Process]::Start($start)
    $out = $process.StandardOutput.ReadToEndAsync(); $err = $process.StandardError.ReadToEndAsync()
    if (-not $process.WaitForExit($Seconds * 1000)) {
        # Its own children (git-remote-https) keep the pipes open: what they wrote is not waited for.
        try { $process.Kill() } catch {}
        return [pscustomobject]@{ Code = -1; Lines = @("git $($Arguments[0]) gave no answer in $Seconds s") }
    }
    $process.WaitForExit()
    $lines = @(($out.Result + "`n" + $err.Result) -split "`r?`n" | Where-Object { $_ -ne '' })
    [pscustomobject]@{ Code = $process.ExitCode; Lines = $lines }
}

# Why a git command failed, as git said it: its lines up to its first fatal: or error: one (ssh's
# own reason comes before git's).
function Get-GitReason($Answer) {
    $said = foreach ($line in $Answer.Lines) { $line.Trim(); if ($line -match '^(fatal|error):') { break } }
    @($said) -join ' '
}

# This checkout's branch and commit as its .git folder records them, read without git, or $null
# (no branch checked out). A worktree's .git names its folder, whose commondir holds the refs.
function Read-GitHead {
    $dir = Join-Path $root '.git'
    if (Test-Path -LiteralPath $dir -PathType Leaf) {
        if ((Get-Content -LiteralPath $dir -TotalCount 1) -notmatch '^gitdir:\s*(.+)$') { return $null }
        $dir = [IO.Path]::GetFullPath([IO.Path]::Combine($root, $Matches[1].Trim()))
    }
    $head = Join-Path $dir 'HEAD'
    if (-not (Test-Path -LiteralPath $head) -or (Get-Content -LiteralPath $head -TotalCount 1) -notmatch '^ref:\s*refs/heads/(\S+)$') { return $null }
    $branch = $Matches[1]
    $common = Join-Path $dir 'commondir'
    $common = if (Test-Path -LiteralPath $common) { [IO.Path]::GetFullPath([IO.Path]::Combine($dir, (Get-Content -LiteralPath $common -TotalCount 1).Trim())) } else { $dir }
    $loose = Join-Path $common "refs\heads\$branch"
    $packed = Join-Path $common 'packed-refs'
    $commit = if (Test-Path -LiteralPath $loose) { (Get-Content -LiteralPath $loose -TotalCount 1).Trim() }
        elseif (Test-Path -LiteralPath $packed) { "$(Get-Content -LiteralPath $packed | Where-Object { $_ -match " refs/heads/$([regex]::Escape($branch))$" } | Select-Object -First 1)".Split(' ')[0] }
    if ("$commit" -match '^[0-9a-f]{40}$') { [pscustomobject]@{ branch = $branch; commit = $commit } } else { $null }
}

# How many commits GitHub's branch of this checkout's branch name has that this checkout does not,
# asked of GitHub itself (its compare API, no git: a second or two), or $null when GitHub cannot say.
function Get-BehindOnGitHub {
    $head = Read-GitHead
    if (-not $head) { return $null }
    try {
        $request = [Net.WebRequest]::Create("https://api.github.com/repos/$ZlcRepository/compare/$($head.commit)...$([Uri]::EscapeDataString($head.branch))")
        $request.Timeout = 10000
        $request.UserAgent = 'zlc-fetch-engine'
        $request.Accept = 'application/vnd.github+json'
        $response = $request.GetResponse()
        try { $text = (New-Object IO.StreamReader $response.GetResponseStream()).ReadToEnd() } finally { $response.Dispose() }
        [int]($text | ConvertFrom-Json).ahead_by
    } catch { $null }
}

# How this checkout stands against the branch it follows on GitHub, after fetching it: commits
# behind and ahead ($null: not learned), the local edits an update would overwrite, and whether a
# fast-forward can bring it up (pullable). 'why' says, in the operator's words, why it cannot; 'url'
# says where git fetched from. $Say: the fetch is announced first.
function Get-PullState([int]$Seconds, [switch]$Say) {
    $state = [ordered]@{ git = $null; branch = $null; upstream = $null; url = $null; behind = $null; ahead = $null; blocked = @(); pullable = $false; why = $null }
    $git = Find-Git
    if (-not $git) { $state.why = 'git was not found on this PC: pull with GitHub Desktop'; return $state }
    $state.git = $git
    $branch = Invoke-Git $git @('symbolic-ref', '--short', '-q', 'HEAD') 10
    if ($branch.Code -ne 0 -or -not $branch.Lines.Count) { $state.why = 'this checkout is on no branch: pull with GitHub Desktop'; return $state }
    $state.branch = $branch.Lines[0]
    $upstream = Invoke-Git $git @('rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}') 10
    if ($upstream.Code -ne 0) { $state.why = "branch $($state.branch) follows no branch on GitHub: pull with GitHub Desktop"; return $state }
    $state.upstream = $upstream.Lines[0]
    $remote = (Invoke-Git $git @('config', "branch.$($state.branch).remote") 10).Lines | Select-Object -First 1
    $state.url = (Invoke-Git $git @('remote', 'get-url', $remote) 10).Lines | Select-Object -First 1
    if ($Say) { Write-Host "Fetching $($state.url) ..." }
    $fetch = Invoke-Git $git @('fetch', '--quiet', $remote) $Seconds
    if ($fetch.Code -ne 0) { $state.why = "git could not fetch $($state.url): $(Get-GitReason $fetch); pull with GitHub Desktop"; return $state }
    $counts = Invoke-Git $git @('rev-list', '--left-right', '--count', 'HEAD...@{u}') 10
    if ("$($counts.Lines | Select-Object -First 1)" -match '^(\d+)\s+(\d+)$') { $state.ahead = [int]$Matches[1]; $state.behind = [int]$Matches[2] }
    else { $state.why = "git could not compare this checkout with $($state.upstream): $(Get-GitReason $counts)"; return $state }
    if ($state.behind -eq 0) { return $state }
    if ($state.ahead -gt 0) { $state.why = "this checkout has $($state.ahead) commit(s) of its own that GitHub does not: merge with GitHub Desktop"; return $state }
    # Local edits (staged, unstaged or new files) to paths the update changes would be overwritten.
    $incoming = @((Invoke-Git $git @('diff', '--name-only', 'HEAD', '@{u}') 30).Lines)
    $local = @((Invoke-Git $git @('diff', '--name-only', 'HEAD') 30).Lines) + @((Invoke-Git $git @('ls-files', '-o', '--exclude-standard') 30).Lines)
    $state.blocked = @($incoming | Where-Object { $local -contains $_ })
    if ($state.blocked.Count) {
        $state.why = "local edits to $($state.blocked.Count) file(s) the update changes ($(($state.blocked | Select-Object -First 3) -join ', ')): commit or discard them first"
        return $state
    }
    $state.pullable = $true
    $state
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
        elseif ((& $current $installed) -and (Test-AtomWorkbenchCurrent)) { 'match' }
        elseif (& $current $staged) { 'staged' }
        elseif (& $current $installed) { 'other' }
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
    $tag = $null
    try {
        $request = [Net.WebRequest]::Create("$releases/latest")
        $request.AllowAutoRedirect = $false
        $request.Timeout = 5000
        $request.UserAgent = 'zlc-fetch-engine'
        $response = $request.GetResponse()
        try { $location = "$($response.Headers['Location'])" } finally { $response.Dispose() }
        if ($location -match '/tag/([^/?#]+)$') { $tag = [Uri]::UnescapeDataString($Matches[1]) }
    } catch {}
    $newer = $tag -and $expected -and (Get-Version ($tag -replace '^v', '')) -gt (Get-Version $expected.version)
    # GitHub's own count first, while git has yet to fetch (the window says it meanwhile): a first
    # line marked fetching, then git's answer.
    $counted = Get-BehindOnGitHub
    [ordered]@{ tag = $tag; newer = [bool]$newer; pull = [ordered]@{ behind = $counted; fetching = $true } } | ConvertTo-Json -Compress -Depth 4
    $standing = Get-PullState 30
    # git could not say how this checkout stands (none, no branch it follows, a fetch that failed):
    # GitHub's count does, so the window never reads up to date when it cannot know.
    if ($null -eq $standing.behind -and $null -ne $counted) { $standing.behind = $counted; if (-not $counted) { $standing.why = $null } }
    [ordered]@{ tag = $tag; newer = [bool]$newer; pull = $standing } | ConvertTo-Json -Compress -Depth 4
    return
}

if ($Pull) {
    $state = Get-PullState 300 -Say
    if ($null -eq $state.behind) { throw $state.why }
    if ($state.behind -eq 0) { Write-Host "Up to date with $($state.upstream)."; return }
    $log = @((Invoke-Git $state.git @('log', '--oneline', '--no-decorate', '-n', '15', 'HEAD..@{u}') 30).Lines)
    Write-Host "$($state.upstream) on GitHub has $($state.behind) commit(s) this checkout does not:"
    $log | ForEach-Object { Write-Host "  $_" }
    if ($state.behind -gt $log.Count) { Write-Host "  ... and $($state.behind - $log.Count) more" }
    if (-not $state.pullable) { throw $state.why }
    if ($DryRun) { Write-Host 'Dry run -- a fast-forward would bring them in; nothing changed.'; return }
    $before = (Invoke-Git $state.git @('rev-parse', '--short', 'HEAD') 10).Lines | Select-Object -First 1
    $merge = Invoke-Git $state.git @('merge', '--ff-only', '--quiet', '@{u}') 300
    if ($merge.Code -ne 0) { throw "git could not fast-forward ($(Get-GitReason $merge)); nothing was changed" }
    $after = (Invoke-Git $state.git @('rev-parse', '--short', 'HEAD') 10).Lines | Select-Object -First 1
    Write-Host "Pulled: $before -> $after. The program step that follows installs the release this checkout now names."
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
