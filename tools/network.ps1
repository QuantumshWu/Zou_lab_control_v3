# Network helpers for the installer. Importing this file performs no IO and
# reads no proxy settings. Policies are resolved per destination when used.
function Protect-ZlcNetworkText([string]$Text) {
    $Text = $Text -replace '(?i)(https?|socks5h?|socks4a?)://[^/\s@]+@', '$1://[redacted]@'
    # Git/curl also accept proxy credentials without a URL scheme and can
    # repeat that spelling in parse errors (for example, a nonnumeric port).
    $Text = $Text -replace '(?<![\w])[^\s/''"<>:@]+:[^\s/''"<>@]+@', '[redacted]@'
    $Text -replace '(?i)([?&](?:token|access_token|password|key|signature|sig)=)[^&\s]+', '$1[redacted]'
}

function ConvertTo-ZlcArgument([string]$Value) {
    # CommandLineToArgvW quoting, including trailing backslashes inside quotes.
    '"' + ([regex]::Replace($Value, '(\\*)"', '$1$1\"') -replace '(\\+)$', '$1$1') + '"'
}

function Initialize-ZlcProcessJob {
    if ('ZlcInstallerProcessJob' -as [type]) { return }
    Add-Type -TypeDefinition @'
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
public sealed class ZlcInstallerProcessJob : IDisposable {
    [StructLayout(LayoutKind.Sequential)] struct Basic {
        public long ProcessTime, JobTime; public uint Flags; public UIntPtr Min, Max;
        public uint Active; public UIntPtr Affinity; public uint Priority, Scheduling;
    }
    [StructLayout(LayoutKind.Sequential)] struct Io { public ulong A,B,C,D,E,F; }
    [StructLayout(LayoutKind.Sequential)] struct Extended {
        public Basic Basic; public Io Io; public UIntPtr ProcessMemory, JobMemory, PeakProcess, PeakJob;
    }
    [DllImport("kernel32.dll")] static extern IntPtr CreateJobObject(IntPtr a, string name);
    [DllImport("kernel32.dll")] static extern bool SetInformationJobObject(IntPtr h, int c, ref Extended i, uint n);
    [DllImport("kernel32.dll")] static extern bool AssignProcessToJobObject(IntPtr h, IntPtr process);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr h);
    IntPtr handle;
    public ZlcInstallerProcessJob(Process process) {
        handle = CreateJobObject(IntPtr.Zero, null);
        var info = new Extended(); info.Basic.Flags = 0x2000; // KILL_ON_JOB_CLOSE
        if (handle == IntPtr.Zero || !SetInformationJobObject(handle, 9, ref info, (uint)Marshal.SizeOf(info))
            || !AssignProcessToJobObject(handle, process.Handle)) {
            Dispose(); throw new InvalidOperationException("Cannot contain the installer subprocess tree");
        }
    }
    public void Dispose() { if (handle != IntPtr.Zero) { CloseHandle(handle); handle = IntPtr.Zero; } }
}
'@
}

function Invoke-ZlcCapturedProcess([string]$File, [string[]]$Arguments, [int]$Seconds = 60, [System.Collections.IDictionary]$Environment = @{}) {
    Initialize-ZlcProcessJob
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = $File
    $start.Arguments = ($Arguments | ForEach-Object { ConvertTo-ZlcArgument $_ }) -join ' '
    $start.UseShellExecute = $false; $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true; $start.RedirectStandardError = $true
    foreach ($key in $Environment.Keys) { $start.EnvironmentVariables[$key] = [string]$Environment[$key] }
    $process = [Diagnostics.Process]::Start($start)
    $job = $null
    try {
        try { $job = New-Object ZlcInstallerProcessJob($process) }
        catch { try { $process.Kill() } catch {}; throw }
        $watch = [Diagnostics.Stopwatch]::StartNew()
        $out = $process.StandardOutput.ReadToEndAsync(); $err = $process.StandardError.ReadToEndAsync()
        $exited = $process.WaitForExit($Seconds * 1000)
        $left = [math]::Max(0, $Seconds * 1000 - [int]$watch.ElapsedMilliseconds)
        # A parent that exited does not imply EOF: helpers can still own both pipes.
        if (-not $exited -or -not [Threading.Tasks.Task]::WaitAll([Threading.Tasks.Task[]]@($out, $err), $left)) {
            return [pscustomobject]@{ Code = -1; Lines = @("The operation timed out after $Seconds s; its subprocesses were stopped.") }
        }
        $lines = @(($out.Result + "`n" + $err.Result) -split "`r?`n" | Where-Object { $_ -ne '' })
        [pscustomobject]@{ Code = $process.ExitCode; Lines = $lines }
    } finally { if ($job) { $job.Dispose() }; $process.Dispose() }
}

function Test-ZlcNoProxy([Uri]$Url, [string]$Rules) {
    foreach ($rule in ($Rules -split '[,;]' | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
        if ($rule -eq '*') { return $true }
        # A literal host, optional port, or a DNS suffix (curl's NO_PROXY forms).
        $hostRule = $rule
        if ($rule -match '^\[([^\]]+)\](?::(\d+))?$') {
            $hostRule = $Matches[1]
            if ($Matches[2] -and [int]$Matches[2] -ne $Url.Port) { continue }
        } elseif ($rule -match '^([^:]+):(\d+)$') {
            $hostRule = $Matches[1]
            if ([int]$Matches[2] -ne $Url.Port) { continue }
        }
        $hostRule = $hostRule.TrimStart('.').TrimEnd('.').ToLowerInvariant()
        $hostName = $Url.DnsSafeHost.Trim('[', ']').TrimEnd('.').ToLowerInvariant()
        if ($hostRule -match '^(.+)/(\d+)$') {
            $network = $null; $address = $null; $bits = [int]$Matches[2]
            if ([Net.IPAddress]::TryParse($Matches[1], [ref]$network) -and [Net.IPAddress]::TryParse($hostName, [ref]$address)) {
                $a = $network.GetAddressBytes(); $b = $address.GetAddressBytes()
                if ($a.Length -eq $b.Length -and $bits -le 8 * $a.Length) {
                    $matchesPrefix = $true
                    for ($i = 0; $i -lt $a.Length -and $bits -gt 0; $i++) {
                        $used = [math]::Min(8, $bits); $mask = (255 -shl (8 - $used)) -band 255
                        if ((($a[$i] -bxor $b[$i]) -band $mask) -ne 0) { $matchesPrefix = $false; break }
                        $bits -= $used
                    }
                    if ($matchesPrefix) { return $true }
                }
            }
            continue
        }
        if ($hostName -eq $hostRule -or $hostName.EndsWith(".$hostRule")) { return $true }
    }
    $false
}

# Pure policy resolver: tests supply all settings and a fake system resolver.
function Resolve-ZlcProxyPolicy([Uri]$Url, [bool]$GitConfigured, [string]$GitProxy,
    [System.Collections.IDictionary]$Environment, [scriptblock]$SystemProxy) {
    if (Test-ZlcNoProxy $Url $Environment['NO_PROXY']) { return @{ Source = 'NO_PROXY'; Address = ''; InjectGit = $false } }
    if ($GitConfigured) { return @{ Source = 'git configuration'; Address = $GitProxy; InjectGit = $false } }
    $names = if ($Url.Scheme -eq 'https') { @('HTTPS_PROXY', 'ALL_PROXY') } else { @('HTTP_PROXY', 'ALL_PROXY') }
    foreach ($name in $names) {
        if ($Environment[$name]) { return @{ Source = $name; Address = [string]$Environment[$name]; InjectGit = $false } }
    }
    $address = & $SystemProxy $Url
    @{ Source = 'Windows system proxy'; Address = [string]$address; InjectGit = $true }
}

function Get-ZlcGitProxy([Uri]$Target, [string]$Git, [string]$RemoteName = '') {
    if ($Git -and $Target.Scheme -in @('http', 'https')) {
        if (-not $RemoteName) {
            # HTTP release requests belong to the tracked repository too. A
            # different mirror host must not inherit that remote-only policy.
            $branch = Invoke-Git $Git @('symbolic-ref', '--short', '-q', 'HEAD') 10
            if ($branch.Code -eq 0 -and $branch.Lines.Count) {
                $remote = Invoke-Git $Git @('config', "branch.$($branch.Lines[0]).remote") 10
                if ($remote.Code -eq 0 -and $remote.Lines.Count) {
                    $url = Invoke-Git $Git @('remote', 'get-url', $remote.Lines[0]) 10
                    $remoteUri = $null
                    if ($url.Code -eq 0 -and $url.Lines.Count -and [Uri]::TryCreate($url.Lines[0], [UriKind]::Absolute, [ref]$remoteUri)) {
                        $prefix = ($remoteUri.AbsolutePath -replace '\.git$', '').TrimEnd('/') + '/'
                        if ($remoteUri.Host -eq $Target.Host -and ($Target.AbsolutePath + '/').StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { $RemoteName = $remote.Lines[0] }
                    }
                }
            }
        }
        if ($RemoteName) {
            $answer = Invoke-Git $Git @('config', '--get', "remote.$RemoteName.proxy") 10
            if ($answer.Code -eq 0) { return @{ Configured = $true; Address = "$($answer.Lines | Select-Object -First 1)"; Source = 'git remote proxy' } }
            if ($answer.Code -ne 1) { throw 'Could not read the tracked remote proxy policy from git; retry the operation.' }
        }
        $lookup = New-Object UriBuilder($Target); $lookup.UserName = ''; $lookup.Password = ''; $lookup.Query = ''; $lookup.Fragment = ''
        $answer = Invoke-Git $Git @('config', '--get-urlmatch', 'http.proxy', $lookup.Uri.AbsoluteUri) 10
        if ($answer.Code -notin @(0, 1)) { throw 'Could not read the destination proxy policy from git; retry the operation.' }
        if ($answer.Code -eq 0) { return @{ Configured = $true; Address = "$($answer.Lines | Select-Object -First 1)"; Source = 'git URL proxy' } }
    }
    @{ Configured = $false; Address = ''; Source = '' }
}

function Get-ZlcProxyPolicy([string]$Url, [string]$Git, [string]$RemoteName = '') {
    $target = [Uri]$Url
    $gitPolicy = Get-ZlcGitProxy $target $Git $RemoteName
    $environment = @{}
    foreach ($name in @('HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY', 'NO_PROXY')) { $environment[$name] = [Environment]::GetEnvironmentVariable($name) }
    $policy = Resolve-ZlcProxyPolicy $target $gitPolicy.Configured $gitPolicy.Address $environment {
        param($destination)
        $proxy = [Net.WebRequest]::GetSystemWebProxy()
        if ($proxy -and -not $proxy.IsBypassed($destination)) {
            $address = $proxy.GetProxy($destination)
            if ($address -and $address -ne $destination) { return $address.AbsoluteUri }
        }
        ''
    }
    if ($policy.Source -eq 'git configuration') { $policy.Source = $gitPolicy.Source }
    $policy
}

function Add-ZlcGitProxyEnvironment([System.Collections.IDictionary]$Environment, $Policy) {
    if (-not $Policy.InjectGit) { return }
    $count = 0
    $existing = [Environment]::GetEnvironmentVariable('GIT_CONFIG_COUNT')
    if ($existing -and (-not [int]::TryParse($existing, [ref]$count) -or $count -lt 0)) { throw 'GIT_CONFIG_COUNT is invalid' }
    $Environment["GIT_CONFIG_KEY_$count"] = 'http.proxy'
    $Environment["GIT_CONFIG_VALUE_$count"] = $Policy.Address
    $Environment['GIT_CONFIG_COUNT'] = [string]($count + 1)
}

function New-ZlcWebRequest([string]$Url, [int]$Timeout = 30000, $Policy = $null) {
    if (-not $Policy) { $Policy = Get-ZlcProxyPolicy $Url (Find-Git) }
    $request = [Net.WebRequest]::Create($Url)
    if ($policy.Address) {
        $address = $policy.Address
        if ($address -notmatch '^\w+://') { $address = "http://$address" }
        $proxyUri = [Uri]$address
        if ($proxyUri.Scheme -notin @('http', 'https')) { throw "The $($policy.Source) uses a proxy protocol unsupported by Windows HTTP; select an HTTP proxy for the installer." }
        $clean = New-Object UriBuilder($proxyUri); $clean.UserName = ''; $clean.Password = ''
        $proxy = New-Object Net.WebProxy($clean.Uri)
        $proxy.Credentials = [Net.CredentialCache]::DefaultNetworkCredentials
        if ($proxyUri.UserInfo) {
            $credentials = $proxyUri.UserInfo -split ':', 2
            $password = if ($credentials.Count -gt 1) { [Uri]::UnescapeDataString($credentials[1]) } else { '' }
            $proxy.Credentials = New-Object Net.NetworkCredential([Uri]::UnescapeDataString($credentials[0]), $password)
        }
        $request.Proxy = $proxy
    } else { $request.Proxy = $null }
    $request.Timeout = $Timeout
    if ($request -is [Net.HttpWebRequest]) { $request.UserAgent = 'zlc-fetch-engine'; $request.ReadWriteTimeout = $Timeout }
    $request
}

# Windows HTTP does not implement SOCKS. Use Windows' curl for an explicit
# SOCKS policy; its proxy credentials stay in the child environment, not argv.
function Invoke-ZlcSocksHttp([string]$Url, $Policy, [string]$Destination = '', [int]$Seconds = 300, [switch]$HeadersOnly) {
    $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
    if (-not (Test-Path -LiteralPath $curl)) { $curl = (Get-Command curl.exe -ErrorAction SilentlyContinue).Source }
    if (-not $curl) { throw 'This explicit SOCKS proxy needs curl.exe (included with current Windows); use an HTTP proxy or install curl.' }
    $environment = @{ HTTPS_PROXY = $Policy.Address; HTTP_PROXY = $Policy.Address; ALL_PROXY = $Policy.Address }
    $arguments = @('--disable', '--silent', '--show-error', '--fail', '--connect-timeout', [string][math]::Min(15, $Seconds), '--max-time', [string]$Seconds)
    if ($HeadersOnly) { $arguments += @('--head', '--dump-header', '-', '--output', 'NUL') }
    else { $arguments += @('--location', '--output', $Destination) }
    $answer = Invoke-ZlcCapturedProcess $curl ($arguments + @('--url', $Url)) ($Seconds + 1) $environment
    if ($answer.Code -ne 0) { throw (Protect-ZlcNetworkText ("HTTP transfer failed: " + ($answer.Lines -join ' '))) }
    if ($HeadersOnly) {
        $location = @($answer.Lines | Where-Object { $_ -match '(?i)^location:\s*(.*)$' } | ForEach-Object { $_ -replace '(?i)^location:\s*', '' }) | Select-Object -Last 1
        return [pscustomobject]@{ Location = $location }
    }
}

function Test-ZlcArchive([string]$Path, $Asset) {
    (Test-Path -LiteralPath $Path -PathType Leaf) -and (Get-Item -LiteralPath $Path).Length -eq $Asset.size -and
        (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -eq $Asset.sha256
}

function Save-ZlcResponse($Response, [string]$Path, [long]$ExpectedSize, [double]$Seconds = 300) {
    $stream = $null; $file = $null; $done = 0L
    $watch = [Diagnostics.Stopwatch]::StartNew()
    try {
        $stream = $Response.GetResponseStream()
        if (-not $stream.CanTimeout) { throw 'The HTTP response cannot enforce a read deadline; retry using a supported HTTP transport.' }
        $file = [IO.File]::Create($Path)
        $buffer = New-Object byte[] (1MB)
        $shown = 0; $lastProgress = 0.0
        while ($true) {
            $left = [int](1000 * ($Seconds - $watch.Elapsed.TotalSeconds))
            if ($left -le 0) { throw "The release download exceeded $Seconds s; retry the install." }
            $stream.ReadTimeout = [math]::Min(30000, $left)
            $count = $stream.Read($buffer, 0, $buffer.Length)
            if ($count -eq 0) { break }
            $file.Write($buffer, 0, $count); $done += $count
            if ($done -gt $ExpectedSize) { throw 'The server returned more bytes than the release manifest permits.' }
            $tenth = [math]::Floor(10 * $done / [math]::Max(1, $ExpectedSize))
            if ($tenth -gt $shown -or $watch.Elapsed.TotalSeconds - $lastProgress -ge 2) {
                $shown = $tenth; $lastProgress = $watch.Elapsed.TotalSeconds
                Write-Host ('  {0,5:N1} of {1:N1} MB' -f ($done / 1MB), ($ExpectedSize / 1MB))
            }
        }
    } catch {
        if ($file) { $file.Dispose(); $file = $null }
        Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
        throw
    } finally {
        if ($file) { $file.Dispose() }; if ($stream) { $stream.Dispose() }; $Response.Dispose()
    }
    $done
}
