# Offline regression tests. No network connection or actual proxy settings are read.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\network.ps1')
function Assert($Condition, [string]$Message) { if (-not $Condition) { throw $Message } }
$none = @{ HTTPS_PROXY = ''; HTTP_PROXY = ''; ALL_PROXY = ''; NO_PROXY = '' }
$system = { param($url) 'http://fake-system:7890' }
$envProxy = @{ HTTPS_PROXY = 'http://fake-env:8000'; ALL_PROXY = 'socks5://fake-all:9000'; NO_PROXY = '' }
$policy = Resolve-ZlcProxyPolicy 'https://example.invalid/file' $true '' $envProxy $system
Assert ($policy.Address -eq '' -and -not $policy.InjectGit) 'Explicit empty git proxy must remain direct'
$policy = Resolve-ZlcProxyPolicy 'https://example.invalid/file' $true 'http://fake-git:8888' $envProxy $system
Assert ($policy.Address -eq 'http://fake-git:8888') 'URL-matched git proxy wins over env/system'
$policy = Resolve-ZlcProxyPolicy 'https://example.invalid/file' $false '' $envProxy $system
Assert ($policy.Address -eq 'http://fake-env:8000' -and -not $policy.InjectGit) 'Explicit env policy must be preserved'
$policy = Resolve-ZlcProxyPolicy 'https://example.invalid/file' $false '' $none $system
Assert ($policy.Address -eq 'http://fake-system:7890' -and $policy.InjectGit) 'System proxy fills missing git policy'
$policy = Resolve-ZlcProxyPolicy 'https://example.invalid/file' $false '' $none { param($url) '' }
Assert ($policy.Address -eq '') 'System direct stays direct'
$script:proxyFixture = 'remote-empty'
function Invoke-Git([string]$Git, [string[]]$Arguments, [int]$Seconds) {
    if ($Arguments[1] -eq '--get') {
        if ($script:proxyFixture -eq 'remote-empty') { return [pscustomobject]@{ Code = 0; Lines = @() } }
        return [pscustomobject]@{ Code = 1; Lines = @() }
    }
    Assert ($Arguments[1] -eq '--get-urlmatch' -and -not $Arguments[3].Contains('fakepass')) 'Proxy lookup strips URL credentials from argv'
    [pscustomobject]@{ Code = 0; Lines = @('http://fake-url-proxy:8888') }
}
$gitPolicy = Get-ZlcGitProxy 'https://user:fakepass@example.invalid/repo.git' 'fake-git' 'origin'
Assert ($gitPolicy.Configured -and $gitPolicy.Address -eq '' -and $gitPolicy.Source -eq 'git remote proxy') 'Remote-specific empty config preserves direct'
$script:proxyFixture = 'url'
$gitPolicy = Get-ZlcGitProxy 'https://user:fakepass@example.invalid/repo.git' 'fake-git' 'origin'
Assert ($gitPolicy.Address -eq 'http://fake-url-proxy:8888') 'Missing remote proxy falls through to URL policy'
foreach ($rule in @('*', 'example.invalid', '.example.invalid', 'example.invalid:443')) {
    Assert (Test-ZlcNoProxy 'https://sub.example.invalid/file' $rule) "NO_PROXY $rule"
}
Assert (-not (Test-ZlcNoProxy 'https://example.invalid/' 'example.invalid:80')) 'NO_PROXY port is significant'
Assert (-not (Test-ZlcNoProxy 'https://notexample.invalid/' 'example.invalid')) 'NO_PROXY must respect domain boundary'
Assert (Test-ZlcNoProxy 'https://192.168.2.3/' '192.168.0.0/16') 'IPv4 NO_PROXY CIDR'
Assert (-not (Test-ZlcNoProxy 'https://192.169.2.3/' '192.168.0.0/16')) 'IPv4 CIDR boundary'
Assert (Test-ZlcNoProxy 'https://[2001:db8::1]/' '2001:db8::/32') 'IPv6 NO_PROXY CIDR'
$text = Protect-ZlcNetworkText 'fatal: https://person:secret@example.invalid/?token=secret&part=1 socks5://a:b@proxy:1'
Assert (-not $text.Contains('secret') -and -not $text.Contains('a:b')) 'Network errors must redact credentials'
$text = Protect-ZlcNetworkText "Unsupported proxy syntax in 'user:proxy-secret@proxy:bad'"
Assert ($text -eq "Unsupported proxy syntax in '[redacted]@proxy:bad'") 'Bare proxy credentials must be redacted in Git/curl diagnostics'
$text = Protect-ZlcNetworkText 'proxy user:p%40ss:word@proxy:1080 failed'
Assert ($text -eq 'proxy [redacted]@proxy:1080 failed') 'Bare proxy passwords with encoded or literal punctuation must be redacted'
$savedCount = $env:GIT_CONFIG_COUNT
try {
    $env:GIT_CONFIG_COUNT = '2'
    $childEnvironment = @{}
    Add-ZlcGitProxyEnvironment $childEnvironment @{ Address = 'http://name:password@fake:1'; InjectGit = $true }
    Assert ($childEnvironment['GIT_CONFIG_COUNT'] -eq '3' -and $childEnvironment['GIT_CONFIG_KEY_2'] -eq 'http.proxy') 'Append, do not replace existing process git config'
    Assert ($childEnvironment['GIT_CONFIG_VALUE_2'] -eq 'http://name:password@fake:1') 'Proxy lives only in child environment'
} finally { $env:GIT_CONFIG_COUNT = $savedCount }

$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('zlc-installer-network-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot | Out-Null
try {
    $fixture = Join-Path $testRoot 'fixture.exe'
    Add-Type -OutputAssembly $fixture -OutputType ConsoleApplication -TypeDefinition @'
using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
public class InstallerNetworkFixture {
    public static void Main(string[] args) {
        if (args[0] == "burst") {
            for (int i=0; i<20000; i++) { Console.Out.WriteLine("stdout"); Console.Error.WriteLine("stderr"); }
            Environment.Exit(17);
        } else if (args[0] == "orphan") {
            Process.Start(new ProcessStartInfo(Environment.GetCommandLineArgs()[0], "child \"" + args[1] + "\"") { UseShellExecute=false, CreateNoWindow=true });
            Thread.Sleep(200);
        } else if (args[0] == "child") {
            File.WriteAllText(args[1], Process.GetCurrentProcess().Id.ToString());
            Thread.Sleep(30000);
        } else { foreach (string argument in args) Console.WriteLine(argument); }
    }
}
'@
    $answer = Invoke-ZlcCapturedProcess $fixture @('burst') 10
    Assert ($answer.Code -eq 17 -and $answer.Lines.Count -eq 40000) 'Drain both full pipes concurrently and preserve nonzero exit'
    $answer = Invoke-ZlcCapturedProcess $fixture @('quotes', 'path with space\', 'a"b', '') 5
    Assert ($answer.Code -eq 0 -and $answer.Lines[1] -eq 'path with space\' -and $answer.Lines[2] -eq 'a"b') 'Windows argument quoting'
    $pidFile = Join-Path $testRoot 'child.pid'
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $answer = Invoke-ZlcCapturedProcess $fixture @('orphan', $pidFile) 2
    Assert ($answer.Code -eq -1 -and $watch.Elapsed.TotalSeconds -lt 8) 'Parent exit with inherited open pipe must be bounded'
    Assert (Test-Path -LiteralPath $pidFile) 'Fixture child ran'
    $childPid = [int](Get-Content -LiteralPath $pidFile)
    Start-Sleep -Milliseconds 200
    Assert (-not (Get-Process -Id $childPid -ErrorAction SilentlyContinue)) 'Timeout must kill the child tree, not only the git parent'

    # No production git or curl is launched below. Evaluate only these two
    # definitions from the script; all IO-dependent collaborators are fake.
    $parseErrors = $null; $tokens = $null
    $ast = [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot '..\fetch_engine.ps1'), [ref]$tokens, [ref]$parseErrors)
    Assert (-not $parseErrors.Count) 'fetch_engine must parse under Windows PowerShell 5.1'
    foreach ($name in @('Get-PullState', 'Get-GitReason')) {
        $definition = $ast.Find({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true)
        Invoke-Expression $definition.Extent.Text
    }
    function Find-Git { 'fake-git' }
    $script:scenario = 'normal'
    $script:networkCalls = 0
    function Invoke-Git([string]$Git, [string[]]$Arguments, [int]$Seconds, [string]$NetworkUrl = '', [switch]$SayNetwork, [string]$RemoteName = '') {
        $code = 0
        $lines = switch ($Arguments[0]) {
            'symbolic-ref' { 'feature/test' }
            'rev-parse' { 'other/tracked-branch' }
            'config' { 'other' }
            'remote' { 'https://test:fakepass@example.invalid/repo.git' }
            'fetch' {
                $script:networkCalls++
                Assert ($NetworkUrl -eq 'https://test:fakepass@example.invalid/repo.git') 'Resolve the configured remote, not a hardcoded repository'
                if ($script:scenario -eq 'fetch-fails') { $code = -1; 'fatal: https://test:fakepass@example.invalid/ timed out' }
            }
            'rev-list' { if ($script:scenario -eq 'ahead') { '1 2' } else { '0 2' } }
            'diff' {
                if ($script:scenario -eq 'diff-fails') { $code = 1; 'fatal: cannot inspect edits' }
                elseif ($Arguments.Count -eq 4 -or $script:scenario -eq 'blocked') { 'changed.py' }
            }
            'ls-files' { }
            default { throw 'Unexpected mock command' }
        }
        [pscustomobject]@{ Code = $code; Lines = @($lines) }
    }
    $state = Get-PullState 3
    Assert ($state.pullable -and $state.upstream -eq 'other/tracked-branch' -and $script:networkCalls -eq 1) 'One fetch against the actual upstream'
    foreach ($scenario in @('fetch-fails', 'ahead', 'blocked', 'diff-fails')) {
        $script:scenario = $scenario
        $state = Get-PullState 3
        Assert (-not $state.pullable -and $state.why) "Fail closed for $scenario"
        Assert (-not (($state | ConvertTo-Json -Depth 4).Contains('fakepass'))) 'Published pull status must not contain URL credentials'
    }
    $captured = ${function:Invoke-ZlcCapturedProcess}
    try {
        function Invoke-ZlcCapturedProcess($File, $Arguments, $Seconds, $Environment) {
            Assert ($Arguments[0] -eq '--disable') 'Do not let a curlrc override the selected route'
            Assert (-not (($Arguments -join ' ').Contains('proxy-secret'))) 'SOCKS credentials must not appear in process argv'
            Assert ($Environment['HTTPS_PROXY'] -eq 'socks5://user:proxy-secret@fake:1080') 'curl gets the explicit SOCKS policy in its environment'
            [pscustomobject]@{ Code = 0; Lines = @('HTTP/1.1 302 Found', 'Location: https://example.invalid/tag/v1') }
        }
        $answer = Invoke-ZlcSocksHttp 'https://example.invalid/latest' @{ Address = 'socks5://user:proxy-secret@fake:1080' } -Seconds 5 -HeadersOnly
        Assert ($answer.Location -eq 'https://example.invalid/tag/v1') 'Mock curl metadata response'
    } finally { ${function:Invoke-ZlcCapturedProcess} = $captured }
    $archive = Join-Path $testRoot 'archive.zip'
    [IO.File]::WriteAllBytes($archive, [byte[]]@(1,2,3))
    $asset = @{ size = 3; sha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() }
    Assert (Test-ZlcArchive $archive $asset) 'Verified cache may avoid another download'
    $asset.size = 4
    Assert (-not (Test-ZlcArchive $archive $asset)) 'Wrong-length cache is never reused'
    function Response([byte[]]$Bytes) {
        $stream = [pscustomobject]@{ Inner = (New-Object IO.MemoryStream(,$Bytes)); CanTimeout = $true; ReadTimeout = 0 }
        $stream | Add-Member ScriptMethod Read { param($buffer, $offset, $count) $this.Inner.Read($buffer, $offset, $count) }
        $stream | Add-Member ScriptMethod Dispose { $this.Inner.Dispose() }
        $response = [pscustomobject]@{ Stream = $stream; Closed = $false }
        $response | Add-Member ScriptMethod GetResponseStream { $this.Stream }
        $response | Add-Member ScriptMethod Dispose { $this.Closed = $true }
        $response
    }
    $part = Join-Path $testRoot 'download.part'
    $response = Response ([byte[]]@(1,2,3))
    Assert ((Save-ZlcResponse $response $part 3) -eq 3 -and $response.Closed) 'Mock body is saved and response closed'
    foreach ($case in @('oversize', 'deadline', 'no-timeout')) {
        $response = Response ([byte[]]@(1,2,3))
        $failed = $false
        try {
            if ($case -eq 'oversize') { Save-ZlcResponse $response $part 2 | Out-Null }
            elseif ($case -eq 'deadline') { Save-ZlcResponse $response $part 3 -Seconds 0 | Out-Null }
            else { $response.Stream.CanTimeout = $false; Save-ZlcResponse $response $part 3 | Out-Null }
        } catch { $failed = $true }
        Assert ($failed -and $response.Closed -and -not (Test-Path -LiteralPath $part)) "Body failure $case cleans its part and response"
    }
    'Offline installer network regressions passed.'
} finally {
    # This GUID-named directory was created above solely for these fixtures.
    if ([IO.Path]::GetFullPath($testRoot).StartsWith([IO.Path]::GetFullPath([IO.Path]::GetTempPath()), [StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $testRoot -Recurse -Force
    }
}
