# Build and exercise the real DOM controls with a mock device transport, without hardware.
param([string]$Browser)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'tools_root.ps1')
if (-not $Browser) {
    $Browser = @(
        "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
        "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
        "$env:ProgramFiles\Google\Chrome\Application\chrome.exe"
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $Browser) { throw 'Install Edge or Chrome, or pass -Browser with its executable path.' }
& (Join-Path $PSScriptRoot 'ui.ps1')
$checkout = Get-ZlcCheckout
$testProfile = Join-Path $checkout '.local\ui-test-profile'
foreach ($test in Get-ChildItem -LiteralPath (Join-Path $checkout 'apps\desktop\ui\tests') -Filter '*.html') {
    $url = ([Uri]$test.FullName).AbsoluteUri
    $dom = (Invoke-Quiet {
        & $Browser --headless --disable-gpu --allow-file-access-from-files --no-first-run "--user-data-dir=$testProfile" --dump-dom --timeout=15000 --virtual-time-budget=10000 $url | Out-String
    }) -join "`n"
    $match = [regex]::Match($dom, '<pre id="result">(.*?)</pre>', 'Singleline')
    if (-not $match.Success) { throw "$($test.Name): browser returned no test result." }
    $result = [System.Net.WebUtility]::HtmlDecode($match.Groups[1].Value) | ConvertFrom-Json
    if (-not $result.passed) { throw "$($test.Name): $($result.error)" }
    Write-Host "$($test.Name): $($result.count) passed"
}
