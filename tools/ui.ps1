# Compile the interface independently; commit dist with its TypeScript sources.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'tools_root.ps1')
$ui = Join-Path (Get-ZlcCheckout) 'apps\desktop\ui'
$npm = Get-ZlcNpm
if (-not $npm) { throw 'Node.js is needed to compile TypeScript. Install Node.js or run tools\dev_setup.ps1.' }
$savedPath = $env:PATH
try {
    $env:PATH = "$(Split-Path -Parent $npm);$env:PATH"
    if (-not (Test-Path -LiteralPath (Join-Path $ui 'node_modules\typescript'))) {
        Invoke-Program { & $npm --prefix $ui ci --ignore-scripts }
        if ($LASTEXITCODE -ne 0) { throw 'npm ci failed.' }
    }
    Invoke-Program { & $npm --prefix $ui run build }
    if ($LASTEXITCODE -ne 0) { throw 'The UI build failed.' }
    $dist = Join-Path $ui 'dist'
    foreach ($item in Get-ChildItem -LiteralPath $dist -Recurse -File -Filter '*.js') {
        $relative = $item.FullName.Substring($dist.Length + 1) -replace '\.js$', '.ts'
        if (-not (Test-Path -LiteralPath (Join-Path (Join-Path $ui 'src') $relative))) {
            Remove-Item -LiteralPath $item.FullName
        }
    }
}
finally { $env:PATH = $savedPath }
