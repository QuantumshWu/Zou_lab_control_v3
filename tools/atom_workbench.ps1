param([switch]$Run,[switch]$RebuildNumerics)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'tools_root.ps1')
$taskRepository=Split-Path -Parent $PSScriptRoot
$taskApp=Join-Path $taskRepository 'apps\atom-workbench'
$taskOutput=Join-Path $taskRepository 'atom_workbench.exe'
$taskSavedPath=$env:PATH
$taskSavedTools=$env:ZLC_V3_TOOLS
try {
    if (-not $env:ZLC_V3_TOOLS -and -not (Test-Path -LiteralPath (Join-Path $taskRepository '.tools'))) {
        $taskCommon=(& git -C $taskRepository rev-parse --path-format=absolute --git-common-dir).Trim()
        $taskSharedTools=Join-Path (Split-Path -Parent $taskCommon) '.tools'
        if (Test-Path -LiteralPath $taskSharedTools) { $env:ZLC_V3_TOOLS=$taskSharedTools }
    }
    $taskNpm=Get-ZlcNpm
    if (-not $taskNpm) { throw 'Node.js is needed to build Atom Workbench. Run bin\scripts\dev_setup.bat first.' }
    $env:PATH="$(Split-Path -Parent $taskNpm);$env:PATH"
    $taskLock=Join-Path $taskApp 'package-lock.json'
    if (-not (Test-Path -LiteralPath $taskLock)) { throw 'The Atom Workbench sources are missing from apps\atom-workbench.' }
    $taskLockHash=(Get-FileHash -LiteralPath $taskLock -Algorithm SHA256).Hash
    $taskStamp=Join-Path $taskApp 'node_modules\.atom-workbench-lock'
    $taskInstalled=if (Test-Path -LiteralPath $taskStamp) { (Get-Content -LiteralPath $taskStamp -Raw).Trim() } else { '' }
    if ($taskInstalled -ne $taskLockHash) {
        Invoke-Program { & $taskNpm --prefix $taskApp ci --ignore-scripts }
        if ($LASTEXITCODE -ne 0) { throw 'Atom Workbench npm ci failed.' }
        Set-Content -LiteralPath $taskStamp -Value $taskLockHash -Encoding ascii
    }
    $taskId=(Get-Date -Format 'yyyyMMdd-HHmmss')+'-'+$PID
    $taskRelease=Join-Path $taskApp ('release\'+$taskId)
    $taskEvidence=Join-Path $taskApp ('.cache\releases\'+$taskId)
    New-Item -ItemType Directory -Force -Path $taskRelease,$taskEvidence | Out-Null
    $taskBuilt=Join-Path $taskRelease 'atom_workbench.exe'
    $taskArguments=@('--prefix',$taskApp,'run','package:single-exe','--','--out',$taskBuilt,'--evidence',$taskEvidence)
    if ($RebuildNumerics) { $taskArguments+='--rebuild-numerics' }
    Invoke-Program { & $taskNpm @taskArguments }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $taskBuilt)) {
        throw 'Atom Workbench build failed. The previous root EXE was not changed.'
    }
    # Publish only a completed build. Replace is atomic on this volume and
    # keeps the previous EXE when an open program prevents replacement.
    $taskPending=Join-Path $taskEvidence 'publish.exe'
    Copy-Item -LiteralPath $taskBuilt -Destination $taskPending
    if (Test-Path -LiteralPath $taskOutput) {
        [IO.File]::Replace($taskPending,$taskOutput,(Join-Path $taskEvidence 'previous.exe'),$true)
    } else { [IO.File]::Move($taskPending,$taskOutput) }
    Write-Host "Built standalone application: $taskOutput"
    Write-Host "Build provenance: $taskEvidence"
    if ($Run) { Start-Process -FilePath $taskOutput }
}
finally { $env:PATH=$taskSavedPath; $env:ZLC_V3_TOOLS=$taskSavedTools }
