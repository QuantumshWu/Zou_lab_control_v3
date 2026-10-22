param(
    # experiment: Devices@Zou lab first, as v2's experiment.bat (Init opens the Task Console).
    # figure_viewer / pulse_editor: v2's standalone apps, that window alone (no apparatus).
    [ValidateSet('experiment', 'figure_viewer', 'pulse_editor')][string]$Mode = 'experiment',
    # A file to open at start (one dropped on the .bat): a Figure archive (.npz), or for
    # pulse_editor a pulse document (.json).
    [Parameter(Position = 0)][string]$Open,
    [switch]$Build,
    [int]$DebugPort = 0
)
# Start zlc-desktop from this checkout: a development build that is current
# for its sources, else the release in bin\engine, else a build when
# this machine has the toolchain -- see Get-ZlcProducts in tools_root.ps1.
# The app runs on its own, as when started from Explorer: this script (and the
# console of the .bat that ran it) ends once it is started. Only finding or
# building it fails here; the app says its own errors (a message box, and its
# log in .local\logs).
$ErrorActionPreference = 'Stop'
trap { Write-Host "ERROR: $($_.Exception.Message)"; exit 1 }
. (Join-Path $PSScriptRoot 'tools_root.ps1')
$executable = Join-Path (Get-ZlcProducts -Build:$Build) 'zlc-desktop.exe'
$savedPort = $env:ZLC_V3_DEBUG_PORT
try {
    if ($DebugPort -gt 0) { $env:ZLC_V3_DEBUG_PORT = [string]$DebugPort }
    else { Remove-Item Env:ZLC_V3_DEBUG_PORT -ErrorAction SilentlyContinue }
    $file = @()
    if ($Open) { $file = @((Resolve-Path -LiteralPath $Open).Path) }
    $arguments = @(switch ($Mode) {
        'figure_viewer' { @('--figure-viewer') + $file }
        'pulse_editor' { @('--pulse-editor') + $file }
        default { if ($Open) { @('--open') + $file } else { @() } }
    })
    # Devices and logic nodes run as plugin processes of the science environment, which the
    # executable finds and sets up itself, as when it is started from Explorer.
    $start = @{ FilePath = $executable }
    # Each argument quoted as the program reads it back (a trailing backslash doubled).
    if ($arguments.Count) { $start.ArgumentList = @($arguments | ForEach-Object { '"' + ($_ -replace '\\+$', '$0$0') + '"' }) }
    Start-Process @start
}
finally { $env:ZLC_V3_DEBUG_PORT = $savedPort }
