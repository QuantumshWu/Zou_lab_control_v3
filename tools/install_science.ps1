param(
    # runtime: what every PC's plugins import (numpy, scipy, Pillow) and the instrument links
    # (devices: serial, VISA, USB). pylon: only Basler's pypylon (~270 MB installed; any
    # version the environment sees counts); slm-gpu: only CuPy and CUDA for the SLM solver;
    # oracle: only a development machine's plotting oracle (tools\dev_setup.ps1) -- these
    # three into an existing environment.
    [ValidateSet('runtime', 'pylon', 'slm-gpu', 'oracle')][string]$Group = 'runtime',
    # Print what would be done and change nothing.
    [switch]$DryRun,
    # installation.exe's check: one JSON line (the environment, the Python it is or would be
    # made from, what each part of science\pyproject.toml lacks -- the rule an install
    # decides by); nothing is changed.
    [switch]$Status
)
# Set up <checkout>\.tools\science-env, the Python environment the plugin
# processes run in: a venv of the Python v2 runs with on this machine (found as
# v2's launchers find it, 3.13 or newer), holding only the third-party packages
# science\pyproject.toml names, at science\constraints.txt's versions. When that
# Python already has them (a v2 machine does), the venv uses them
# (--system-site-packages) and pip installs nothing. The checkout's own packages
# are not installed: plugins import them from science\src. The base Python is
# never modified, and no global PATH or other environment is changed.
$ErrorActionPreference = 'Stop'
trap { Write-Host "ERROR: $($_.Exception.Message)"; exit 1 }
. (Join-Path $PSScriptRoot 'tools_root.ps1')
$root = Get-ZlcCheckout
$science = Join-Path $root 'science'
$constraints = Join-Path $science 'constraints.txt'
$venv = Join-Path (Get-ZlcToolsRoot) 'science-env'
$python = Join-Path $venv 'Scripts\python.exe'
# v2's stored interpreter: one line, the python.exe (v2 _resolve_tools.bat reads the same file).
$stored = Join-Path $root '.zlc_python_path'

# pip and the venv see neither the caller's PYTHONHOME nor PYTHONPATH (what the plugins would
# see is read as they run, Invoke-ZlcAsPlugin), and pip refuses to run outside a venv: the base
# interpreter is never installed into.
Remove-Item Env:PYTHONHOME, Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:PIP_REQUIRE_VIRTUALENV = 'true'

# One candidate, believed only once it has run (a python.exe under WindowsApps is the Store's
# stub): a 64-bit CPython of v2's version, 3.13 or newer, with venv. Answers the base
# interpreter itself -- the one behind py.exe, a .bat or a venv -- or $null, a wrong one noted.
$script:Rejected = @()
function Test-BasePython([string]$Candidate, [string[]]$Arguments = @()) {
    if (-not $Candidate) { return $null }
    $Candidate = $Candidate.Trim()
    if ($Candidate -match '^call\s+(.+)$') { $Candidate = $Matches[1] }
    $Candidate = $Candidate.Trim().Trim('"')
    if (-not $Candidate -or $Candidate -like '*\WindowsApps\*' -or -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { return $null }
    $answer = @(Invoke-Quiet { & $Candidate @Arguments -c "import sys, venv, struct; print(getattr(sys, '_base_executable', '') or sys.executable); print(sys.implementation.name, *sys.version_info[:2], struct.calcsize('P') * 8)" })
    if ($LASTEXITCODE -ne 0 -or $answer.Count -lt 2) { return $null }
    $executable = "$($answer[0])".Trim()
    if ("$($answer[1])".Trim() -notmatch '^(\S+) (\d+) (\d+) (\d+)$') { return $null }
    if ($Matches[1] -ne 'cpython' -or [int]$Matches[2] -ne 3 -or [int]$Matches[3] -lt 13 -or $Matches[4] -ne '64') {
        $script:Rejected += "$executable ($($Matches[1]) $($Matches[2]).$($Matches[3]), $($Matches[4])-bit)"
        return $null
    }
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) { return $null }
    $executable
}

# The Python v2 runs with, in v2 _resolve_tools.bat's order: ZLC_PY_CMD and ZLC_FPGA_PYTHON
# (v2's), ZLC_PYTHON (this checkout's), the stored .zlc_python_path, python and py on PATH,
# conda's base (CONDA_PREFIX, CONDA_EXE, conda info --base), the Anaconda, Miniconda and
# Miniforge folders, then python.org's. PATH is where an interpreter announces itself, not
# where it lives: Anaconda recommends leaving PATH alone and ships no py.exe.
function Find-BasePython {
    function Found([string]$Path, [string]$Source) { [pscustomobject]@{ Path = $Path; Source = $Source } }
    foreach ($name in 'ZLC_PY_CMD', 'ZLC_FPGA_PYTHON', 'ZLC_PYTHON') {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ($value) { $hit = Test-BasePython $value; if ($hit) { return Found $hit $name } }
    }
    if (Test-Path -LiteralPath $stored -PathType Leaf) {
        $line = @(Get-Content -LiteralPath $stored -Encoding UTF8 -TotalCount 1)
        $text = if ($line.Count) { "$($line[0])".Trim() } else { '' }
        if ($text) {
            $hit = Test-BasePython $text
            if ($hit) { return Found $hit '.zlc_python_path' }
            Write-Host "Ignoring stale .zlc_python_path: $text"
        }
    }
    foreach ($name in 'python', 'py') {
        foreach ($candidate in @(Invoke-Quiet { & where.exe $name })) {
            $hit = Test-BasePython $candidate
            if (-not $hit -and $name -eq 'py') { $hit = Test-BasePython $candidate @('-3') }
            if ($hit) { return Found $hit "PATH ($name)" }
        }
    }
    if ($env:CONDA_PREFIX) { $hit = Test-BasePython (Join-Path $env:CONDA_PREFIX 'python.exe'); if ($hit) { return Found $hit 'CONDA_PREFIX' } }
    if ($env:CONDA_EXE) {
        $base = Split-Path -Parent (Split-Path -Parent $env:CONDA_EXE)
        if ($base) { $hit = Test-BasePython (Join-Path $base 'python.exe'); if ($hit) { return Found $hit 'CONDA_EXE' } }
    }
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        foreach ($base in @(Invoke-Quiet { & conda info --base })) {
            if ("$base".Trim()) { $hit = Test-BasePython (Join-Path "$base".Trim() 'python.exe'); if ($hit) { return Found $hit 'conda info --base' } }
        }
    }
    $roots = @($env:USERPROFILE, $env:LOCALAPPDATA, $(if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA 'Continuum' }),
               $env:ProgramData, $env:ProgramFiles, 'C:\') | Where-Object { $_ }
    foreach ($folder in $roots) {
        foreach ($distribution in 'anaconda3', 'miniconda3', 'miniforge3') {
            $hit = Test-BasePython (Join-Path (Join-Path $folder $distribution) 'python.exe')
            if ($hit) { return Found $hit 'the conda folders' }
        }
    }
    $installs = @($(if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA 'Programs\Python' }), $env:ProgramFiles, ${env:ProgramFiles(x86)}, 'C:\') | Where-Object { $_ }
    foreach ($folder in $installs) {
        if (-not (Test-Path -LiteralPath $folder -PathType Container)) { continue }
        foreach ($install in @(Get-ChildItem -LiteralPath $folder -Directory -Filter 'Python3*' -ErrorAction SilentlyContinue | Sort-Object Name)) {
            $hit = Test-BasePython (Join-Path $install.FullName 'python.exe')
            if ($hit) { return Found $hit 'the python.org folders' }
        }
    }
    $null
}

$fresh = -not (Test-Path -LiteralPath $python)
if ($fresh -and $Group -ne 'runtime') {
    # installation.exe's plan runs the environment's step first; a plan of it made nothing.
    if ($DryRun) { Write-Host "Dry run -- there is no science environment yet ($venv): once it is made, the $Group packages are installed into it."; return }
    throw "There is no science environment yet ($venv). Run installation.exe (Science environment) first."
}
$base = $null
if ($fresh) {
    $base = Find-BasePython
    if (-not $base) {
        $seen = if ($script:Rejected.Count) { " Found, but not a 64-bit CPython 3.13 or newer: $(($script:Rejected | Select-Object -Unique) -join '; ')." } else { '' }
        throw ("No Python 3.13 or newer was found (v2 needs the same). Searched: ZLC_PY_CMD, ZLC_FPGA_PYTHON, ZLC_PYTHON, .zlc_python_path, " +
               "python and py on PATH, conda's base and the Anaconda, Miniconda, Miniforge and python.org folders.$seen " +
               'Point ZLC_PY_CMD at the python.exe v2 runs with (setx ZLC_PY_CMD "C:\Users\you\anaconda3\python.exe") and run this again.')
    }
    $interpreter = $base.Path
}
else { $interpreter = $python }
$baseHome = Get-ZlcBaseHome $interpreter
$baseFolders = Get-ZlcBaseFolders $baseHome
# This process only (the window closes with it): pip and the venv find the base's DLLs. A conda
# Python (3.8+) searches PATH for an extension's DLLs only when asked to.
$env:PATH = (@($baseFolders) + @($env:PATH)) -join ';'
if ((Test-Path -LiteralPath (Join-Path $baseHome 'conda-meta') -PathType Container) -and -not $env:CONDA_DLL_SEARCH_MODIFICATION_ENABLE) { $env:CONDA_DLL_SEARCH_MODIFICATION_ENABLE = '1' }

# The parts of science\pyproject.toml: core = its dependencies, the rest its optional groups.
# (Python snippets here hold no double quotes: Windows PowerShell passes them to a program unescaped.)
$pyproject = Join-Path $science 'pyproject.toml'
$parts = @('core', 'devices', 'pylon', 'slm-gpu', 'oracle')
$groupParts = @(if ($Group -eq 'runtime') { 'core', 'devices' } else { $Group })

# What pip would find unmet of each part, read from the package metadata an interpreter sees
# when it runs as a plugin process (nothing is imported or changed): each requirement, and
# recursively each dependency its installed distribution declares (extras and markers as pip
# evaluates them), must be installed at a version its specifier and constraints.txt allow.
# Arguments: constraints.txt, pyproject.toml, the parts. One JSON line: the interpreter's version
# and, per part, its unmet items (none = pip has nothing to do).
$unmet = @'
import sys, json, tomllib
from importlib import metadata
try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name

def parse(text):
    try:
        return Requirement(text)
    except Exception:
        return None

def applies(requirement, extras):
    if requirement.marker is None:
        return True
    return any(requirement.marker.evaluate({'extra': extra}) for extra in ('',) + tuple(extras))

constrained = {}
for raw in open(sys.argv[1], encoding='utf-8'):
    pin = parse(raw.split('#', 1)[0].strip()) if raw.split('#', 1)[0].strip() else None
    if pin is not None and applies(pin, ()):
        constrained[canonicalize_name(pin.name)] = pin.specifier
project = tomllib.load(open(sys.argv[2], 'rb'))['project']
installed = {}
for dist in metadata.distributions():
    name = dist.metadata['Name']
    if name:
        installed.setdefault(canonicalize_name(name), dist)

def unmet_of(lines):
    unmet, done = [], set()
    pending = [requirement for requirement in map(parse, lines) if requirement is not None and applies(requirement, ())]
    while pending:
        requirement = pending.pop()
        name = canonicalize_name(requirement.name)
        key = (name, str(requirement.specifier), tuple(sorted(requirement.extras)))
        if key in done:
            continue
        done.add(key)
        dist = installed.get(name)
        if dist is None:
            unmet.append(requirement.name + ' is not installed')
            continue
        version = dist.version or '0'
        for specifier, why in ((requirement.specifier, 'wanted'), (constrained.get(name), 'constrained')):
            if specifier and not specifier.contains(version, prereleases=True):
                unmet.append(requirement.name + ' ' + version + ' (' + why + ' ' + str(specifier) + ')')
        for line in dist.requires or []:
            dependency = parse(line)
            if dependency is not None and applies(dependency, requirement.extras):
                dependency.marker = None
                pending.append(dependency)
    return list(dict.fromkeys(unmet))

parts = {part: unmet_of(project['dependencies'] if part == 'core' else project['optional-dependencies'][part]) for part in sys.argv[3:]}
print(json.dumps({'version': sys.version.split()[0], 'unmet': parts}))
'@
function Get-Unmet([string]$Interpreter, [string[]]$Of) {
    $answer = Invoke-ZlcAsPlugin $Interpreter { param($executable) @(& $executable -c $unmet $constraints $pyproject @Of) }
    if ($LASTEXITCODE -ne 0) { throw "$Interpreter could not report its packages." }
    ($answer | Where-Object { "$_".StartsWith('{') } | Select-Object -Last 1) | ConvertFrom-Json
}
# The unmet items of some parts, each once.
function Get-Lacking($Answer, [string[]]$Of) { @($Of | ForEach-Object { $Answer.unmet.$_ } | Select-Object -Unique) }
# What each part lacks: in the venv as it is, or (no venv yet) in the base a new venv would use.
$seen = Get-Unmet $interpreter $parts
# A new venv uses the base's packages when the base already satisfies numpy, scipy and Pillow
# with everything they need (a v2 machine satisfies every runtime requirement): pip then has at
# most the instrument links left to fetch -- with nothing unmet, it does not run at all.
$inherit = $fresh -and -not @(Get-Lacking $seen 'core').Count
$withoutPip = $false
if ($fresh) {
    # A base without ensurepip (some conda builds) still has pip, which an inheriting venv uses.
    Invoke-Quiet { & $interpreter -I -c 'import ensurepip' } | Out-Null
    $withoutPip = $LASTEXITCODE -ne 0
    if ($withoutPip -and -not $inherit) { throw "$interpreter has no ensurepip, so a venv of it gets no pip: point ZLC_PY_CMD at a full Python 3.13 installation." }
}
if ($Status) {
    $lacking = [ordered]@{}
    foreach ($part in $parts) { $lacking[$part] = @(if ($fresh -and -not $inherit) { 'not installed' } else { Get-Lacking $seen $part }) }
    [ordered]@{ environment = $venv; exists = -not $fresh; python = $interpreter; source = $(if ($base) { $base.Source } else { $null }); version = $seen.version
                inherit = $inherit; unmet = $lacking } | ConvertTo-Json -Compress -Depth 4
    return
}
$venvArguments = @('-m', 'venv')
if ($inherit) { $venvArguments += '--system-site-packages' }
if ($withoutPip) { $venvArguments += '--without-pip' }
$venvArguments += $venv
# The group's requirement lines (pip's arguments), read from pyproject.toml by the interpreter itself.
$read = @'
import sys, tomllib
project = tomllib.load(open(sys.argv[1], 'rb'))['project']
print(chr(10).join(line for part in sys.argv[2:] for line in (project['dependencies'] if part == 'core' else project['optional-dependencies'][part])))
'@
$requirements = @(& $interpreter -c $read $pyproject @groupParts)
if ($LASTEXITCODE -ne 0 -or -not $requirements.Count) { throw 'Could not read the requirements from science\pyproject.toml.' }
# What the group still needs: in the venv as it is, or (no venv yet) in the base a new venv inherits.
$needed = @(if (-not $fresh -or $inherit) { Get-Lacking $seen $groupParts } else { 'a venv of its own: every requirement' })

Write-Host "Science environment: $venv"
Write-Host "Group:               $Group"
if ($fresh) {
    Write-Host "Base interpreter:    $interpreter (found through $($base.Source); a new venv is made from it, the base is not changed)"
    if ($inherit) { Write-Host 'Base packages:       numpy, scipy and Pillow satisfied; the venv uses the base''s packages (--system-site-packages)' }
    else { Write-Host 'Base packages:       numpy, scipy and Pillow not all satisfied; the venv gets its own' }
}
else { Write-Host 'Existing environment: kept; only what is unmet is installed.' }
Write-Host "Base DLL folders:    $($baseFolders -join '; ')"
Write-Host "Requirements:        $($requirements -join ', ')"
Write-Host "Constraints:         $constraints"
if ($needed.Count) { Write-Host "Unmet:               $($needed -join '; ')" }
else { Write-Host 'Unmet:               nothing (requirements, their dependencies and the constraints): pip install is skipped' }
# The plugins' sources and the venv's packages are compiled to bytecode here, once, so that a first
# start compiles nothing (a base Python the venv uses is left as it is).
$compiled = @((Join-Path $science 'src'), (Join-Path $venv 'Lib\site-packages'))
if ($DryRun) {
    Write-Host ''
    Write-Host 'Dry run -- these steps would run:'
    if ($fresh) {
        Write-Host "  `"$interpreter`" $($venvArguments -join ' ')"
        Write-Host "  store `"$interpreter`" in $stored"
    }
    if ($needed.Count) { Write-Host "  `"$python`" -m pip install --constraint `"$constraints`" $($requirements -join ' ')" }
    Write-Host "  `"$python`" -m pip check$(if ($inherit -or -not $needed.Count) { ' (a warning only)' })"
    Write-Host "  `"$python`" -m compileall -q -j 0 $($compiled -join ' ')"
    Write-Host ('  the plugin check (python -m zlc_plugin describe zlc_atom' + $(if ($Group -eq 'slm-gpu') { ', a GPU dot product and FFT), then tools\warm_slm_gpu.ps1' } else { ')' }))
    return
}

if ($fresh) {
    Write-Host "[1/5] Making the venv ..."
    Invoke-Program { & $interpreter @venvArguments }
    if ($LASTEXITCODE -ne 0) { throw "Could not make the venv $venv." }
    # Stored as v2 stores it, so the next run finds the same Python first.
    [System.IO.File]::WriteAllText($stored, "$interpreter`r`n")
    $needed = @(Get-Lacking (Get-Unmet $python $groupParts) $groupParts)
}
Write-Host "[2/5] Installing $Group packages ..."
if ($Group -eq 'slm-gpu') { Write-Host 'This does not install or update the NVIDIA display driver.' }
$installed = $false
if ($needed.Count) {
    # As v2's installer: pip from the index, through the constraints.
    Invoke-Program { & $python -m pip install --disable-pip-version-check --constraint $constraints @requirements }
    if ($LASTEXITCODE -ne 0) { throw "pip install failed with code $LASTEXITCODE." }
    $installed = $true
}
else { Write-Host '  already satisfied (requirements, their dependencies and the constraints): nothing installed.' }
Write-Host '[3/5] Checking dependency consistency ...'
Invoke-Program { & $python -m pip check --disable-pip-version-check }
if ($LASTEXITCODE -ne 0) {
    # Only what pip just installed into a venv of its own is this script's to answer for: a venv
    # that uses the base's packages sees every package of the base (never changed here), and an
    # environment left as it was is the one that already ran. The plugin check below decides.
    $inherits = @(Get-Content -LiteralPath (Join-Path $venv 'pyvenv.cfg') -ErrorAction SilentlyContinue | Where-Object { $_ -match '^\s*include-system-site-packages\s*=\s*true' }).Count -gt 0
    if ($installed -and -not $inherits) { throw 'pip check found inconsistent packages.' }
    Write-Host '  (a warning only: these packages were left as they are; the plugin check decides)'
}
Write-Host '[4/5] Compiling the bytecode of the plugins and the packages ...'
Invoke-Program { & $python -m compileall -q -j 0 @compiled }
if ($LASTEXITCODE -ne 0) { Write-Host '  (a warning only: Python compiles the files above as it imports them)' }
Write-Host '[5/5] Checking the plugins in the environment they run in ...'
Invoke-ZlcPluginEnvironment {
    param($interpreter)
    $catalog = (& $interpreter -m zlc_plugin describe zlc_atom) -join "`n" | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or -not $catalog.types) { throw 'python -m zlc_plugin describe zlc_atom failed.' }
    foreach ($failure in @($catalog.errors)) { Write-Host ($failure | ConvertTo-Json -Depth 4) }
    if (@($catalog.errors).Count) { throw "$(@($catalog.errors).Count) plugin module(s) do not import (above)." }
    Write-Host "  $(@($catalog.types).Count) plugin types import."
    $probe = switch ($Group) {
        'runtime' { "import serial, pyvisa, pyvisa_py, usb, libusb_package; print('  instrument links import')" }
        'pylon' { "import pypylon.pylon; from importlib.metadata import version; print('  pypylon', version('pypylon'), 'imports')" }
        'oracle' { "import numba, matplotlib; print('  numba', numba.__version__, 'and matplotlib', matplotlib.__version__, 'import')" }
        'slm-gpu' { @'
import cupy as cp
from zlc_atom.devices.slm import solver
info = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
x = cp.arange(4, dtype=cp.float32)
value = float(cp.dot(x, x).item())
spectrum = cp.asnumpy(cp.fft.fft(x))
assert value == 14.0 and float(spectrum[0].real) == 6.0
print('  GPU', info['name'].decode(), 'CuPy', cp.__version__, 'dot + FFT: PASS')
'@ }
    }
    & $interpreter -c $probe
    if ($LASTEXITCODE -ne 0) { throw "The $Group check failed." }
}
# The GPU is warmed here once, as installation.exe's Re-warm (bin\scripts\warm_slm_gpu.bat) does alone.
if ($Group -eq 'slm-gpu') {
    & (Join-Path $PSScriptRoot 'warm_slm_gpu.ps1')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Host ''
Write-Host "Installed. Run experiment.exe."
