# Warm the SLM rearrangement GPU code for the X15213 panel (the virtual SLM is
# the same size) in the environment the SLM plugin runs in, as a first
# preparation would: compile the CUDA module (CuPy keeps the binary) and
# measure, per band count, whether cuBLAS or a warp kernel computes the analysis
# product faster on this GPU, written where the solver reads it
# (<checkout>\.local\slm_gpu_cache\analysis.json; CuPy's in .local\cupy). The choices are
# keyed by GPU, driver, cuBLAS, panel and CUDA source; a preparation measures
# any band count its key lacks, so the plugin also re-warms on first use.
# installation.exe's Re-warm (bin\scripts\warm_slm_gpu.bat) runs this alone; its SLM GPU install after installing.
$ErrorActionPreference = 'Stop'
trap { Write-Host "ERROR: $($_.Exception.Message)"; exit 1 }
. (Join-Path $PSScriptRoot 'tools_root.ps1')
# No double quotes: Windows PowerShell passes them to a program unescaped.
$warm = @'
import sys, traceback
try:
    from zlc_atom.devices.slm.device import X15213_SHAPE_YX
    from zlc_atom.devices.slm.solver import warm_rearrangement_gpu
    print(warm_rearrangement_gpu(X15213_SHAPE_YX))
except Exception as error:
    traceback.print_exc()
    print(chr(10) + 'SLM GPU warmup failed: ' + type(error).__name__ + ': ' + str(error))
    sys.exit(1)
'@
Write-Host 'Warming the SLM rearrangement GPU code on this machine (close the SLM task console first) ...'
Invoke-ZlcPluginEnvironment {
    param($python)
    & $python -c $warm
    if ($LASTEXITCODE -ne 0) { throw 'The SLM GPU warmup failed (above).' }
}
